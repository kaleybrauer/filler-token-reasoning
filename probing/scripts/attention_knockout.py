"""
attention_knockout.py

Causal test: zero out attention to filler positions and measure accuracy.

Tests whether the model needs to READ from filler KV entries to get the
accuracy benefit. Unlike removing filler entirely, this preserves:
- Sequence length (same number of tokens)
- Positional encoding (answer_prompt at the same position)
- Filler KV entries exist (filler was processed), just not read

Three conditions:
1. dots_50: normal filler (control, ~75%)
2. dots_50 + knockout at answer_prompt only: block filler reading at the
   last prefill position, but generation tokens can still read filler
3. dots_50 + knockout at ALL positions: block filler reading everywhere
   (answer_prompt + all generation steps)

Requires eager attention (not flash). Uses monkey-patched forward method
to modify attention weights between softmax and V matmul.

Usage:
    source /workspace/config/probing_env.sh
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python probing/scripts/attention_knockout.py \
        --model-path /workspace/models/deepseek-v3-awq \
        --max-examples 500 \
        --output-dir probing/results/attention_knockout
"""

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

# Patch for autoawq compat
import transformers.activations as _act
if not hasattr(_act, "PytorchGELUTanh"):
    _act.PytorchGELUTanh = _act.GELUTanh

sys.path.insert(0, str(Path(__file__).parent))
from extract_hidden_states import (
    build_messages_for_condition,
    find_filler_boundaries,
    load_tokenizer,
)
from extract_attention import load_model_eager
from prompt_utils import build_messages, extract_answer


def find_pre_padding_boundaries(tokenizer, input_ids, k):
    """Find the start and end positions of pre-padding dots in the LAST user turn."""
    ids = input_ids[0].tolist()
    tokens = [tokenizer.decode([tid]) for tid in ids]
    seq_len = len(tokens)

    # Find the last "Padding" token
    last_padding_pos = -1
    for i in range(seq_len - 1, -1, -1):
        if "Padding" in tokens[i]:
            last_padding_pos = i
            break

    if last_padding_pos < 0:
        return -1, -1

    # Find the colon after "Padding"
    colon_pos = -1
    for i in range(last_padding_pos + 1, min(last_padding_pos + 3, seq_len)):
        if ":" in tokenizer.decode([ids[i]]):
            colon_pos = i
            break

    # First dot after colon
    pad_start = colon_pos + 1 if colon_pos >= 0 else last_padding_pos + 1

    # Find "Question" token after padding
    question_pos = -1
    for i in range(pad_start, seq_len):
        if "Question" in tokenizer.decode([ids[i]]):
            question_pos = i
            break

    # Last dot before Question (scan backward past whitespace)
    pad_end = pad_start
    if question_pos >= 0:
        for i in range(question_pos - 1, pad_start - 1, -1):
            decoded = tokenizer.decode([ids[i]])
            if decoded.strip() and "." in decoded:
                pad_end = i
                break

    return pad_start, pad_end


# ==============================================================================
# Position ID correction
# ==============================================================================

def build_corrected_position_ids(
    seq_len: int,
    filler_start: int,
    filler_end: int,
    device: torch.device,
) -> torch.Tensor:
    """Build position IDs that undo the positional shift from filler tokens.

    Filler tokens keep their natural position IDs (doesn't matter if knockout
    is also applied). Post-filler tokens get the position IDs they would have
    had without filler — as if filler tokens were invisible to RoPE.

    Example with filler at positions 479-528 (50 tokens):
        Normal:    [0, 1, ..., 478, 479, ..., 528, 529, 530, 531]
        Corrected: [0, 1, ..., 478, 479, ..., 528, 479, 480, 481]
    """
    position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
    n_filler = filler_end - filler_start + 1
    # Shift post-filler positions back by n_filler
    position_ids[filler_end + 1:] -= n_filler
    return position_ids.unsqueeze(0)  # (1, seq_len)


# ==============================================================================
# Attention knockout via attention mask modification
# ==============================================================================

def generate_with_knockout(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    filler_start: int,
    filler_end: int,
    mode: str = "all",
    max_new_tokens: int = 20,
    correct_positions: bool = False,
) -> Tuple[str, Optional[int]]:
    """
    Generate with attention to filler positions knocked out.

    Uses persistent pre-forward hooks on self_attn to modify the 4D attention
    mask, setting filler columns to -inf so they get zero weight after softmax.
    Works with KV cache (model.generate) — hooks fire on both prefill and
    each generation step.

    Modes:
        "all": block filler attention at every step (prefill + generation)
        "answer_only": block only during prefill, generation can read filler

    If correct_positions=True, overrides position IDs so post-filler tokens
    use the same positions they would have had without filler (undoes the
    RoPE shift from filler tokens).
    """
    seq_len = input_ids.shape[1]

    def mask_hook(module, args, kwargs):
        if 'attention_mask' in kwargs and kwargs['attention_mask'] is not None:
            mask = kwargs['attention_mask']
            if mask.dim() == 4 and mask.shape[-1] > filler_end:
                should_block = False
                is_prefill = (mask.shape[2] > 1)  # q_len > 1 means prefill

                if mode == "all":
                    should_block = True
                elif mode == "answer_only" and is_prefill:
                    should_block = True

                if should_block:
                    mask = mask.clone()
                    neg_inf = torch.finfo(mask.dtype).min
                    if mode == "answer_only":
                        # Only block the last query position from reading filler
                        mask[:, :, -1, filler_start:filler_end + 1] = neg_inf
                    else:
                        mask[:, :, :, filler_start:filler_end + 1] = neg_inf
                    kwargs['attention_mask'] = mask
        return args, kwargs

    # Register hooks
    hook_handles = []
    for layer in model.model.layers:
        h = layer.self_attn.register_forward_pre_hook(mask_hook, with_kwargs=True)
        hook_handles.append(h)

    # If correcting positions, zero out filler in the 2D attention_mask.
    # This makes prepare_inputs_for_generation compute position_ids via cumsum
    # that skip filler positions (correcting RoPE), AND blocks attention to filler
    # in the 4D causal mask (equivalent to knockout).
    if correct_positions:
        attention_mask = attention_mask.clone()
        attention_mask[0, filler_start:filler_end + 1] = 0

    try:
        with torch.no_grad():
            gen_output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
    finally:
        for h in hook_handles:
            h.remove()

    new_tokens = gen_output[0, seq_len:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    m = re.search(r"Answer:\s*(-?\d+)", raw)
    if m:
        return raw, int(m.group(1))
    m = re.search(r"(-?\d+)", raw)
    if m:
        return raw, int(m.group(1))
    return raw, None



# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Attention knockout causal test")
    parser.add_argument("--model-path", type=str,
                        default="/workspace/models/deepseek-v3-awq")
    parser.add_argument("--dataset", type=Path,
                        default=Path("probing/data/1hop_addition_dataset.json"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("probing/results/knockout_bare"))
    parser.add_argument("--max-examples", type=int, default=500)
    parser.add_argument("--filler-k", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=5)
    parser.add_argument("--only", nargs="+", default=None,
                        help="Run only these conditions (by name)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.dataset) as f:
        dataset = json.load(f)
    few_shot = dataset["few_shot_facts"]
    problems = dataset["examples"][:args.max_examples]
    print(f"Loaded {len(problems)} problems")

    model, tokenizer = load_model_eager(args.model_path)
    first_param = next(model.parameters())
    input_device = first_param.device

    conditions = [
        # (name, k, ko_mode, correct_pos, keep_last_n)
        # keep_last_n: leave the last N filler tokens unblocked
        (f"dots_{args.filler_k}_ko_all", args.filler_k, "all", False, 0),
        (f"dots_{args.filler_k}_ko_corrpos", args.filler_k, "all", True, 0),
        ("ko_k1", 1, "all", False, 0),
        ("ko_k5", 5, "all", False, 0),
        ("ko_k10", 10, "all", False, 0),
        ("ko_k25", 25, "all", False, 0),
        # Partial knockout: block most dots, keep last N visible
        (f"ko_k{args.filler_k}_keep5", args.filler_k, "all", False, 5),
        (f"ko_k{args.filler_k}_keep1", args.filler_k, "all", False, 1),
    ]

    if args.only:
        conditions = [c for c in conditions if c[0] in args.only]

    all_results = {}

    for cond_name, k, ko_mode, correct_pos, keep_last_n in conditions:
        results_file = args.output_dir / f"{cond_name}.json"
        if results_file.exists():
            with open(results_file) as f:
                existing = json.load(f)
            if existing["total"] >= len(problems):
                print(f"\n  {cond_name}: already done, skipping")
                all_results[cond_name] = existing
                continue

        print(f"\n{'='*60}")
        print(f"  {cond_name} (k={k}, ko={ko_mode}, corrpos={correct_pos}, keep_last={keep_last_n})")
        print(f"{'='*60}")

        results = []
        correct = 0
        t0 = time.time()

        for prob_idx, problem in enumerate(tqdm(problems, desc=cond_name)):
            messages = build_messages(few_shot[:5], problem, "dots", k)
            full_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(full_text, return_tensors="pt")
            input_ids = inputs["input_ids"].to(input_device)
            attention_mask = inputs["attention_mask"].to(input_device)
            seq_len = input_ids.shape[1]

            _, filler_start, filler_end = find_filler_boundaries(
                tokenizer, input_ids, k
            )

            # For partial knockout, shrink the blocked range
            ko_filler_end = filler_end - keep_last_n if keep_last_n > 0 else filler_end

            if ko_mode and filler_start >= 0 and filler_start <= ko_filler_end:
                raw, answer = generate_with_knockout(
                    model, tokenizer, input_ids, attention_mask,
                    filler_start, ko_filler_end,
                    mode=ko_mode,
                    max_new_tokens=args.max_new_tokens,
                    correct_positions=correct_pos,
                )
            else:
                with torch.no_grad():
                    gen_output = model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                    )
                new_tokens = gen_output[0, seq_len:]
                raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                answer = extract_answer(raw)

            is_correct = answer is not None and answer == problem["answer"]
            if is_correct:
                correct += 1

            results.append({
                "idx": prob_idx,
                "expected": problem["answer"],
                "predicted": answer,
                "correct": is_correct,
                "raw_response": raw,
                "fact_value": problem["fact_value"],
                "x": problem["x"],
            })

            del input_ids, attention_mask
            torch.cuda.empty_cache()

        elapsed = time.time() - t0
        acc = correct / len(problems)
        print(f"\n  {cond_name}: {correct}/{len(problems)} ({acc:.1%})")
        print(f"  Time: {elapsed:.0f}s ({elapsed/len(problems):.1f}s/example)")

        cond_result = {
            "accuracy": acc,
            "correct": correct,
            "total": len(problems),
            "time": elapsed,
            "results": results,
        }
        all_results[cond_name] = cond_result

        with open(results_file, "w") as f:
            json.dump(cond_result, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for cond_name, data in all_results.items():
        print(f"  {cond_name:<25} {data['accuracy']:.1%}")

    summary = {k: {sk: sv for sk, sv in v.items() if sk != "results"}
               for k, v in all_results.items()}
    with open(args.output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
