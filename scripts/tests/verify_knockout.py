"""
verify_knockout.py

Verify the attention knockout mechanism:
1. Check that post-softmax attention weights to filler are zero during knockout
2. Compare use_cache=True vs use_cache=False on a few examples
3. Optionally run a full condition with use_cache=False for comparison

Usage:
    source /workspace/config/probing_env.sh
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python scripts/verify_knockout.py \
        --model-path /workspace/models/deepseek-v3-awq \
        --verify-n 10 \
        --run-nocache-k 25 \
        --max-examples 500 \
        --output-dir results/verify_knockout
"""

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
from tqdm import tqdm

import transformers.activations as _act
if not hasattr(_act, "PytorchGELUTanh"):
    _act.PytorchGELUTanh = _act.GELUTanh

sys.path.insert(0, str(Path(__file__).parent))
from extract_hidden_states import (
    build_messages_for_condition,
    find_filler_boundaries,
)
from extract_attention import load_model_eager


def extract_answer(text: str) -> Optional[int]:
    m = re.search(r"Answer:\s*(-?\d+)", text)
    if m:
        return int(m.group(1))
    m = re.search(r"(-?\d+)", text)
    if m:
        return int(m.group(1))
    return None


# ==============================================================================
# Test 1: Verify attention weights are zero at filler positions
# ==============================================================================

def verify_attention_zeros(model, tokenizer, input_ids, attention_mask,
                           filler_start, filler_end, max_new_tokens=5):
    """Run knockout generation and verify filler attention is zero at every layer/step."""
    seq_len = input_ids.shape[1]
    violations = []
    step_count = [0]

    def mask_and_verify_hook(module, args, kwargs):
        layer_idx = None
        for i, layer in enumerate(model.model.layers):
            if layer.self_attn is module:
                layer_idx = i
                break

        if 'attention_mask' in kwargs and kwargs['attention_mask'] is not None:
            mask = kwargs['attention_mask']
            if mask.dim() == 4 and mask.shape[-1] > filler_end:
                # Block filler
                mask = mask.clone()
                neg_inf = torch.finfo(mask.dtype).min
                mask[:, :, :, filler_start:filler_end + 1] = neg_inf
                kwargs['attention_mask'] = mask
        return args, kwargs

    def verify_output_hook(module, input, output):
        """Check post-forward: did the attention weights have zero at filler positions?"""
        layer_idx = None
        for i, layer in enumerate(model.model.layers):
            if layer.self_attn is module:
                layer_idx = i
                break

        if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
            attn_weights = output[1]  # (B, n_heads, q_len, kv_len)
            if attn_weights.shape[-1] > filler_end:
                filler_attn = attn_weights[0, :, :, filler_start:filler_end + 1]
                max_filler_attn = filler_attn.max().item()
                sum_filler_attn = filler_attn.sum().item()
                if max_filler_attn > 1e-6:
                    violations.append({
                        "step": step_count[0],
                        "layer": layer_idx,
                        "max_filler_attn": max_filler_attn,
                        "sum_filler_attn": sum_filler_attn,
                        "q_len": attn_weights.shape[2],
                        "kv_len": attn_weights.shape[-1],
                    })

    # Register hooks
    pre_handles = []
    post_handles = []
    for layer in model.model.layers:
        h1 = layer.self_attn.register_forward_pre_hook(mask_and_verify_hook, with_kwargs=True)
        h2 = layer.self_attn.register_forward_hook(verify_output_hook)
        pre_handles.append(h1)
        post_handles.append(h2)

    try:
        model.config.output_attentions = True
        with torch.no_grad():
            gen_output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
    finally:
        model.config.output_attentions = False
        for h in pre_handles + post_handles:
            h.remove()

    new_tokens = gen_output[0, seq_len:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    answer = extract_answer(raw)

    return raw, answer, violations


# ==============================================================================
# Test 2: use_cache=True vs use_cache=False comparison
# ==============================================================================

def generate_knockout_nocache(model, tokenizer, input_ids, attention_mask,
                               filler_start, filler_end, max_new_tokens=5):
    """Knockout with use_cache=False (full recomputation each step)."""
    device = input_ids.device
    seq_len = input_ids.shape[1]
    current_ids = input_ids.clone()
    current_mask = attention_mask.clone()

    for step in range(max_new_tokens):
        hook_handles = []

        def mask_hook(module, args, kwargs):
            if 'attention_mask' in kwargs and kwargs['attention_mask'] is not None:
                mask = kwargs['attention_mask']
                if mask.dim() == 4 and mask.shape[-1] > filler_end:
                    mask = mask.clone()
                    neg_inf = torch.finfo(mask.dtype).min
                    mask[:, :, :, filler_start:filler_end + 1] = neg_inf
                    kwargs['attention_mask'] = mask
            return args, kwargs

        for layer in model.model.layers:
            h = layer.self_attn.register_forward_pre_hook(mask_hook, with_kwargs=True)
            hook_handles.append(h)

        try:
            with torch.no_grad():
                outputs = model(
                    input_ids=current_ids,
                    attention_mask=current_mask,
                    use_cache=False,
                )
        finally:
            for h in hook_handles:
                h.remove()

        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        if next_token.item() == tokenizer.eos_token_id:
            break

        current_ids = torch.cat([current_ids, next_token], dim=1)
        current_mask = torch.cat([
            current_mask,
            torch.ones((1, 1), dtype=current_mask.dtype, device=device)
        ], dim=1)

    new_tokens = current_ids[0, seq_len:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    answer = extract_answer(raw)
    return raw, answer


def generate_knockout_cache(model, tokenizer, input_ids, attention_mask,
                             filler_start, filler_end, max_new_tokens=5):
    """Knockout with use_cache=True (KV cache, persistent hooks)."""
    from attention_knockout import generate_with_knockout
    return generate_with_knockout(
        model, tokenizer, input_ids, attention_mask,
        filler_start, filler_end,
        mode="all", max_new_tokens=max_new_tokens,
    )


# ==============================================================================
# Test 3: Full condition run with use_cache=False
# ==============================================================================

def run_nocache_condition(model, tokenizer, problems, few_shot, k,
                          input_device, max_new_tokens=5):
    """Run full knockout condition with use_cache=False."""
    results = []
    correct = 0

    for prob_idx, problem in enumerate(tqdm(problems, desc=f"k={k} ko (no cache)")):
        example_rng = random.Random(prob_idx)
        messages = build_messages_for_condition(
            few_shot[:5], problem, "dots", k, rng=example_rng
        )
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(full_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(input_device)
        attention_mask = inputs["attention_mask"].to(input_device)
        seq_len = input_ids.shape[1]

        _, filler_start, filler_end = find_filler_boundaries(tokenizer, input_ids, k)

        raw, answer = generate_knockout_nocache(
            model, tokenizer, input_ids, attention_mask,
            filler_start, filler_end, max_new_tokens=max_new_tokens,
        )

        is_correct = answer is not None and answer == problem["answer"]
        if is_correct:
            correct += 1

        results.append({
            "idx": prob_idx,
            "expected": problem["answer"],
            "predicted": answer,
            "correct": is_correct,
            "raw_response": raw,
        })

        del input_ids, attention_mask
        torch.cuda.empty_cache()

    acc = correct / len(problems)
    return results, acc, correct


def main():
    parser = argparse.ArgumentParser(description="Verify attention knockout")
    parser.add_argument("--model-path", type=str,
                        default="/workspace/models/deepseek-v3-awq")
    parser.add_argument("--dataset", type=Path,
                        default=Path("data/1hop_addition_dataset.json"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/verify_knockout"))
    parser.add_argument("--filler-k", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=5)

    # Test 1: verify attention zeros
    parser.add_argument("--verify-n", type=int, default=10,
                        help="Number of examples for attention weight verification")

    # Test 2: cache vs no-cache comparison
    parser.add_argument("--compare-n", type=int, default=20,
                        help="Number of examples for cache vs no-cache comparison")

    # Test 3: full no-cache run
    parser.add_argument("--run-nocache-k", type=int, default=None,
                        help="If set, run full knockout condition at this k with use_cache=False")
    parser.add_argument("--max-examples", type=int, default=500)

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.dataset) as f:
        dataset = json.load(f)
    few_shot = dataset["few_shot_facts"]
    problems = dataset["examples"]

    model, tokenizer = load_model_eager(args.model_path)
    input_device = next(model.parameters()).device

    # =========================================================================
    # Test 1: Verify attention weights are zero at filler positions
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"  Test 1: Verify filler attention is zero ({args.verify_n} examples)")
    print(f"{'='*60}")

    all_violations = []
    for i in range(args.verify_n):
        problem = problems[i]
        example_rng = random.Random(i)
        messages = build_messages_for_condition(
            few_shot[:5], problem, "dots", args.filler_k, rng=example_rng
        )
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(full_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(input_device)
        attention_mask = inputs["attention_mask"].to(input_device)

        _, filler_start, filler_end = find_filler_boundaries(
            tokenizer, input_ids, args.filler_k
        )

        raw, answer, violations = verify_attention_zeros(
            model, tokenizer, input_ids, attention_mask,
            filler_start, filler_end, max_new_tokens=args.max_new_tokens,
        )

        if violations:
            print(f"  [{i}] VIOLATIONS: {len(violations)} (answer={answer}, expected={problem['answer']})")
            for v in violations[:3]:
                print(f"      step={v['step']} L{v['layer']} max_attn={v['max_filler_attn']:.6f}")
            all_violations.extend(violations)
        else:
            print(f"  [{i}] OK (answer={answer}, expected={problem['answer']})")

        del input_ids, attention_mask
        torch.cuda.empty_cache()

    print(f"\n  Total violations: {len(all_violations)}")

    # =========================================================================
    # Test 2: Cache vs no-cache comparison
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"  Test 2: use_cache=True vs False ({args.compare_n} examples)")
    print(f"{'='*60}")

    mismatches = 0
    for i in range(args.compare_n):
        problem = problems[i]
        example_rng = random.Random(i)
        messages = build_messages_for_condition(
            few_shot[:5], problem, "dots", args.filler_k, rng=example_rng
        )
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(full_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(input_device)
        attention_mask = inputs["attention_mask"].to(input_device)

        _, filler_start, filler_end = find_filler_boundaries(
            tokenizer, input_ids, args.filler_k
        )

        raw_cache, ans_cache = generate_knockout_cache(
            model, tokenizer, input_ids, attention_mask,
            filler_start, filler_end, max_new_tokens=args.max_new_tokens,
        )
        raw_nocache, ans_nocache = generate_knockout_nocache(
            model, tokenizer, input_ids, attention_mask,
            filler_start, filler_end, max_new_tokens=args.max_new_tokens,
        )

        match = ans_cache == ans_nocache
        if not match:
            mismatches += 1
            print(f"  [{i}] MISMATCH cache={ans_cache} nocache={ans_nocache} expected={problem['answer']}")
        else:
            print(f"  [{i}] match={ans_cache} expected={problem['answer']}")

        del input_ids, attention_mask
        torch.cuda.empty_cache()

    print(f"\n  Mismatches: {mismatches}/{args.compare_n}")

    # =========================================================================
    # Test 3: Full no-cache run (optional)
    # =========================================================================
    if args.run_nocache_k is not None:
        k = args.run_nocache_k
        n = args.max_examples
        print(f"\n{'='*60}")
        print(f"  Test 3: Full knockout k={k} with use_cache=False ({n} examples)")
        print(f"{'='*60}")

        t0 = time.time()
        results, acc, correct = run_nocache_condition(
            model, tokenizer, problems[:n], few_shot, k,
            input_device, max_new_tokens=args.max_new_tokens,
        )
        elapsed = time.time() - t0

        print(f"\n  k={k} ko (no cache): {correct}/{n} ({acc:.1%}) [{elapsed:.0f}s]")

        with open(args.output_dir / f"k{k}_ko_nocache.json", "w") as f:
            json.dump({
                "k": k, "accuracy": acc, "correct": correct, "total": n,
                "time": elapsed, "use_cache": False, "results": results,
            }, f, indent=2)

    # Save summary
    summary = {
        "test1_violations": len(all_violations),
        "test2_mismatches": mismatches,
        "test2_n": args.compare_n,
    }
    with open(args.output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
