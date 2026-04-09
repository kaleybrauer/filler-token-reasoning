"""
knockout_truncation.py

Truncation curve with attention knockout: for each k in [1, 5, 10, 25, 50],
run filler with k dots but block all attention to filler positions.

If accuracy increases with k even when filler can't be read → positional effect.
If accuracy stays flat → the original truncation curve was from filler computation.

Also runs normal filler (no knockout) at each k for direct comparison on
the same engine.

Usage:
    source /workspace/config/probing_env.sh
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python scripts/knockout_truncation.py \
        --model-path /workspace/models/deepseek-v3-awq \
        --max-examples 500 \
        --output-dir results/knockout_truncation
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
from extract_hidden_states import find_filler_boundaries
from extract_hidden_states import load_model
from attention_knockout import generate_with_knockout
from prompt_utils import build_messages, extract_answer


def run_condition(model, tokenizer, problems, few_shot, k, knockout,
                  input_device, max_new_tokens=5, format_only=False,
                  format_only_k=50):
    """Run one (k, knockout) condition. Returns list of result dicts."""
    results = []
    correct = 0

    desc = "format_only" if format_only else f"k={k} {'ko' if knockout else 'normal'}"
    for prob_idx, problem in enumerate(tqdm(
        problems, desc=desc
    )):
        if format_only:
            messages = build_messages(few_shot[:5], problem, "format_only", format_only_k)
        else:
            messages = build_messages(few_shot[:5], problem, "dots", k)
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(full_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(input_device)
        attention_mask = inputs["attention_mask"].to(input_device)
        seq_len = input_ids.shape[1]

        filler_start, filler_end = -1, -1
        if k > 0:
            _, filler_start, filler_end = find_filler_boundaries(
                tokenizer, input_ids, k
            )

        if knockout and filler_start >= 0:
            raw, answer = generate_with_knockout(
                model, tokenizer, input_ids, attention_mask,
                filler_start, filler_end,
                mode="all",
                max_new_tokens=max_new_tokens,
            )
        else:
            with torch.no_grad():
                gen_output = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
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
        })

        del input_ids, attention_mask
        torch.cuda.empty_cache()

    acc = correct / len(problems)
    return results, acc, correct


def main():
    parser = argparse.ArgumentParser(description="Truncation curve with attention knockout")
    parser.add_argument("--model-path", type=str,
                        default="/workspace/models/deepseek-v3-awq")
    parser.add_argument("--dataset", type=Path,
                        default=Path("data/1hop_addition_dataset.json"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/knockout_truncation"))
    parser.add_argument("--max-examples", type=int, default=500)
    parser.add_argument("--ks", type=str, default="0,1,5,10,25,50",
                        help="Comma-separated k values")
    parser.add_argument("--max-new-tokens", type=int, default=5)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    k_values = [int(x) for x in args.ks.split(",")]

    with open(args.dataset) as f:
        dataset = json.load(f)
    few_shot = dataset["few_shot_facts"]
    problems = dataset["examples"][:args.max_examples]
    print(f"Loaded {len(problems)} problems, k_values={k_values}")

    model, tokenizer = load_model(args.model_path)
    input_device = next(model.parameters()).device

    all_results = {}

    # Build condition list: (name, k, knockout, format_only)
    conditions = []
    for k in k_values:
        conditions.append((f"k{k}_normal", k, False, False))
        if k == 0:
            # format_only: filler format with 0 dots, few-shots have full filler
            conditions.append(("format_only_k0", 0, False, True))
        else:
            conditions.append((f"k{k}_ko", k, True, False))

    for cond_name, k, knockout, format_only in conditions:
        # Resume support
        results_file = args.output_dir / f"{cond_name}.json"
        if results_file.exists():
            with open(results_file) as f:
                existing = json.load(f)
            if existing["total"] >= len(problems):
                print(f"\n  {cond_name}: already done ({existing['accuracy']:.1%}), skipping")
                all_results[cond_name] = existing
                continue

        print(f"\n{'='*60}")
        print(f"  {cond_name}")
        print(f"{'='*60}")

        t0 = time.time()
        results, acc, correct = run_condition(
            model, tokenizer, problems, few_shot, k, knockout,
            input_device, args.max_new_tokens,
            format_only=format_only,
            format_only_k=max(k_values),
        )
        elapsed = time.time() - t0

        print(f"  {cond_name}: {correct}/{len(problems)} ({acc:.1%}) "
              f"[{elapsed:.0f}s, {elapsed/len(problems):.1f}s/ex]")

        cond_data = {
            "k": k,
            "knockout": knockout,
            "format_only": format_only,
            "accuracy": acc,
            "correct": correct,
            "total": len(problems),
            "time": elapsed,
            "results": results,
        }
        all_results[cond_name] = cond_data

        with open(results_file, "w") as f:
            json.dump(cond_data, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Condition':<16}  {'Normal':>8}  {'Knockout':>8}  {'Δ':>8}")
    print(f"{'-'*48}")
    for k in k_values:
        normal = all_results.get(f"k{k}_normal", {})
        ko = all_results.get(f"k{k}_ko", {})
        n_acc = f"{normal['accuracy']:.1%}" if normal else "-"
        k_acc = f"{ko['accuracy']:.1%}" if ko else "-"
        if normal and ko:
            delta = f"{ko['accuracy'] - normal['accuracy']:+.1%}"
        else:
            delta = "-"
        print(f"k={k:<13}  {n_acc:>8}  {k_acc:>8}  {delta:>8}")
        if k == 0:
            fmt = all_results.get("format_only_k0", {})
            f_acc = f"{fmt['accuracy']:.1%}" if fmt else "-"
            print(f"{'format_only':<16}  {f_acc:>8}  {'-':>8}  {'-':>8}")

    with open(args.output_dir / "summary.json", "w") as f:
        summary = {k: {sk: sv for sk, sv in v.items() if sk != "results"}
                   for k, v in all_results.items()}
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
