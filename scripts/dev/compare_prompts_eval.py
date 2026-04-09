"""
compare_prompts_eval.py

Run a systematic comparison of prompt formats with bare assistant turns
(assistant just outputs the number, no "Answer: " prefix).

Conditions:
  1. baseline (k=0, no filler)
  2. dots k=1,10,50 (few-shots match k)
  3. format_only with few-shot k=10,50
  4. pre_padding with k=10,50
  5. no_newline with few-shot k=10,50

Usage:
    source /workspace/config/probing_env.sh
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python probing/scripts/compare_prompts_eval.py \
        --model-path /workspace/models/deepseek-v3-awq \
        --max-examples 500 \
        --output-dir probing/results/prompt_comparison
"""

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

import transformers.activations as _act
if not hasattr(_act, "PytorchGELUTanh"):
    _act.PytorchGELUTanh = _act.GELUTanh

sys.path.insert(0, str(Path(__file__).parent))
from extract_hidden_states import load_tokenizer
from extract_hidden_states import load_model
from prompt_utils import build_messages, extract_answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/workspace/models/deepseek-v3-awq")
    parser.add_argument("--dataset", type=Path,
                        default=Path("probing/data/1hop_addition_dataset.json"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("probing/results/prompt_comparison"))
    parser.add_argument("--max-examples", type=int, default=500)
    parser.add_argument("--max-new-tokens", type=int, default=5)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.dataset) as f:
        dataset = json.load(f)
    few_shot = dataset["few_shot_facts"]
    problems = dataset["examples"][:args.max_examples]
    print(f"Loaded {len(problems)} problems")

    model, tokenizer = load_model(args.model_path)
    first_param = next(model.parameters())
    input_device = first_param.device

    conditions = [
        # (name, mode, k)
        ("baseline", "baseline", 0),
        ("dots_k1", "dots", 1),
        ("dots_k10", "dots", 10),
        ("dots_k50", "dots", 50),
        ("format_only_fs10", "format_only", 10),
        ("format_only_fs50", "format_only", 50),
        ("pre_padding_k10", "pre_padding", 10),
        ("pre_padding_k50", "pre_padding", 50),
        ("no_newline_fs10", "no_newline", 10),
        ("no_newline_fs50", "no_newline", 50),
    ]

    all_results = {}

    for cond_name, mode, k in conditions:
        results_file = args.output_dir / f"{cond_name}.json"
        if results_file.exists():
            with open(results_file) as f:
                existing = json.load(f)
            if existing["total"] >= len(problems):
                print(f"\n  {cond_name}: already done ({existing['total']} examples), skipping")
                all_results[cond_name] = existing
                continue

        print(f"\n{'='*60}")
        print(f"  {cond_name} (mode={mode}, k={k})")
        print(f"{'='*60}")

        # Print first example prompt for verification
        prob0 = problems[0]
        msgs0 = build_messages(few_shot[:5], prob0, mode, k)
        print(f"  Example target: {msgs0[-1]['content']!r}")
        print(f"  Example asst[0]: {msgs0[2]['content']!r}")

        results = []
        correct = 0
        t0 = time.time()

        for prob_idx, problem in enumerate(tqdm(problems, desc=cond_name)):
            messages = build_messages(few_shot[:5], problem, mode, k)
            full_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(full_text, return_tensors="pt")
            input_ids = inputs["input_ids"].to(input_device)
            attention_mask = inputs["attention_mask"].to(input_device)
            seq_len = input_ids.shape[1]

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
        acc = data["accuracy"] if isinstance(data["accuracy"], float) else data["accuracy"]
        print(f"  {cond_name:<25} {acc:.1%}")

    summary = {k: {sk: sv for sk, sv in v.items() if sk != "results"}
               for k, v in all_results.items()}
    with open(args.output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
