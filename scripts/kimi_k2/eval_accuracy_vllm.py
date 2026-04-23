"""
eval_accuracy_vllm.py

Behavioral eval only — no hidden-state extraction. Runs vLLM batched generation
on one condition and prints/saves accuracy.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import torch

# Reuse helpers
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SCRIPTS_DIR))

from extract.extract_hidden_states import CONDITIONS, build_messages_for_condition  # noqa: E402


def parse_answer(text: str) -> int | None:
    m = re.search(r"Answer:\s*(-?\d+)", text)
    if m:
        return int(m.group(1))
    m = re.search(r"(-?\d+)", text)
    return int(m.group(1)) if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/workspace/models/kimi-k2-w4a16", type=Path)
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--condition", default=None,
                    choices=list(CONDITIONS.keys()),
                    help="Single condition (legacy). Use --conditions for multi.")
    ap.add_argument("--conditions", nargs="+", default=None,
                    choices=list(CONDITIONS.keys()),
                    help="Multiple conditions; loads vLLM once and evals each.")
    ap.add_argument("--dataset-type", default="2fact", choices=["1hop", "2fact"])
    ap.add_argument("--max-problems", type=int, default=1000)
    ap.add_argument("--output-file", type=Path,
                    default=Path("results/kimi_k2_baseline_accuracy.json"))
    ap.add_argument("--tensor-parallel-size", type=int, default=None)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.97)
    ap.add_argument("--max-model-len", type=int, default=1024)
    args = ap.parse_args()

    if args.conditions:
        cond_list = list(args.conditions)
    elif args.condition:
        cond_list = [args.condition]
    else:
        ap.error("Pass --condition or --conditions")
    print(f"Conditions: {cond_list}")

    with open(args.dataset) as f:
        data = json.load(f)
    problems = data["examples"] if isinstance(data, dict) else data
    few_shot = data.get("few_shot_facts", []) if isinstance(data, dict) else []
    if args.max_problems:
        problems = problems[:args.max_problems]
    print(f"  {len(problems)} problems, {len(few_shot)} few-shot facts")

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    tp = args.tensor_parallel_size or torch.cuda.device_count()
    print(f"\nLoading vLLM (TP={tp}) from {args.model_path}...")
    t0 = time.time()
    llm = LLM(
        model=str(args.model_path),
        tensor_parallel_size=tp,
        dtype="bfloat16",
        quantization="compressed-tensors",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        trust_remote_code=True,
        disable_custom_all_reduce=True,
    )
    print(f"Loaded in {time.time() - t0:.0f}s")

    tokenizer = llm.get_tokenizer()
    sp = SamplingParams(temperature=0, max_tokens=20, detokenize=True)

    all_results: dict[str, dict] = {}

    for cond_name in cond_list:
        k, filler_type = CONDITIONS[cond_name]
        print(f"\n--- Condition: {cond_name} (k={k}, filler_type={filler_type}) ---")

        prompts = []
        for i, p in enumerate(problems):
            msgs = build_messages_for_condition(
                few_shot[:5], p, filler_type, k,
                rng=random.Random(i), dataset_type=args.dataset_type,
            )
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            ids = tokenizer(text)["input_ids"]
            prompts.append(TokensPrompt(prompt_token_ids=ids))

        t_gen = time.time()
        outs = llm.generate(prompts, sp)
        elapsed = time.time() - t_gen

        per_problem = []
        correct = 0
        for i, (p, o) in enumerate(zip(problems, outs)):
            resp = o.outputs[0].text.strip()
            model_ans = parse_answer(resp)
            is_corr = (model_ans == p["answer"]) if model_ans is not None else False
            correct += int(is_corr)
            per_problem.append({
                "idx": i, "answer": p["answer"],
                "model_response": resp, "model_answer": model_ans, "correct": is_corr,
            })
        acc = correct / len(problems)
        print(f"  Generation: {elapsed:.0f}s ({elapsed / len(problems):.2f}s/example)")
        print(f"  Accuracy: {correct}/{len(problems)} = {acc:.1%}")

        all_results[cond_name] = {
            "condition": cond_name,
            "k": k,
            "filler_type": filler_type,
            "n_problems": len(problems),
            "n_correct": correct,
            "accuracy": acc,
            "gen_time_s": elapsed,
            "per_problem": per_problem,
        }

    summary = {
        "model_path": str(args.model_path),
        "dataset": str(args.dataset),
        "dataset_type": args.dataset_type,
        "n_problems": len(problems),
        "conditions": all_results,
        "accuracy_by_condition": {c: all_results[c]["accuracy"] for c in cond_list},
    }

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Summary ===")
    for c in cond_list:
        r = all_results[c]
        print(f"  {c}: {r['n_correct']}/{r['n_problems']} = {r['accuracy']:.1%}")
    print(f"Saved {args.output_file}")


if __name__ == "__main__":
    main()
