"""
eval_counting.py

Evaluate counting filler (1 2 3 4 5 ...) at various k values.

Usage:
    source /workspace/config/probing_env.sh
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python probing/scripts/eval_counting.py \
        --model-path /workspace/models/deepseek-v3-awq \
        --max-examples 500 \
        --output-dir probing/results/counting
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

import transformers.activations as _act
if not hasattr(_act, "PytorchGELUTanh"):
    _act.PytorchGELUTanh = _act.GELUTanh

sys.path.insert(0, str(Path(__file__).parent))
from extract_hidden_states import load_model
from prompt_utils import build_messages, extract_answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/workspace/models/deepseek-v3-awq")
    parser.add_argument("--dataset", type=Path,
                        default=Path("probing/data/1hop_addition_dataset.json"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("probing/results/counting"))
    parser.add_argument("--max-examples", type=int, default=500)
    parser.add_argument("--max-new-tokens", type=int, default=5)
    parser.add_argument("--ks", type=str, default="5,10,25,50")
    args = parser.parse_args()

    k_values = [int(x) for x in args.ks.split(",")]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.dataset) as f:
        dataset = json.load(f)
    few_shot = dataset["few_shot_facts"]
    problems = dataset["examples"][:args.max_examples]
    print(f"Loaded {len(problems)} problems")

    model, tokenizer = load_model(args.model_path)
    first_param = next(model.parameters())
    input_device = first_param.device

    all_results = {}

    for k in k_values:
        name = f"counting_k{k}"
        results_file = args.output_dir / f"{name}.json"
        if results_file.exists():
            with open(results_file) as f:
                existing = json.load(f)
            if existing["total"] >= len(problems):
                print(f"\n  {name}: already done, skipping")
                all_results[name] = existing
                continue

        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

        results = []
        correct = 0
        t0 = time.time()

        for prob_idx, problem in enumerate(tqdm(problems, desc=name)):
            messages = build_messages(few_shot[:5], problem, "counting", k)
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
            })

            del input_ids, attention_mask
            torch.cuda.empty_cache()

        elapsed = time.time() - t0
        acc = correct / len(problems)
        print(f"\n  {name}: {correct}/{len(problems)} ({acc:.1%})")

        cond_result = {
            "accuracy": acc,
            "correct": correct,
            "total": len(problems),
            "time": elapsed,
            "results": results,
        }
        all_results[name] = cond_result

        with open(results_file, "w") as f:
            json.dump(cond_result, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, data in all_results.items():
        print(f"  {name:<25} {data['accuracy']:.1%}")

    with open(args.output_dir / "summary.json", "w") as f:
        json.dump({k: {sk: sv for sk, sv in v.items() if sk != "results"}
                   for k, v in all_results.items()}, f, indent=2)

    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
