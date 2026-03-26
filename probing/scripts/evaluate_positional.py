"""
evaluate_positional.py

Positional encoding test: add pre-padding dots BEFORE the question (not between
question and answer) and measure if accuracy improves. If it does, the filler
effect is at least partly positional (RoPE), not computational.

Uses transformers (not vLLM) for generation.

Usage:
    source /workspace/config/probing_env.sh
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python probing/scripts/evaluate_positional.py \
        --filler-k 50 --pre-padding 50 \
        --outdir probing/results/positional_test
"""

import argparse
import json
import re
import time
from pathlib import Path

import torch
from tqdm import tqdm

import transformers.activations as _act
if not hasattr(_act, "PytorchGELUTanh"):
    _act.PytorchGELUTanh = _act.GELUTanh

from extract_hidden_states import load_model, load_tokenizer


def build_system_message(k: int) -> str:
    base = (
        "You will be given a question. Answer immediately using the format "
        "'Answer: [ANSWER]' where [ANSWER] is just the number, nothing else. "
        "No explanation, no words, no reasoning, just the number."
    )
    if k > 0:
        base += (
            f" After the question, there will be {k} filler tokens "
            f"(a sequence of dots) to give you extra space to process "
            f"the problem before answering."
        )
    return base


def build_user_turn(fact_phrase: str, x: int, k: int, pre_padding: int = 0) -> str:
    question_line = f"Question: What is {fact_phrase} plus {x}?"
    prefix = ""
    if pre_padding > 0:
        padding = " ".join(["."] * pre_padding)
        prefix = f"Padding: {padding}\n\n"
    if k > 0:
        filler = " ".join(["."] * k)
        return f"{prefix}{question_line}\n\nFiller: {filler}\n\nAnswer:"
    else:
        return f"{prefix}{question_line}\n\nAnswer:"


def build_messages(few_shot_items, target, k, pre_padding=0):
    messages = [{"role": "system", "content": build_system_message(k)}]
    for fs in few_shot_items[:5]:
        user_content = build_user_turn(fs["fact_phrase"], fs["x"], k, pre_padding)
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": f"Answer: {fs['answer']}"})
    user_content = build_user_turn(target["fact_phrase"], target["x"], k, pre_padding)
    messages.append({"role": "user", "content": user_content})
    return messages


def extract_answer(text: str):
    m = re.search(r"Answer:\s*(-?\d+)", text)
    if m:
        return int(m.group(1))
    m = re.search(r"(-?\d+)", text)
    if m:
        return int(m.group(1))
    return None


def run_eval(model, tokenizer, examples, few_shot, k, pre_padding, device):
    """Run evaluation and return per-example results."""
    results = []
    correct = 0
    total = 0

    for ex in tqdm(examples, desc=f"k={k}, pre_pad={pre_padding}"):
        messages = build_messages(few_shot, ex, k, pre_padding)
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(full_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            gen = model.generate(
                input_ids, attention_mask=attention_mask,
                max_new_tokens=20, do_sample=False,
            )

        new_tokens = gen[0][input_ids.shape[1]:]
        raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        answer = extract_answer(raw)
        is_correct = (answer == ex["answer"])
        if is_correct:
            correct += 1
        total += 1

        results.append({
            "idx": ex.get("idx", total - 1),
            "fact_value": ex["fact_value"],
            "x": ex["x"],
            "answer": ex["answer"],
            "model_answer": answer,
            "model_correct": is_correct,
            "raw": raw,
        })

        del input_ids, attention_mask
        torch.cuda.empty_cache()

    return results, correct, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/workspace/models/deepseek-v3-awq")
    parser.add_argument("--dataset", type=Path,
                        default=Path("probing/data/1hop_addition_dataset.json"))
    parser.add_argument("--outdir", type=Path,
                        default=Path("probing/results/positional_test"))
    parser.add_argument("--max-examples", type=int, default=500)
    parser.add_argument("--filler-k", type=int, default=50)
    parser.add_argument("--pre-padding", type=int, default=50)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    with open(args.dataset) as f:
        data = json.load(f)
    few_shot = data["few_shot_facts"]
    examples = data["examples"][:args.max_examples]
    for i, ex in enumerate(examples):
        if "idx" not in ex:
            ex["idx"] = i
    print(f"Loaded {len(examples)} examples")

    model, tokenizer = load_model(args.model_path)
    device = next(model.parameters()).device

    conditions = {
        f"pre_pad_{args.pre_padding}": (0, args.pre_padding),
        f"dots_{args.filler_k}_pre_pad_{args.pre_padding}": (args.filler_k, args.pre_padding),
    }

    all_results = {}
    for cond_name, (k, pre_pad) in conditions.items():
        print(f"\n{'='*60}")
        print(f"Condition: {cond_name} (k={k}, pre_padding={pre_pad})")
        print(f"{'='*60}")

        t0 = time.time()
        results, correct, total = run_eval(
            model, tokenizer, examples, few_shot, k, pre_pad, device
        )
        elapsed = time.time() - t0

        accuracy = correct / total
        print(f"  Accuracy: {correct}/{total} = {accuracy:.1%}")
        print(f"  Time: {elapsed:.0f}s ({elapsed/total:.1f}s/example)")

        all_results[cond_name] = {
            "k": k,
            "pre_padding": pre_pad,
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "results": results,
        }

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for cond_name, data in all_results.items():
        print(f"  {cond_name:<40} {data['accuracy']:.1%} ({data['correct']}/{data['total']})")

    # Save
    with open(args.outdir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {args.outdir}/results.json")


if __name__ == "__main__":
    main()
