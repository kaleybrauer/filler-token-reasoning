#!/usr/bin/env python3
"""Evaluate DeepSeek V3 (4-bit) on 1-hop addition: baseline vs dot filler.

Usage:
    source /workspace/config/probing_env.sh
    uv run --project /workspace/filler-token-reasoning/probing python \
        probing/scripts/evaluate_1hop.py \
        --max-examples 200 --filler-k 250
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MODEL_PATH = "/workspace/models/deepseek-v3-gptq"
DATASET_PATH = Path("/workspace/filler-token-reasoning/probing/data/1hop_addition_dataset.json")
RESULTS_DIR = Path("/workspace/filler-token-reasoning/probing/results")


# ---------------------------------------------------------------------------
# Prompt building (mirrors generate_1hop_dataset.py)
# ---------------------------------------------------------------------------

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


def build_user_turn(fact_phrase: str, x: int, k: int) -> str:
    question_line = f"Question: What is {fact_phrase} plus {x}?"
    if k > 0:
        filler = " ".join(["."] * k)
        return f"{question_line}\n\nFiller: {filler}\n\nAnswer:"
    else:
        return f"{question_line}\n\nAnswer:"


def build_messages(few_shot_items: list[dict], target: dict, k: int) -> list[dict]:
    """Build chat messages for a single evaluation example."""
    messages = [{"role": "system", "content": build_system_message(k)}]

    # Few-shot examples (5 of 10, to keep prompt shorter)
    for fs in few_shot_items[:5]:
        user_content = build_user_turn(fs["fact_phrase"], fs["x"], k)
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": f"Answer: {fs['answer']}"})

    # Target question
    user_content = build_user_turn(target["fact_phrase"], target["x"], k)
    messages.append({"role": "user", "content": user_content})

    return messages


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_answer(text: str) -> int | None:
    """Extract the numeric answer from model output."""
    # Try "Answer: <number>" format first
    m = re.search(r"Answer:\s*(-?\d+)", text)
    if m:
        return int(m.group(1))
    # Fall back to first integer in the output
    m = re.search(r"(-?\d+)", text)
    if m:
        return int(m.group(1))
    return None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_path: str):
    """Load a pre-quantized GPTQ/AWQ model, or fall back to bitsandbytes NF4."""
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    n_gpus = torch.cuda.device_count()
    print(f"GPUs available: {n_gpus}")
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}, {props.total_memory / 1e9:.1f} GB")

    # Check if this is a pre-quantized model (GPTQ/AWQ) by looking at config
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_path)
    is_prequantized = hasattr(config, "quantization_config") and config.quantization_config is not None

    t0 = time.time()
    if is_prequantized:
        print(f"Detected pre-quantized model, loading directly...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
    else:
        print(f"Loading with bitsandbytes NF4 quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
        max_memory = {
            i: f"{int(torch.cuda.get_device_properties(i).total_memory * 0.95 / 1e9)}GiB"
            for i in range(n_gpus)
        }
        max_memory["cpu"] = "300GiB"
        offload_dir = "/workspace/.offload"
        Path(offload_dir).mkdir(parents=True, exist_ok=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            max_memory=max_memory,
            torch_dtype=torch.bfloat16,
            offload_folder=offload_dir,
        )

    model.eval()
    print(f"Model loaded in {time.time() - t0:.1f}s")

    for i in range(n_gpus):
        alloc = torch.cuda.memory_allocated(i) / 1e9
        print(f"  GPU {i}: {alloc:.1f} GB allocated")

    return model, tokenizer


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate_condition(
    model,
    tokenizer,
    few_shot_items: list[dict],
    examples: list[dict],
    k: int,
    max_new_tokens: int = 20,
) -> list[dict]:
    """Evaluate all examples under a single filler condition."""
    results = []

    for ex in tqdm(examples, desc=f"k={k}"):
        messages = build_messages(few_shot_items, ex, k)
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        response = tokenizer.decode(
            outputs[0][input_len:], skip_special_tokens=True
        ).strip()

        predicted = extract_answer(response)
        correct = predicted == ex["answer"]

        results.append({
            "idx": ex["idx"],
            "expected": ex["answer"],
            "predicted": predicted,
            "correct": correct,
            "raw_response": response,
            "input_tokens": input_len,
            "fact_value": ex["fact_value"],
            "x": ex["x"],
            "kind": ex["kind"],
        })

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--outdir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--max-examples", type=int, default=200)
    parser.add_argument("--filler-k", type=int, default=250,
                        help="Number of dot filler tokens for the filler condition")
    args = parser.parse_args()

    # Load dataset
    with open(args.dataset) as f:
        dataset = json.load(f)
    few_shot = dataset["few_shot_facts"]
    examples = dataset["examples"][:args.max_examples]
    print(f"Using {len(examples)} examples, {len(few_shot)} few-shot facts")

    # Load model
    model, tokenizer = load_model(args.model_path)

    # Quick sanity check
    print("\nSanity check: generating '2+2' ...")
    sanity_msgs = [{"role": "user", "content": "What is 2+2? Answer with just the number."}]
    sanity_text = tokenizer.apply_chat_template(sanity_msgs, tokenize=False, add_generation_prompt=True)
    sanity_inputs = tokenizer(sanity_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        sanity_out = model.generate(**sanity_inputs, max_new_tokens=10, do_sample=False)
    sanity_resp = tokenizer.decode(sanity_out[0][sanity_inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"  Response: {sanity_resp!r}")

    # Print a sample prompt for each condition
    sample = examples[0]
    for cond_k, label in [(0, "baseline"), (args.filler_k, f"dots k={args.filler_k}")]:
        msgs = build_messages(few_shot, sample, cond_k)
        txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        n_tok = len(tokenizer(txt)["input_ids"])
        print(f"\nSample prompt ({label}): {n_tok} tokens")

    # Evaluate baseline (k=0)
    print(f"\n{'='*60}")
    print(f"Evaluating BASELINE (k=0) on {len(examples)} examples...")
    print(f"{'='*60}")
    baseline_results = evaluate_condition(model, tokenizer, few_shot, examples, k=0)
    baseline_acc = sum(r["correct"] for r in baseline_results) / len(baseline_results)
    print(f"Baseline accuracy: {baseline_acc:.1%} ({sum(r['correct'] for r in baseline_results)}/{len(baseline_results)})")

    # Evaluate filler (k=args.filler_k)
    print(f"\n{'='*60}")
    print(f"Evaluating FILLER (k={args.filler_k} dots) on {len(examples)} examples...")
    print(f"{'='*60}")
    filler_results = evaluate_condition(model, tokenizer, few_shot, examples, k=args.filler_k)
    filler_acc = sum(r["correct"] for r in filler_results) / len(filler_results)
    print(f"Filler accuracy: {filler_acc:.1%} ({sum(r['correct'] for r in filler_results)}/{len(filler_results)})")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Baseline (k=0):            {baseline_acc:.1%}")
    print(f"  Filler (k={args.filler_k} dots):  {filler_acc:.1%}")
    print(f"  Δ accuracy:                {filler_acc - baseline_acc:+.1%}")

    # Save results
    args.outdir.mkdir(parents=True, exist_ok=True)
    output = {
        "config": {
            "model_path": args.model_path,
            "dataset": str(args.dataset),
            "max_examples": args.max_examples,
            "filler_k": args.filler_k,
            "n_few_shot": len(few_shot),
        },
        "summary": {
            "baseline_accuracy": baseline_acc,
            "filler_accuracy": filler_acc,
            "delta": filler_acc - baseline_acc,
            "n_examples": len(examples),
        },
        "baseline_results": baseline_results,
        "filler_results": filler_results,
    }
    out_path = args.outdir / "eval_1hop_baseline_vs_filler.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
