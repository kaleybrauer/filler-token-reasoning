#!/usr/bin/env python3
"""Evaluate DeepSeek V3 (quantized) on 1-hop addition using vLLM.

Usage:
    source /workspace/config/probing_env.sh
    uv run --project /workspace/filler-token-reasoning/probing python \
        probing/scripts/evaluate_1hop_vllm.py \
        --model-path /workspace/models/deepseek-v3-awq \
        --max-examples 200 --filler-k 250
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

# CUDA 12.8 forward compatibility for CUDA 12.4 drivers
_compat = "/usr/local/cuda-12.8/compat"
if os.path.isdir(_compat):
    os.environ["LD_LIBRARY_PATH"] = _compat + ":" + os.environ.get("LD_LIBRARY_PATH", "")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MODEL_PATH = "/workspace/models/deepseek-v3-awq"
DATASET_PATH = Path("/workspace/filler-token-reasoning/probing/data/1hop_addition_dataset.json")
RESULTS_DIR = Path("/workspace/filler-token-reasoning/probing/results")


# ---------------------------------------------------------------------------
# Prompt building (identical to evaluate_1hop.py)
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
    messages = [{"role": "system", "content": build_system_message(k)}]
    for fs in few_shot_items[:5]:
        user_content = build_user_turn(fs["fact_phrase"], fs["x"], k)
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": f"Answer: {fs['answer']}"})
    user_content = build_user_turn(target["fact_phrase"], target["x"], k)
    messages.append({"role": "user", "content": user_content})
    return messages


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_answer(text: str) -> int | None:
    m = re.search(r"Answer:\s*(-?\d+)", text)
    if m:
        return int(m.group(1))
    m = re.search(r"(-?\d+)", text)
    if m:
        return int(m.group(1))
    return None


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
    parser.add_argument("--tensor-parallel-size", type=int, default=None,
                        help="TP size (default: auto-detect from GPU count)")
    parser.add_argument("--pipeline-parallel-size", type=int, default=None,
                        help="PP size (default: auto-detect from GPU count)")
    args = parser.parse_args()

    import torch
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    # Load dataset
    with open(args.dataset) as f:
        dataset = json.load(f)
    few_shot = dataset["few_shot_facts"]
    examples = dataset["examples"][:args.max_examples]
    print(f"Using {len(examples)} examples, {len(few_shot)} few-shot facts")

    # Auto-detect parallelism: 128 heads must be divisible by TP
    n_gpus = torch.cuda.device_count()
    if args.tensor_parallel_size and args.pipeline_parallel_size:
        tp_size = args.tensor_parallel_size
        pp_size = args.pipeline_parallel_size
    elif n_gpus >= 4:
        tp_size = 4
        pp_size = 1
    elif n_gpus == 2:
        tp_size = 2
        pp_size = 1
    else:
        tp_size = 1
        pp_size = n_gpus
    print(f"Using TP={tp_size}, PP={pp_size} on {n_gpus} GPUs")

    # Load tokenizer (for chat template formatting)
    print(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # For PP > 1, vLLM 0.8.x requires AsyncLLMEngine (served via HTTP).
    # For TP-only, we can use the sync LLM class directly.
    use_server = pp_size > 1

    if use_server:
        import subprocess, signal, atexit
        from openai import OpenAI

        port = 8192
        print(f"Starting vLLM server (PP={pp_size} requires async engine)...")
        server_cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", args.model_path,
            "--tensor-parallel-size", str(tp_size),
            "--pipeline-parallel-size", str(pp_size),
            "--max-model-len", "4096",
            "--gpu-memory-utilization", "0.95",
            "--enforce-eager",
            "--dtype", "float16",
            "--port", str(port),
            "--disable-log-requests",
        ]
        server_log = open("/tmp/vllm_server.log", "w")
        server_proc = subprocess.Popen(server_cmd, stdout=server_log, stderr=subprocess.STDOUT)
        atexit.register(lambda: server_proc.kill())

        # Wait for server to be ready
        client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="dummy")
        print("Waiting for server to load model...")
        t0 = time.time()
        for attempt in range(600):  # up to 10 min
            time.sleep(1)
            try:
                client.models.list()
                break
            except Exception:
                # Check if server crashed
                if server_proc.poll() is not None:
                    with open("/tmp/vllm_server.log") as f:
                        log = f.read()
                    print(f"Server crashed! Output:\n{log[-3000:]}")
                    sys.exit(1)
        else:
            print("Server failed to start in 10 minutes")
            sys.exit(1)
        print(f"Server ready in {time.time() - t0:.0f}s")

        def generate_batch(prompts):
            results = []
            for prompt in prompts:
                resp = client.completions.create(
                    model=args.model_path,
                    prompt=prompt,
                    max_tokens=20,
                    temperature=0,
                )
                results.append(resp.choices[0].text.strip())
            return results
    else:
        from vllm import LLM, SamplingParams
        print(f"Loading model from {args.model_path}...")
        t0 = time.time()
        llm = LLM(
            model=args.model_path,
            tensor_parallel_size=tp_size,
            trust_remote_code=True,
            max_model_len=4096,
            gpu_memory_utilization=0.95,
            enforce_eager=True,
            dtype="float16",
        )
        print(f"Model loaded in {time.time() - t0:.1f}s")

        sampling_params = SamplingParams(max_tokens=20, temperature=0)

        def generate_batch(prompts):
            outputs = llm.generate(prompts, sampling_params)
            return [out.outputs[0].text.strip() for out in outputs]

    # Sanity check
    print("\nSanity check: '2+2' ...")
    sanity_msgs = [{"role": "user", "content": "What is 2+2? Answer with just the number."}]
    sanity_prompt = tokenizer.apply_chat_template(sanity_msgs, tokenize=False, add_generation_prompt=True)
    sanity_results = generate_batch([sanity_prompt])
    print(f"  Response: {sanity_results[0]!r}")

    # Helper to run a condition
    def evaluate_condition(k: int) -> list[dict]:
        prompts = []
        for ex in examples:
            msgs = build_messages(few_shot, ex, k)
            prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)

        sample_ids = tokenizer(prompts[0])["input_ids"]
        print(f"  Sample prompt: {len(sample_ids)} tokens")

        t0 = time.time()
        responses = generate_batch(prompts)
        elapsed = time.time() - t0
        print(f"  Generated {len(responses)} responses in {elapsed:.1f}s ({elapsed/len(responses):.2f}s/example)")

        results = []
        for ex, response in zip(examples, responses):
            predicted = extract_answer(response)
            correct = predicted == ex["answer"]
            results.append({
                "idx": ex["idx"],
                "expected": ex["answer"],
                "predicted": predicted,
                "correct": correct,
                "raw_response": response,
                "fact_value": ex["fact_value"],
                "x": ex["x"],
                "kind": ex["kind"],
            })
        return results

    # Evaluate baseline (k=0)
    print(f"\n{'='*60}")
    print(f"Evaluating BASELINE (k=0) on {len(examples)} examples...")
    print(f"{'='*60}")
    baseline_results = evaluate_condition(k=0)
    baseline_acc = sum(r["correct"] for r in baseline_results) / len(baseline_results)
    print(f"Baseline accuracy: {baseline_acc:.1%} ({sum(r['correct'] for r in baseline_results)}/{len(baseline_results)})")

    # Evaluate filler (k=args.filler_k)
    print(f"\n{'='*60}")
    print(f"Evaluating FILLER (k={args.filler_k} dots) on {len(examples)} examples...")
    print(f"{'='*60}")
    filler_results = evaluate_condition(k=args.filler_k)
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
            "tensor_parallel_size": tp_size,
            "pipeline_parallel_size": pp_size,
            "engine": "vllm",
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
