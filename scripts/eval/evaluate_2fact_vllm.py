#!/usr/bin/env python3
"""Evaluate DeepSeek V3 (quantized) on 2-fact addition using vLLM.

Task: "What is [atomic number of X] plus [atomic number of Y]?"
Model must retrieve two facts and add them.

Usage:
    python scripts/evaluate_2fact_vllm.py \
        --filler-type dots --filler-k 0,1,5,10,25,50,100,200

    python scripts/evaluate_2fact_vllm.py \
        --filler-type counting --filler-k 0,10,25,50,100
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# CUDA 12.8 forward compatibility for CUDA 12.4 drivers
_compat = "/usr/local/cuda-12.8/compat"
if os.path.isdir(_compat):
    os.environ["LD_LIBRARY_PATH"] = _compat + ":" + os.environ.get("LD_LIBRARY_PATH", "")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.generate_2fact_dataset import build_prompt_messages_2fact
from prompt_utils import extract_answer

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MODEL_PATH = "/workspace/models/deepseek-v3-awq"
DATASET_PATH = Path("data/2fact_addition_dataset.json")
RESULTS_DIR = Path("results/eval_2fact")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--outdir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--max-examples", type=int, default=500)
    parser.add_argument("--filler-k", type=str, default="0,1,5,10,25,50,100,200",
                        help="Filler lengths, comma-separated (0 = baseline)")
    parser.add_argument("--filler-type", type=str, default="dots",
                        choices=["dots", "counting", "alphabet"])
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--pipeline-parallel-size", type=int, default=None)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--dtype", type=str, default="float16",
                        help="Model dtype (float16 for V3, bfloat16 for V3.2)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    args = parser.parse_args()

    import torch
    from transformers import AutoTokenizer

    # Load dataset
    with open(args.dataset) as f:
        dataset = json.load(f)
    few_shot = dataset["few_shot_facts"]
    examples = dataset["examples"][:args.max_examples]
    print(f"Loaded {len(examples)} examples, {len(few_shot)} few-shot facts")

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

    # Load tokenizer
    print(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    use_server = pp_size > 1

    if use_server:
        import subprocess
        import signal
        import atexit
        from openai import OpenAI

        port = 8192
        client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="dummy")
        server_proc = None
        try:
            client.models.list()
            print(f"vLLM server already running on port {port}")
        except Exception:
            print(f"Starting vLLM server (PP={pp_size} requires async engine)...")
            server_cmd = [
                sys.executable, "-m", "vllm.entrypoints.openai.api_server",
                "--model", args.model_path,
                "--tensor-parallel-size", str(tp_size),
                "--pipeline-parallel-size", str(pp_size),
                "--max-model-len", str(args.max_model_len),
                "--gpu-memory-utilization", str(args.gpu_memory_utilization),
                "--enforce-eager",
                "--dtype", args.dtype,
                "--port", str(port),
                "--disable-log-requests",
                "--trust-remote-code",
            ]
            server_log = open("/tmp/vllm_server.log", "w")
            server_proc = subprocess.Popen(
                server_cmd, stdout=server_log, stderr=subprocess.STDOUT,
                env={**os.environ, "VLLM_USE_V1": "0"},
                start_new_session=True,
            )
            def _cleanup_server():
                if server_proc and server_proc.poll() is None:
                    # Kill entire process group to catch PP child workers
                    try:
                        os.killpg(os.getpgid(server_proc.pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            atexit.register(_cleanup_server)
            signal.signal(signal.SIGTERM, lambda *_: (_cleanup_server(), sys.exit(0)))

        client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="dummy")
        print("Waiting for server to load model...")
        t0 = time.time()
        for attempt in range(600):
            time.sleep(1)
            try:
                client.models.list()
                break
            except Exception:
                if server_proc and server_proc.poll() is not None:
                    with open("/tmp/vllm_server.log") as f:
                        log = f.read()
                    print(f"Server crashed! Output:\n{log[-3000:]}")
                    sys.exit(1)
        else:
            print("Server failed to start in 10 minutes")
            sys.exit(1)
        print(f"Server ready in {time.time() - t0:.0f}s")

        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _single_request(prompt):
            resp = client.completions.create(
                model=args.model_path,
                prompt=prompt,
                max_tokens=20,
                temperature=0,
            )
            return resp.choices[0].text.strip()

        def generate_batch(prompts, max_workers=32):
            results = [None] * len(prompts)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_single_request, p): i
                           for i, p in enumerate(prompts)}
                for future in as_completed(futures):
                    results[futures[future]] = future.result()
            return results
    else:
        from vllm import LLM, SamplingParams
        print(f"Loading model from {args.model_path}...")
        t0 = time.time()
        llm = LLM(
            model=args.model_path,
            tensor_parallel_size=tp_size,
            trust_remote_code=True,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=True,
            dtype=args.dtype,
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

    k_values = sorted(set(int(x) for x in args.filler_k.split(",")))
    filler_type = args.filler_type

    def evaluate_condition(k: int) -> list[dict]:
        prompts = []
        for ex in examples:
            msgs = build_prompt_messages_2fact(few_shot, ex, filler_type, k)
            prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)

        sample_ids = tokenizer(prompts[0])["input_ids"]
        print(f"  Sample prompt: {len(sample_ids)} tokens")

        t0 = time.time()
        responses = generate_batch(prompts)
        elapsed = time.time() - t0
        print(f"  Generated {len(responses)} in {elapsed:.1f}s ({elapsed/len(responses):.2f}s/ex)")

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
                "fact_value_1": ex["fact_value_1"],
                "fact_value_2": ex["fact_value_2"],
            })
        return results

    # Evaluate all k values
    all_results = {}
    for k in k_values:
        label = "baseline" if k == 0 else f"{filler_type}_{k}"
        print(f"\n{'='*60}")
        print(f"Evaluating {label} (k={k}) on {len(examples)} examples...")
        print(f"{'='*60}")
        results = evaluate_condition(k)
        acc = sum(r["correct"] for r in results) / len(results)
        print(f"{label} accuracy: {acc:.1%} ({sum(r['correct'] for r in results)}/{len(results)})")
        all_results[label] = {"accuracy": acc, "results": results}

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    baseline_acc = all_results.get("baseline", {}).get("accuracy")
    for label, data in all_results.items():
        delta = ""
        if baseline_acc is not None and label != "baseline":
            delta = f"  (Δ {data['accuracy'] - baseline_acc:+.1%})"
        print(f"  {label:<25} {data['accuracy']:.1%}{delta}")

    # Save
    args.outdir.mkdir(parents=True, exist_ok=True)
    output = {
        "config": {
            "model_path": args.model_path,
            "dataset": str(args.dataset),
            "max_examples": args.max_examples,
            "filler_type": filler_type,
            "k_values": k_values,
            "n_few_shot": len(few_shot),
            "tp": tp_size,
            "pp": pp_size,
            "engine": "vllm",
        },
        "summary": {label: data["accuracy"] for label, data in all_results.items()},
    }
    for label, data in all_results.items():
        output[label + "_results"] = data["results"]

    out_path = args.outdir / f"eval_2fact_{filler_type}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
