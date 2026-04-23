#!/usr/bin/env python3
"""Evaluate DeepSeek V3 (quantized) on the element letter-position task via vLLM.

Task: "What is the {position} letter of the chemical element with atomic number N?"
Answer is a single lowercase letter.

Usage:
    python scripts/evaluate_letterpos_vllm.py \
        --conditions baseline dots_100 dots_200 dots_300 counting_100 counting_200 counting_300
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.generate_letterpos_dataset import build_prompt_messages_letterpos

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MODEL_PATH = "/workspace/models/deepseek-v3-awq"
DATASET_PATH = Path("data/element_letter_positions.json")
RESULTS_DIR = Path("results/eval_letterpos")


def parse_condition(cond: str) -> tuple[int, str]:
    """baseline -> (0, 'dots'); dots_N -> (N, 'dots'); counting_N -> (N, 'counting')."""
    if cond == "baseline":
        return 0, "dots"
    for prefix in ("dots_", "counting_", "alphabet_"):
        if cond.startswith(prefix):
            return int(cond[len(prefix):]), prefix.rstrip("_")
    raise ValueError(f"Unknown condition: {cond!r}")


def parse_letter(response: str) -> str | None:
    """Parse a single lowercase letter from a model response. Unicode-aware."""
    m = re.search(r"Answer:\s*(\S)", response)
    if m and m.group(1).isalpha():
        return m.group(1).lower()
    for ch in response:
        if ch.isalpha():
            return ch.lower()
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--outdir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--max-examples", type=int, default=285)
    parser.add_argument("--conditions", nargs="+", required=True,
                        help="e.g. baseline dots_100 counting_200")
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--pipeline-parallel-size", type=int, default=None)
    parser.add_argument("--max-model-len", type=int, default=5120,
                        help="Needs to cover 6 turns × (~20 question + k filler + "
                             "~15 template) at k_max. 5120 covers counting_300.")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    args = parser.parse_args()

    import torch
    from transformers import AutoTokenizer

    with open(args.dataset) as f:
        dataset = json.load(f)
    few_shot = dataset["few_shot_examples"]
    examples = dataset["examples"][:args.max_examples]
    print(f"Loaded {len(examples)} examples, {len(few_shot)} few-shot.")

    # V3 has 128 attention heads → TP must divide 128.
    # V3 AWQ is ~328 GB, so 2×H200 (280 GB total) can't hold it. On 3 GPUs use PP=3.
    n_gpus = torch.cuda.device_count()
    if args.tensor_parallel_size and args.pipeline_parallel_size:
        tp_size = args.tensor_parallel_size
        pp_size = args.pipeline_parallel_size
    elif n_gpus >= 4:
        tp_size, pp_size = 4, 1
    elif n_gpus == 2:
        tp_size, pp_size = 2, 1
    else:
        tp_size, pp_size = 1, n_gpus
    print(f"Using TP={tp_size}, PP={pp_size} on {n_gpus} GPUs")

    print(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    use_server = pp_size > 1

    if use_server:
        import subprocess, signal, atexit
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
            server_log = open("/tmp/vllm_server_letterpos.log", "w")
            server_proc = subprocess.Popen(
                server_cmd, stdout=server_log, stderr=subprocess.STDOUT,
                env={**os.environ, "VLLM_USE_V1": "0"},
                start_new_session=True,
            )

            def _cleanup_server():
                if server_proc and server_proc.poll() is None:
                    try:
                        os.killpg(os.getpgid(server_proc.pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass

            atexit.register(_cleanup_server)
            signal.signal(signal.SIGTERM, lambda *_: (_cleanup_server(), sys.exit(0)))

        client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="dummy")
        print("Waiting for server to load model...")
        t0 = time.time()
        for _ in range(1200):
            time.sleep(1)
            try:
                client.models.list()
                break
            except Exception:
                if server_proc and server_proc.poll() is not None:
                    with open("/tmp/vllm_server_letterpos.log") as f:
                        log = f.read()
                    print(f"Server crashed! Output:\n{log[-3000:]}")
                    sys.exit(1)
        else:
            print("Server failed to start in 20 minutes")
            sys.exit(1)
        print(f"Server ready in {time.time() - t0:.0f}s")

        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _single_request(prompt):
            resp = client.completions.create(
                model=args.model_path,
                prompt=prompt,
                max_tokens=10,
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
        sampling_params = SamplingParams(max_tokens=10, temperature=0)

        def generate_batch(prompts):
            outputs = llm.generate(prompts, sampling_params)
            return [out.outputs[0].text.strip() for out in outputs]

    # Sanity
    print("\nSanity: 'first letter of apple?' ...")
    sanity_msgs = [{"role": "user", "content":
                    "What is the first letter of the word 'apple'? "
                    "Answer with just the single lowercase letter."}]
    sanity_prompt = tokenizer.apply_chat_template(sanity_msgs, tokenize=False,
                                                  add_generation_prompt=True)
    print(f"  Response: {generate_batch([sanity_prompt])[0]!r}")

    def evaluate_condition(cond: str) -> dict:
        k, filler_type = parse_condition(cond)
        prompts = []
        for ex in examples:
            msgs = build_prompt_messages_letterpos(few_shot, ex, filler_type, k)
            prompt = tokenizer.apply_chat_template(msgs, tokenize=False,
                                                   add_generation_prompt=True)
            prompts.append(prompt)

        sample_ids = tokenizer(prompts[0])["input_ids"]
        print(f"  Sample prompt: {len(sample_ids)} tokens")

        t0 = time.time()
        responses = generate_batch(prompts)
        elapsed = time.time() - t0
        print(f"  Generated {len(responses)} in {elapsed:.1f}s "
              f"({elapsed/len(responses):.3f}s/ex)")

        results = []
        for ex, response in zip(examples, responses):
            predicted = parse_letter(response)
            correct = predicted == ex["answer"]
            results.append({
                "element": ex["element"],
                "atomic_number": ex["atomic_number"],
                "position": ex["position"],
                "expected": ex["answer"],
                "predicted": predicted,
                "correct": correct,
                "raw_response": response,
            })
        acc = sum(r["correct"] for r in results) / len(results)
        return {"accuracy": acc, "results": results, "k": k, "filler_type": filler_type}

    all_results = {}
    for cond in args.conditions:
        print(f"\n{'='*60}\nEvaluating {cond} on {len(examples)} examples...\n{'='*60}")
        all_results[cond] = evaluate_condition(cond)
        n_ok = sum(r["correct"] for r in all_results[cond]["results"])
        print(f"{cond} accuracy: {all_results[cond]['accuracy']:.1%} "
              f"({n_ok}/{len(examples)})")

    # Summary
    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    baseline_acc = all_results.get("baseline", {}).get("accuracy")
    for cond, data in all_results.items():
        delta = ""
        if baseline_acc is not None and cond != "baseline":
            delta = f"  (Δ {data['accuracy'] - baseline_acc:+.1%})"
        print(f"  {cond:<20} {data['accuracy']:.1%}{delta}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    output = {
        "config": {
            "model_path": args.model_path,
            "dataset": str(args.dataset),
            "max_examples": args.max_examples,
            "conditions": args.conditions,
            "n_few_shot": len(few_shot),
            "tp": tp_size,
            "pp": pp_size,
            "engine": "vllm",
        },
        "summary": {cond: data["accuracy"] for cond, data in all_results.items()},
    }
    for cond, data in all_results.items():
        output[cond + "_results"] = data["results"]

    out_path = args.outdir / "eval_letterpos.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
