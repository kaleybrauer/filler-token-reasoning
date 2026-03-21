#!/usr/bin/env python3
"""Evaluate accuracy vs filler truncation length.

Few-shot examples always use the full filler length for their condition.
Only the target problem's filler is truncated to k tokens.
The system message describes the full filler length (matching the few-shots).

Usage:
    source /workspace/config/probing_env.sh

    # Dots only (fastest)
    uv run --project /workspace/filler-token-reasoning/probing python \
        probing/scripts/evaluate_truncation.py \
        --model-path /workspace/models/deepseek-v3-awq \
        --filler-types dots \
        --outdir probing/results/truncation

    # All filler types
    uv run --project /workspace/filler-token-reasoning/probing python \
        probing/scripts/evaluate_truncation.py \
        --model-path /workspace/models/deepseek-v3-awq \
        --filler-types dots counting counting-scrambled \
        --outdir probing/results/truncation
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

# CUDA 12.8 forward compatibility for CUDA 12.4 drivers
_compat = "/usr/local/cuda-12.8/compat"
if os.path.isdir(_compat):
    os.environ["LD_LIBRARY_PATH"] = _compat + ":" + os.environ.get("LD_LIBRARY_PATH", "")

# Add scripts dir to path so we can import prompt helpers
sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_1hop_dataset import build_system_message, build_user_turn, make_filler_tokens

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MODEL_PATH = "/workspace/models/deepseek-v3-awq"
DATASET_PATH = Path("/workspace/filler-token-reasoning/probing/data/1hop_addition_dataset.json")
RESULTS_DIR = Path("/workspace/filler-token-reasoning/probing/results/truncation")

# Full filler lengths per type
FULL_K = {
    "dots": 250,
    "counting": 150,
    "counting-scrambled": 150,
}

# Truncation levels per type
TRUNCATION_KS = {
    "dots": [0, 10, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250],
    "counting": [0, 10, 25, 50, 75, 100, 125, 150],
    "counting-scrambled": [0, 10, 25, 50, 75, 100, 125, 150],
}


# ---------------------------------------------------------------------------
# Prompt building with split few-shot / target filler lengths
# ---------------------------------------------------------------------------

def make_truncated_filler(filler_type: str, full_k: int, target_k: int,
                          rng: random.Random | None = None) -> str:
    """Generate filler of length target_k by truncating a full-length sequence.

    For scrambled counting, this generates the full shuffle of [1..full_k] and
    takes the first target_k elements, so all truncation levels are prefixes of
    the same sequence.
    """
    if target_k == 0:
        return ""
    if filler_type == "counting-scrambled":
        # Generate full-length scramble, take prefix
        full_filler = make_filler_tokens(filler_type, full_k, rng=rng)
        tokens = full_filler.split()
        return " ".join(tokens[:target_k])
    else:
        # Dots and ordered counting: truncation is just generating with smaller k
        return make_filler_tokens(filler_type, target_k, rng=rng)


def build_user_turn_with_filler(fact_phrase: str, x: int, filler: str) -> str:
    """Build a user turn with a pre-built filler string."""
    question_line = f"Question: What is {fact_phrase} plus {x}?"
    if filler:
        return f"{question_line}\n\nFiller: {filler}\n\nAnswer:"
    else:
        return f"{question_line}\n\nAnswer:"


def build_messages_truncated(
    few_shot_items: list[dict],
    target: dict,
    filler_type: str,
    full_k: int,
    target_k: int,
    rng: random.Random | None = None,
) -> list[dict]:
    """Build chat messages with full filler for few-shots, truncated for target.

    The system message uses full_k (matching the few-shot demonstrations).
    For scrambled counting, the target filler is a prefix of the full scramble.
    """
    system_msg = build_system_message(filler_type, full_k)
    messages = [{"role": "system", "content": system_msg}]

    # Few-shot examples: always full filler length
    for fs in few_shot_items[:5]:
        user_content = build_user_turn(
            fs["fact_phrase"], fs["x"], filler_type, full_k, rng=rng
        )
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": f"Answer: {fs['answer']}"})

    # Target: truncated filler (prefix of full sequence for scrambled)
    target_filler = make_truncated_filler(filler_type, full_k, target_k, rng=rng)
    user_content = build_user_turn_with_filler(
        target["fact_phrase"], target["x"], target_filler
    )
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
    parser = argparse.ArgumentParser(description="Filler truncation evaluation")
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--outdir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--max-examples", type=int, default=500)
    parser.add_argument("--filler-types", nargs="+",
                        choices=["dots", "counting", "counting-scrambled"],
                        default=["dots"])
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--pipeline-parallel-size", type=int, default=None)
    parser.add_argument("--ks", type=str, default=None,
                        help="Comma-separated truncation levels (overrides defaults)")
    args = parser.parse_args()

    import torch
    from transformers import AutoTokenizer

    # Load dataset
    with open(args.dataset) as f:
        dataset = json.load(f)
    few_shot = dataset["few_shot_facts"]
    examples = dataset["examples"][:args.max_examples]
    print(f"Using {len(examples)} examples, {len(few_shot)} few-shot facts")
    print(f"Filler types: {args.filler_types}")

    # Auto-detect parallelism
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

    # Set up inference engine
    use_server = pp_size > 1

    if use_server:
        import subprocess
        import atexit
        from openai import OpenAI

        port = 8192
        client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="dummy")

        # Check if a server is already running on this port
        server_proc = None
        try:
            client.models.list()
            print(f"Found existing vLLM server on port {port}")
        except Exception:
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
                "--trust-remote-code",
            ]
            server_log = open("/tmp/vllm_server.log", "w")
            server_proc = subprocess.Popen(server_cmd, stdout=server_log, stderr=subprocess.STDOUT)
            atexit.register(lambda: server_proc.kill())

            print("Waiting for server to load model...")
            t0 = time.time()
            for attempt in range(600):
                time.sleep(1)
                try:
                    client.models.list()
                    break
                except Exception:
                    if server_proc.poll() is not None:
                        with open("/tmp/vllm_server.log") as f:
                            log = f.read()
                        print(f"Server crashed! Output:\n{log[-3000:]}")
                        sys.exit(1)
            else:
                print("Server failed to start in 10 minutes")
                sys.exit(1)
            print(f"Server ready in {time.time() - t0:.0f}s")

        def _complete_one(prompt):
            resp = client.completions.create(
                model=args.model_path,
                prompt=prompt,
                max_tokens=20,
                temperature=0,
            )
            return resp.choices[0].text.strip()

        def generate_batch(prompts, max_workers=32):
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                return list(pool.map(_complete_one, prompts))
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

    # Run truncation evaluation
    args.outdir.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for filler_type in args.filler_types:
        full_k = FULL_K[filler_type]
        if args.ks:
            ks = [int(x) for x in args.ks.split(",")]
        else:
            ks = TRUNCATION_KS[filler_type]
        print(f"\n{'='*60}")
        print(f"Filler type: {filler_type} (full_k={full_k})")
        print(f"Truncation levels: {ks}")
        print(f"{'='*60}")

        type_results = {}
        for k in ks:
            print(f"\n  k={k} ({filler_type})...")

            prompts = []
            for i, ex in enumerate(examples):
                # Per-example RNG for scrambled counting (reproducible)
                example_rng = random.Random(ex["idx"])
                msgs = build_messages_truncated(
                    few_shot, ex, filler_type, full_k, target_k=k, rng=example_rng
                )
                prompt = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
                prompts.append(prompt)

            # Log token count for first prompt
            sample_ids = tokenizer(prompts[0])["input_ids"]
            print(f"    Sample prompt: {len(sample_ids)} tokens")

            t0 = time.time()
            responses = generate_batch(prompts)
            elapsed = time.time() - t0
            print(f"    Generated {len(responses)} in {elapsed:.1f}s ({elapsed/len(responses):.2f}s/ex)")

            results = []
            n_correct = 0
            for ex, response in zip(examples, responses):
                predicted = extract_answer(response)
                correct = predicted == ex["answer"]
                n_correct += correct
                results.append({
                    "idx": ex["idx"],
                    "expected": ex["answer"],
                    "predicted": predicted,
                    "correct": correct,
                    "raw_response": response,
                    "fact_value": ex["fact_value"],
                    "x": ex["x"],
                })

            acc = n_correct / len(examples)
            print(f"    Accuracy: {acc:.1%} ({n_correct}/{len(examples)})")
            type_results[k] = {
                "accuracy": acc,
                "n_correct": n_correct,
                "n_total": len(examples),
                "per_example": results,
            }

        all_results[filler_type] = {
            "full_k": full_k,
            "truncation_ks": ks,
            "results": type_results,
        }

        # Save per-type results incrementally
        out_path = args.outdir / f"truncation_{filler_type.replace('-', '_')}.json"
        output = {
            "config": {
                "model_path": args.model_path,
                "dataset": str(args.dataset),
                "n_examples": len(examples),
                "filler_type": filler_type,
                "full_k": full_k,
                "truncation_ks": ks,
                "few_shot_k": full_k,
                "note": "Few-shot examples use full_k; only target is truncated",
            },
            "summary": {
                k: {"accuracy": type_results[k]["accuracy"],
                    "n_correct": type_results[k]["n_correct"]}
                for k in ks
            },
            "per_example": {
                k: type_results[k]["per_example"]
                for k in ks
            },
        }
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n  Saved {filler_type} results to {out_path}")

    # Print final summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for filler_type in args.filler_types:
        print(f"\n{filler_type}:")
        type_data = all_results[filler_type]
        for k in type_data["truncation_ks"]:
            acc = type_data["results"][k]["accuracy"]
            print(f"  k={k:>4d}: {acc:.1%}")


if __name__ == "__main__":
    main()
