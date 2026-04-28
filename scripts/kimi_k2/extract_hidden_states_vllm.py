"""
extract_hidden_states_vllm.py

vLLM-based hidden-state extraction for Kimi K2 W4A16. Produces pickle files with
the same layout as scripts/extract_hidden_states.py so downstream decode scripts
(decode_2fact_heatmap.py, cross_example_consistency.py, etc.) work unchanged.

Why vLLM: the QuixiAI AWQ checkpoint is broken (inf layer-norm weights), and the
transformers+compressed-tensors path dequants W4A16 to bf16 on first forward (~2 TB,
doesn't fit). vLLM's Marlin int4 kernels keep the weights packed on H200.

Usage:
    source /workspace/config/probing_env.sh
    NCCL_NVLS_ENABLE=0 VLLM_USE_V1=0 /root/.venvs/filler-probing/bin/python \
        scripts/kimi_k2/extract_hidden_states_vllm.py \
        --model-path /workspace/models/kimi-k2-w4a16 \
        --dataset data/2fact_addition_dataset.json --dataset-type 2fact \
        --output-dir data/extracted_states_2fact_allpos_kimi_k2 \
        --conditions dots_10 dots_50 counting_10 \
        --max-problems 1000 --all-positions
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Reuse helpers from the shared transformers pipeline
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SCRIPTS_DIR))

from extract.extract_hidden_states import (  # noqa: E402
    CONDITIONS,
    build_messages_for_condition,
    compute_all_positions,
    compute_baseline_positions,
    compute_extraction_positions,
    find_filler_boundaries,
    problem_metadata,
)


# ------------------------------------------------------------------------------
# lm_head / rms_norm extraction from safetensors (bypasses vLLM's sharded params)
# ------------------------------------------------------------------------------

def save_lm_head_and_norm(model_path: Path, family_subdir: str = "kimi_k2") -> None:
    from safetensors import safe_open

    out_dir = Path("data/model_weights") / family_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    lm_head_path = out_dir / "lm_head_weight.npy"
    norm_path = out_dir / "rms_norm_weight.npy"
    if lm_head_path.exists() and norm_path.exists():
        print(f"Weights already saved at {out_dir}, skipping")
        return

    index = json.loads((model_path / "model.safetensors.index.json").read_text())
    wm = index["weight_map"]

    def _load(key: str) -> torch.Tensor:
        with safe_open(model_path / wm[key], framework="pt") as f:
            return f.get_tensor(key)

    lm_head = _load("lm_head.weight").to(torch.float16).cpu().numpy()
    np.save(lm_head_path, lm_head)
    print(f"Saved {lm_head_path}: {lm_head.shape} {lm_head.dtype}")

    norm = _load("model.norm.weight").to(torch.float32).cpu().numpy()
    np.save(norm_path, norm)
    print(f"Saved {norm_path}: {norm.shape} {norm.dtype}")


# ------------------------------------------------------------------------------
# vLLM model load + forward hooks
# ------------------------------------------------------------------------------

class LayerCapture:
    """Forward hooks on every decoder layer. Captures per-layer hidden states
    from the single prefill forward pass of the current request."""

    def __init__(self, model, n_layers: int) -> None:
        self.n_layers = n_layers
        # Per-layer list; outer index = layer, each stores the captured tensor for the
        # last forward pass. We clear() before each request.
        self.captured: dict[int, torch.Tensor] = {}
        self.handles = []
        layers = model.model.layers
        for i, layer in enumerate(layers):
            self.handles.append(
                layer.register_forward_hook(self._make_hook(i))
            )

    def _make_hook(self, idx: int):
        def hook(mod, inp, output):
            # Capture only the first forward call per layer after clear() —
            # that's the prefill. Subsequent calls are per-token decodes
            # whose output is shape (1, hidden) and would overwrite prefill.
            if idx in self.captured:
                return
            # vLLM decoder layer returns a tuple (hidden_states, residual) in
            # some versions, or a single tensor. Handle both.
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            self.captured[idx] = hidden.detach()
        return hook

    def clear(self) -> None:
        self.captured.clear()

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()


def reach_model(llm):
    """Return the underlying torch.nn.Module of the vLLM driver worker."""
    return llm.llm_engine.model_executor.driver_worker.model_runner.model


# ------------------------------------------------------------------------------
# Per-problem extraction
# ------------------------------------------------------------------------------

def extract_states_at_positions(
    captured: dict[int, torch.Tensor],
    positions: dict[str, int],
    layer_indices: list[int],
    hidden_size: int,
) -> dict[str, dict[int, np.ndarray]]:
    """Slice captured per-layer hidden states at the target positions."""
    out: dict[str, dict[int, np.ndarray]] = {}
    for pos_name, pos_idx in positions.items():
        layer_dict: dict[int, np.ndarray] = {}
        for li in layer_indices:
            h = captured.get(li)
            if h is None:
                continue
            # h shape is (seq_len, hidden) when batch is flat (vLLM prefill),
            # or (batch, seq_len, hidden) depending on engine path.
            if h.dim() == 3:
                vec = h[0, pos_idx]
            else:
                vec = h[pos_idx]
            layer_dict[li] = vec.to(torch.float16).cpu().numpy()
        out[pos_name] = layer_dict
    return out


# ------------------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/workspace/models/kimi-k2-w4a16", type=Path)
    ap.add_argument("--dataset", required=True, type=Path, nargs="+",
                    help="One or more dataset JSON paths. Multiple = sequential extraction "
                         "with a single vLLM load.")
    ap.add_argument("--output-dir", type=Path, nargs="+",
                    default=[Path("data/extracted_states_2fact_allpos_kimi_k2")],
                    help="One output dir per --dataset (1:1 pairing).")
    ap.add_argument("--layers", default="all",
                    help='Layer indices (comma-separated) or "all"')
    ap.add_argument("--conditions", nargs="+", required=True,
                    choices=list(CONDITIONS.keys()))
    ap.add_argument("--max-problems", type=int, default=None)
    ap.add_argument("--no-skip-existing", action="store_true")
    ap.add_argument("--filler-positions", type=str, default=None)
    ap.add_argument("--all-positions", action="store_true")
    ap.add_argument("--dataset-type", choices=["1hop", "2fact", "letterpos"],
                    default=["2fact"], nargs="+",
                    help="One per --dataset, or a single value to broadcast.")
    ap.add_argument("--tensor-parallel-size", type=int, default=None,
                    help="Override TP size (default: all visible GPUs)")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--no-generate", action="store_true")
    args = ap.parse_args()

    # Pair up (dataset, output_dir, dataset_type)
    if len(args.output_dir) != len(args.dataset):
        ap.error(f"--output-dir count ({len(args.output_dir)}) must match "
                 f"--dataset count ({len(args.dataset)})")
    if len(args.dataset_type) == 1:
        dataset_types = args.dataset_type * len(args.dataset)
    elif len(args.dataset_type) == len(args.dataset):
        dataset_types = list(args.dataset_type)
    else:
        ap.error(f"--dataset-type count ({len(args.dataset_type)}) must be 1 "
                 f"or equal --dataset count ({len(args.dataset)})")

    def _load_dataset(dataset_path):
        print(f"Loading dataset from {dataset_path}")
        with open(dataset_path) as f:
            data = json.load(f)
        problems = data["examples"] if isinstance(data, dict) else data
        if isinstance(data, dict):
            few_shot = data.get("few_shot_facts") or data.get("few_shot_examples", [])
        else:
            few_shot = []
        if args.max_problems:
            problems = problems[:args.max_problems]
        print(f"  {len(problems)} problems, {len(few_shot)} few-shot facts")
        return problems, few_shot

    # Load first dataset for smoke test (others loaded inside the per-dataset loop)
    first_problems, first_few_shot = _load_dataset(args.dataset[0])
    first_dataset_type = dataset_types[0]

    selected_conditions = {c: CONDITIONS[c] for c in args.conditions}
    print(f"  conditions: {selected_conditions}")

    # lm_head + rms_norm from safetensors (cheap, idempotent)
    save_lm_head_and_norm(args.model_path)

    # vLLM load
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    tp_size = args.tensor_parallel_size or torch.cuda.device_count()
    print(f"\nLoading vLLM (TP={tp_size}, gpu_util={args.gpu_memory_utilization}, "
          f"max_len={args.max_model_len}) from {args.model_path}...")
    t0 = time.time()
    llm = LLM(
        model=str(args.model_path),
        tensor_parallel_size=tp_size,
        dtype="bfloat16",
        quantization="compressed-tensors",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        trust_remote_code=True,  # tokenizer uses custom TikTokenTokenizer
        disable_custom_all_reduce=True,
    )
    print(f"Loaded in {time.time() - t0:.0f}s")

    tokenizer = llm.get_tokenizer()
    model = reach_model(llm)
    n_layers = len(model.model.layers)
    hidden_size = model.model.layers[0].hidden_size \
        if hasattr(model.model.layers[0], "hidden_size") else 7168

    # Layers to save
    if args.layers == "all":
        layer_indices = list(range(n_layers))
    else:
        layer_indices = [int(x) for x in args.layers.split(",")]
    print(f"Extracting from {len(layer_indices)} layers")

    capture = LayerCapture(model, n_layers)

    # Filler position override
    filler_pos = None
    if args.filler_positions and not args.all_positions:
        filler_pos = [int(x) for x in args.filler_positions.split(",")]

    # Smoke-test forward pass (uses first dataset)
    print("\n=== Smoke test forward pass ===")
    smoke_k, smoke_ftype = list(selected_conditions.values())[0]
    smoke_msgs = build_messages_for_condition(
        first_few_shot[:5], first_problems[0], smoke_ftype, smoke_k,
        rng=random.Random(0), dataset_type=first_dataset_type,
    )
    smoke_text = tokenizer.apply_chat_template(smoke_msgs, tokenize=False,
                                               add_generation_prompt=True)
    smoke_ids = tokenizer(smoke_text)["input_ids"]
    print(f"  smoke input_len={len(smoke_ids)}")
    sp = SamplingParams(temperature=0, max_tokens=1, detokenize=False)
    capture.clear()
    _ = llm.generate([TokensPrompt(prompt_token_ids=smoke_ids)], sp, use_tqdm=False)
    print(f"  captured layers: {sorted(capture.captured.keys())[:5]}...{sorted(capture.captured.keys())[-3:]}")
    for li in [0, 1, 2, n_layers - 1]:
        h = capture.captured.get(li)
        if h is None:
            print(f"  layer {li}: NOT CAPTURED")
            continue
        h32 = h.to(torch.float32)
        finite = h32[torch.isfinite(h32)]
        n_nan = torch.isnan(h32).sum().item()
        print(f"  layer {li}: shape={tuple(h.shape)} nan={n_nan} "
              f"absmax={finite.abs().max().item() if finite.numel() else float('nan'):.3g}")

    def _extra_meta(p, dt):
        if dt == "2fact":
            return {"fact_phrase_1": p["fact_phrase_1"], "fact_phrase_2": p["fact_phrase_2"]}
        if dt == "letterpos":
            # problem_metadata already passed through all task-specific fields
            return {}
        return {"fact_phrase": p["fact_phrase"], "kind": p["kind"]}

    gen_sp = SamplingParams(temperature=0, max_tokens=20, detokenize=True)
    prefill_sp = SamplingParams(temperature=0, max_tokens=1, detokenize=False)

    for ds_idx, (dataset_path, output_dir, dataset_type) in enumerate(
        zip(args.dataset, args.output_dir, dataset_types)
    ):
        print(f"\n{'#' * 70}\n# Dataset {ds_idx + 1}/{len(args.dataset)}: {dataset_path} "
              f"(type={dataset_type})\n# Output: {output_dir}\n{'#' * 70}")

        if ds_idx == 0:
            problems_all, few_shot = first_problems, first_few_shot
        else:
            problems_all, few_shot = _load_dataset(dataset_path)

        # Metadata
        output_dir.mkdir(parents=True, exist_ok=True)
        meta = [
            {
                "idx": i,
                **problem_metadata(p, dataset_type),
                **_extra_meta(p, dataset_type),
            }
            for i, p in enumerate(problems_all)
        ]
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        for cond_name, (k, filler_type) in selected_conditions.items():
            cond_dir = output_dir / cond_name
            cond_dir.mkdir(exist_ok=True)
            print(f"\n{'=' * 60}\nCondition: {cond_name} (k={k}, filler_type={filler_type})\n{'=' * 60}")

            todo = []
            for i in range(len(problems_all)):
                save_path = cond_dir / f"prob_{i:04d}.pkl"
                if not args.no_skip_existing and save_path.exists():
                    continue
                todo.append(i)
            if not todo:
                print(f"  All {len(problems_all)} problems already extracted, skipping")
                continue
            print(f"  {len(todo)} problems to extract")

            t_start = time.time()
            correct = 0
            total = 0

            for prob_idx in tqdm(todo, desc=cond_name):
                problem = problems_all[prob_idx]
                rng = random.Random(prob_idx)
                messages = build_messages_for_condition(
                    few_shot[:5], problem, filler_type, k, rng=rng,
                    dataset_type=dataset_type,
                )
                full_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                input_ids = tokenizer(full_text)["input_ids"]
                seq_len = len(input_ids)

                # Compute positions
                filler_start = filler_end = question_end_pos = None
                boundaries = None
                if k > 0:
                    ids_tensor = torch.tensor([input_ids])
                    question_end_pos, filler_start, filler_end = find_filler_boundaries(
                        tokenizer, ids_tensor, k
                    )
                    if args.all_positions:
                        positions, boundaries = compute_all_positions(
                            question_end_pos, seq_len,
                            filler_start=filler_start, filler_end=filler_end,
                        )
                    else:
                        positions = compute_extraction_positions(
                            filler_start, filler_end, seq_len,
                            absolute_positions=filler_pos,
                            question_end_pos=question_end_pos,
                        )
                else:
                    positions = compute_baseline_positions(seq_len)

                # Sanity check positions
                positions = {
                    p: idx for p, idx in positions.items() if 0 <= idx < seq_len
                }

                # Run forward + extract states
                capture.clear()
                if args.no_generate:
                    _ = llm.generate([TokensPrompt(prompt_token_ids=input_ids)],
                                     prefill_sp, use_tqdm=False)
                    model_response, model_answer, model_correct = None, None, None
                else:
                    outs = llm.generate([TokensPrompt(prompt_token_ids=input_ids)],
                                        gen_sp, use_tqdm=False)
                    model_response = outs[0].outputs[0].text.strip()
                    import re
                    if dataset_type == "letterpos":
                        m = re.search(r"Answer:\s*(\S)", model_response)
                        if m and m.group(1).isalpha():
                            model_answer = m.group(1).lower()
                        else:
                            model_answer = next((ch.lower() for ch in model_response if ch.isalpha()), None)
                    else:
                        m = re.search(r"Answer:\s*(-?\d+)", model_response)
                        if not m:
                            m = re.search(r"(-?\d+)", model_response)
                        model_answer = int(m.group(1)) if m else None
                    model_correct = (model_answer == problem["answer"]) if model_answer is not None else False
                    total += 1
                    correct += int(model_correct)

                states = extract_states_at_positions(
                    capture.captured, positions, layer_indices, hidden_size
                )

                result = {
                    "problem_idx": prob_idx,
                    "condition": cond_name,
                    "k": k,
                    **problem_metadata(problem, dataset_type),
                    "positions": positions,
                    "states": states,
                    "model_response": model_response,
                    "model_answer": model_answer,
                    "model_correct": model_correct,
                    "seq_len": seq_len,
                    "filler_start": filler_start,
                    "filler_end": filler_end,
                    "gen_prefix": None,
                }
                if boundaries is not None:
                    result["boundaries"] = boundaries

                with open(cond_dir / f"prob_{prob_idx:04d}.pkl", "wb") as f:
                    pickle.dump(result, f)

            elapsed = time.time() - t_start
            acc = correct / total if total else float("nan")
            print(f"  {cond_name}: done in {elapsed:.0f}s "
                  f"({elapsed / max(total, 1):.1f}s/example, {acc:.1%} correct)")


if __name__ == "__main__":
    main()
