"""
extract_kv_latents.py

Extract MLA compressed KV latents at filler and answer positions.

DeepSeek V3's MLA compresses KV into a 512-dim latent (kv_lora_rank)
plus a 64-dim RoPE key. This is what gets stored in the KV cache and
what the answer position reads via attention.

Hooks kv_a_proj_with_mqa at each layer to capture the raw projection
before it's split into compressed_kv and k_pe.

Usage:
    source /workspace/config/probing_env.sh
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python scripts/extract_kv_latents.py \
        --model-path /workspace/models/deepseek-v3-awq \
        --conditions baseline dots_100 \
        --max-examples 500 \
        --batch-size 4 \
        --output-dir data/extracted_kv_latents
"""

import argparse
import json
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

import transformers.activations as _act
if not hasattr(_act, "PytorchGELUTanh"):
    _act.PytorchGELUTanh = _act.GELUTanh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extract.extract_hidden_states import (
    find_filler_boundaries, load_model, load_tokenizer,
    compute_extraction_positions, compute_baseline_positions,
    build_messages_for_condition,
)


CONDITIONS = {
    "baseline": (0, "dots"),
    "dots_10": (10, "dots"),
    "dots_100": (100, "dots"),
    "counting_25": (25, "counting"),
}


class KVLatentExtractor:
    """Hook kv_a_proj_with_mqa to capture compressed KV latents."""

    def __init__(self, model, layer_indices: List[int]):
        self.model = model
        self.layer_indices = layer_indices
        self._captured = {}
        self._hooks = []

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            # output of kv_a_proj_with_mqa: (batch, seq_len, kv_lora_rank + qk_rope_head_dim)
            # = (batch, seq_len, 576)
            self._captured[layer_idx] = output.detach().cpu()
        return hook_fn

    def register_hooks(self):
        self.remove_hooks()
        for li in self.layer_indices:
            module = self.model.model.layers[li].self_attn.kv_a_proj_with_mqa
            hook = module.register_forward_hook(self._make_hook(li))
            self._hooks.append(hook)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self._captured = {}

    def extract(self, input_ids, attention_mask, positions: Dict[str, int]):
        """Run forward pass and extract KV latents at specified positions.

        Returns:
            {position_name: {layer_idx: np.ndarray of shape (576,)}}
        """
        self._captured = {}
        with torch.no_grad():
            self.model(input_ids=input_ids, attention_mask=attention_mask)

        result = {}
        for pos_name, pos_idx in positions.items():
            result[pos_name] = {}
            for li in self.layer_indices:
                vec = self._captured[li][0, pos_idx]  # (576,)
                result[pos_name][li] = vec.to(torch.float16).numpy()

        self._captured = {}
        torch.cuda.empty_cache()
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/workspace/models/deepseek-v3-awq")
    parser.add_argument("--dataset", type=Path,
                        default=Path("data/1hop_addition_dataset.json"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/extracted_kv_latents"))
    parser.add_argument("--conditions", nargs="+",
                        default=["baseline", "dots_100"])
    parser.add_argument("--layers", type=str, default="all")
    parser.add_argument("--max-examples", type=int, default=500)
    parser.add_argument("--filler-positions", type=str, default="1,3,5,10,25,50,100")
    args = parser.parse_args()

    num_layers = 61
    if args.layers == "all":
        layer_indices = list(range(num_layers))
    else:
        layer_indices = [int(x) for x in args.layers.split(",")]

    filler_pos = [int(x) for x in args.filler_positions.split(",")]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.dataset) as f:
        dataset = json.load(f)
    few_shot = dataset["few_shot_facts"]
    problems = dataset["examples"][:args.max_examples]

    selected = {k: CONDITIONS[k] for k in args.conditions if k in CONDITIONS}

    # Storage estimate: 576 dims × 2 bytes × ~9 positions × 61 layers × 500 examples × n_conditions
    n_pos = len(filler_pos) + 2  # filler positions + question_end + answer_prompt
    bytes_per_example = n_pos * len(layer_indices) * 576 * 2
    total_gb = len(problems) * len(selected) * bytes_per_example / 1e9
    print(f"Estimated storage: {total_gb:.2f} GB")

    model, tokenizer = load_model(args.model_path)
    device = next(model.parameters()).device

    extractor = KVLatentExtractor(model, layer_indices)
    extractor.register_hooks()

    for cond_name, (k, filler_type) in selected.items():
        cond_dir = args.output_dir / cond_name
        cond_dir.mkdir(exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  {cond_name} (k={k}, {filler_type})")
        print(f"{'='*60}")

        t0 = time.time()
        for prob_idx, problem in enumerate(tqdm(problems, desc=cond_name)):
            save_path = cond_dir / f"kv_{prob_idx:04d}.pkl"
            if save_path.exists():
                continue

            example_rng = random.Random(prob_idx)
            messages = build_messages_for_condition(
                few_shot[:5], problem, filler_type, k, rng=example_rng
            )
            full_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(full_text, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            seq_len = input_ids.shape[1]

            if k > 0:
                qend, filler_start, filler_end = find_filler_boundaries(
                    tokenizer, input_ids, k
                )
                positions = compute_extraction_positions(
                    filler_start, filler_end, seq_len,
                    absolute_positions=filler_pos,
                    question_end_pos=qend,
                )
            else:
                positions = compute_baseline_positions(seq_len)

            kv_latents = extractor.extract(input_ids, attention_mask, positions)

            with open(save_path, "wb") as f:
                pickle.dump({
                    "problem_idx": prob_idx,
                    "fact_value": problem["fact_value"],
                    "x": problem["x"],
                    "answer": problem["answer"],
                    "positions": positions,
                    "kv_latents": kv_latents,
                    "seq_len": seq_len,
                }, f)

            del input_ids, attention_mask
            torch.cuda.empty_cache()

        elapsed = time.time() - t0
        print(f"  Time: {elapsed:.0f}s ({elapsed/len(problems):.1f}s per example)")

    extractor.remove_hooks()
    print("\nExtraction complete!")


if __name__ == "__main__":
    main()
