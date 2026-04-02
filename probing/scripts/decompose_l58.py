"""
decompose_l58.py

Decompose the A→A+Y computation at L58 pre_answer into attention vs MLP
contributions. For each example:

1. Run generation up to pre_answer (generate "Answer: ")
2. At the pre_answer forward pass, hook L58's decoder layer to capture:
   - h_in: residual stream input to L58 (after L57)
   - attn_out: attention output (before adding to residual)
   - mlp_out: MLP output (before adding to residual)
   - h_out: h_in + attn_out + mlp_out
3. Project each through RMSNorm + lm_head to get logits
4. Record the logit/rank of A and A+Y tokens for each component

This reveals whether the addition (A→A+Y) is performed by attention
(reading Y from somewhere) or by MLP (nonlinear transformation).

Usage:
    source /workspace/config/probing_env.sh
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python probing/scripts/decompose_l58.py \
        --model-path /workspace/models/deepseek-v3-awq \
        --max-examples 500 \
        --output-dir probing/results/decompose_l58
"""

import argparse
import json
import pickle
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

import transformers.activations as _act
if not hasattr(_act, "PytorchGELUTanh"):
    _act.PytorchGELUTanh = _act.GELUTanh

sys.path.insert(0, str(Path(__file__).parent))
from extract_hidden_states import (
    build_messages_for_condition,
    find_filler_boundaries,
    load_tokenizer,
)
from extract_hidden_states import load_model


def get_token_id(tokenizer, val: int) -> Optional[int]:
    ids = tokenizer.encode(str(val), add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None


def rms_norm_np(x, weight, eps=1e-6):
    rms = np.sqrt(np.mean(x ** 2) + eps)
    return (x / rms) * weight


class L58Decomposer:
    """Hook into a specific decoder layer to capture attention and MLP outputs separately."""

    def __init__(self, model, layer_idx: int):
        self.model = model
        self.layer_idx = layer_idx
        self.captured = {}
        self._hooks = []

    def _attn_hook(self, module, input, output):
        """Capture attention output (before residual add)."""
        # DeepseekV3DecoderLayer.forward:
        #   residual = hidden_states
        #   hidden_states = self.input_layernorm(hidden_states)
        #   hidden_states, _, _ = self.self_attn(hidden_states, ...)
        #   hidden_states = residual + hidden_states  <-- attn residual add
        #   residual = hidden_states
        #   hidden_states = self.post_attention_layernorm(hidden_states)
        #   hidden_states = self.mlp(hidden_states)
        #   hidden_states = residual + hidden_states  <-- mlp residual add
        #
        # self_attn returns (attn_output, attn_weights, past_kv)
        if isinstance(output, tuple):
            attn_out = output[0]
        else:
            attn_out = output
        # Last position only
        self.captured["attn_out"] = attn_out[0, -1].detach().cpu().float().numpy()

    def _mlp_hook(self, module, input, output):
        """Capture MLP output (before residual add)."""
        self.captured["mlp_out"] = output[0, -1].detach().cpu().float().numpy()

    def _layer_pre_hook(self, module, args, kwargs):
        """Capture input to this layer (residual stream from previous layer)."""
        if args:
            hidden = args[0]
        elif 'hidden_states' in kwargs:
            hidden = kwargs['hidden_states']
        else:
            return args, kwargs
        self.captured["h_in"] = hidden[0, -1].detach().cpu().float().numpy()
        return args, kwargs

    def _layer_post_hook(self, module, input, output):
        """Capture output of this layer (after both residual adds)."""
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        self.captured["h_out"] = hidden[0, -1].detach().cpu().float().numpy()

    def register(self):
        self.remove()
        layer = self.model.model.layers[self.layer_idx]

        # Pre-hook on the decoder layer to get h_in
        h1 = layer.register_forward_pre_hook(self._layer_pre_hook, with_kwargs=True)
        self._hooks.append(h1)

        # Hook on self_attn to get attention output
        h2 = layer.self_attn.register_forward_hook(self._attn_hook)
        self._hooks.append(h2)

        # Hook on MLP to get MLP output
        h3 = layer.mlp.register_forward_hook(self._mlp_hook)
        self._hooks.append(h3)

        # Post-hook on the decoder layer to get h_out
        h4 = layer.register_forward_hook(self._layer_post_hook)
        self._hooks.append(h4)

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self.captured = {}


def run_decomposition(
    model, tokenizer, problems, few_shot,
    layer_idx, lm_head, norm_weight,
    conditions, output_dir, max_examples=500,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    first_param = next(model.parameters())
    input_device = first_param.device

    decomposer = L58Decomposer(model, layer_idx)

    for cond_name, (k, filler_type) in conditions.items():
        print(f"\n{'='*60}")
        print(f"  {cond_name} (k={k}) — decomposing L{layer_idx} at pre_answer")
        print(f"{'='*60}")

        results = []

        for prob_idx in tqdm(range(min(max_examples, len(problems))), desc=cond_name):
            problem = problems[prob_idx]
            example_rng = random.Random(prob_idx)

            messages = build_messages_for_condition(
                few_shot[:5], problem, filler_type, k, rng=example_rng
            )
            full_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(full_text, return_tensors="pt")
            input_ids = inputs["input_ids"].to(input_device)
            attention_mask = inputs["attention_mask"].to(input_device)
            seq_len = input_ids.shape[1]

            # Step 1: Generate "Answer: " (3 tokens) without hooks
            with torch.no_grad():
                gen_output = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=3,
                    do_sample=False,
                )
            prefix_ids = gen_output[:, :seq_len + 3]
            prefix_mask = torch.ones_like(prefix_ids)

            # Step 2: Forward pass with hooks to decompose at pre_answer
            decomposer.captured = {}
            decomposer.register()

            try:
                with torch.no_grad():
                    model(input_ids=prefix_ids, attention_mask=prefix_mask)
            finally:
                captured = dict(decomposer.captured)
                decomposer.remove()
            if not all(k in captured for k in ["h_in", "attn_out", "mlp_out", "h_out"]):
                print(f"  WARNING: missing captures for {prob_idx}")
                continue

            # Step 3: Project each component through RMSNorm + lm_head
            a_tok = get_token_id(tokenizer, problem["fact_value"])
            ay_tok = get_token_id(tokenizer, problem["answer"])
            if a_tok is None or ay_tok is None:
                continue

            component_results = {}
            for comp_name in ["h_in", "attn_out", "mlp_out", "h_out"]:
                vec = captured[comp_name]

                # For h_in and h_out: project through norm + lm_head (standard logit lens)
                # For attn_out and mlp_out: these are CONTRIBUTIONS (deltas),
                # so we project raw (no norm) to see their effect on logits
                if comp_name in ("h_in", "h_out"):
                    vec_normed = rms_norm_np(vec, norm_weight)
                    logits = vec_normed @ lm_head.T
                else:
                    # Raw projection shows the logit SHIFT from this component
                    logits = vec @ lm_head.T

                a_logit = float(logits[a_tok])
                ay_logit = float(logits[ay_tok])
                a_rank = int((logits > a_logit).sum()) + 1
                ay_rank = int((logits > ay_logit).sum()) + 1

                # Top 5 tokens
                top5_idx = np.argpartition(logits, -5)[-5:]
                top5_idx = top5_idx[np.argsort(logits[top5_idx])[::-1]]
                top5 = [(tokenizer.decode([int(i)]), float(logits[i])) for i in top5_idx]

                component_results[comp_name] = {
                    "a_logit": a_logit,
                    "ay_logit": ay_logit,
                    "a_rank": a_rank,
                    "ay_rank": ay_rank,
                    "top5": top5,
                }

            results.append({
                "problem_idx": prob_idx,
                "fact_value": problem["fact_value"],
                "x": problem["x"],
                "answer": problem["answer"],
                "a_token_id": a_tok,
                "ay_token_id": ay_tok,
                "components": component_results,
            })

            del input_ids, attention_mask, prefix_ids, prefix_mask
            torch.cuda.empty_cache()

        # Save and summarize
        with open(output_dir / f"{cond_name}_decomposition.json", "w") as f:
            json.dump(results, f, indent=2)

        # Print summary
        print(f"\n  L{layer_idx} decomposition summary ({len(results)} examples):")
        print(f"  {'Component':<12} {'A medRank':>10} {'A+Y medRank':>12} {'A+Y logit':>10}")
        print(f"  {'-'*48}")
        for comp in ["h_in", "attn_out", "mlp_out", "h_out"]:
            a_ranks = [r["components"][comp]["a_rank"] for r in results]
            ay_ranks = [r["components"][comp]["ay_rank"] for r in results]
            ay_logits = [r["components"][comp]["ay_logit"] for r in results]
            print(f"  {comp:<12} {np.median(a_ranks):>10.0f} {np.median(ay_ranks):>12.0f} "
                  f"{np.median(ay_logits):>10.2f}")


def main():
    parser = argparse.ArgumentParser(description="Decompose L58 computation at pre_answer")
    parser.add_argument("--model-path", type=str,
                        default="/workspace/models/deepseek-v3-awq")
    parser.add_argument("--dataset", type=Path,
                        default=Path("probing/data/1hop_addition_dataset.json"))
    parser.add_argument("--lm-head", type=Path,
                        default=Path("probing/lm_head_weight.npy"))
    parser.add_argument("--rms-norm", type=Path,
                        default=Path("probing/rms_norm_weight.npy"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("probing/results/decompose_l58"))
    parser.add_argument("--layer", type=int, default=58)
    parser.add_argument("--max-examples", type=int, default=500)
    parser.add_argument("--conditions", nargs="+", default=["baseline", "dots_50"])
    args = parser.parse_args()

    CONDITIONS = {
        "baseline": (0, "dots"),
        "dots_50": (50, "dots"),
    }

    with open(args.dataset) as f:
        dataset = json.load(f)
    few_shot = dataset["few_shot_facts"]
    problems = dataset["examples"]

    model, tokenizer = load_model(args.model_path)

    lm_head = np.load(args.lm_head).astype(np.float32)
    norm_weight = np.load(args.rms_norm).astype(np.float32)

    selected = {k: CONDITIONS[k] for k in args.conditions}

    run_decomposition(
        model, tokenizer, problems, few_shot,
        args.layer, lm_head, norm_weight,
        selected, args.output_dir,
        max_examples=args.max_examples,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
