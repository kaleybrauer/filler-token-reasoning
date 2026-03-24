"""
extract_attention.py

Extract attention weight distributions at answer_prompt and pre_answer positions,
aggregated into position groups (filler, question, few_shot, answer_prefix, other).

Requires eager attention (not flash) to materialize attention weights.
DeepSeek V3: 61 layers × 128 heads × ~530 seq_len per example.

Usage:
    source /workspace/config/probing_env.sh
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python probing/scripts/extract_attention.py \
        --model-path /workspace/models/deepseek-v3-awq \
        --conditions baseline dots_50 \
        --output-dir probing/attention_results
"""

import argparse
import json
import pickle
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

# Patch for autoawq compat
import transformers.activations as _act
if not hasattr(_act, "PytorchGELUTanh"):
    _act.PytorchGELUTanh = _act.GELUTanh

sys.path.insert(0, str(Path(__file__).parent))
from extract_hidden_states import (
    build_messages_for_condition,
    find_filler_boundaries,
    load_tokenizer,
)
from generate_1hop_dataset import build_system_message


# ==============================================================================
# Position group classification
# ==============================================================================

def classify_positions(
    tokenizer,
    input_ids: torch.Tensor,
    filler_start: int,
    filler_end: int,
    seq_len: int,
    n_generated: int = 0,
) -> Dict[str, List[int]]:
    """
    Classify each token position into a group.

    Groups:
        system    — system prompt tokens
        few_shot  — few-shot example tokens
        question  — target question tokens (user turn)
        filler    — filler tokens (dots)
        format    — "Filler:", "Answer:", newlines between sections
        answer_prefix — generated "Answer: " tokens (only for pre_answer)
        assistant — assistant header token
        other     — anything not classified above

    Returns dict mapping group name -> list of token indices.
    """
    ids = input_ids[0].tolist()
    tokens = [tokenizer.decode([tid]) for tid in ids]

    groups = defaultdict(list)

    # Find role boundaries by looking for chat template markers
    # DeepSeek uses <｜begin▁of▁sentence｜>, <｜User｜>, <｜Assistant｜>, etc.
    role_boundaries = []
    for i, tid in enumerate(ids):
        decoded = tokenizer.decode([tid])
        if "User" in decoded and "｜" in decoded:
            role_boundaries.append(("user", i))
        elif "Assistant" in decoded and "｜" in decoded:
            role_boundaries.append(("assistant", i))
        elif "begin" in decoded and "sentence" in decoded:
            role_boundaries.append(("bos", i))

    # Count user turns to identify few-shot vs target question
    user_turns = [pos for role, pos in role_boundaries if role == "user"]
    # First user turn is after system prompt; last user turn is the target question
    # Everything between is few-shot

    # Simple approach: classify based on known boundaries
    input_len = seq_len  # prefill length

    for i in range(input_len):
        if filler_start >= 0 and filler_start <= i <= filler_end:
            groups["filler"].append(i)
        elif len(user_turns) > 0 and i >= user_turns[-1]:
            # Last user turn = target question region
            if filler_start >= 0 and i < filler_start:
                groups["question"].append(i)
            elif filler_start < 0:
                # Baseline: no filler, everything in last user turn is question
                groups["question"].append(i)
            else:
                groups["format"].append(i)
        elif any(role == "assistant" and pos == i for role, pos in role_boundaries):
            groups["assistant"].append(i)
        elif len(user_turns) > 1 and i >= user_turns[0] and i < user_turns[-1]:
            groups["few_shot"].append(i)
        else:
            # System prompt or BOS
            groups["system"].append(i)

    # Generated tokens (for pre_answer)
    for i in range(input_len, input_len + n_generated):
        groups["answer_prefix"].append(i)

    return dict(groups)


def aggregate_attention_by_group(
    attn_weights: torch.Tensor,
    position_groups: Dict[str, List[int]],
    query_positions: List[int],
) -> Dict[str, np.ndarray]:
    """
    Aggregate attention weights by position group.

    Args:
        attn_weights: (n_heads, q_len, kv_len) — post-softmax attention for one layer
        position_groups: group_name -> list of token indices
        query_positions: which query positions to extract (e.g., [answer_prompt_idx])

    Returns:
        {group_name: np.ndarray of shape (n_heads, n_query_positions)}
        Each value is the sum of attention weights to that group.
    """
    n_heads = attn_weights.shape[0]
    n_query = len(query_positions)

    result = {}
    for group_name, indices in position_groups.items():
        if not indices:
            result[group_name] = np.zeros((n_heads, n_query), dtype=np.float32)
            continue
        idx_tensor = torch.tensor(indices, device=attn_weights.device)
        # Sum attention to this group of positions
        group_attn = attn_weights[:, query_positions, :][:, :, idx_tensor].sum(dim=-1)
        result[group_name] = group_attn.cpu().float().numpy()

    return result


# ==============================================================================
# Attention extraction hooks
# ==============================================================================

class AttentionExtractor:
    """
    Hooks into attention layers to capture post-softmax attention weights.

    The model MUST use eager attention (not flash) for weights to be materialized.
    """

    def __init__(self, model, layer_indices: List[int]):
        self.model = model
        self.layer_indices = layer_indices
        self._captured = {}
        self._hooks = []

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            # DeepseekV3Attention.forward returns (attn_output, attn_weights, past_kv)
            # attn_weights is None unless output_attentions=True
            if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                # output[1] shape: (batch, n_heads, q_len, kv_len)
                self._captured[layer_idx] = output[1][0].detach()  # remove batch dim
        return hook_fn

    def register_hooks(self):
        self.remove_hooks()
        for layer_idx in self.layer_indices:
            # Hook on the attention module specifically
            module = self.model.model.layers[layer_idx].self_attn
            hook = module.register_forward_hook(self._make_hook(layer_idx))
            self._hooks.append(hook)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self._captured = {}


# ==============================================================================
# Model loading (eager attention)
# ==============================================================================

def load_model_eager(model_path: str):
    """Load model with eager attention to get attention weights."""
    from transformers import AutoModelForCausalLM

    tokenizer = load_tokenizer(model_path)

    n_gpus = torch.cuda.device_count()
    print(f"Loading model (EAGER attention) on {n_gpus} GPUs...")
    for i in range(n_gpus):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU {i}: {name}, {mem:.0f} GB")

    max_memory = {
        i: f"{int(torch.cuda.get_device_properties(i).total_memory * 0.80 / 1e9)}GiB"
        for i in range(n_gpus)
    }

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        max_memory=max_memory,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    elapsed = time.time() - t0
    print(f"Model loaded in {elapsed:.0f}s")

    # Verify eager attention
    attn_class = type(model.model.layers[0].self_attn).__name__
    print(f"Attention class: {attn_class}")
    if "Flash" in attn_class:
        raise RuntimeError("Flash attention detected — need eager for attention weights")

    return model, tokenizer


# ==============================================================================
# Main extraction
# ==============================================================================

CONDITIONS = {
    "baseline": (0, "dots"),
    "dots_50": (50, "dots"),
}


def run_extraction(
    model,
    tokenizer,
    problems: list,
    few_shot: list,
    conditions: Dict[str, Tuple[int, str]],
    layer_indices: List[int],
    output_dir: Path,
    max_examples: int = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    extractor = AttentionExtractor(model, layer_indices)
    extractor.register_hooks()

    first_param = next(model.parameters())
    input_device = first_param.device

    if max_examples:
        problems = problems[:max_examples]

    for cond_name, (k, filler_type) in conditions.items():
        cond_dir = output_dir / cond_name
        cond_dir.mkdir(exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Condition: {cond_name} (k={k})")
        print(f"{'='*60}")

        t_start = time.time()

        for prob_idx, problem in enumerate(tqdm(problems, desc=cond_name)):
            save_path = cond_dir / f"attn_{prob_idx:04d}.pkl"
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
            input_ids = inputs["input_ids"].to(input_device)
            attention_mask = inputs["attention_mask"].to(input_device)
            seq_len = input_ids.shape[1]

            # Find position boundaries
            filler_start, filler_end = -1, -1
            if k > 0:
                _, filler_start, filler_end = find_filler_boundaries(
                    tokenizer, input_ids, k
                )

            # Classify positions
            position_groups = classify_positions(
                tokenizer, input_ids, filler_start, filler_end, seq_len
            )

            # Query positions for answer_prompt
            answer_prompt_pos = seq_len - 1

            # ---- Forward pass 1: extract attention at answer_prompt ----
            extractor._captured = {}
            with torch.no_grad():
                model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=True,
                )

            answer_prompt_attn = {}
            for layer_idx in layer_indices:
                if layer_idx in extractor._captured:
                    attn_w = extractor._captured[layer_idx]  # (n_heads, q_len, kv_len)
                    agg = aggregate_attention_by_group(
                        attn_w, position_groups, [answer_prompt_pos]
                    )
                    # Squeeze out the single query position dim
                    answer_prompt_attn[layer_idx] = {
                        g: v[:, 0] for g, v in agg.items()  # (n_heads,)
                    }

            extractor._captured = {}
            torch.cuda.empty_cache()

            # ---- Forward pass 2: generate "Answer: " then extract at pre_answer ----
            with torch.no_grad():
                gen_output = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=3,
                    do_sample=False,
                )

            prefix_ids = gen_output[:, :seq_len + 3]
            prefix_mask = torch.ones_like(prefix_ids)
            pre_answer_pos = prefix_ids.shape[1] - 1

            # Update position groups with generated tokens
            pa_groups = dict(position_groups)
            pa_groups["answer_prefix"] = list(range(seq_len, seq_len + 3))

            extractor._captured = {}
            with torch.no_grad():
                model(
                    input_ids=prefix_ids,
                    attention_mask=prefix_mask,
                    output_attentions=True,
                )

            pre_answer_attn = {}
            for layer_idx in layer_indices:
                if layer_idx in extractor._captured:
                    attn_w = extractor._captured[layer_idx]
                    agg = aggregate_attention_by_group(
                        attn_w, pa_groups, [pre_answer_pos]
                    )
                    pre_answer_attn[layer_idx] = {
                        g: v[:, 0] for g, v in agg.items()
                    }

            extractor._captured = {}
            torch.cuda.empty_cache()

            # Save
            result = {
                "problem_idx": prob_idx,
                "condition": cond_name,
                "answer": problem["answer"],
                "fact_value": problem["fact_value"],
                "x": problem["x"],
                "seq_len": seq_len,
                "filler_start": filler_start,
                "filler_end": filler_end,
                "position_groups": {g: len(idxs) for g, idxs in position_groups.items()},
                "answer_prompt_attn": answer_prompt_attn,
                "pre_answer_attn": pre_answer_attn,
            }

            with open(save_path, "wb") as f:
                pickle.dump(result, f)

            del input_ids, attention_mask, prefix_ids, prefix_mask
            torch.cuda.empty_cache()

        elapsed = time.time() - t_start
        print(f"  Time: {elapsed:.0f}s ({elapsed/len(problems):.1f}s per example)")

    extractor.remove_hooks()
    print("\nExtraction complete!")


# ==============================================================================
# Analysis
# ==============================================================================

def analyze_results(output_dir: Path, conditions: list, categories_file: Path = None):
    """Load results and print summary tables."""

    categories = {}
    if categories_file and categories_file.exists():
        with open(categories_file) as f:
            cat_data = json.load(f)
        for ex in cat_data["examples"]:
            cat_field = "category_dots_50" if "category_dots_50" in ex else "category"
            categories[ex["problem_idx"]] = ex.get(cat_field, "unknown")

    group_order = ["system", "few_shot", "question", "filler", "format",
                   "assistant", "answer_prefix", "other"]

    for cond_name in conditions:
        cond_dir = output_dir / cond_name
        files = sorted(cond_dir.glob("attn_*.pkl"))
        if not files:
            continue

        print(f"\n{'='*80}")
        print(f"  {cond_name} — {len(files)} examples")
        print(f"{'='*80}")

        for pos_key in ["answer_prompt_attn", "pre_answer_attn"]:
            # Collect per-layer, per-group mean attention (averaged over heads and examples)
            layer_group_attn = defaultdict(lambda: defaultdict(list))
            # Also per-head for finding filler-reading heads
            layer_head_filler = defaultdict(list)

            by_cat = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

            for f in files:
                with open(f, "rb") as fp:
                    data = pickle.load(fp)
                attn_data = data[pos_key]
                cat = categories.get(data["problem_idx"], "unknown")

                for layer, groups in attn_data.items():
                    for group, head_values in groups.items():
                        mean_across_heads = float(np.mean(head_values))
                        layer_group_attn[layer][group].append(mean_across_heads)
                        by_cat[cat][layer][group].append(mean_across_heads)

                    if "filler" in groups:
                        layer_head_filler[layer].append(groups["filler"])

            pos_label = pos_key.replace("_attn", "")
            print(f"\n--- {pos_label} (mean attention to each group, averaged over heads) ---")

            layers = sorted(layer_group_attn.keys())
            present_groups = [g for g in group_order
                              if any(g in layer_group_attn[l] for l in layers)]

            header = f"{'Layer':>6}"
            for g in present_groups:
                header += f"  {g[:8]:>8}"
            print(header)
            print("-" * len(header))

            for layer in layers:
                if layer % 5 != 0 and layer < 55 and layer not in [53]:
                    continue
                row = f"L{layer:>4}"
                for g in present_groups:
                    vals = layer_group_attn[layer].get(g, [])
                    mean = np.mean(vals) if vals else 0
                    row += f"  {mean:>8.4f}"
                print(row)

            # Top filler-attending heads
            if layer_head_filler:
                print(f"\n  Top filler-attending heads at {pos_label}:")
                head_scores = []
                for layer in layers:
                    if layer not in layer_head_filler:
                        continue
                    stacked = np.stack(layer_head_filler[layer], axis=0)  # (n_examples, n_heads)
                    mean_per_head = stacked.mean(axis=0)
                    for head_idx, score in enumerate(mean_per_head):
                        head_scores.append((layer, head_idx, score))

                head_scores.sort(key=lambda x: -x[2])
                print(f"  {'Layer':>6} {'Head':>6} {'MeanAttnToFiller':>18}")
                for layer, head, score in head_scores[:20]:
                    print(f"  L{layer:>4}  H{head:>4}  {score:>18.4f}")

            # By category (filler attention only)
            if "filler" in present_groups:
                print(f"\n  Filler attention by category at {pos_label} (L53):")
                for cat in ["filler_helped", "both_correct", "both_wrong"]:
                    if cat in by_cat and 53 in by_cat[cat] and "filler" in by_cat[cat][53]:
                        vals = by_cat[cat][53]["filler"]
                        print(f"    {cat:<25} n={len(vals):>3}  "
                              f"mean={np.mean(vals):.4f}  med={np.median(vals):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Extract attention weights")
    parser.add_argument("--model-path", type=str,
                        default="/workspace/models/deepseek-v3-awq")
    parser.add_argument("--dataset", type=Path,
                        default=Path("probing/data/1hop_addition_dataset.json"))
    parser.add_argument("--conditions", nargs="+", default=["baseline", "dots_50"])
    parser.add_argument("--output-dir", type=Path,
                        default=Path("probing/attention_results"))
    parser.add_argument("--categories", type=Path,
                        default=Path("probing/probe_results/categories.json"))
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--layers", default="all",
                        help="'all' or comma-separated layer indices")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Skip extraction, just analyze existing results")
    args = parser.parse_args()

    # Determine layers
    num_layers = 61
    if args.layers == "all":
        layer_indices = list(range(num_layers))
    else:
        layer_indices = [int(x) for x in args.layers.split(",")]

    if args.analyze_only:
        analyze_results(args.output_dir, args.conditions, args.categories)
        return

    # Load dataset
    with open(args.dataset) as f:
        dataset = json.load(f)
    few_shot = dataset["few_shot_facts"]
    problems = dataset["examples"]
    print(f"Loaded {len(problems)} problems")

    # Select conditions
    selected = {k: CONDITIONS[k] for k in args.conditions}

    # Load model with eager attention
    model, tokenizer = load_model_eager(args.model_path)

    # Run extraction
    run_extraction(
        model, tokenizer, problems, few_shot,
        selected, layer_indices, args.output_dir,
        max_examples=args.max_examples,
    )

    # Analyze
    analyze_results(args.output_dir, args.conditions, args.categories)


if __name__ == "__main__":
    main()
