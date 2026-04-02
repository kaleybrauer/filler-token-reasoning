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
    find_filler_boundaries,
    load_model,
    load_tokenizer,
)
from prompt_utils import build_messages


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
        system       — system prompt tokens
        few_shot     — few-shot example tokens
        question     — target question tokens (user turn)
        filler       — filler tokens (dots)
        filler_label — "Filler:", newlines before dots
        answer_label — "Answer:", newlines after dots (present in both conditions)
        answer_prefix — generated "Answer: " tokens (only for pre_answer)
        assistant    — assistant header token
        other        — anything not classified above

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

    # Find "Answer" and "Filler"/"iller" token positions in last user turn
    answer_token_pos = -1
    filler_label_start = -1
    if len(user_turns) > 0:
        for i in range(seq_len - 1, user_turns[-1] - 1, -1):
            if "Answer" in tokens[i]:
                answer_token_pos = i
                break
        if filler_start >= 0:
            # Find the "Filler" / "F" token before filler_start
            # Walk back from filler_start to find where the label begins
            for i in range(filler_start - 1, user_turns[-1] - 1, -1):
                t = tokens[i].strip()
                if t in ("F", "Filler", "iller", ":") or "Filler" in tokens[i]:
                    filler_label_start = i
                elif "\n" in tokens[i] and filler_label_start > 0:
                    # newline before "Filler:" — include it
                    filler_label_start = i
                    break
                else:
                    break

    for i in range(input_len):
        if filler_start >= 0 and filler_start <= i <= filler_end:
            groups["filler"].append(i)
        elif len(user_turns) > 0 and i >= user_turns[-1]:
            # Last user turn = target question region
            if filler_start < 0:
                # Baseline: no filler
                if answer_token_pos >= 0 and i >= answer_token_pos:
                    groups["answer_label"].append(i)
                else:
                    groups["question"].append(i)
            else:
                # Has filler — split into question, filler_label, answer_label
                if answer_token_pos >= 0 and i >= answer_token_pos:
                    groups["answer_label"].append(i)
                elif filler_label_start >= 0 and i >= filler_label_start:
                    groups["filler_label"].append(i)
                elif i < filler_start:
                    groups["question"].append(i)
                else:
                    # After filler_end but before answer — filler_label
                    groups["filler_label"].append(i)
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
    """Deprecated: use load_model from extract_hidden_states instead."""
    return load_model(model_path)


# ==============================================================================
# Main extraction
# ==============================================================================

CONDITIONS = {
    "baseline": ("baseline", 0),
    "dots_5": ("dots", 5),
    "dots_10": ("dots", 10),
    "dots_25": ("dots", 25),
    "dots_100": ("dots", 100),
}


def run_extraction(
    model,
    tokenizer,
    problems: list,
    few_shot: list,
    conditions: Dict[str, Tuple[str, int]],
    layer_indices: List[int],
    output_dir: Path,
    max_examples: int = None,
):
    """Extract attention patterns at the answer position (last input token).

    Uses bare assistant format (no "Answer: " prefix). Single forward pass
    per example — no generation step needed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    extractor = AttentionExtractor(model, layer_indices)
    extractor.register_hooks()

    first_param = next(model.parameters())
    input_device = first_param.device

    if max_examples:
        problems = problems[:max_examples]

    for cond_name, (mode, k) in conditions.items():
        cond_dir = output_dir / cond_name
        cond_dir.mkdir(exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Condition: {cond_name} (mode={mode}, k={k})")
        print(f"{'='*60}")

        t_start = time.time()

        for prob_idx, problem in enumerate(tqdm(problems, desc=cond_name)):
            save_path = cond_dir / f"attn_{prob_idx:04d}.pkl"
            if save_path.exists():
                continue

            messages = build_messages(few_shot[:5], problem, mode, k)
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

            # Answer position = last token
            answer_pos = seq_len - 1

            # Single forward pass with attention weights
            extractor._captured = {}
            with torch.no_grad():
                model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=True,
                )

            answer_attn = {}
            for layer_idx in layer_indices:
                if layer_idx in extractor._captured:
                    attn_w = extractor._captured[layer_idx]
                    agg = aggregate_attention_by_group(
                        attn_w, position_groups, [answer_pos]
                    )
                    answer_attn[layer_idx] = {
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
                "answer_attn": answer_attn,
            }

            with open(save_path, "wb") as f:
                pickle.dump(result, f)

            del input_ids, attention_mask
            torch.cuda.empty_cache()

        elapsed = time.time() - t_start
        print(f"  Time: {elapsed:.0f}s ({elapsed/len(problems):.1f}s per example)")

    extractor.remove_hooks()
    print("\nExtraction complete!")


# ==============================================================================
# Analysis
# ==============================================================================

def analyze_results(output_dir: Path, conditions: list):
    """Load results and print summary tables."""

    group_order = ["system", "few_shot", "question", "filler", "filler_label",
                   "answer_label", "assistant", "other"]

    for cond_name in conditions:
        cond_dir = output_dir / cond_name
        files = sorted(cond_dir.glob("attn_*.pkl"))
        if not files:
            continue

        print(f"\n{'='*80}")
        print(f"  {cond_name} — {len(files)} examples")
        print(f"{'='*80}")

        layer_group_attn = defaultdict(lambda: defaultdict(list))
        layer_head_filler = defaultdict(list)

        for f in files:
            with open(f, "rb") as fp:
                data = pickle.load(fp)
            attn_data = data["answer_attn"]

            for layer, groups in attn_data.items():
                for group, head_values in groups.items():
                    mean_across_heads = float(np.mean(head_values))
                    layer_group_attn[layer][group].append(mean_across_heads)

                if "filler" in groups:
                    layer_head_filler[layer].append(groups["filler"])

        print(f"\n--- answer position (mean attention to each group, averaged over heads) ---")

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
            print(f"\n  Top filler-attending heads:")
            head_scores = []
            for layer in layers:
                if layer not in layer_head_filler:
                    continue
                stacked = np.stack(layer_head_filler[layer], axis=0)
                mean_per_head = stacked.mean(axis=0)
                for head_idx, score in enumerate(mean_per_head):
                    head_scores.append((layer, head_idx, score))

            head_scores.sort(key=lambda x: -x[2])
            print(f"  {'Layer':>6} {'Head':>6} {'MeanAttnToFiller':>18}")
            for layer, head, score in head_scores[:20]:
                print(f"  L{layer:>4}  H{head:>4}  {score:>18.4f}")


def main():
    parser = argparse.ArgumentParser(description="Extract attention weights")
    parser.add_argument("--model-path", type=str,
                        default="/workspace/models/deepseek-v3-awq")
    parser.add_argument("--dataset", type=Path,
                        default=Path("probing/data/1hop_addition_dataset.json"))
    parser.add_argument("--conditions", nargs="+", default=["baseline", "dots_100"])
    parser.add_argument("--output-dir", type=Path,
                        default=Path("probing/results/attention"))
    parser.add_argument("--max-examples", type=int, default=100)
    parser.add_argument("--layers", default="0,10,20,30,40,50,55,58,60",
                        help="Comma-separated layer indices")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Skip extraction, just analyze existing results")
    args = parser.parse_args()

    layer_indices = [int(x) for x in args.layers.split(",")]

    if args.analyze_only:
        analyze_results(args.output_dir, args.conditions)
        return

    with open(args.dataset) as f:
        dataset = json.load(f)
    few_shot = dataset["few_shot_facts"]
    problems = dataset["examples"]
    print(f"Loaded {len(problems)} problems")

    selected = {k: CONDITIONS[k] for k in args.conditions if k in CONDITIONS}

    model, tokenizer = load_model(args.model_path)

    run_extraction(
        model, tokenizer, problems, few_shot,
        selected, layer_indices, args.output_dir,
        max_examples=args.max_examples,
    )

    analyze_results(args.output_dir, args.conditions)


if __name__ == "__main__":
    main()
