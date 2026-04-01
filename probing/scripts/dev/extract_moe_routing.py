"""
extract_moe_routing.py

Extract MoE router decisions from DeepSeek V3 during filler token processing.
For each MoE layer (3-60), hooks into the MoEGate to capture top-8 expert
indices. Three analyses:

1. Filler diversity (dots_50 only): how many unique experts across 50 filler
   positions, compared between accuracy categories.
2. Answer-position routing (baseline vs dots_50): which experts are active at
   answer_prompt and pre_answer — the positions where the answer is determined.
3. Cross-condition overlap: does filler change which experts are active at the
   answer position? Measured by Jaccard similarity between baseline and dots_50
   expert sets.

To get pre_answer routing without generation, we append "Answer: " to the
input text before tokenizing. The causal mask ensures positions before the
appended tokens compute identically to the original input.

Usage:
    source /workspace/config/probing_env.sh
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    uv run --project /workspace/filler-token-reasoning/probing \
        python probing/scripts/extract_moe_routing.py \
        --model-path /workspace/models/deepseek-v3-awq \
        --output-dir probing/results/router_diversity \
        --max-examples 500
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Patch: autoawq imports PytorchGELUTanh which was renamed in older transformers
import transformers.activations as _act
if not hasattr(_act, "PytorchGELUTanh"):
    _act.PytorchGELUTanh = _act.GELUTanh

sys.path.insert(0, str(Path(__file__).parent))
from extract_hidden_states import (
    build_messages_for_condition,
    find_filler_boundaries,
    load_model,
    load_tokenizer,
)


# ==========================================================================
# Router extraction
# ==========================================================================

class MoERouterExtractor:
    """Hook into MoEGate modules to capture routing decisions."""

    def __init__(self, model, moe_layer_indices: List[int], save_logits: bool = False):
        self.model = model
        self.moe_layer_indices = moe_layer_indices
        self.save_logits = save_logits
        self._captured: Dict[int, dict] = {}
        self._hooks = []

    def _make_gate_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            topk_idx, topk_weight = output
            entry = {
                "topk_idx": topk_idx.detach().cpu(),
                "topk_weight": topk_weight.detach().cpu(),
            }
            if self.save_logits:
                hs = input[0]
                hs_flat = hs.view(-1, hs.shape[-1])
                logits = F.linear(hs_flat.float(), module.weight.float())
                entry["logits"] = logits.detach().cpu()
            self._captured[layer_idx] = entry
        return hook_fn

    def register_hooks(self):
        self.remove_hooks()
        for layer_idx in self.moe_layer_indices:
            gate = self.model.model.layers[layer_idx].mlp.gate
            hook = gate.register_forward_hook(self._make_gate_hook(layer_idx))
            self._hooks.append(hook)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._captured.clear()

    def extract(self, input_ids, attention_mask, positions: List[int]):
        """Run forward pass and return routing at specified positions.

        Returns:
            {layer_idx: {"topk_idx": np.array(n_pos, 8),
                         "topk_weight": np.array(n_pos, 8),
                         "logits": np.array(n_pos, 256)}}
        """
        self._captured = {}
        with torch.no_grad():
            self.model(input_ids=input_ids, attention_mask=attention_mask)

        pos_tensor = torch.tensor(positions, dtype=torch.long)
        result = {}
        for layer_idx in self.moe_layer_indices:
            cap = self._captured[layer_idx]
            entry = {
                "topk_idx": cap["topk_idx"][pos_tensor].numpy().astype(np.int16),
                "topk_weight": cap["topk_weight"][pos_tensor].numpy().astype(np.float16),
            }
            if self.save_logits and "logits" in cap:
                entry["logits"] = cap["logits"][pos_tensor].numpy().astype(np.float16)
            result[layer_idx] = entry

        self._captured = {}
        torch.cuda.empty_cache()
        return result


# ==========================================================================
# Metrics
# ==========================================================================

def compute_diversity_metrics(topk_idx: np.ndarray) -> dict:
    """Compute routing diversity metrics over multiple positions.

    Args:
        topk_idx: shape (n_positions, 8) — top-8 expert indices per position
    """
    n_pos = len(topk_idx)

    diversity = int(len(np.unique(topk_idx)))

    jaccards = []
    for i in range(n_pos - 1):
        set_a = set(topk_idx[i])
        set_b = set(topk_idx[i + 1])
        jaccards.append(len(set_a & set_b) / len(set_a | set_b))
    mean_jaccard = float(np.mean(jaccards)) if jaccards else 0.0

    flat = topk_idx.astype(np.int32).flatten()
    counts = np.bincount(flat, minlength=256)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    entropy = float(-np.sum(probs * np.log2(probs)))

    return {
        "diversity": diversity,
        "mean_jaccard": mean_jaccard,
        "expert_entropy": entropy,
    }


def expert_overlap(experts_a: np.ndarray, experts_b: np.ndarray) -> dict:
    """Compare two sets of top-8 expert selections.

    Args:
        experts_a, experts_b: shape (8,) — expert indices for one position
    Returns:
        jaccard, n_shared, n_different
    """
    set_a = set(experts_a.astype(int))
    set_b = set(experts_b.astype(int))
    n_shared = len(set_a & set_b)
    n_union = len(set_a | set_b)
    return {
        "jaccard": n_shared / n_union if n_union > 0 else 0.0,
        "n_shared": n_shared,
        "n_different": 8 - n_shared,
    }


# ==========================================================================
# Main
# ==========================================================================

def detect_moe_layers(model) -> List[int]:
    """Find which layers have MoE (gate attribute on mlp)."""
    moe_layers = []
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer.mlp, "gate"):
            moe_layers.append(i)
    return moe_layers


def main():
    parser = argparse.ArgumentParser(
        description="Extract MoE router decisions during filler token processing"
    )
    parser.add_argument("--model-path", type=str,
                        default="/workspace/models/deepseek-v3-awq")
    parser.add_argument("--dataset", type=Path,
                        default=Path("probing/data/1hop_addition_dataset.json"))
    parser.add_argument("--categories", type=Path,
                        default=Path("probing/results/probe_results/categories.json"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("probing/results/router_diversity"))
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--save-logits", action="store_true",
                        help="Save full 256-dim router logits at key layers")
    parser.add_argument("--key-layers", type=str, default="53,57,58,59,60",
                        help="Layers to highlight and save detailed data for")
    parser.add_argument("--k", type=int, default=50)
    args = parser.parse_args()

    key_layers = [int(x) for x in args.key_layers.split(",")]

    # ---- Load dataset ----
    with open(args.dataset) as f:
        dataset = json.load(f)
    few_shot = dataset["few_shot_facts"]
    problems = dataset["examples"]
    n_examples = min(args.max_examples or len(problems), len(problems))

    # ---- Load categories ----
    cat_field = "category_dots_50"
    with open(args.categories) as f:
        cat_data = json.load(f)
    cat_lookup = {}
    for ex in cat_data["examples"]:
        cat_lookup[ex["problem_idx"]] = ex.get(cat_field, ex.get("category", "unknown"))

    # ---- Load model ----
    model, tokenizer = load_model(args.model_path)
    first_device = next(model.parameters()).device

    # ---- Detect MoE layers ----
    moe_layers = detect_moe_layers(model)
    print(f"Detected {len(moe_layers)} MoE layers: {moe_layers[0]}-{moe_layers[-1]}")

    # ---- Setup extractor ----
    extractor = MoERouterExtractor(model, moe_layers, save_logits=args.save_logits)
    extractor.register_hooks()

    # ---- Output dir ----
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_logits:
        logits_dir = args.output_dir / "router_logits"
        logits_dir.mkdir(exist_ok=True)

    # ---- Main loop ----
    per_example = []
    t0 = time.time()

    for prob_idx in tqdm(range(n_examples), desc="Extracting routing"):
        problem = problems[prob_idx]
        category = cat_lookup.get(prob_idx, "unknown")

        # ==============================================================
        # dots_50: extended input (with "Answer: " appended)
        # One forward pass captures filler + answer_prompt + pre_answer
        # ==============================================================
        rng_d50 = random.Random(prob_idx)
        msgs_d50 = build_messages_for_condition(
            few_shot[:5], problem, "dots", args.k, rng=rng_d50
        )
        text_d50 = tokenizer.apply_chat_template(
            msgs_d50, tokenize=False, add_generation_prompt=True
        )
        text_d50_ext = text_d50 + "Answer: "

        ids_d50_orig = tokenizer(text_d50, return_tensors="pt")["input_ids"]
        ids_d50_ext = tokenizer(text_d50_ext, return_tensors="pt")["input_ids"]

        orig_len_d50 = ids_d50_orig.shape[1]
        ext_len_d50 = ids_d50_ext.shape[1]
        n_prefix = ext_len_d50 - orig_len_d50

        # Verify tokenization consistency: prefix of extended should match original
        assert torch.equal(ids_d50_ext[0, :orig_len_d50], ids_d50_orig[0]), \
            f"Example {prob_idx}: extended tokenization changed prefix tokens"

        # Find filler boundaries from original tokenization
        q_end, filler_start, filler_end = find_filler_boundaries(
            tokenizer, ids_d50_orig, args.k
        )
        n_filler = filler_end - filler_start + 1
        filler_positions = list(range(filler_start, filler_end + 1))

        answer_prompt_d50 = orig_len_d50 - 1
        pre_answer_d50 = ext_len_d50 - 1

        # Forward pass
        input_ids = ids_d50_ext.to(first_device)
        attention_mask = torch.ones_like(input_ids)

        d50_positions = filler_positions + [answer_prompt_d50, pre_answer_d50]
        d50_routing = extractor.extract(input_ids, attention_mask, d50_positions)

        del input_ids, attention_mask

        # ==============================================================
        # baseline: extended input (with "Answer: " appended)
        # Forward pass captures answer_prompt + pre_answer
        # ==============================================================
        rng_bl = random.Random(prob_idx)
        msgs_bl = build_messages_for_condition(
            few_shot[:5], problem, "dots", 0, rng=rng_bl
        )
        text_bl = tokenizer.apply_chat_template(
            msgs_bl, tokenize=False, add_generation_prompt=True
        )
        text_bl_ext = text_bl + "Answer: "

        ids_bl_orig = tokenizer(text_bl, return_tensors="pt")["input_ids"]
        ids_bl_ext = tokenizer(text_bl_ext, return_tensors="pt")["input_ids"]

        orig_len_bl = ids_bl_orig.shape[1]
        ext_len_bl = ids_bl_ext.shape[1]

        assert torch.equal(ids_bl_ext[0, :orig_len_bl], ids_bl_orig[0]), \
            f"Example {prob_idx}: baseline extended tokenization changed prefix"

        answer_prompt_bl = orig_len_bl - 1
        pre_answer_bl = ext_len_bl - 1

        input_ids = ids_bl_ext.to(first_device)
        attention_mask = torch.ones_like(input_ids)

        bl_positions = [answer_prompt_bl, pre_answer_bl]
        bl_routing = extractor.extract(input_ids, attention_mask, bl_positions)

        del input_ids, attention_mask
        torch.cuda.empty_cache()

        # ==============================================================
        # Compute metrics
        # ==============================================================
        n_filler_slots = len(filler_positions)
        example_data = {
            "problem_idx": prob_idx,
            "category": category,
            "fact_value": problem["fact_value"],
            "x": problem["x"],
            "answer": problem["answer"],
            "n_filler_tokens": n_filler,
            "n_prefix_tokens": n_prefix,
        }

        # --- Filler diversity (dots_50 only) ---
        filler_diversity = {}
        filler_metrics = {}
        for layer_idx in moe_layers:
            filler_idx = d50_routing[layer_idx]["topk_idx"][:n_filler_slots]
            metrics = compute_diversity_metrics(filler_idx)
            filler_diversity[layer_idx] = metrics["diversity"]
            filler_metrics[layer_idx] = metrics
        example_data["filler_diversity"] = {str(l): d for l, d in filler_diversity.items()}

        # --- Cross-condition overlap at answer_prompt and pre_answer ---
        answer_overlap = {}
        pre_answer_overlap = {}
        # d50_positions layout: [filler_0..filler_49, answer_prompt, pre_answer]
        d50_ap_idx = n_filler_slots      # index into extracted positions
        d50_pa_idx = n_filler_slots + 1
        # bl_positions layout: [answer_prompt, pre_answer]
        bl_ap_idx = 0
        bl_pa_idx = 1

        for layer_idx in moe_layers:
            d50_ap_experts = d50_routing[layer_idx]["topk_idx"][d50_ap_idx]
            bl_ap_experts = bl_routing[layer_idx]["topk_idx"][bl_ap_idx]
            answer_overlap[layer_idx] = expert_overlap(d50_ap_experts, bl_ap_experts)

            d50_pa_experts = d50_routing[layer_idx]["topk_idx"][d50_pa_idx]
            bl_pa_experts = bl_routing[layer_idx]["topk_idx"][bl_pa_idx]
            pre_answer_overlap[layer_idx] = expert_overlap(d50_pa_experts, bl_pa_experts)

        example_data["answer_prompt_overlap"] = {
            str(l): o for l, o in answer_overlap.items()
        }
        example_data["pre_answer_overlap"] = {
            str(l): o for l, o in pre_answer_overlap.items()
        }

        # --- Key layer details ---
        key_details = {}
        for l in key_layers:
            if l in d50_routing and l in bl_routing:
                key_details[str(l)] = {
                    "d50_answer_experts": d50_routing[l]["topk_idx"][d50_ap_idx].tolist(),
                    "d50_pre_answer_experts": d50_routing[l]["topk_idx"][d50_pa_idx].tolist(),
                    "bl_answer_experts": bl_routing[l]["topk_idx"][bl_ap_idx].tolist(),
                    "bl_pre_answer_experts": bl_routing[l]["topk_idx"][bl_pa_idx].tolist(),
                    "filler_diversity": filler_diversity.get(l, -1),
                    "answer_jaccard": answer_overlap.get(l, {}).get("jaccard", -1),
                    "pre_answer_jaccard": pre_answer_overlap.get(l, {}).get("jaccard", -1),
                }
        example_data["key_layers"] = key_details

        # --- Save logits if requested ---
        if args.save_logits:
            save_data = {}
            for l in key_layers:
                if l in d50_routing and "logits" in d50_routing[l]:
                    save_data[f"d50_filler_L{l}"] = d50_routing[l]["logits"][:n_filler_slots]
                    save_data[f"d50_answer_L{l}"] = d50_routing[l]["logits"][d50_ap_idx:d50_ap_idx+1]
                    save_data[f"d50_pre_answer_L{l}"] = d50_routing[l]["logits"][d50_pa_idx:d50_pa_idx+1]
                if l in bl_routing and "logits" in bl_routing[l]:
                    save_data[f"bl_answer_L{l}"] = bl_routing[l]["logits"][bl_ap_idx:bl_ap_idx+1]
                    save_data[f"bl_pre_answer_L{l}"] = bl_routing[l]["logits"][bl_pa_idx:bl_pa_idx+1]
            np.savez_compressed(logits_dir / f"prob_{prob_idx:04d}.npz", **save_data)

        per_example.append(example_data)

        # --- First example sanity check ---
        if prob_idx == 0:
            print(f"\n  Example 0: category={category}")
            print(f"  dots_50: orig_len={orig_len_d50}, ext_len={ext_len_d50}, "
                  f"filler={filler_start}..{filler_end} ({n_filler} tokens), "
                  f"prefix_tokens={n_prefix}")
            print(f"  baseline: orig_len={orig_len_bl}, ext_len={ext_len_bl}")
            for l in key_layers:
                kd = key_details.get(str(l), {})
                fm = filler_metrics.get(l, {})
                print(f"  L{l}: filler_div={kd.get('filler_diversity', '?')}, "
                      f"ap_jaccard={kd.get('answer_jaccard', '?'):.3f}, "
                      f"pa_jaccard={kd.get('pre_answer_jaccard', '?'):.3f}, "
                      f"filler_entropy={fm.get('expert_entropy', 0):.2f}")
                print(f"    d50 pre_answer experts: {kd.get('d50_pre_answer_experts', [])}")
                print(f"    bl  pre_answer experts: {kd.get('bl_pre_answer_experts', [])}")

    extractor.remove_hooks()
    elapsed = time.time() - t0
    print(f"\nExtraction complete: {n_examples} examples in {elapsed:.0f}s "
          f"({elapsed / n_examples:.1f}s/example)")

    # ==================================================================
    # Aggregate by category
    # ==================================================================
    categories = ["filler_helped", "both_correct", "both_wrong", "filler_hurt"]
    cat_examples = {c: [e for e in per_example if e["category"] == c] for c in categories}

    # --- Filler diversity aggregation ---
    filler_div_results = {}
    for layer_idx in moe_layers:
        lk = str(layer_idx)
        filler_div_results[lk] = {}
        for cat in categories:
            vals = [e["filler_diversity"][lk] for e in cat_examples[cat]]
            if vals:
                filler_div_results[lk][cat] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "median": float(np.median(vals)),
                    "n": len(vals),
                }

    # --- Cross-condition overlap aggregation ---
    overlap_results = {"answer_prompt": {}, "pre_answer": {}}
    for pos_key in ["answer_prompt", "pre_answer"]:
        data_key = f"{pos_key}_overlap"
        for layer_idx in moe_layers:
            lk = str(layer_idx)
            overlap_results[pos_key][lk] = {}
            for cat in categories:
                jaccards = [e[data_key][lk]["jaccard"] for e in cat_examples[cat]]
                n_shared = [e[data_key][lk]["n_shared"] for e in cat_examples[cat]]
                if jaccards:
                    overlap_results[pos_key][lk][cat] = {
                        "mean_jaccard": float(np.mean(jaccards)),
                        "std_jaccard": float(np.std(jaccards)),
                        "mean_shared": float(np.mean(n_shared)),
                        "n": len(jaccards),
                    }

    # --- Statistical tests ---
    try:
        from scipy.stats import mannwhitneyu
        has_scipy = True
    except ImportError:
        has_scipy = False

    stat_tests = {"filler_diversity": {}, "answer_prompt_overlap": {}, "pre_answer_overlap": {}}
    if has_scipy:
        for layer_idx in moe_layers:
            lk = str(layer_idx)

            # Filler diversity: helped vs correct, helped vs wrong
            helped_div = [e["filler_diversity"][lk] for e in cat_examples["filler_helped"]]
            correct_div = [e["filler_diversity"][lk] for e in cat_examples["both_correct"]]
            wrong_div = [e["filler_diversity"][lk] for e in cat_examples["both_wrong"]]
            tests = {}
            if helped_div and correct_div:
                u, p = mannwhitneyu(helped_div, correct_div, alternative="two-sided")
                tests["helped_vs_correct"] = {"U": float(u), "p": float(p)}
            if helped_div and wrong_div:
                u, p = mannwhitneyu(helped_div, wrong_div, alternative="two-sided")
                tests["helped_vs_wrong"] = {"U": float(u), "p": float(p)}
            stat_tests["filler_diversity"][lk] = tests

            # Cross-condition overlap: does overlap differ by category?
            for pos_key in ["answer_prompt", "pre_answer"]:
                data_key = f"{pos_key}_overlap"
                helped_j = [e[data_key][lk]["jaccard"] for e in cat_examples["filler_helped"]]
                correct_j = [e[data_key][lk]["jaccard"] for e in cat_examples["both_correct"]]
                wrong_j = [e[data_key][lk]["jaccard"] for e in cat_examples["both_wrong"]]
                tests = {}
                if helped_j and correct_j:
                    u, p = mannwhitneyu(helped_j, correct_j, alternative="two-sided")
                    tests["helped_vs_correct"] = {"U": float(u), "p": float(p)}
                if helped_j and wrong_j:
                    u, p = mannwhitneyu(helped_j, wrong_j, alternative="two-sided")
                    tests["helped_vs_wrong"] = {"U": float(u), "p": float(p)}
                stat_tests[f"{pos_key}_overlap"][lk] = tests

    # ==================================================================
    # Save results
    # ==================================================================
    with open(args.output_dir / "results.json", "w") as f:
        json.dump({
            "moe_layers": moe_layers,
            "categories": categories,
            "filler_diversity": filler_div_results,
            "cross_condition_overlap": overlap_results,
            "stat_tests": stat_tests,
        }, f, indent=2)

    # Per-example data (slim — diversity + overlap, no per-position experts)
    per_example_slim = []
    for e in per_example:
        slim = {
            "problem_idx": e["problem_idx"],
            "category": e["category"],
            "fact_value": e["fact_value"],
            "x": e["x"],
            "answer": e["answer"],
            "filler_diversity": e["filler_diversity"],
            "answer_prompt_jaccard": {
                l: e["answer_prompt_overlap"][l]["jaccard"]
                for l in e["answer_prompt_overlap"]
            },
            "pre_answer_jaccard": {
                l: e["pre_answer_overlap"][l]["jaccard"]
                for l in e["pre_answer_overlap"]
            },
        }
        per_example_slim.append(slim)
    with open(args.output_dir / "per_example.json", "w") as f:
        json.dump(per_example_slim, f)

    # Key layer routing details (per-position experts at key layers)
    key_layer_data = [
        {"problem_idx": e["problem_idx"], "category": e["category"],
         "key_layers": e["key_layers"]}
        for e in per_example
    ]
    with open(args.output_dir / "key_layer_routing.json", "w") as f:
        json.dump(key_layer_data, f)

    # Run config
    with open(args.output_dir / "run_config.json", "w") as f:
        json.dump({
            "model_path": args.model_path,
            "k": args.k,
            "n_examples": n_examples,
            "moe_layers": moe_layers,
            "key_layers": key_layers,
            "save_logits": args.save_logits,
            "elapsed_seconds": elapsed,
            "seconds_per_example": elapsed / n_examples,
        }, f, indent=2)

    # ==================================================================
    # Print summary
    # ==================================================================
    lines = []
    lines.append("MoE Router Analysis")
    lines.append("=" * 100)
    lines.append(f"500 examples, dots_50 (k={args.k}) vs baseline, {len(moe_layers)} MoE layers")
    lines.append("")

    # --- Part 1: Filler diversity ---
    lines.append("PART 1: Filler Routing Diversity (unique experts across 50 filler positions)")
    lines.append("-" * 100)
    header = (f"{'Layer':>6}  {'filler_helped':>20}  {'both_correct':>20}  "
              f"{'both_wrong':>20}  {'p(h vs c)':>12}  {'p(h vs w)':>12}")
    lines.append(header)
    lines.append("-" * len(header))

    for layer_idx in moe_layers:
        lk = str(layer_idx)
        parts = [f"L{layer_idx:>4}"]
        for cat in ["filler_helped", "both_correct", "both_wrong"]:
            d = filler_div_results[lk].get(cat, {})
            if d:
                parts.append(f"{d['mean']:6.1f} +/- {d['std']:5.1f} (n={d['n']:>3})")
            else:
                parts.append(f"{'---':>20}")

        for test_key in ["helped_vs_correct", "helped_vs_wrong"]:
            t = stat_tests["filler_diversity"].get(lk, {}).get(test_key, {})
            if t:
                p = t["p"]
                p_str = f"{p:.4f}" if p >= 0.001 else f"{p:.1e}"
                if p < 0.001:
                    p_str += " ***"
                elif p < 0.01:
                    p_str += " **"
                elif p < 0.05:
                    p_str += " *"
            else:
                p_str = ""
            parts.append(f"{p_str:>12}")

        marker = " <--" if layer_idx in key_layers else ""
        lines.append("  ".join(parts) + marker)

    # --- Part 2: Cross-condition overlap ---
    lines.append("")
    lines.append("PART 2: Cross-Condition Expert Overlap (Jaccard: baseline vs dots_50)")
    lines.append("-" * 100)

    for pos_label, pos_key in [("answer_prompt", "answer_prompt"),
                                ("pre_answer", "pre_answer")]:
        lines.append(f"\n  Position: {pos_label}")
        header = (f"  {'Layer':>6}  {'filler_helped':>18}  {'both_correct':>18}  "
                  f"{'both_wrong':>18}  {'p(h vs c)':>12}")
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))

        for layer_idx in moe_layers:
            lk = str(layer_idx)
            parts = [f"  L{layer_idx:>4}"]
            for cat in ["filler_helped", "both_correct", "both_wrong"]:
                d = overlap_results[pos_key][lk].get(cat, {})
                if d:
                    parts.append(f"{d['mean_jaccard']:.3f} ({d['mean_shared']:.1f}/8)")
                else:
                    parts.append(f"{'---':>18}")

            t = stat_tests.get(f"{pos_key}_overlap", {}).get(lk, {}).get("helped_vs_correct", {})
            if t:
                p = t["p"]
                p_str = f"{p:.4f}" if p >= 0.001 else f"{p:.1e}"
                if p < 0.001:
                    p_str += " ***"
                elif p < 0.01:
                    p_str += " **"
                elif p < 0.05:
                    p_str += " *"
            else:
                p_str = ""
            parts.append(f"{p_str:>12}")

            marker = " <--" if layer_idx in key_layers else ""
            lines.append("  ".join(parts) + marker)

    lines.append("")
    lines.append(f"Key layers ({args.key_layers}) marked with <--")

    summary_text = "\n".join(lines)
    print(f"\n{summary_text}")

    with open(args.output_dir / "summary.txt", "w") as f:
        f.write(summary_text + "\n")

    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
