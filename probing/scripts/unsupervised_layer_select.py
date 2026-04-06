"""
unsupervised_layer_select.py

Four unsupervised criteria for selecting which layer best decodes A from
averaged filler hidden states via RMSNorm → lm_head → top number token.

Criteria:
1. Self-consistency: decode(avg_filler, L) + Y == decode(answer_prompt, L_final)
2. Cross-condition agreement: dots_100 and dots_10 give same prediction
3. Entropy: how peaked is the number-token logit distribution
4. Filler subset stability: early vs late filler subsets agree

None of these use ground-truth A labels. Post-hoc MAE is reported for
comparison but not used in any criterion.

Usage:
    python probing/scripts/unsupervised_layer_select.py \
        --extraction-dir probing/extracted_states \
        --output-dir probing/results/unsupervised_layer_select
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.stats import entropy
from tqdm import tqdm


def rms_norm(x, weight, eps=1e-6):
    """Apply RMSNorm with learned weight."""
    rms = np.sqrt(np.mean(x ** 2) + eps)
    return (x / rms) * weight


def build_number_token_map(tokenizer, max_val=601):
    """Map token IDs to integer values for all single-token numbers."""
    number_tokens = {}
    for val in range(0, max_val):
        ids = tokenizer.encode(str(val), add_special_tokens=False)
        if len(ids) == 1:
            number_tokens[ids[0]] = val
    return number_tokens


def decode_states(states, lm_head, norm_weight, number_tok_ids, number_tok_vals):
    """Decode hidden states to number predictions via RMSNorm + lm_head.

    Args:
        states: (N, D) array
        lm_head: (vocab, D) array
        norm_weight: (D,) array
        number_tok_ids: list of token IDs for numbers
        number_tok_vals: array of corresponding integer values

    Returns:
        predictions: (N,) int array
        entropies: (N,) float array — entropy of number-token softmax
    """
    predictions = np.zeros(len(states), dtype=np.float32)
    entropies = np.zeros(len(states), dtype=np.float32)

    for i, vec in enumerate(states):
        h = rms_norm(vec, norm_weight)
        logits = h @ lm_head.T
        num_logits = logits[number_tok_ids]
        predictions[i] = number_tok_vals[np.argmax(num_logits)]

        # Entropy of softmax over number tokens
        shifted = num_logits - num_logits.max()
        probs = np.exp(shifted) / np.exp(shifted).sum()
        entropies[i] = entropy(probs)

    return predictions, entropies


def load_condition_data(extraction_dir, condition, filter_idx=None):
    """Load all examples for a condition. Returns per-layer avg filler states,
    answer_prompt states, and metadata."""
    cond_dir = Path(extraction_dir) / condition
    files = sorted(cond_dir.glob("prob_*.pkl"))

    # First pass: identify layers and filler positions
    with open(files[0], "rb") as f:
        d0 = pickle.load(f)
    filler_positions = [p for p in d0["states"]
                        if p.startswith("filler_k") or p == "pre_filler"]
    all_layers = sorted(d0["states"][filler_positions[0]].keys())

    # Identify early vs late filler positions for subset stability
    filler_k_positions = [p for p in filler_positions if p.startswith("filler_k")]
    filler_k_positions.sort(key=lambda p: int(p.split("k")[1]))
    mid = len(filler_k_positions) // 2
    early_filler = filler_k_positions[:mid] if mid > 0 else filler_k_positions
    late_filler = filler_k_positions[mid:] if mid > 0 else filler_k_positions

    # Second pass: load all data
    metadata = []
    # avg_filler[layer] = list of (D,) vectors
    avg_filler = {l: [] for l in all_layers}
    avg_early = {l: [] for l in all_layers}
    avg_late = {l: [] for l in all_layers}
    answer_prompt = {l: [] for l in all_layers}

    for f in files:
        with open(f, "rb") as fp:
            d = pickle.load(fp)

        if filter_idx is not None and d["problem_idx"] not in filter_idx:
            continue

        metadata.append({
            "problem_idx": d["problem_idx"],
            "fact_value": d["fact_value"],
            "x": d["x"],
            "answer": d["answer"],
            "model_correct": d.get("model_correct", False),
        })

        for layer in all_layers:
            # Full average
            vecs = [d["states"][p][layer].astype(np.float32)
                    for p in filler_positions if layer in d["states"].get(p, {})]
            avg_filler[layer].append(np.mean(vecs, axis=0))

            # Early/late subsets
            early_vecs = [d["states"][p][layer].astype(np.float32)
                          for p in early_filler if layer in d["states"].get(p, {})]
            late_vecs = [d["states"][p][layer].astype(np.float32)
                         for p in late_filler if layer in d["states"].get(p, {})]
            if early_vecs:
                avg_early[layer].append(np.mean(early_vecs, axis=0))
            if late_vecs:
                avg_late[layer].append(np.mean(late_vecs, axis=0))

            # Answer prompt
            if "answer_prompt" in d["states"] and layer in d["states"]["answer_prompt"]:
                answer_prompt[layer].append(d["states"]["answer_prompt"][layer].astype(np.float32))

    # Stack
    for layer in all_layers:
        avg_filler[layer] = np.stack(avg_filler[layer])
        if avg_early[layer]:
            avg_early[layer] = np.stack(avg_early[layer])
        if avg_late[layer]:
            avg_late[layer] = np.stack(avg_late[layer])
        if answer_prompt[layer]:
            answer_prompt[layer] = np.stack(answer_prompt[layer])

    return avg_filler, avg_early, avg_late, answer_prompt, metadata, all_layers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extraction-dir", type=Path, default=Path("probing/extracted_states"))
    parser.add_argument("--conditions", nargs="+", default=["dots_100", "dots_10"])
    parser.add_argument("--lm-head", type=Path, default=Path("probing/lm_head_weight.npy"))
    parser.add_argument("--rms-norm", type=Path, default=Path("probing/rms_norm_weight.npy"))
    parser.add_argument("--model-path", type=str, default="/workspace/models/deepseek-v3-awq")
    parser.add_argument("--output-dir", type=Path, default=Path("probing/results/unsupervised_layer_select"))
    parser.add_argument("--filter-categories", nargs="+", default=["both_correct", "filler_helped"])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    from transformers import PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model_path)
    number_tokens = build_number_token_map(tokenizer)
    number_tok_ids = sorted(number_tokens.keys())
    number_tok_vals = np.array([number_tokens[tid] for tid in number_tok_ids], dtype=np.float32)
    print(f"Number tokens: {len(number_tokens)}")

    # Load weights
    lm_head = np.load(args.lm_head).astype(np.float32)
    norm_weight = np.load(args.rms_norm).astype(np.float32)
    print(f"lm_head: {lm_head.shape}, norm: {norm_weight.shape}")

    # Build category filter
    filter_idx = None
    if args.filter_categories:
        bl_dir = args.extraction_dir / "baseline"
        cond_dir = args.extraction_dir / args.conditions[0]
        bl_files = sorted(bl_dir.glob("prob_*.pkl"))
        cond_files = sorted(cond_dir.glob("prob_*.pkl"))
        filter_idx = set()
        for bf, cf in zip(bl_files, cond_files):
            with open(bf, "rb") as f: bd = pickle.load(f)
            with open(cf, "rb") as f: cd = pickle.load(f)
            idx = bd["problem_idx"]
            bc = bd.get("model_correct", False)
            fc = cd.get("model_correct", False)
            if bc and fc: cat = "both_correct"
            elif not bc and fc: cat = "filler_helped"
            elif bc and not fc: cat = "filler_hurt"
            else: cat = "both_wrong"
            if cat in args.filter_categories:
                filter_idx.add(idx)
        print(f"Category filter: {len(filter_idx)} examples ({args.filter_categories})")

    # Load data for each condition
    cond_data = {}
    for cond in args.conditions:
        print(f"\nLoading {cond}...")
        avg_f, avg_e, avg_l, ans_p, meta, layers = load_condition_data(
            args.extraction_dir, cond, filter_idx)
        cond_data[cond] = {
            "avg_filler": avg_f, "avg_early": avg_e, "avg_late": avg_l,
            "answer_prompt": ans_p, "metadata": meta, "layers": layers
        }
        print(f"  {len(meta)} examples, {len(layers)} layers")

    primary = args.conditions[0]
    layers = cond_data[primary]["layers"]
    meta_primary = cond_data[primary]["metadata"]
    A = np.array([m["fact_value"] for m in meta_primary], dtype=np.float32)
    Y = np.array([m["x"] for m in meta_primary], dtype=np.float32)
    n = len(A)

    # --- Criterion 1: Self-consistency ---
    # Decode answer_prompt at the final layer to get the model's "answer"
    print("\nDecoding answer_prompt at final layer...")
    final_layer = layers[-1]
    ans_states = cond_data[primary]["answer_prompt"][final_layer]
    model_answers, _ = decode_states(ans_states, lm_head, norm_weight,
                                      number_tok_ids, number_tok_vals)

    # --- Criterion 2: Cross-condition matching ---
    # Match examples by problem_idx
    if len(args.conditions) >= 2:
        secondary = args.conditions[1]
        idx_pri = {m["problem_idx"]: i for i, m in enumerate(meta_primary)}
        idx_sec = {m["problem_idx"]: i for i, m
                   in enumerate(cond_data[secondary]["metadata"])}
        shared_pidx = sorted(set(idx_pri.keys()) & set(idx_sec.keys()))
        map_pri = [idx_pri[p] for p in shared_pidx]
        map_sec = [idx_sec[p] for p in shared_pidx]
        print(f"Cross-condition shared examples: {len(shared_pidx)}")
    else:
        shared_pidx = []

    # --- Run all criteria per layer ---
    print(f"\n{'Layer':>5} {'MAE':>6} {'±10':>5} {'±5':>5}"
          f" {'self_con':>9} {'cross_ag':>9} {'entropy':>8}"
          f" {'subset_ag':>10}")
    print("-" * 75)

    results = {}

    for layer in tqdm(layers, desc="Layers"):
        # Decode filler at this layer
        filler_states = cond_data[primary]["avg_filler"][layer]
        preds, ents = decode_states(filler_states, lm_head, norm_weight,
                                     number_tok_ids, number_tok_vals)

        # Ground truth metrics (for comparison only)
        mae = float(np.mean(np.abs(preds - A)))
        frac10 = float(np.mean(np.abs(preds - A) <= 10))
        frac5 = float(np.mean(np.abs(preds - A) <= 5))

        # Criterion 1: Self-consistency
        # Does pred + Y == model_answer?
        expected_answer = preds + Y
        self_consistent = float(np.mean(np.abs(expected_answer - model_answers) <= 5))

        # Criterion 2: Cross-condition agreement
        cross_agree = 0.0
        if shared_pidx:
            sec_states = cond_data[secondary]["avg_filler"][layer]
            preds_sec, _ = decode_states(sec_states, lm_head, norm_weight,
                                          number_tok_ids, number_tok_vals)
            p_pri = preds[map_pri]
            p_sec = preds_sec[map_sec]
            cross_agree = float(np.mean(np.abs(p_pri - p_sec) <= 5))

        # Criterion 3: Mean entropy (lower = more confident)
        mean_entropy = float(np.mean(ents))

        # Criterion 4: Filler subset stability
        subset_agree = 0.0
        early_states = cond_data[primary]["avg_early"][layer]
        late_states = cond_data[primary]["avg_late"][layer]
        if isinstance(early_states, np.ndarray) and isinstance(late_states, np.ndarray):
            preds_early, _ = decode_states(early_states, lm_head, norm_weight,
                                            number_tok_ids, number_tok_vals)
            preds_late, _ = decode_states(late_states, lm_head, norm_weight,
                                           number_tok_ids, number_tok_vals)
            subset_agree = float(np.mean(np.abs(preds_early - preds_late) <= 5))

        results[layer] = {
            "mae": mae, "frac_within_10": frac10, "frac_within_5": frac5,
            "self_consistency": self_consistent,
            "cross_condition_agreement": cross_agree,
            "mean_entropy": mean_entropy,
            "subset_stability": subset_agree,
        }

        print(f"L{layer:>3}  {mae:>6.1f} {frac10:>4.1%} {frac5:>4.1%}"
              f" {self_consistent:>9.1%} {cross_agree:>9.1%} {mean_entropy:>8.2f}"
              f" {subset_agree:>10.1%}")

    # Find best layer per criterion
    print(f"\n{'Criterion':<25} {'Best layer':>10} {'Value':>10}")
    print("-" * 50)

    criteria = [
        ("self_consistency", True, "Self-consistency"),
        ("cross_condition_agreement", True, "Cross-condition agree"),
        ("mean_entropy", False, "Entropy (lower=better)"),
        ("subset_stability", True, "Subset stability"),
        ("frac_within_10", True, "±10 (ground truth)"),
    ]
    for key, higher_better, label in criteria:
        if higher_better:
            best_l = max(results.keys(), key=lambda l: results[l][key])
        else:
            best_l = min(results.keys(), key=lambda l: results[l][key])
        print(f"{label:<25} L{best_l:>8} {results[best_l][key]:>10.3f}")

    # Save
    save_results = {str(k): v for k, v in results.items()}
    save_results["_criteria_best"] = {}
    for key, higher_better, label in criteria:
        if higher_better:
            best_l = max(results.keys(), key=lambda l: results[l][key])
        else:
            best_l = min(results.keys(), key=lambda l: results[l][key])
        save_results["_criteria_best"][key] = {"layer": best_l, "value": results[best_l][key]}

    with open(args.output_dir / "layer_selection_criteria.json", "w") as f:
        json.dump(save_results, f, indent=2)

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layer_list = sorted(results.keys())

    plot_criteria = [
        ("self_consistency", "Self-consistency\npred + Y ≈ model answer", True),
        ("cross_condition_agreement", f"Cross-condition agreement\n{args.conditions[0]} vs {args.conditions[1]}", True),
        ("mean_entropy", "Mean entropy\n(lower = more confident)", False),
        ("subset_stability", "Subset stability\nearly vs late filler agree", True),
    ]

    fig, axes = plt.subplots(len(plot_criteria) + 1, 1, figsize=(14, 3 * (len(plot_criteria) + 1)),
                              sharex=True)

    for ax, (key, ylabel, higher_better) in zip(axes, plot_criteria):
        vals = [results[l][key] for l in layer_list]
        ax.plot(layer_list, vals, ".-", markersize=5, linewidth=1.5)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(alpha=0.3)

        if higher_better:
            best_l = layer_list[np.argmax(vals)]
        else:
            best_l = layer_list[np.argmin(vals)]
        ax.axvline(x=best_l, color="red", linestyle="--", alpha=0.5)
        ax.annotate(f"L{best_l}", xy=(best_l, vals[layer_list.index(best_l)]),
                    fontsize=10, color="red",
                    xytext=(best_l + 1.5, vals[layer_list.index(best_l)]))

    axes[0].set_title("Unsupervised layer selection — avg filler → RMSNorm → lm_head", fontsize=13)

    # Ground truth MAE
    ax = axes[-1]
    maes = [results[l]["mae"] for l in layer_list]
    frac10s = [results[l]["frac_within_10"] for l in layer_list]
    ax.plot(layer_list, frac10s, "k.-", markersize=5, linewidth=1.5, label="% within ±10 (ground truth)")
    ax.set_ylabel("% within ±10\n(ground truth)", fontsize=9)
    ax.set_xlabel("Layer")
    ax.grid(alpha=0.3)
    best_l = layer_list[np.argmax(frac10s)]
    ax.axvline(x=best_l, color="black", linestyle="--", alpha=0.5)
    ax.annotate(f"L{best_l}", xy=(best_l, frac10s[layer_list.index(best_l)]),
                fontsize=10, color="black",
                xytext=(best_l + 1.5, frac10s[layer_list.index(best_l)]))

    plt.tight_layout()
    plt.savefig(args.output_dir / "layer_selection_criteria.png", dpi=150, bbox_inches="tight")
    plt.savefig(args.output_dir / "layer_selection_criteria.pdf", bbox_inches="tight")
    print(f"\nPlot saved to {args.output_dir}/layer_selection_criteria.png")
    print(f"Results saved to {args.output_dir}/layer_selection_criteria.json")


if __name__ == "__main__":
    main()
