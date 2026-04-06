"""
unsupervised_decode_filler.py

Fully unsupervised pipeline to decode hidden intermediate values from
filler token hidden states. No labels used at any point.

Pipeline:
1. Average hidden states across filler positions per example
2. For each layer: apply RMSNorm → lm_head → argmax over number tokens (0-999)
3. Select best layer (unsupervised):
   - Multiple conditions: cross-condition consistency × diversity
     (do different filler types decode to the same value for the same example?)
   - Single condition: neighbor-layer consistency × diversity (fallback)
4. Output: per-layer decodes + best-layer decode per condition

Post-hoc evaluation (optional, uses labels) checks accuracy against ground truth.

Usage:
    python probing/scripts/unsupervised_decode_filler.py \
        --condition dots_100 dots_10 counting_25 \
        --output-dir probing/results/unsupervised_decode
"""

import argparse
import json
import pickle
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm


# ==============================================================================
# Core decode functions
# ==============================================================================

def rms_norm(x, weight, eps=1e-6):
    """Apply RMSNorm: x / RMS(x) * weight."""
    rms = np.sqrt(np.mean(x ** 2) + eps)
    return (x / rms) * weight


def build_number_token_map(tokenizer, max_val=1000):
    """Map token IDs to integer values for all single-token numbers 0-999."""
    number_tokens = {}
    for val in range(0, max_val):
        ids = tokenizer.encode(str(val), add_special_tokens=False)
        if len(ids) == 1:
            number_tokens[ids[0]] = val
    return number_tokens


def decode_layer(avg_states, lm_head, norm_weight, number_tok_ids, number_tok_vals):
    """Decode all examples at one layer. Returns (predictions, confidences).

    predictions: (N,) int array of predicted values
    confidences: (N,) float array of softmax probability of predicted token
    """
    predictions = []
    confidences = []
    for vec in avg_states:
        h_normed = rms_norm(vec, norm_weight)
        logits = h_normed @ lm_head.T
        num_logits = logits[number_tok_ids]
        # Softmax over number tokens
        num_logits_shifted = num_logits - num_logits.max()
        probs = np.exp(num_logits_shifted) / np.exp(num_logits_shifted).sum()
        best_idx = np.argmax(probs)
        predictions.append(number_tok_vals[best_idx])
        confidences.append(float(probs[best_idx]))
    return np.array(predictions, dtype=np.float32), np.array(confidences)


# ==============================================================================
# Layer selection
# ==============================================================================

def consistency_x_diversity(all_preds, n_examples):
    """Compute consistency×diversity score for each layer.

    consistency: mean fraction agreeing (±5) with L-1 and L+1
    diversity: n_unique_predictions / n_examples
    score: consistency × diversity

    Returns dict: {layer: {"consistency": float, "diversity": float, "score": float}}
    """
    layers = sorted(all_preds.keys())
    scores = {}

    for layer in layers:
        preds = all_preds[layer]
        diversity = len(np.unique(preds)) / n_examples

        # Consistency with neighbors
        neighbors = []
        if layer - 1 in all_preds:
            neighbors.append(np.mean(np.abs(preds - all_preds[layer - 1]) <= 5))
        if layer + 1 in all_preds:
            neighbors.append(np.mean(np.abs(preds - all_preds[layer + 1]) <= 5))
        consistency = np.mean(neighbors) if neighbors else 0.0

        scores[layer] = {
            "consistency": float(consistency),
            "diversity": float(diversity),
            "score": float(consistency * diversity),
        }

    return scores


def cross_condition_scores(all_cond_preds, conditions, n_examples, tol=5):
    """Compute cross-condition agreement × diversity for each layer.

    all_cond_preds: {condition: {layer: np.array(n_examples,)}}

    For each layer, consistency = mean pairwise agreement (±tol) across
    condition pairs for the same example. Diversity = mean fraction of
    unique predictions across conditions.

    Returns: {layer: {"consistency": float, "diversity": float, "score": float}}
    """
    layer_sets = [set(all_cond_preds[c].keys()) for c in conditions]
    common_layers = sorted(set.intersection(*layer_sets))
    cond_pairs = list(combinations(conditions, 2))

    scores = {}
    for layer in common_layers:
        # Pairwise agreement
        pair_agreements = []
        for c1, c2 in cond_pairs:
            agree = np.mean(np.abs(all_cond_preds[c1][layer] - all_cond_preds[c2][layer]) <= tol)
            pair_agreements.append(agree)
        consistency = float(np.mean(pair_agreements))

        # Average diversity across conditions
        diversities = [len(np.unique(all_cond_preds[c][layer])) / n_examples
                       for c in conditions]
        diversity = float(np.mean(diversities))

        scores[layer] = {
            "consistency": consistency,
            "diversity": diversity,
            "score": consistency * diversity,
        }

    return scores


# ==============================================================================
# Data loading
# ==============================================================================

def load_and_average_filler(extraction_dir, condition, layer, filter_idx=None):
    """Load hidden states, average across filler positions.

    Returns:
        avg_states: list of (7168,) float32 arrays
        metadata: list of dicts
    """
    cond_dir = Path(extraction_dir) / condition
    files = sorted(cond_dir.glob("prob_*.pkl"))

    avg_states = []
    metadata = []

    for f in files:
        with open(f, "rb") as fp:
            d = pickle.load(fp)

        if filter_idx is not None and d["problem_idx"] not in filter_idx:
            continue

        filler_positions = [p for p in d["states"]
                            if p.startswith("filler_k") or p == "pre_filler"]
        if not filler_positions:
            continue

        vecs = [d["states"][p][layer].astype(np.float32)
                for p in filler_positions if layer in d["states"][p]]
        if not vecs:
            continue
        avg_states.append(np.mean(vecs, axis=0))
        metadata.append({
            "problem_idx": d["problem_idx"],
            "fact_value": d["fact_value"],
            "x": d["x"],
            "answer": d["answer"],
            "model_correct": d.get("model_correct", False),
        })

    return avg_states, metadata


def build_category_filter(extraction_dir, condition, categories=("both_correct", "filler_helped")):
    """Build filter set from baseline vs condition correctness."""
    bl_dir = Path(extraction_dir) / "baseline"
    cond_dir = Path(extraction_dir) / condition

    if not bl_dir.exists():
        return None

    bl_files = sorted(bl_dir.glob("prob_*.pkl"))
    cond_files = sorted(cond_dir.glob("prob_*.pkl"))

    filter_idx = set()
    for bf, cf in zip(bl_files, cond_files):
        with open(bf, "rb") as f:
            bd = pickle.load(f)
        with open(cf, "rb") as f:
            cd = pickle.load(f)
        idx = bd["problem_idx"]
        bc = bd.get("model_correct", False)
        fc = cd.get("model_correct", False)
        if bc and fc:
            cat = "both_correct"
        elif not bc and fc:
            cat = "filler_helped"
        elif bc and not fc:
            cat = "filler_hurt"
        else:
            cat = "both_wrong"
        if cat in categories:
            filter_idx.add(idx)

    return filter_idx


# ==============================================================================
# Evaluation (post-hoc, uses labels)
# ==============================================================================

def evaluate(predictions, A):
    """Post-hoc evaluation metrics. Only used for reporting, not for any decisions."""
    errors = np.abs(predictions - A)
    return {
        "mae": float(np.mean(errors)),
        "median_ae": float(np.median(errors)),
        "frac_exact": float(np.mean(predictions == A)),
        "frac_within_5": float(np.mean(errors <= 5)),
        "frac_within_10": float(np.mean(errors <= 10)),
        "r": float(stats.pearsonr(predictions, A)[0]) if np.std(predictions) > 0 else 0.0,
    }


# ==============================================================================
# Main pipeline
# ==============================================================================

def run_decode_pipeline(extraction_dir, condition, lm_head, norm_weight,
                        number_tok_ids, number_tok_vals, filter_idx=None,
                        layers=None):
    """Run the full unsupervised decode pipeline for one condition.

    Returns:
        results: dict with per-layer decodes, layer scores, and best-layer selection
    """
    # Determine available layers from first file
    cond_dir = Path(extraction_dir) / condition
    first_file = sorted(cond_dir.glob("prob_*.pkl"))[0]
    with open(first_file, "rb") as f:
        d = pickle.load(f)
    filler_pos = [p for p in d["states"] if p.startswith("filler_k") or p == "pre_filler"]
    available_layers = sorted(d["states"][filler_pos[0]].keys())

    if layers is None:
        layers = available_layers

    # Decode at every layer
    all_preds = {}
    all_confidences = {}
    metadata = None
    n_examples = None

    for layer in tqdm(layers, desc=f"{condition} decode"):
        avg_states, meta = load_and_average_filler(
            extraction_dir, condition, layer, filter_idx
        )
        if metadata is None:
            metadata = meta
            n_examples = len(meta)

        preds, confs = decode_layer(
            avg_states, lm_head, norm_weight, number_tok_ids, number_tok_vals
        )
        all_preds[layer] = preds
        all_confidences[layer] = confs

    # Layer selection via consistency×diversity
    layer_scores = consistency_x_diversity(all_preds, n_examples)
    best_layer = max(layer_scores.keys(), key=lambda l: layer_scores[l]["score"])

    # Build results
    A = np.array([m["fact_value"] for m in metadata], dtype=np.float32)

    results = {
        "condition": condition,
        "n_examples": n_examples,
        "n_layers": len(layers),
        "filler_positions": filler_pos,
        "best_layer": int(best_layer),
        "best_layer_score": layer_scores[best_layer],
        "layer_scores": {str(l): s for l, s in layer_scores.items()},
        "per_layer": {},
        "best_decode": {
            "layer": int(best_layer),
            "predictions": all_preds[best_layer].tolist(),
            "confidences": all_confidences[best_layer].tolist(),
        },
        "metadata": [{"problem_idx": m["problem_idx"], "fact_value": m["fact_value"],
                       "x": m["x"], "answer": m["answer"]}
                      for m in metadata],
    }

    # Per-layer results
    for layer in layers:
        layer_result = {
            "predictions": all_preds[layer].tolist(),
            "score": layer_scores[layer],
        }
        # Post-hoc evaluation
        layer_result["eval"] = evaluate(all_preds[layer], A)
        results["per_layer"][str(layer)] = layer_result

    # Best-layer evaluation
    results["best_decode"]["eval"] = evaluate(all_preds[best_layer], A)

    # Chance baselines
    baseline_mae = mean_absolute_error(A, np.full(n_examples, A.mean()))
    results["chance_baselines"] = {
        "predict_mean_mae": float(baseline_mae),
        "predict_mean_frac5": float(np.mean(np.abs(A - A.mean()) <= 5)),
        "predict_mean_frac10": float(np.mean(np.abs(A - A.mean()) <= 10)),
    }

    return results


def print_summary(results):
    """Print a readable summary."""
    cond = results["condition"]
    n = results["n_examples"]
    best_l = results["best_layer"]
    best_s = results["best_layer_score"]
    best_e = results["best_decode"]["eval"]
    chance = results["chance_baselines"]

    print(f"\n{'='*70}")
    print(f"  {cond} — {n} examples")
    print(f"{'='*70}")

    # Layer selection
    print(f"\n  Layer selection (consistency × diversity):")
    print(f"  Best layer: L{best_l} (consistency={best_s['consistency']:.1%}, "
          f"diversity={best_s['diversity']:.1%}, score={best_s['score']:.3f})")

    # Best-layer results
    print(f"\n  Best-layer decode (L{best_l}):")
    print(f"    MAE:       {best_e['mae']:.1f}  (chance: {chance['predict_mean_mae']:.1f})")
    print(f"    Median AE: {best_e['median_ae']:.1f}")
    print(f"    Exact:     {best_e['frac_exact']:.1%}  (chance: ~1%)")
    print(f"    ±5:        {best_e['frac_within_5']:.1%}  (chance: {chance['predict_mean_frac5']:.1%})")
    print(f"    ±10:       {best_e['frac_within_10']:.1%}  (chance: {chance['predict_mean_frac10']:.1%})")

    # Per-layer trajectory (every 5th + best)
    print(f"\n  Per-layer trajectory:")
    print(f"  {'Layer':>6} {'MAE':>7} {'MedAE':>7} {'Exact':>7} {'±5':>6} {'±10':>6} {'Score':>7}")
    print(f"  {'-'*48}")
    layers = sorted(int(l) for l in results["per_layer"].keys())
    for layer in layers:
        e = results["per_layer"][str(layer)]["eval"]
        s = results["per_layer"][str(layer)]["score"]["score"]
        marker = " ◀" if layer == best_l else ""
        if layer % 5 == 0 or layer == best_l or e["frac_within_5"] > 0.3:
            print(f"  L{layer:>4} {e['mae']:>7.1f} {e['median_ae']:>7.1f} "
                  f"{e['frac_exact']:>6.1%} {e['frac_within_5']:>5.1%} "
                  f"{e['frac_within_10']:>5.1%} {s:>7.3f}{marker}")


def main():
    parser = argparse.ArgumentParser(
        description="Unsupervised decode of hidden values from filler hidden states"
    )
    parser.add_argument("--extraction-dir", type=Path,
                        default=Path("probing/extracted_states"))
    parser.add_argument("--condition", type=str, nargs="+", default=["dots_100"])
    parser.add_argument("--lm-head", type=Path,
                        default=Path("probing/lm_head_weight.npy"))
    parser.add_argument("--rms-norm", type=Path,
                        default=Path("probing/rms_norm_weight.npy"))
    parser.add_argument("--model-path", type=str,
                        default="/workspace/models/deepseek-v3-awq")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("probing/results/unsupervised_decode"))
    parser.add_argument("--layers", type=str, default="all",
                        help="Comma-separated layers or 'all'")
    parser.add_argument("--filter-categories", nargs="+",
                        default=["both_correct", "filler_helped"])
    parser.add_argument("--no-filter", action="store_true",
                        help="Use all examples (no category filter)")
    parser.add_argument("--tol", type=int, default=5,
                        help="Tolerance for cross-condition agreement (default: ±5)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer and number map
    from transformers import PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model_path)
    number_tokens = build_number_token_map(tokenizer)
    number_tok_ids = sorted(number_tokens.keys())
    number_tok_vals = np.array([number_tokens[tid] for tid in number_tok_ids])
    print(f"Number tokens: {len(number_tokens)} (0-{max(number_tokens.values())})")

    # Load weights
    lm_head = np.load(args.lm_head).astype(np.float32)
    norm_weight = np.load(args.rms_norm).astype(np.float32)
    print(f"lm_head: {lm_head.shape}, norm: {norm_weight.shape}")

    # Parse layers
    layers = None
    if args.layers != "all":
        layers = [int(l) for l in args.layers.split(",")]

    # Build common filter (intersection across all conditions)
    filter_idx = None
    if not args.no_filter:
        filter_sets = []
        for cond in args.condition:
            fset = build_category_filter(
                args.extraction_dir, cond, tuple(args.filter_categories)
            )
            if fset is not None:
                filter_sets.append(fset)
        if filter_sets:
            filter_idx = set.intersection(*filter_sets)
            print(f"Common filtered examples: {len(filter_idx)} ({args.filter_categories})")

    # Decode all conditions at every layer
    all_cond_preds = {}   # {condition: {layer: preds}}
    all_cond_confs = {}   # {condition: {layer: confs}}
    all_cond_meta = {}    # {condition: metadata}

    for condition in args.condition:
        all_cond_preds[condition] = {}
        all_cond_confs[condition] = {}

        # Determine layers from first file
        cond_dir = Path(args.extraction_dir) / condition
        first_file = sorted(cond_dir.glob("prob_*.pkl"))[0]
        with open(first_file, "rb") as f:
            d = pickle.load(f)
        filler_pos = [p for p in d["states"]
                      if p.startswith("filler_k") or p == "pre_filler"]
        available_layers = sorted(d["states"][filler_pos[0]].keys())
        cond_layers = layers if layers is not None else available_layers

        for layer in tqdm(cond_layers, desc=f"Decode {condition}"):
            avg_states, meta = load_and_average_filler(
                args.extraction_dir, condition, layer, filter_idx
            )
            preds, confs = decode_layer(
                avg_states, lm_head, norm_weight, number_tok_ids, number_tok_vals
            )
            all_cond_preds[condition][layer] = preds
            all_cond_confs[condition][layer] = confs

        all_cond_meta[condition] = meta

    # Layer selection
    n_examples = len(all_cond_meta[args.condition[0]])
    use_cross = len(args.condition) >= 2

    if use_cross:
        layer_scores = cross_condition_scores(
            all_cond_preds, args.condition, n_examples, tol=args.tol
        )
        selection_method = "cross-condition"
        print(f"\nLayer selection: cross-condition consistency × diversity "
              f"({len(args.condition)} conditions, ±{args.tol})")
    else:
        layer_scores = consistency_x_diversity(
            all_cond_preds[args.condition[0]], n_examples
        )
        selection_method = "within-condition"
        print(f"\nLayer selection: within-condition neighbor consistency × diversity")

    best_layer = max(layer_scores.keys(), key=lambda l: layer_scores[l]["score"])
    bs = layer_scores[best_layer]
    print(f"Best layer: L{best_layer} (consistency={bs['consistency']:.1%}, "
          f"diversity={bs['diversity']:.1%}, score={bs['score']:.3f})")

    # Build and save per-condition results using the shared best layer
    for condition in args.condition:
        A = np.array([m["fact_value"] for m in all_cond_meta[condition]],
                     dtype=np.float32)
        cond_layers = sorted(all_cond_preds[condition].keys())

        results = {
            "condition": condition,
            "n_examples": n_examples,
            "n_layers": len(cond_layers),
            "selection_method": selection_method,
            "best_layer": int(best_layer),
            "best_layer_score": layer_scores[best_layer],
            "layer_scores": {str(l): s for l, s in layer_scores.items()},
            "per_layer": {},
            "best_decode": {
                "layer": int(best_layer),
                "predictions": all_cond_preds[condition][best_layer].tolist(),
                "confidences": all_cond_confs[condition][best_layer].tolist(),
            },
            "metadata": [{"problem_idx": m["problem_idx"],
                           "fact_value": m["fact_value"],
                           "x": m["x"], "answer": m["answer"]}
                          for m in all_cond_meta[condition]],
        }

        # Per-layer eval
        for layer in cond_layers:
            results["per_layer"][str(layer)] = {
                "score": layer_scores.get(layer, {}),
                "eval": evaluate(all_cond_preds[condition][layer], A),
            }

        # Best-layer eval
        results["best_decode"]["eval"] = evaluate(
            all_cond_preds[condition][best_layer], A
        )

        # Chance baselines
        baseline_mae = mean_absolute_error(A, np.full(n_examples, A.mean()))
        results["chance_baselines"] = {
            "predict_mean_mae": float(baseline_mae),
            "predict_mean_frac5": float(np.mean(np.abs(A - A.mean()) <= 5)),
            "predict_mean_frac10": float(np.mean(np.abs(A - A.mean()) <= 10)),
        }

        print_summary(results)

        # Save (slim — drop per-layer predictions)
        save_results = dict(results)
        for layer_key in save_results["per_layer"]:
            save_results["per_layer"][layer_key].pop("predictions", None)
        outfile = args.output_dir / f"decode_{condition}.json"
        with open(outfile, "w") as f:
            json.dump(save_results, f, indent=2)
        print(f"\n  Saved to {outfile}")

    print("\nDone.")


if __name__ == "__main__":
    main()
