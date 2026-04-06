"""
cross_condition_layer_select.py

Cross-condition consistency as an unsupervised layer selection criterion.

Idea: for a given layer, decode each example under multiple filler conditions.
If the decoded values agree across conditions for the same example, the layer
is extracting a real signal (the underlying fact value A) rather than
condition-specific noise. This is a stronger unsupervised signal than
within-condition neighbor consistency because unrelated filler types shouldn't
produce the same artifact.

Metrics:
- cross_agreement: fraction of examples where all condition-pairs agree (±5)
- diversity: average fraction of unique predictions across conditions
- score: cross_agreement × diversity

Usage:
    python probing/scripts/cross_condition_layer_select.py \
        --condition dots_10 dots_100 counting_25

    # Compare with within-condition selection:
    python probing/scripts/cross_condition_layer_select.py \
        --condition dots_10 dots_100 counting_25 --compare-within
"""

import argparse
import json
import pickle
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats
from tqdm import tqdm

from unsupervised_decode_filler import (
    build_category_filter,
    build_number_token_map,
    decode_layer,
    evaluate,
    load_and_average_filler,
    rms_norm,
)


def cross_condition_scores(all_preds, conditions, n_examples, tol=5):
    """Compute cross-condition agreement × diversity for each layer.

    all_preds: {condition: {layer: np.array(n_examples,)}}

    Returns: {layer: {"cross_agreement": float, "diversity": float, "score": float}}
    """
    # Find layers common to all conditions
    layer_sets = [set(all_preds[c].keys()) for c in conditions]
    common_layers = sorted(set.intersection(*layer_sets))
    cond_pairs = list(combinations(conditions, 2))

    scores = {}
    for layer in common_layers:
        # Cross-condition agreement: mean pairwise agreement (±tol)
        pair_agreements = []
        for c1, c2 in cond_pairs:
            agree = np.mean(np.abs(all_preds[c1][layer] - all_preds[c2][layer]) <= tol)
            pair_agreements.append(agree)
        cross_agreement = np.mean(pair_agreements)

        # Diversity: average across conditions
        diversities = []
        for c in conditions:
            diversities.append(len(np.unique(all_preds[c][layer])) / n_examples)
        diversity = np.mean(diversities)

        scores[layer] = {
            "cross_agreement": float(cross_agreement),
            "diversity": float(diversity),
            "score": float(cross_agreement * diversity),
            "pair_agreements": {
                f"{c1}_vs_{c2}": float(a) for (c1, c2), a in zip(cond_pairs, pair_agreements)
            },
        }

    return scores


def main():
    parser = argparse.ArgumentParser(
        description="Cross-condition layer selection for unsupervised decode"
    )
    parser.add_argument("--extraction-dir", type=Path,
                        default=Path("probing/extracted_states"))
    parser.add_argument("--condition", type=str, nargs="+",
                        default=["dots_10", "dots_100", "counting_25"])
    parser.add_argument("--lm-head", type=Path,
                        default=Path("probing/lm_head_weight.npy"))
    parser.add_argument("--rms-norm", type=Path,
                        default=Path("probing/rms_norm_weight.npy"))
    parser.add_argument("--model-path", type=str,
                        default="/workspace/models/deepseek-v3-awq")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("probing/results/unsupervised_decode"))
    parser.add_argument("--filter-categories", nargs="+",
                        default=["both_correct", "filler_helped"])
    parser.add_argument("--no-filter", action="store_true")
    parser.add_argument("--compare-within", action="store_true",
                        help="Also show within-condition consistency×diversity for comparison")
    parser.add_argument("--tol", type=int, default=5,
                        help="Tolerance for agreement (default: ±5)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    assert len(args.condition) >= 2, "Need at least 2 conditions for cross-condition comparison"

    # Load tokenizer and number map
    from transformers import PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model_path)
    number_tokens = build_number_token_map(tokenizer)
    number_tok_ids = sorted(number_tokens.keys())
    number_tok_vals = np.array([number_tokens[tid] for tid in number_tok_ids])
    print(f"Number tokens: {len(number_tokens)}")

    # Load weights
    lm_head = np.load(args.lm_head).astype(np.float32)
    norm_weight = np.load(args.rms_norm).astype(np.float32)

    # Get common examples across all conditions
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
            print(f"Common filtered examples: {len(filter_idx)}")

    # Determine common layers
    cond_dir = Path(args.extraction_dir) / args.condition[0]
    first_file = sorted(cond_dir.glob("prob_*.pkl"))[0]
    with open(first_file, "rb") as f:
        d = pickle.load(f)
    filler_pos = [p for p in d["states"] if p.startswith("filler_k") or p == "pre_filler"]
    layers = sorted(d["states"][filler_pos[0]].keys())
    print(f"Layers: {len(layers)} ({layers[0]}-{layers[-1]})")

    # Decode all conditions × all layers
    all_preds = {}  # {condition: {layer: np.array}}
    all_metadata = {}

    for cond in args.condition:
        all_preds[cond] = {}
        for layer in tqdm(layers, desc=f"Decode {cond}"):
            avg_states, meta = load_and_average_filler(
                args.extraction_dir, cond, layer, filter_idx
            )
            preds, _ = decode_layer(avg_states, lm_head, norm_weight,
                                    number_tok_ids, number_tok_vals)
            all_preds[cond][layer] = preds

        all_metadata[cond] = meta

    n_examples = len(all_metadata[args.condition[0]])
    A = np.array([m["fact_value"] for m in all_metadata[args.condition[0]]], dtype=np.float32)
    print(f"\n{n_examples} examples")

    # Cross-condition scores
    cc_scores = cross_condition_scores(all_preds, args.condition, n_examples, tol=args.tol)
    best_layer = max(cc_scores.keys(), key=lambda l: cc_scores[l]["score"])

    # Within-condition scores (for comparison)
    from unsupervised_decode_filler import consistency_x_diversity
    within_scores = {}
    within_best = {}
    for cond in args.condition:
        ws = consistency_x_diversity(all_preds[cond], n_examples)
        within_scores[cond] = ws
        within_best[cond] = max(ws.keys(), key=lambda l: ws[l]["score"])

    # Print results
    print(f"\n{'='*75}")
    print(f"  CROSS-CONDITION LAYER SELECTION ({' × '.join(args.condition)})")
    print(f"{'='*75}")

    print(f"\n  Best layer: L{best_layer}")
    bs = cc_scores[best_layer]
    print(f"  Cross-agreement: {bs['cross_agreement']:.1%}")
    print(f"  Diversity:       {bs['diversity']:.1%}")
    print(f"  Score:           {bs['score']:.3f}")
    for pair, val in bs["pair_agreements"].items():
        print(f"    {pair}: {val:.1%}")

    # Compare with within-condition picks
    print(f"\n  Within-condition picks for reference:")
    for cond in args.condition:
        wl = within_best[cond]
        ws = within_scores[cond][wl]
        print(f"    {cond}: L{wl} (cons={ws['consistency']:.2f}, div={ws['diversity']:.2f})")

    # Evaluate cross-condition pick on each condition
    print(f"\n  {'':>16} {'MAE':>7} {'Exact':>7} {'±5':>6} {'±10':>6}")
    print(f"  {'-'*44}")
    for cond in args.condition:
        preds = all_preds[cond][best_layer]
        ev = evaluate(preds, A)
        print(f"  {cond:>16} {ev['mae']:>7.1f} {ev['frac_exact']:>6.1%} "
              f"{ev['frac_within_5']:>5.1%} {ev['frac_within_10']:>5.1%}")

    # For comparison: evaluate within-condition picks
    if args.compare_within:
        print(f"\n  Within-condition picks evaluated:")
        print(f"  {'':>16} {'Layer':>6} {'MAE':>7} {'Exact':>7} {'±5':>6} {'±10':>6}")
        print(f"  {'-'*50}")
        for cond in args.condition:
            wl = within_best[cond]
            preds = all_preds[cond][wl]
            ev = evaluate(preds, A)
            print(f"  {cond:>16} L{wl:>4} {ev['mae']:>7.1f} {ev['frac_exact']:>6.1%} "
                  f"{ev['frac_within_5']:>5.1%} {ev['frac_within_10']:>5.1%}")

    # Layer trajectory table
    print(f"\n  Layer trajectory (score = cross_agreement × diversity):")
    print(f"  {'Layer':>6} {'Cross':>7} {'Div':>6} {'Score':>7}  ", end="")
    for cond in args.condition:
        print(f"  {cond[:8]:>8}", end="")
    print()
    print(f"  {'-'*70}")
    for layer in layers:
        s = cc_scores[layer]
        marker = " ◀" if layer == best_layer else ""
        # Only show every 5th layer + best + high scorers
        if layer % 5 == 0 or layer == best_layer or s["score"] > 0.25:
            print(f"  L{layer:>4} {s['cross_agreement']:>6.1%} {s['diversity']:>5.1%} "
                  f"{s['score']:>7.3f}", end="  ")
            # Per-condition exact A accuracy at this layer
            for cond in args.condition:
                ev = evaluate(all_preds[cond][layer], A)
                print(f"  {ev['frac_exact']:>7.1%}", end="")
            print(marker)

    # Save results
    results = {
        "conditions": args.condition,
        "n_examples": n_examples,
        "tolerance": args.tol,
        "best_layer": int(best_layer),
        "best_layer_score": cc_scores[best_layer],
        "within_condition_best": {c: int(within_best[c]) for c in args.condition},
        "layer_scores": {str(l): s for l, s in cc_scores.items()},
        "per_condition_eval_at_best": {},
    }
    for cond in args.condition:
        preds = all_preds[cond][best_layer]
        results["per_condition_eval_at_best"][cond] = evaluate(preds, A)

    outfile = args.output_dir / "cross_condition_layer_select.json"
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {outfile}")


if __name__ == "__main__":
    main()
