"""
per_position_layer_select.py

Simple approach: for each filler position, pick the best layer via neighbor-layer
consistency (per-example agreement with L±1 predictions). Then optionally dedup
positions that encode the same variable (>threshold pairwise agreement).

Unsupervised layer selection without clustering — just per-position consistency
+ optional dedup step.

Usage:
    python scripts/per_position_layer_select.py \
        --condition dots_10 \
        --extraction-dir data/extracted_states_2fact_allpos
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", type=str, default="dots_10")
    parser.add_argument("--npz", type=Path, default=None,
                        help="Path to cross_example_consistency npz "
                             "(reuses cached predictions)")
    parser.add_argument("--dedup-threshold", type=float, default=0.6,
                        help="If two positions have pairwise exact match >= this, "
                             "treat as same variable and keep the one with higher "
                             "neighbor consistency")
    parser.add_argument("--tol", type=int, default=5,
                        help="Tolerance for neighbor consistency matching")
    parser.add_argument("--min-diversity", type=float, default=0.10,
                        help="Minimum prediction diversity (unique/total) to keep a position")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.npz is None:
        args.npz = Path(f"results/cross_example_consistency/cross_example_consistency_{args.condition}.npz")

    data = np.load(args.npz, allow_pickle=True)
    settings = data["settings"]
    predictions = data["predictions"]
    A1 = data["A1"]
    A2 = data["A2"]

    # Build {pos: {layer: idx}} mapping
    pos_layer_idx = {}
    for i, s in enumerate(settings):
        pos, layer = str(s[0]), int(s[1])
        pos_layer_idx.setdefault(pos, {})[layer] = i

    positions = sorted(pos_layer_idx.keys(),
                       key=lambda p: int(p.split("_")[1]) if "_" in p else -1)

    # Step 1: per-position neighbor-layer consistency
    print(f"\n=== Step 1: per-position neighbor-layer consistency ===")
    print(f"{'Position':>12}  {'Best L':>6}  {'Consistency':>11}  {'A1 exact':>8}  {'A2 exact':>8}")
    best_per_pos = {}
    for pos in positions:
        layers = sorted(pos_layer_idx[pos].keys())
        scores = {}
        for layer in layers:
            idx = pos_layer_idx[pos][layer]
            preds = predictions[idx]
            neighbor_agrees = []
            for dl in [-1, 1]:
                if layer + dl in pos_layer_idx[pos]:
                    npreds = predictions[pos_layer_idx[pos][layer + dl]]
                    neighbor_agrees.append(np.mean(np.abs(preds - npreds) <= args.tol))
            if neighbor_agrees:
                scores[layer] = np.mean(neighbor_agrees)

        if not scores:
            continue
        best_layer = max(scores, key=scores.get)
        best_idx = pos_layer_idx[pos][best_layer]
        preds = predictions[best_idx]
        diversity = len(np.unique(preds)) / len(preds)
        if diversity < args.min_diversity:
            continue  # skip degenerate constant-output positions
        a1 = np.mean(preds == A1)
        a2 = np.mean(preds == A2)
        best_per_pos[pos] = {
            "layer": best_layer,
            "consistency": scores[best_layer],
            "a1": a1, "a2": a2,
            "idx": best_idx,
            "diversity": diversity,
        }
        print(f"{pos:>12}  {best_layer:>6}  {scores[best_layer]:>10.1%}  "
              f"{a1:>7.1%}  {a2:>7.1%}")

    # Step 2: dedup by pairwise agreement
    print(f"\n=== Step 2: dedup (threshold={args.dedup_threshold:.0%} pairwise agreement) ===")
    # Rank by consistency (best first), greedily keep settings that disagree with
    # already-kept settings.
    ranked = sorted(best_per_pos.items(),
                    key=lambda x: -x[1]["consistency"])

    kept = []
    for pos, info in ranked:
        preds = predictions[info["idx"]]
        is_dup = False
        for kept_pos, kept_info in kept:
            kept_preds = predictions[kept_info["idx"]]
            agree = np.mean(preds == kept_preds)
            if agree >= args.dedup_threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append((pos, info))

    print(f"Kept {len(kept)} distinct positions (deduped {len(ranked) - len(kept)}):")
    print(f"{'Position':>12}  {'Layer':>6}  {'Consistency':>11}  {'A1 exact':>8}  {'A2 exact':>8}")
    for pos, info in kept:
        print(f"{pos:>12}  {info['layer']:>6}  {info['consistency']:>10.1%}  "
              f"{info['a1']:>7.1%}  {info['a2']:>7.1%}")

    # Save
    out = args.output or Path(f"results/per_position_select_{args.condition}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    save = {
        "condition": args.condition,
        "per_position": {p: {"layer": int(i["layer"]),
                             "consistency": float(i["consistency"]),
                             "a1_exact": float(i["a1"]),
                             "a2_exact": float(i["a2"])}
                         for p, i in best_per_pos.items()},
        "deduped": [{"position": p, "layer": int(i["layer"]),
                     "consistency": float(i["consistency"]),
                     "a1_exact": float(i["a1"]),
                     "a2_exact": float(i["a2"])}
                    for p, i in kept],
        "dedup_threshold": args.dedup_threshold,
    }
    with open(out, "w") as f:
        json.dump(save, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
