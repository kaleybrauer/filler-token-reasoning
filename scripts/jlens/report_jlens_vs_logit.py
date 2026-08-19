"""
report_jlens_vs_logit.py — head-to-head on the 2-fact decode heatmap.

Reads the two decode_2fact_*.json files written by
scripts/analysis/decode_2fact_heatmap.py (with and without --jlens) and reports,
per target, both comparisons that matter:

  best-cell   the peak (layer, position) each arm reaches — what the heatmap
              figure shows, and what the published numbers quote
  same-cell   each arm evaluated at the OTHER arm's peak, plus the paired
              cell-wise difference over the shared grid, so a win cannot come
              from picking a different cell

Both arms are restricted to the layers the lens was fitted at; accuracies are
reported per arm (never as a delta alone).

Usage:
    python scripts/jlens/report_jlens_vs_logit.py \
        --logit results/unsupervised_decode_2fact/decode_2fact_dots_10.json \
        --jlens results/jlens_decode_2fact/decode_2fact_dots_10_jlens.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TARGETS = [("frac_A1_exact", "A1 exact"), ("frac_A2_exact", "A2 exact"),
           ("frac_A1A2_exact", "sum exact"), ("frac_A1_within5", "A1 +/-5"),
           ("frac_A2_within5", "A2 +/-5"), ("frac_A1A2_within5", "sum +/-5")]


def grid(data: dict, metric: str, positions: list[str], layers: list[str]) -> np.ndarray:
    out = np.full((len(layers), len(positions)), np.nan)
    for j, pos in enumerate(positions):
        for i, layer in enumerate(layers):
            cell = data.get(pos, {}).get(layer)
            if cell is not None:
                out[i, j] = cell[metric]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logit", type=Path, required=True)
    ap.add_argument("--jlens", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    a = json.loads(args.logit.read_text())
    b = json.loads(args.jlens.read_text())

    positions = [p for p in a["_positions"] if p in b["_positions"]]
    layers = sorted({str(l) for l in a["_layers"]} & {str(l) for l in b["_layers"]},
                    key=int)
    print(f"condition {a.get('_condition')}  n={a.get('_n')} (logit) / {b.get('_n')} (jlens)")
    print(f"shared grid: {len(layers)} layers [{layers[0]}..{layers[-1]}] x "
          f"{len(positions)} positions\n")
    if a.get("_n") != b.get("_n"):
        print("WARNING: the two arms scored different example counts\n")

    rows = []
    header = (f"{'target':<11} {'arm':<6} {'peak cell':<16} {'peak':>7} "
              f"{'other@peak':>11} {'mean diff':>10}")
    print(header)
    print("-" * len(header))
    for metric, label in TARGETS:
        A, B = grid(a, metric, positions, layers), grid(b, metric, positions, layers)
        both = ~(np.isnan(A) | np.isnan(B))
        mean_diff = float(np.mean(B[both] - A[both])) if both.any() else float("nan")
        for arm, (G, other) in (("logit", (A, B)), ("jlens", (B, A))):
            if not np.isfinite(G).any():
                continue
            i, j = np.unravel_index(np.nanargmax(G), G.shape)
            cell = f"L{layers[i]},{positions[j]}"
            rows.append({"target": label, "arm": arm, "peak_cell": cell,
                         "peak": float(G[i, j]), "other_at_peak": float(other[i, j]),
                         "mean_cellwise_diff_jlens_minus_logit": mean_diff})
            print(f"{label:<11} {arm:<6} {cell:<16} {G[i, j] * 100:6.1f}% "
                  f"{other[i, j] * 100:10.1f}% {mean_diff * 100:9.2f}pp")
        print()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"logit_file": str(args.logit), "jlens_file": str(args.jlens),
             "layers": layers, "positions": positions, "rows": rows}, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
