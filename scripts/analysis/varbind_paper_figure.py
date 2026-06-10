"""Paper figure: baseline vs filler decoded-value maps, side by side.

Composes the proportional (fractional) decoded-value overlay for two (or more)
conditions into one figure with a shared layer axis and a single legend. Each
(layer, position) cell is a convex blend of the value colours weighted by the
exact-match decode fraction (white = nothing decodes), so saturation = how
strongly something is encoded — honest about magnitude (no per-cell boosting).

Reads the ALL-examples decode JSONs by default (the unbiased population —
correct-only cherry-picks examples the model already solved, inflating baseline).

Usage:
    python scripts/analysis/varbind_paper_figure.py
    python scripts/analysis/varbind_paper_figure.py --conditions baseline dots_10 dots_50
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

# (json key, legend label, RGB) — computation order, cool -> warm. Labels are
# intentionally terse; define them in the caption.
TARGETS = [
    ("frac_B_exact",   "x",      (0.12, 0.47, 0.71)),  # blue  — base value (given)
    ("frac_C1B_exact", "c₁·x",   (0.17, 0.63, 0.17)),  # green — chain product (hidden)
    ("frac_QV_exact",  "y",      (0.85, 0.69, 0.00)),  # gold  — queried value (hidden)
    ("frac_QC_exact",  "c₂·y",   (1.00, 0.50, 0.00)),  # orange — question product (hidden)
    ("frac_ANS_exact", "answer", (0.84, 0.15, 0.16)),  # red   — output
]
KEYS = [t[0] for t in TARGETS]
COLORS = np.array([t[2] for t in TARGETS])


def off(p):
    if p == "question_end": return -2
    if p == "pre_filler": return -1
    if p.startswith("pos_"): return int(p.split("_")[1])
    if p == "answer_prompt": return 99999
    return 9999


def pos_label(p):
    if p in ("question_end",): return "q_end"
    if p == "pre_filler": return "filler_label"
    if p == "answer_prompt": return "ans"
    if p.startswith("pos_"): return p.split("_")[1].lstrip("0") or "0"
    return p


def proportional_rgb(r, layers):
    positions = sorted(r["_positions"], key=off)
    img = np.ones((len(layers), len(positions), 3))
    for j, p in enumerate(positions):
        for i, l in enumerate(layers):
            cell = r.get(p, {}).get(str(l))
            if not cell:
                continue
            fr = np.array([cell.get(k) or 0.0 for k in KEYS])
            total = float(fr.sum())
            img[i, j] = np.clip(max(0.0, 1.0 - total) * np.ones(3)
                                + (fr[:, None] * COLORS).sum(0), 0.0, 1.0)
    return img, positions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", default=["baseline", "dots_10"])
    ap.add_argument("--titles", nargs="+",
                    default=["no filler", "filler (k=10)"])
    ap.add_argument("--subset", default="all", choices=["all", "correct", "incorrect"],
                    help="Which decode JSON to read; 'all' = unbiased population (default).")
    ap.add_argument("--decode-dir", type=Path, default=Path("results/varbind_decode_heatmap"))
    ap.add_argument("--min-layer", type=int, default=30)
    ap.add_argument("--output", type=Path,
                    default=Path("results/varbind_decode_heatmap/figure_baseline_vs_filler"))
    args = ap.parse_args()

    sfx = "" if args.subset == "correct" else f"_{args.subset}"
    data, layers = [], None
    for cond in args.conditions:
        r = json.load(open(args.decode_dir / f"decode_varbind_{cond}{sfx}.json"))
        layers = [l for l in r["_layers"] if l >= args.min_layer]
        img, positions = proportional_rgb(r, layers)
        data.append((cond, r, img, positions))

    n_pos = [len(d[3]) for d in data]
    widths = [max(2.5, n) for n in n_pos]                    # keep narrow panels legible
    fig = plt.figure(figsize=(sum(widths) * 0.40 + 0.6, 6.4))
    gs = fig.add_gridspec(1, len(data), width_ratios=widths, wspace=0.05)

    for k, (cond, r, img, positions) in enumerate(data):
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(img, aspect="auto", interpolation="nearest")
        # x ticks
        if len(positions) > 6:
            step = max(1, len(positions) // 12)
            idxs = list(range(0, len(positions), step))
            if (len(positions) - 1) not in idxs:
                idxs.append(len(positions) - 1)
        else:
            idxs = list(range(len(positions)))
        ax.set_xticks(idxs)
        ax.set_xticklabels([pos_label(positions[i]) for i in idxs], rotation=45, ha="right",
                           fontsize=10)
        ax.set_xlabel("token position", fontsize=12)
        b = r.get("_boundaries") or {}
        fe = b.get("filler_end_offset")
        if fe is not None:
            ax.axvline(fe + 0.5, color="black", lw=1.0, ls="--", alpha=0.5)
        title = args.titles[k] if k < len(args.titles) else cond
        ax.set_title(title, fontsize=14)
        if k == 0:
            ax.set_yticks(range(len(layers)))
            ax.set_yticklabels([str(l) if l % 5 == 0 else "" for l in layers], fontsize=10)
            ax.set_ylabel("layer", fontsize=12)
        else:
            ax.set_yticks([])

    fig.legend(handles=[Patch(facecolor=c, label=name) for _, name, c in TARGETS],
               loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=len(TARGETS),
               fontsize=13, frameon=False, handlelength=1.3, columnspacing=1.7,
               handletextpad=0.5)
    fig.subplots_adjust(top=0.93, bottom=0.17, left=0.07, right=0.985)
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.output}.{ext}", dpi=200, bbox_inches="tight")
    print(f"saved {args.output}.png / .pdf")


if __name__ == "__main__":
    main()
