"""Plot per-position best-layer results as a consistency × diversity scatter.

Usage:
    python plotting/plot_per_position_scatter.py --condition dots_10
"""

import argparse
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

try:
    from adjustText import adjust_text
    HAS_ADJUSTTEXT = True
except ImportError:
    HAS_ADJUSTTEXT = False

plt.rcParams.update({"font.size": 20})


def classify(a1, a2, sum_v):
    # Dominant sum
    if sum_v > 0.3 and sum_v > max(a1, a2) * 1.5:
        return "A₁+A₂", "#d62728", f"{sum_v:.0%}"
    # Dominant A1
    if a1 > 0.3 and a1 > a2 * 1.5 and a1 > sum_v * 1.5:
        return "A₁", "#228B22", f"{a1:.0%}"
    # Dominant A2
    if a2 > 0.3 and a2 > a1 * 1.5 and a2 > sum_v * 1.5:
        return "A₂", "#1f77b4", f"{a2:.0%}"
    # Mixed — show whichever metrics are ≥10%
    if a1 >= 0.10 or a2 >= 0.10 or sum_v >= 0.10:
        parts = []
        if a1 >= 0.10: parts.append(f"A₁={a1:.0%}")
        if a2 >= 0.10: parts.append(f"A₂={a2:.0%}")
        if sum_v >= 0.10: parts.append(f"Σ={sum_v:.0%}")
        return "mixed", "#D4A03C", ", ".join(parts)
    return "unclear", "#888888", None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", type=str, default="dots_10")
    parser.add_argument("--per-pos-file", type=Path, default=None)
    parser.add_argument("--npz", type=Path, default=None)
    parser.add_argument("--extraction-dir", type=Path,
                        default=Path("data/extracted_states_2fact_allpos"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--include-post-filler", action="store_true")
    args = parser.parse_args()

    if args.per_pos_file is None:
        args.per_pos_file = Path(f"results/per_position_select_{args.condition}.json")
    if args.npz is None:
        args.npz = Path(f"results/cross_example_consistency/cross_example_consistency_{args.condition}.npz")

    # Get filler boundary
    ex_file = sorted((args.extraction_dir / args.condition).glob("prob_*.pkl"))[0]
    with open(ex_file, "rb") as f:
        d = pickle.load(f)
    filler_end = d["boundaries"]["filler_end_offset"]

    with open(args.per_pos_file) as f:
        per_pos = json.load(f)["per_position"]
    data = np.load(args.npz, allow_pickle=True)
    settings = data["settings"]
    predictions = data["predictions"]
    A1 = data["A1"]
    A2 = data["A2"]
    A1A2 = data["A1A2"]

    fig, ax = plt.subplots(figsize=(12, 8.5))
    texts = []
    seen_labels = set()

    for pos_str, info in per_pos.items():
        pos_num = int(pos_str.split("_")[1])
        if not args.include_post_filler and pos_num > filler_end:
            continue

        layer = info["layer"]
        idx = [i for i in range(len(settings))
               if str(settings[i][0]) == pos_str and int(settings[i][1]) == layer][0]
        preds = predictions[idx]
        diversity = len(np.unique(preds)) / len(preds)
        a1 = info["a1_exact"]
        a2 = info["a2_exact"]
        sum_v = np.mean(preds == A1A2)
        consistency = info["consistency"]

        label, color, pct_str = classify(a1, a2, sum_v)
        seen_labels.add(label)
        ax.scatter(consistency, diversity, c=color, s=700, alpha=0.75,
                   edgecolor="black", linewidth=1.5)

        text_str = f"pos {pos_num}"
        if pct_str is not None:
            text_str += f"\n{pct_str}"
        t = ax.text(consistency, diversity, text_str, fontsize=16, fontweight="bold",
                    color=color if pct_str else "black")
        texts.append(t)

    # Expand axis limits to give annotations room
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_pad = (xlim[1] - xlim[0]) * 0.08
    y_pad = (ylim[1] - ylim[0]) * 0.10
    ax.set_xlim(xlim[0] - x_pad, xlim[1] + x_pad)
    ax.set_ylim(ylim[0] - y_pad, ylim[1] + y_pad)

    if HAS_ADJUSTTEXT:
        adjust_text(texts, ax=ax,
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.5, alpha=0.5),
                    expand_points=(1.4, 1.8))

    all_legend = [
        ("A₁", "#228B22", "A₁"),
        ("A₂", "#1f77b4", "A₂"),
        ("A₁+A₂", "#d62728", "A₁+A₂"),
        ("Mixed", "#D4A03C", "mixed"),
        ("Unclear", "#888888", "unclear"),
    ]
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=22, label=l)
        for l, c, key in all_legend if key in seen_labels
    ]
    ax.legend(handles=legend_elements, fontsize=20, loc="lower right", framealpha=0.9)
    ax.set_xlabel("Neighbor-layer consistency (best layer)", fontsize=22)
    ax.set_ylabel("Prediction diversity (unique / total)", fontsize=22)
    suffix = " (filler region only)" if not args.include_post_filler else ""
    ax.set_title(f"{args.condition}: per-position best layer{suffix}", fontsize=24)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=16)

    plt.tight_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        outpath = args.output_dir / f"per_position_scatter_{args.condition}.{ext}"
        fig.savefig(outpath, dpi=200, bbox_inches="tight")
        print(f"Saved {outpath}")
    plt.close()


if __name__ == "__main__":
    main()
