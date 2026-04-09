"""
plot_rank_scatter.py

Paired scatter plot: baseline rank vs dots_50 rank of the A+Y answer token
at pre_answer, per example. Colored by category.

Usage:
    uv run --project /workspace/filler-token-reasoning/probing \
        python scripts/plot_rank_scatter.py \
        --output-dir plots
"""

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def load_ranks(per_example_file: Path, position: str, layer: str):
    """Load per-example ranks at a given (position, layer)."""
    with open(per_example_file) as f:
        data = json.load(f)
    ranks = {}
    for ex in data:
        entry = ex["positions"].get(position, {}).get(layer)
        if entry:
            ranks[ex["problem_idx"]] = {
                "rank": entry["answer_rank"],
                "category": ex.get("category", "unknown"),
                "answer": ex["answer"],
                "fact_value": ex["fact_value"],
                "x": ex["x"],
            }
    return ranks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-file", type=Path,
                        default=Path("logit_lens_results/all_baseline/per_example.json"))
    parser.add_argument("--dots50-file", type=Path,
                        default=Path("logit_lens_results/all_dots50/per_example.json"))
    parser.add_argument("--position", default="pre_answer")
    parser.add_argument("--layer", default="60")
    parser.add_argument("--output-dir", type=Path, default=Path("plots"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    bl_ranks = load_ranks(args.baseline_file, args.position, args.layer)
    d50_ranks = load_ranks(args.dots50_file, args.position, args.layer)

    # Match examples present in both
    common_idxs = sorted(set(bl_ranks.keys()) & set(d50_ranks.keys()))
    print(f"{len(common_idxs)} examples in common")

    # Category colors and markers
    cat_style = {
        "both_correct":  {"color": "#2196F3", "marker": "o", "alpha": 0.5, "zorder": 2},
        "filler_helped": {"color": "#4CAF50", "marker": "D", "alpha": 0.8, "zorder": 4},
        "both_wrong":    {"color": "#F44336", "marker": "s", "alpha": 0.5, "zorder": 3},
        "filler_hurt":   {"color": "#FF9800", "marker": "^", "alpha": 0.8, "zorder": 4},
    }

    # Build arrays by category (use dots_50 categories as canonical)
    by_cat = {}
    for idx in common_idxs:
        cat = d50_ranks[idx]["category"]
        if cat not in by_cat:
            by_cat[cat] = {"bl": [], "d50": []}
        by_cat[cat]["bl"].append(bl_ranks[idx]["rank"])
        by_cat[cat]["d50"].append(d50_ranks[idx]["rank"])

    # === Plot 1: Paired scatter ===
    fig, ax = plt.subplots(figsize=(8, 8))

    for cat in ["both_correct", "both_wrong", "filler_hurt", "filler_helped"]:
        if cat not in by_cat:
            continue
        style = cat_style.get(cat, {"color": "gray", "marker": "o", "alpha": 0.5, "zorder": 1})
        bl = np.array(by_cat[cat]["bl"])
        d50 = np.array(by_cat[cat]["d50"])
        # Add small jitter to avoid overplotting at rank 1
        jitter_bl = bl * (1 + np.random.randn(len(bl)) * 0.05)
        jitter_d50 = d50 * (1 + np.random.randn(len(d50)) * 0.05)
        jitter_bl = np.maximum(jitter_bl, 0.8)
        jitter_d50 = np.maximum(jitter_d50, 0.8)
        ax.scatter(jitter_bl, jitter_d50, c=style["color"], marker=style["marker"],
                   alpha=style["alpha"], s=30, zorder=style["zorder"],
                   label=f"{cat} (n={len(bl)})")

    # Diagonal line (no change)
    lims = [0.8, max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, "k--", alpha=0.3, linewidth=1, zorder=1)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"Baseline rank (L{args.layer}, {args.position})", fontsize=12)
    ax.set_ylabel(f"Dots_50 rank (L{args.layer}, {args.position})", fontsize=12)
    ax.set_title("A+Y answer token rank: baseline vs dots_50", fontsize=13)
    ax.legend(loc="upper left", fontsize=10)
    ax.set_aspect("equal")

    # Annotations
    ax.text(0.98, 0.02, "below diagonal = filler helped",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, fontstyle="italic", alpha=0.6)

    plt.tight_layout()
    plt.savefig(args.output_dir / "rank_scatter_pre_answer.png", dpi=200, bbox_inches="tight")
    plt.savefig(args.output_dir / "rank_scatter_pre_answer.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved to {args.output_dir}/rank_scatter_pre_answer.png")

    # === Plot 2: CDF ===
    fig, ax = plt.subplots(figsize=(10, 5))

    all_bl = np.array([bl_ranks[i]["rank"] for i in common_idxs])
    all_d50 = np.array([d50_ranks[i]["rank"] for i in common_idxs])

    max_k = 50
    ks = np.arange(1, max_k + 1)
    bl_cdf = [np.mean(all_bl <= k) for k in ks]
    d50_cdf = [np.mean(all_d50 <= k) for k in ks]

    ax.plot(ks, bl_cdf, "o-", color="#F44336", linewidth=2, markersize=4, label="Baseline")
    ax.plot(ks, d50_cdf, "o-", color="#4CAF50", linewidth=2, markersize=4, label="Dots_50")
    ax.fill_between(ks, bl_cdf, d50_cdf, alpha=0.15, color="#4CAF50")

    ax.set_xlabel("Rank threshold k", fontsize=12)
    ax.set_ylabel("Fraction with A+Y rank ≤ k", fontsize=12)
    ax.set_title(f"Cumulative rank distribution at L{args.layer} {args.position}", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, max_k)
    ax.set_ylim(0, 1.02)

    plt.tight_layout()
    plt.savefig(args.output_dir / "rank_cdf_pre_answer.png", dpi=200, bbox_inches="tight")
    plt.savefig(args.output_dir / "rank_cdf_pre_answer.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved to {args.output_dir}/rank_cdf_pre_answer.png")

    # === Plot 3: Category breakdown bar chart ===
    fig, ax = plt.subplots(figsize=(10, 5))

    cats_order = ["both_correct", "filler_helped", "both_wrong"]
    cats_present = [c for c in cats_order if c in by_cat]
    x = np.arange(len(cats_present))
    width = 0.35

    bl_top1 = [np.mean(np.array(by_cat[c]["bl"]) == 1) for c in cats_present]
    d50_top1 = [np.mean(np.array(by_cat[c]["d50"]) == 1) for c in cats_present]

    bars1 = ax.bar(x - width/2, bl_top1, width, label="Baseline", color="#F44336", alpha=0.7)
    bars2 = ax.bar(x + width/2, d50_top1, width, label="Dots_50", color="#4CAF50", alpha=0.7)

    ax.set_ylabel("Fraction with A+Y as top-1 token", fontsize=12)
    ax.set_title(f"Top-1 accuracy by category at L{args.layer} {args.position}", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n(n={len(by_cat[c]['bl'])})" for c in cats_present], fontsize=10)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")

    # Add percentage labels on bars
    for bar, val in zip(bars1, bl_top1):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.0%}", ha="center", va="bottom", fontsize=9)
    for bar, val in zip(bars2, d50_top1):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.0%}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(args.output_dir / "rank_by_category_pre_answer.png", dpi=200, bbox_inches="tight")
    plt.savefig(args.output_dir / "rank_by_category_pre_answer.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved to {args.output_dir}/rank_by_category_pre_answer.png")

    # === Plot 4: Scatter with marginal histograms ===
    fig = plt.figure(figsize=(9, 9))
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                          hspace=0.05, wspace=0.05)
    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    for cat in ["both_correct", "both_wrong", "filler_hurt", "filler_helped"]:
        if cat not in by_cat:
            continue
        style = cat_style[cat]
        bl = np.array(by_cat[cat]["bl"])
        d50 = np.array(by_cat[cat]["d50"])
        jbl = bl * (1 + np.random.randn(len(bl)) * 0.05)
        jd50 = d50 * (1 + np.random.randn(len(d50)) * 0.05)
        jbl = np.maximum(jbl, 0.8)
        jd50 = np.maximum(jd50, 0.8)
        ax_main.scatter(jbl, jd50, c=style["color"], marker=style["marker"],
                        alpha=style["alpha"], s=25, zorder=style["zorder"],
                        label=f"{cat} (n={len(bl)})")

    ax_main.plot([0.8, 1.2e2], [0.8, 1.2e2], "k--", alpha=0.3, linewidth=1, zorder=1)
    ax_main.set_xscale("log")
    ax_main.set_yscale("log")
    ax_main.set_xlabel(f"Baseline rank", fontsize=11)
    ax_main.set_ylabel(f"Dots_50 rank", fontsize=11)
    ax_main.legend(loc="upper left", fontsize=9)

    ax_main.text(1.2, 0.2, "below diagonal = filler helped",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=9, fontstyle="italic", alpha=0.6)

    # Marginal histograms (log-spaced bins)
    bins = np.logspace(0, np.log10(max(all_bl.max(), all_d50.max()) + 1), 40)
    ax_top.hist(all_bl, bins=bins, color="#F44336", alpha=0.5, label="Baseline")
    ax_top.hist(all_d50, bins=bins, color="#4CAF50", alpha=0.5, label="Dots_50")
    ax_top.set_xscale("log")
    ax_top.set_ylabel("Count")
    ax_top.legend(fontsize=8)
    plt.setp(ax_top.get_xticklabels(), visible=False)

    ax_right.hist(all_d50, bins=bins, orientation="horizontal", color="#4CAF50", alpha=0.5)
    ax_right.hist(all_bl, bins=bins, orientation="horizontal", color="#F44336", alpha=0.3)
    ax_right.set_yscale("log")
    ax_right.set_xlabel("Count")
    plt.setp(ax_right.get_yticklabels(), visible=False)

    fig.suptitle(f"A+Y rank: baseline vs dots_50 (L{args.layer}, {args.position})", fontsize=13, y=0.98)
    plt.savefig(args.output_dir / "rank_scatter_marginals.png", dpi=200, bbox_inches="tight")
    plt.savefig(args.output_dir / "rank_scatter_marginals.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved to {args.output_dir}/rank_scatter_marginals.png")

    # === Plot 5: Violin/strip plot per category ===
    fig, ax = plt.subplots(figsize=(10, 6))

    cats_plot = ["both_correct", "filler_helped", "both_wrong"]
    cats_present_v = [c for c in cats_plot if c in by_cat]
    cat_colors = {"both_correct": "#2196F3", "filler_helped": "#4CAF50", "both_wrong": "#F44336"}

    positions_x = []
    violin_data = []
    violin_colors = []
    labels = []

    for i, cat in enumerate(cats_present_v):
        bl = np.array(by_cat[cat]["bl"], dtype=float)
        d50 = np.array(by_cat[cat]["d50"], dtype=float)
        # Use log ranks for violin (otherwise rank 1 vs rank 10000 makes violins unreadable)
        bl_log = np.log10(np.maximum(bl, 0.5))
        d50_log = np.log10(np.maximum(d50, 0.5))

        # Baseline strip
        x_bl = i * 3
        x_d50 = i * 3 + 1
        jitter_bl = np.random.randn(len(bl)) * 0.12
        jitter_d50 = np.random.randn(len(d50)) * 0.12
        ax.scatter(x_bl + jitter_bl, bl_log, c="#F44336", alpha=0.35, s=12, zorder=3)
        ax.scatter(x_d50 + jitter_d50, d50_log, c="#4CAF50", alpha=0.35, s=12, zorder=3)

        # Median lines
        ax.hlines(np.median(bl_log), x_bl - 0.3, x_bl + 0.3, color="#B71C1C", linewidth=2, zorder=4)
        ax.hlines(np.median(d50_log), x_d50 - 0.3, x_d50 + 0.3, color="#1B5E20", linewidth=2, zorder=4)

        positions_x.extend([x_bl, x_d50])
        labels.extend([f"BL", f"D50"])

    # Set up x-axis
    group_centers = [i * 3 + 0.5 for i in range(len(cats_present_v))]
    ax.set_xticks(group_centers)
    ax.set_xticklabels([f"{c}\n(n={len(by_cat[c]['bl'])})" for c in cats_present_v], fontsize=10)

    # Add BL/D50 labels
    for i in range(len(cats_present_v)):
        ax.text(i * 3, -0.45, "BL", ha="center", fontsize=8, color="#F44336")
        ax.text(i * 3 + 1, -0.45, "D50", ha="center", fontsize=8, color="#4CAF50")

    # Y-axis: log10(rank) with readable labels
    yticks = [0, np.log10(2), np.log10(5), 1, np.log10(20), np.log10(50), 2]
    yticklabels = ["1", "2", "5", "10", "20", "50", "100"]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_ylabel("A+Y answer token rank", fontsize=12)
    ax.set_title(f"Rank distribution by category (L{args.layer}, {args.position})", fontsize=13)
    ax.grid(True, alpha=0.3, axis="y")
    ax.invert_yaxis()  # rank 1 at top

    # Legend
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#F44336", alpha=0.5, label="Baseline"),
        Patch(color="#4CAF50", alpha=0.5, label="Dots_50"),
    ], fontsize=10)

    plt.tight_layout()
    plt.savefig(args.output_dir / "rank_strip_by_category.png", dpi=200, bbox_inches="tight")
    plt.savefig(args.output_dir / "rank_strip_by_category.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved to {args.output_dir}/rank_strip_by_category.png")

    # === Plot 6: Rank change histogram per category ===
    fig, axes = plt.subplots(1, len(cats_present_v), figsize=(4 * len(cats_present_v), 4),
                              sharey=True)
    if len(cats_present_v) == 1:
        axes = [axes]

    for ax_i, cat in enumerate(cats_present_v):
        ax = axes[ax_i]
        bl = np.array(by_cat[cat]["bl"])
        d50 = np.array(by_cat[cat]["d50"])
        # Use log ratio: log10(d50_rank / bl_rank). Negative = filler helped.
        # Handle rank 0 edge case
        bl_safe = np.maximum(bl, 0.5)
        d50_safe = np.maximum(d50, 0.5)
        log_change = np.log10(d50_safe / bl_safe)

        color = cat_colors.get(cat, "gray")
        ax.hist(log_change, bins=30, color=color, alpha=0.7, edgecolor="white", linewidth=0.5)
        ax.axvline(0, color="black", linestyle="--", alpha=0.5, linewidth=1)
        ax.axvline(np.median(log_change), color="black", linestyle="-", linewidth=2,
                   label=f"median={np.median(log_change):.2f}")

        ax.set_xlabel("log10(dots_50 rank / baseline rank)", fontsize=10)
        ax.set_title(f"{cat} (n={len(bl)})", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    axes[0].set_ylabel("Count", fontsize=11)
    fig.suptitle(f"Rank change distribution (L{args.layer}, {args.position})", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(args.output_dir / "rank_change_histogram.png", dpi=200, bbox_inches="tight")
    plt.savefig(args.output_dir / "rank_change_histogram.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved to {args.output_dir}/rank_change_histogram.png")

    # Print summary stats
    print(f"\nSummary at L{args.layer} {args.position}:")
    print(f"  Overall: baseline top-1={np.mean(all_bl == 1):.1%}, dots_50 top-1={np.mean(all_d50 == 1):.1%}")
    for cat in cats_present_v:
        bl = np.array(by_cat[cat]["bl"])
        d50 = np.array(by_cat[cat]["d50"])
        print(f"  {cat:<16} n={len(bl):>3}  "
              f"bl top-1={np.mean(bl==1):.1%} med={np.median(bl):.0f}  "
              f"d50 top-1={np.mean(d50==1):.1%} med={np.median(d50):.0f}")


if __name__ == "__main__":
    main()
