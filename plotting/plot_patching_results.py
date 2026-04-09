"""
plot_patching_results.py

Plots for the activation patching intervention results.

Usage:
    uv run --project /workspace/filler-token-reasoning/probing \
        python scripts/plot_patching_results.py \
        --output-dir plots
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_results(path):
    with open(path) as f:
        return json.load(f)


def get_phase1_from_checkpoint(path):
    """Load from checkpoint or results file."""
    with open(path) as f:
        data = json.load(f)
    return data.get("results", data) if isinstance(data, dict) else data


# ==============================================================================
# Plot 1: Phase 1 position sweep
# ==============================================================================

def plot_position_sweep(results, output_dir):
    """Bar chart: donor%, original%, novel% by start position."""
    order = ["question_end", "k1", "k5", "k10", "k25", "k50", "answer_prompt", "pre_answer"]
    short_labels = ["q_end", "k1", "k5", "k10", "k25", "k50", "prompt", "answer"]

    by_cond = defaultdict(list)
    for r in results:
        by_cond[r["start_condition"]].append(r)

    donor_pcts = []
    orig_pcts = []
    novel_pcts = []
    parse_pcts = []
    ns = []

    for cond in order:
        items = by_cond[cond]
        n = len(items)
        ns.append(n)
        n_parsed = sum(1 for r in items if not r["parse_failure"])
        donor_pcts.append(sum(1 for r in items if r["exact_match_donor"]) / max(n_parsed, 1))
        orig_pcts.append(sum(1 for r in items if r["exact_match_original"]) / max(n_parsed, 1))
        novel_pcts.append(sum(1 for r in items if r["novel"]) / max(n_parsed, 1))

    x = np.arange(len(order))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))

    bars_d = ax.bar(x - width, donor_pcts, width, label="Donor answer", color="#4CAF50", alpha=0.8)
    bars_o = ax.bar(x, orig_pcts, width, label="Original answer", color="#2196F3", alpha=0.8)
    bars_n = ax.bar(x + width, novel_pcts, width, label="Novel answer", color="#9E9E9E", alpha=0.6)

    # Add percentage labels on donor bars
    for bar, val in zip(bars_d, donor_pcts):
        if val > 0.02:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{val:.0%}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=10)
    ax.set_ylabel("Fraction of examples", fontsize=12)
    ax.set_title("All-layer patching by start position", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.01)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_dir / "patching_position_sweep.png", dpi=200, bbox_inches="tight")
    plt.savefig(output_dir / "patching_position_sweep.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved patching_position_sweep")


# ==============================================================================
# Plot 2: Phase 3 layer profile at pre_answer
# ==============================================================================

def plot_layer_profile(results, output_dir):
    """Line plot: donor% across layers at pre_answer and k1."""
    by_cond_layer = defaultdict(lambda: defaultdict(list))
    for r in results:
        cond = r["start_condition"]
        layer = int(r["layer_band"].replace("L", ""))
        by_cond_layer[cond][layer].append(r)

    fig, ax = plt.subplots(figsize=(12, 4))

    for cond, color, label in [
        ("pre_answer", "#4CAF50", "pre_answer (single token)"),
        ("k1", "#2196F3", "k1 (filler + answer_prompt)"),
    ]:
        if cond not in by_cond_layer:
            continue
        layers = sorted(by_cond_layer[cond].keys())
        donor_pcts = []
        for layer in layers:
            items = by_cond_layer[cond][layer]
            n_parsed = sum(1 for r in items if not r["parse_failure"])
            donor = sum(1 for r in items if r["exact_match_donor"]) / max(n_parsed, 1)
            donor_pcts.append(donor)

        ax.plot(layers, donor_pcts, "o-", color=color, linewidth=2, markersize=4, label=label)

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Donor match %", fontsize=12)
    ax.set_title("Per-layer patching — donor match by layer", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(-1, 61)
    ax.set_ylim(-0.05, 1.1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "patching_layer_profile.png", dpi=200, bbox_inches="tight")
    plt.savefig(output_dir / "patching_layer_profile.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved patching_layer_profile")


# ==============================================================================
# Plot 3: Cross-condition comparison
# ==============================================================================

def plot_cross_condition(within_results, cross_results, output_dir):
    """Grouped bars: within-dots_50 vs cross-condition (dots_50 → baseline)."""

    # Within dots_50: extract pre_answer and answer_prompt
    within_by_cond = defaultdict(list)
    for r in within_results:
        within_by_cond[r["start_condition"]].append(r)

    # Cross-condition: extract by condition name
    cross_by_cond = defaultdict(list)
    for r in cross_results:
        cross_by_cond[r["cond_name"]].append(r)

    # Conditions to show
    conditions = [
        ("pre_answer\n(all layers)", "pre_answer", "pre_answer_all"),
        ("pre_answer\n(L60 only)", "pre_answer", "pre_answer_L60"),
        ("answer_prompt\n(all layers)", "answer_prompt", "ans_prompt_all"),
        ("question_end\n(all layers)", "question_end", "qend_all"),
    ]

    labels = []
    within_donor = []
    cross_donor = []
    cross_correct = []

    for label, within_key, cross_key in conditions:
        labels.append(label)

        # Within dots_50
        items = within_by_cond.get(within_key, [])
        n_parsed = sum(1 for r in items if not r["parse_failure"])
        within_donor.append(
            sum(1 for r in items if r["exact_match_donor"]) / max(n_parsed, 1)
        )

        # Cross-condition
        items = cross_by_cond.get(cross_key, [])
        n_parsed = sum(1 for r in items if not r["parse_failure"])
        cross_donor.append(
            sum(1 for r in items if r["matches_donor"]) / max(n_parsed, 1)
        )
        cross_correct.append(
            sum(1 for r in items if r["matches_correct"]) / max(n_parsed, 1)
        )

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))

    bars1 = ax.bar(x - width, within_donor, width,
                   label="Within dots_50: donor match", color="#4CAF50", alpha=0.8)
    bars2 = ax.bar(x, cross_donor, width,
                   label="Cross (D50→BL): donor match", color="#2196F3", alpha=0.8)
    bars3 = ax.bar(x + width, cross_correct, width,
                   label="Cross (D50→BL): correct answer", color="#FF9800", alpha=0.7)

    for bars, vals in [(bars1, within_donor), (bars2, cross_donor), (bars3, cross_correct)]:
        for bar, val in zip(bars, vals):
            if val > 0.02:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f"{val:.0%}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Fraction of examples", fontsize=12)
    ax.set_title("Cross-condition patching: dots_50 source → baseline target", fontsize=13)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_dir / "patching_cross_condition.png", dpi=200, bbox_inches="tight")
    plt.savefig(output_dir / "patching_cross_condition.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved patching_cross_condition")


# ==============================================================================
# Plot 4: Combined two-panel (position sweep + layer profile)
# ==============================================================================

def plot_combined(phase1_results, phase3_results, output_dir):
    """Two-panel figure: left = position sweep, right = layer profile at pre_answer."""

    # Left panel data: position sweep
    order = ["question_end", "k1", "k5", "k10", "k25", "k50", "answer_prompt", "pre_answer"]
    short_labels = ["q_end", "k1", "k5", "k10", "k25", "k50", "prompt", "pre_ans"]

    by_cond = defaultdict(list)
    for r in phase1_results:
        by_cond[r["start_condition"]].append(r)

    donor_pcts_pos = []
    for cond in order:
        items = by_cond[cond]
        n_parsed = sum(1 for r in items if not r["parse_failure"])
        donor_pcts_pos.append(
            sum(1 for r in items if r["exact_match_donor"]) / max(n_parsed, 1)
        )

    # Right panel data: layer profile at pre_answer
    by_layer = defaultdict(list)
    for r in phase3_results:
        if r["start_condition"] == "pre_answer":
            layer = int(r["layer_band"].replace("L", ""))
            by_layer[layer].append(r)

    layers = sorted(by_layer.keys())
    donor_pcts_layer = []
    for layer in layers:
        items = by_layer[layer]
        n_parsed = sum(1 for r in items if not r["parse_failure"])
        donor_pcts_layer.append(
            sum(1 for r in items if r["exact_match_donor"]) / max(n_parsed, 1)
        )

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5),
                                     gridspec_kw={"width_ratios": [1.2, 1]})

    # Left: position sweep
    # colors = ["#9E9E9E"] * 7 + ["#4CAF50"]  # highlight pre_answer
    colors = ["#4CAF50"] * len(order)  # all same color
    bars = ax1.bar(range(len(order)), donor_pcts_pos, color=colors, alpha=0.8, edgecolor="white")
    for bar, val in zip(bars, donor_pcts_pos):
        if val > 0.02:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f"{val:.0%}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax1.set_xticks(range(len(order)))
    ax1.set_xticklabels(short_labels, fontsize=9, rotation=45, ha="right")
    ax1.set_ylabel("Donor match %", fontsize=11)
    ax1.set_title("A. Patching from each start position\n(all 61 layers)", fontsize=12)
    ax1.set_ylim(0, 1.15)
    ax1.grid(True, alpha=0.3, axis="y")

    # Right: layer profile
    ax2.plot(layers, donor_pcts_layer, "o-", color="#4CAF50", linewidth=2, markersize=3)
    ax2.fill_between(layers, donor_pcts_layer, alpha=0.15, color="#4CAF50")
    ax2.set_xlabel("Layer", fontsize=11)
    ax2.set_ylabel("Donor match %", fontsize=11)
    ax2.set_title("B. Patching at pre_answer\n(one layer at a time)", fontsize=12)
    ax2.set_xlim(-1, 61)
    ax2.set_ylim(-0.05, 1.15)
    ax2.grid(True, alpha=0.3)

    # Annotate L60
    l60_idx = layers.index(60)
    ax2.annotate(f"L60: {donor_pcts_layer[l60_idx]:.0%}",
                 xy=(60, donor_pcts_layer[l60_idx]),
                 xytext=(45, 0.7), fontsize=10,
                 arrowprops=dict(arrowstyle="->", color="black", alpha=0.6),
                 fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_dir / "patching_combined.png", dpi=200, bbox_inches="tight")
    plt.savefig(output_dir / "patching_combined.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved patching_combined")


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1", type=Path,
                        default=Path("patching_results_v2/phase1_results.json"))
    parser.add_argument("--phase3", type=Path,
                        default=Path("patching_results_v2/phase3_results.json"))
    parser.add_argument("--cross", type=Path,
                        default=Path("patching_results_v2/baseline_cross/results.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("plots"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    phase1 = load_results(args.phase1)
    print(f"Phase 1: {len(phase1)} results")

    plot_position_sweep(phase1, args.output_dir)

    if args.phase3.exists():
        phase3 = load_results(args.phase3)
        print(f"Phase 3: {len(phase3)} results")
        plot_layer_profile(phase3, args.output_dir)
        plot_combined(phase1, phase3, args.output_dir)
    else:
        print(f"Phase 3 not found at {args.phase3}, skipping layer plots")

    if args.cross.exists():
        cross = load_results(args.cross)
        print(f"Cross-condition: {len(cross)} results")
        plot_cross_condition(phase1, cross, args.output_dir)
    else:
        # Try checkpoint
        checkpoint = args.cross.parent / "checkpoint.json"
        if checkpoint.exists():
            cross = get_phase1_from_checkpoint(checkpoint)
            print(f"Cross-condition (from checkpoint): {len(cross)} results")
            plot_cross_condition(phase1, cross, args.output_dir)
        else:
            print(f"Cross-condition not found, skipping")

    print("Done.")


if __name__ == "__main__":
    main()
