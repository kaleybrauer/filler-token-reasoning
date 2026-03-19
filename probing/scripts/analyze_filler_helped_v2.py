"""
analyze_filler_helped_v2.py

Cross-condition analysis with better metrics:
- MAE (not R²) as primary metric
- Fraction within tolerance
- Per-example error trajectory plots
- Box plots showing distributions
"""

import pickle
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="Ill-conditioned")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def load_metadata(extraction_dir, condition):
    cond_dir = extraction_dir / condition
    files = sorted(cond_dir.glob("prob_*.pkl"))
    meta = []
    for f in files:
        with open(f, "rb") as fp:
            data = pickle.load(fp)
        meta.append({
            "problem_idx": data["problem_idx"],
            "fact_value": data["fact_value"],
            "x": data["x"],
            "answer": data["answer"],
            "model_correct": data["model_correct"],
            "model_answer": data.get("model_answer"),
        })
    return meta


def get_best_layer_predictions(results, target, pos):
    """Get predictions from the best layer (by overall R²) at a given position."""
    layers = sorted(results[target][pos].keys())
    best_layer = max(layers, key=lambda l: results[target][pos][l]["r2"])
    return (results[target][pos][best_layer]["y_pred"],
            results[target][pos][best_layer]["y_test"],
            best_layer)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--extraction-dir", type=Path, default=Path("probing/extracted_states"))
    parser.add_argument("--results-dir", type=Path, default=Path("probing/probe_results"))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.results_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.results_dir / "all_probe_results.pkl", "rb") as f:
        all_results = pickle.load(f)

    bl_meta = load_metadata(args.extraction_dir, "baseline")
    dt_meta = load_metadata(args.extraction_dir, "dots_250")

    n = len(bl_meta)
    bl_correct = np.array([m["model_correct"] for m in bl_meta], dtype=bool)
    dt_correct = np.array([m["model_correct"] for m in dt_meta], dtype=bool)

    rng = np.random.RandomState(42)
    indices = rng.permutation(n)
    split = int(0.75 * n)
    test_idx = indices[split:]

    bl_test = bl_correct[test_idx]
    dt_test = dt_correct[test_idx]

    categories = {
        "filler_helped\n(wrong→right)": ~bl_test & dt_test,
        "both_correct": bl_test & dt_test,
        "both_wrong": ~bl_test & ~dt_test,
    }
    cat_colors = {
        "filler_helped\n(wrong→right)": "#4CAF50",
        "both_correct": "#2196F3",
        "both_wrong": "#9E9E9E",
    }

    dots_results = all_results["dots_250"]

    filler_positions = sorted(
        [p for p in dots_results["A"].keys() if p.startswith("filler_")],
        key=lambda p: float(p.split("_")[1])
    )
    all_positions = ["question_end"] + filler_positions + ["answer_prompt"]
    all_positions = [p for p in all_positions if p in dots_results["A"]]
    pos_labels = [p.replace("filler_", "f").replace("question_end", "q_end").replace("answer_prompt", "ans")
                  for p in all_positions]

    for cat_name, mask in categories.items():
        print(f"{cat_name.replace(chr(10), ' ')}: n={mask.sum()}")

    # =====================================================================
    # Plot 1: Per-example absolute error trajectories for A
    # =====================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for ax, (cat_name, mask) in zip(axes, categories.items()):
        n_cat = mask.sum()
        if n_cat < 1:
            continue

        # Collect per-example errors at each position
        errors = np.zeros((n_cat, len(all_positions)))
        for j, pos in enumerate(all_positions):
            y_pred, y_true, _ = get_best_layer_predictions(dots_results, "A", pos)
            errors[:, j] = np.abs(y_pred[mask] - y_true[mask])

        # Plot each example as a thin line
        x = np.arange(len(all_positions))
        for i in range(n_cat):
            ax.plot(x, errors[i], color=cat_colors[cat_name], alpha=0.15, linewidth=0.8)

        # Bold mean line
        ax.plot(x, errors.mean(axis=0), color=cat_colors[cat_name],
                linewidth=3, marker="o", markersize=6, label=f"mean (n={n_cat})")

        # Median line
        ax.plot(x, np.median(errors, axis=0), color=cat_colors[cat_name],
                linewidth=2, linestyle="--", marker="s", markersize=4, label="median")

        ax.set_xticks(x)
        ax.set_xticklabels(pos_labels, rotation=45, ha="right")
        ax.set_title(cat_name, fontsize=13)
        ax.set_ylabel("Absolute probe error for A", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    fig.suptitle("Per-example probe error trajectories (A target, dots_250 hidden states)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(args.output_dir / "v2_error_trajectories.png", dpi=150, bbox_inches="tight")
    plt.savefig(args.output_dir / "v2_error_trajectories.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: v2_error_trajectories")

    # =====================================================================
    # Plot 2: Box plots of probe error at key positions
    # =====================================================================
    key_positions = ["question_end", "filler_0.50", "filler_1.00", "answer_prompt"]
    key_positions = [p for p in key_positions if p in dots_results["A"]]
    key_labels = [p.replace("filler_", "f").replace("question_end", "q_end").replace("answer_prompt", "ans")
                  for p in key_positions]

    fig, axes = plt.subplots(1, len(key_positions), figsize=(4 * len(key_positions), 5), sharey=True)

    for ax, pos, label in zip(axes, key_positions, key_labels):
        y_pred, y_true, best_l = get_best_layer_predictions(dots_results, "A", pos)
        abs_errors = np.abs(y_pred - y_true)

        box_data = []
        box_labels = []
        box_colors = []
        for cat_name, mask in categories.items():
            if mask.sum() < 3:
                continue
            box_data.append(abs_errors[mask])
            short_name = cat_name.split("\n")[0]
            box_labels.append(f"{short_name}\n(n={mask.sum()})")
            box_colors.append(cat_colors[cat_name])

        bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True,
                        widths=0.6, showfliers=True, flierprops=dict(markersize=3))
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)

        ax.set_title(f"{label} (L{best_l})", fontsize=12)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(bottom=0)

    axes[0].set_ylabel("Absolute probe error for A", fontsize=11)
    fig.suptitle("Probe error distributions by category and position", fontsize=14)
    plt.tight_layout()
    plt.savefig(args.output_dir / "v2_boxplots.png", dpi=150, bbox_inches="tight")
    plt.savefig(args.output_dir / "v2_boxplots.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: v2_boxplots")

    # =====================================================================
    # Plot 3: MAE + fraction within tolerance at each position
    # =====================================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    tolerances = [5, 10, 20]

    # Left: MAE
    ax = axes[0]
    for cat_name, mask in categories.items():
        if mask.sum() < 3:
            continue
        mae_vals = []
        for pos in all_positions:
            y_pred, y_true, _ = get_best_layer_predictions(dots_results, "A", pos)
            mae_vals.append(np.mean(np.abs(y_pred[mask] - y_true[mask])))
        short = cat_name.split("\n")[0]
        ax.plot(range(len(all_positions)), mae_vals, marker="o",
                color=cat_colors[cat_name], linewidth=2,
                label=f"{short} (n={mask.sum()})")

    ax.set_xticks(range(len(all_positions)))
    ax.set_xticklabels(pos_labels, rotation=45, ha="right")
    ax.set_ylabel("MAE for A", fontsize=12)
    ax.set_title("Mean Absolute Error", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # Right: fraction within ±10
    ax = axes[1]
    tol = 10
    for cat_name, mask in categories.items():
        if mask.sum() < 3:
            continue
        frac_vals = []
        for pos in all_positions:
            y_pred, y_true, _ = get_best_layer_predictions(dots_results, "A", pos)
            errs = np.abs(y_pred[mask] - y_true[mask])
            frac_vals.append(np.mean(errs <= tol))
        short = cat_name.split("\n")[0]
        ax.plot(range(len(all_positions)), frac_vals, marker="o",
                color=cat_colors[cat_name], linewidth=2,
                label=f"{short} (n={mask.sum()})")

    ax.set_xticks(range(len(all_positions)))
    ax.set_xticklabels(pos_labels, rotation=45, ha="right")
    ax.set_ylabel(f"Fraction with |error| ≤ {tol}", fontsize=12)
    ax.set_title(f"Probe accuracy (within ±{tol})", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    fig.suptitle("Probe for A across filler positions (dots_250)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(args.output_dir / "v2_mae_and_tolerance.png", dpi=150, bbox_inches="tight")
    plt.savefig(args.output_dir / "v2_mae_and_tolerance.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: v2_mae_and_tolerance")

    # =====================================================================
    # Plot 4: Error reduction from question_end to answer_prompt (scatter)
    # =====================================================================
    if "question_end" in dots_results["A"] and "answer_prompt" in dots_results["A"]:
        pred_qe, true_qe, _ = get_best_layer_predictions(dots_results, "A", "question_end")
        pred_ap, true_ap, _ = get_best_layer_predictions(dots_results, "A", "answer_prompt")
        err_qe = np.abs(pred_qe - true_qe)
        err_ap = np.abs(pred_ap - true_ap)

        fig, ax = plt.subplots(figsize=(7, 7))
        for cat_name, mask in categories.items():
            if mask.sum() < 3:
                continue
            short = cat_name.split("\n")[0]
            ax.scatter(err_qe[mask], err_ap[mask], color=cat_colors[cat_name],
                       label=f"{short} (n={mask.sum()})", alpha=0.6, s=40, edgecolors="white", linewidth=0.5)

        lim = max(err_qe.max(), err_ap.max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", alpha=0.3, label="no change")
        ax.set_xlabel("Probe |error| at question_end", fontsize=12)
        ax.set_ylabel("Probe |error| at answer_prompt", fontsize=12)
        ax.set_title("Error reduction: question_end → answer_prompt\n(below diagonal = filler helped probe)", fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect("equal")

        plt.tight_layout()
        plt.savefig(args.output_dir / "v2_error_reduction_scatter.png", dpi=150, bbox_inches="tight")
        plt.savefig(args.output_dir / "v2_error_reduction_scatter.pdf", bbox_inches="tight")
        plt.close()
        print("Saved: v2_error_reduction_scatter")

    # =====================================================================
    # Print summary table with better metrics
    # =====================================================================
    print("\n" + "=" * 90)
    print("SUMMARY: Probe for A — MAE and fraction within ±10")
    print("=" * 90)
    header = f"{'Category':<25} {'n':>3}"
    for pos in key_positions:
        short = pos.replace("filler_", "f").replace("question_end", "q_end").replace("answer_prompt", "ans")
        header += f"  {short+' MAE':>10} {short+' ±10':>8}"
    print(header)
    print("-" * 90)

    for cat_name, mask in categories.items():
        if mask.sum() < 3:
            continue
        short_cat = cat_name.replace("\n", " ")
        row = f"{short_cat:<25} {mask.sum():>3}"
        for pos in key_positions:
            y_pred, y_true, _ = get_best_layer_predictions(dots_results, "A", pos)
            errs = np.abs(y_pred[mask] - y_true[mask])
            row += f"  {errs.mean():>10.1f} {np.mean(errs <= 10):>7.0%}"
        print(row)

    print()


if __name__ == "__main__":
    main()
