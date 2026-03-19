"""
analyze_filler_helped.py

Cross-condition analysis: compare probe accuracy on examples where
filler tokens flipped the model's answer from wrong to right.

Categories:
  - filler_helped:  baseline wrong, dots_250 correct
  - filler_hurt:    baseline correct, dots_250 wrong
  - both_correct:   baseline correct, dots_250 correct
  - both_wrong:     baseline wrong, dots_250 wrong

Uses existing probe predictions from all_probe_results.pkl.
"""

import pickle
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore", message="Ill-conditioned")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score


def load_metadata(extraction_dir: Path, condition: str):
    cond_dir = extraction_dir / condition
    files = sorted(cond_dir.glob("prob_*.pkl"))
    metadata = []
    for f in files:
        with open(f, "rb") as fp:
            data = pickle.load(fp)
        metadata.append({
            "problem_idx": data["problem_idx"],
            "model_correct": data["model_correct"],
        })
    return metadata


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--extraction-dir", type=Path, default=Path("probing/extracted_states"))
    parser.add_argument("--results-dir", type=Path, default=Path("probing/probe_results"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--train-fraction", type=float, default=0.75)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.results_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load probe results
    with open(args.results_dir / "all_probe_results.pkl", "rb") as f:
        all_results = pickle.load(f)

    # Load metadata for both conditions
    baseline_meta = load_metadata(args.extraction_dir, "baseline")
    dots_meta = load_metadata(args.extraction_dir, "dots_250")

    n = len(baseline_meta)
    baseline_correct = np.array([m["model_correct"] for m in baseline_meta], dtype=bool)
    dots_correct = np.array([m["model_correct"] for m in dots_meta], dtype=bool)

    # Reproduce the same train/test split
    rng = np.random.RandomState(42)
    indices = rng.permutation(n)
    split = int(args.train_fraction * n)
    test_idx = indices[split:]

    # Correctness on test set only
    bl_test = baseline_correct[test_idx]
    dt_test = dots_correct[test_idx]

    filler_helped = ~bl_test & dt_test    # wrong -> right
    filler_hurt = bl_test & ~dt_test      # right -> wrong
    both_correct = bl_test & dt_test
    both_wrong = ~bl_test & ~dt_test

    print(f"Test set: {len(test_idx)} examples")
    print(f"  filler_helped (wrong->right): {filler_helped.sum()}")
    print(f"  filler_hurt   (right->wrong): {filler_hurt.sum()}")
    print(f"  both_correct:                 {both_correct.sum()}")
    print(f"  both_wrong:                   {both_wrong.sum()}")
    print(f"  baseline acc:  {bl_test.mean():.1%}")
    print(f"  dots_250 acc:  {dt_test.mean():.1%}")
    print(f"  uplift:        {dt_test.mean() - bl_test.mean():.1%}")

    # --- Plot 1: R² for A across filler positions, split by category ---
    dots_results = all_results["dots_250"]
    target = "A"
    if target not in dots_results:
        print("No A target in dots_250 results")
        return

    filler_positions = sorted(
        [p for p in dots_results[target].keys() if p.startswith("filler_")],
        key=lambda p: float(p.split("_")[1])
    )
    all_positions = ["question_end"] + filler_positions + ["answer_prompt"]
    all_positions = [p for p in all_positions if p in dots_results[target]]

    categories = {
        "filler_helped": filler_helped,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
    }
    if filler_hurt.sum() >= 5:
        categories["filler_hurt"] = filler_hurt

    colors = {
        "filler_helped": "#4CAF50",
        "filler_hurt": "#F44336",
        "both_correct": "#2196F3",
        "both_wrong": "#9E9E9E",
    }
    markers = {
        "filler_helped": "o",
        "filler_hurt": "x",
        "both_correct": "s",
        "both_wrong": "^",
    }

    # For each position, find best layer, then compute subset MAE/R²
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, metric_name in zip(axes, ["mae", "r2"]):
        for cat_name, mask in categories.items():
            if mask.sum() < 3:
                continue

            values = []
            x_labels = []
            for pos in all_positions:
                layers = sorted(dots_results[target][pos].keys())
                # Find best layer by overall R²
                best_layer = max(layers, key=lambda l: dots_results[target][pos][l]["r2"])

                y_pred = dots_results[target][pos][best_layer]["y_pred"]
                y_test = dots_results[target][pos][best_layer]["y_test"]

                subset_pred = y_pred[mask]
                subset_true = y_test[mask]

                if metric_name == "mae":
                    values.append(mean_absolute_error(subset_true, subset_pred))
                else:
                    if len(subset_true) > 1 and np.std(subset_true) > 0:
                        values.append(r2_score(subset_true, subset_pred))
                    else:
                        values.append(float("nan"))

                x_labels.append(pos.replace("filler_", "f").replace("question_end", "q_end").replace("answer_prompt", "ans"))

            ax.plot(range(len(all_positions)), values, marker=markers[cat_name],
                    color=colors[cat_name], label=f"{cat_name} (n={mask.sum()})",
                    linewidth=2, markersize=8)

        ax.set_xticks(range(len(all_positions)))
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.set_ylabel("MAE" if metric_name == "mae" else "R²", fontsize=12)
        ax.set_title(f"Probe for A: {metric_name.upper()} by category\n(dots_250 hidden states, best layer per position)",
                     fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = args.output_dir / "filler_helped_A.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.savefig(str(out_path).replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out_path}")

    # --- Plot 2: A+Y same analysis ---
    target = "A+Y"
    if target in dots_results:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        for ax, metric_name in zip(axes, ["mae", "r2"]):
            for cat_name, mask in categories.items():
                if mask.sum() < 3:
                    continue

                values = []
                for pos in all_positions:
                    if pos not in dots_results[target]:
                        continue
                    layers = sorted(dots_results[target][pos].keys())
                    best_layer = max(layers, key=lambda l: dots_results[target][pos][l]["r2"])
                    y_pred = dots_results[target][pos][best_layer]["y_pred"]
                    y_test = dots_results[target][pos][best_layer]["y_test"]
                    subset_pred = y_pred[mask]
                    subset_true = y_test[mask]
                    if metric_name == "mae":
                        values.append(mean_absolute_error(subset_true, subset_pred))
                    else:
                        if len(subset_true) > 1 and np.std(subset_true) > 0:
                            values.append(r2_score(subset_true, subset_pred))
                        else:
                            values.append(float("nan"))

                ax.plot(range(len(values)), values, marker=markers[cat_name],
                        color=colors[cat_name], label=f"{cat_name} (n={mask.sum()})",
                        linewidth=2, markersize=8)

            ax.set_xticks(range(len(all_positions)))
            ax.set_xticklabels(x_labels, rotation=45, ha="right")
            ax.set_ylabel("MAE" if metric_name == "mae" else "R²", fontsize=12)
            ax.set_title(f"Probe for A+Y: {metric_name.upper()} by category\n(dots_250 hidden states)",
                         fontsize=12)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = args.output_dir / "filler_helped_AY.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.savefig(str(out_path).replace(".png", ".pdf"), bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")

    # --- Plot 3: Layerwise R² for A at answer_prompt, split by category ---
    target = "A"
    pos = "answer_prompt"
    if pos in dots_results.get(target, {}):
        fig, ax = plt.subplots(figsize=(12, 5))
        layers = sorted(dots_results[target][pos].keys())

        for cat_name, mask in categories.items():
            if mask.sum() < 3:
                continue

            r2_vals = []
            for layer in layers:
                y_pred = dots_results[target][pos][layer]["y_pred"]
                y_test = dots_results[target][pos][layer]["y_test"]
                subset_pred = y_pred[mask]
                subset_true = y_test[mask]
                if np.std(subset_true) > 0:
                    r2_vals.append(r2_score(subset_true, subset_pred))
                else:
                    r2_vals.append(float("nan"))

            ax.plot(layers, r2_vals, color=colors[cat_name],
                    label=f"{cat_name} (n={mask.sum()})", alpha=0.8, linewidth=1.5)

        ax.set_xlabel("Layer", fontsize=12)
        ax.set_ylabel("R²", fontsize=12)
        ax.set_title("Layerwise probe R² for A at answer_prompt\n(dots_250, split by filler effect)", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = args.output_dir / "filler_helped_layerwise.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.savefig(str(out_path).replace(".png", ".pdf"), bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")

    # --- Print summary table ---
    print("\n\nSummary: Probe MAE for A at key positions, by category")
    print("=" * 80)
    header = f"{'Category':<20} {'n':>4}  "
    for pos in ["question_end", "filler_0.50", "filler_1.00", "answer_prompt"]:
        header += f"  {pos:>15}"
    print(header)
    print("-" * 80)

    for cat_name, mask in categories.items():
        if mask.sum() < 3:
            continue
        row = f"{cat_name:<20} {mask.sum():>4}  "
        for pos in ["question_end", "filler_0.50", "filler_1.00", "answer_prompt"]:
            if pos not in dots_results["A"]:
                row += f"  {'N/A':>15}"
                continue
            layers = sorted(dots_results["A"][pos].keys())
            best_layer = max(layers, key=lambda l: dots_results["A"][pos][l]["r2"])
            y_pred = dots_results["A"][pos][best_layer]["y_pred"]
            y_test = dots_results["A"][pos][best_layer]["y_test"]
            mae = mean_absolute_error(y_test[mask], y_pred[mask])
            row += f"  {mae:>15.2f}"
        print(row)

    print()
    print("Summary: Probe R² for A at key positions, by category")
    print("=" * 80)
    header = f"{'Category':<20} {'n':>4}  "
    for pos in ["question_end", "filler_0.50", "filler_1.00", "answer_prompt"]:
        header += f"  {pos:>15}"
    print(header)
    print("-" * 80)

    for cat_name, mask in categories.items():
        if mask.sum() < 3:
            continue
        row = f"{cat_name:<20} {mask.sum():>4}  "
        for pos in ["question_end", "filler_0.50", "filler_1.00", "answer_prompt"]:
            if pos not in dots_results["A"]:
                row += f"  {'N/A':>15}"
                continue
            layers = sorted(dots_results["A"][pos].keys())
            best_layer = max(layers, key=lambda l: dots_results["A"][pos][l]["r2"])
            y_pred = dots_results["A"][pos][best_layer]["y_pred"]
            y_test = dots_results["A"][pos][best_layer]["y_test"]
            subset_pred = y_pred[mask]
            subset_true = y_test[mask]
            if np.std(subset_true) > 0:
                r2 = r2_score(subset_true, subset_pred)
                row += f"  {r2:>15.4f}"
            else:
                row += f"  {'N/A':>15}"
        print(row)


if __name__ == "__main__":
    main()
