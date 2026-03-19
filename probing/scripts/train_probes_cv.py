"""
train_probes_cv.py

Train linear probes using k-fold cross-validation to get out-of-sample
predictions for ALL examples (not just a held-out test set).

For each (target, position, layer) combo:
  - Split data into k folds
  - For each fold: train on k-1 folds, predict held-out fold
  - Concatenate predictions → full-dataset out-of-sample predictions

This gives ~4x the effective test set compared to a single 75/25 split.

Usage:
    uv run --project probing python probing/scripts/train_probes_cv.py \
        --extraction-dir probing/extracted_states \
        --output-dir probing/probe_results/cv
"""

import argparse
import json
import os
import pickle
import time
import warnings
from collections import defaultdict
from pathlib import Path

# Limit BLAS threads so joblib workers don't oversubscribe CPUs
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", message="Ill-conditioned")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold


# ==============================================================================
# Data loading (same as train_probes.py)
# ==============================================================================

def load_condition_data(extraction_dir: Path, condition_name: str):
    cond_dir = extraction_dir / condition_name
    files = sorted(cond_dir.glob("prob_*.pkl"))
    if not files:
        raise FileNotFoundError(f"No files found in {cond_dir}")

    states_by_pos = defaultdict(lambda: defaultdict(list))
    metadata = []

    for f in files:
        with open(f, "rb") as fp:
            data = pickle.load(fp)
        metadata.append({
            "problem_idx": data["problem_idx"],
            "fact_value": data["fact_value"],
            "x": data["x"],
            "answer": data["answer"],
            "model_correct": data["model_correct"],
            "model_answer": data.get("model_answer"),
        })
        for pos_name, layer_dict in data["states"].items():
            for layer_idx, state_vec in layer_dict.items():
                states_by_pos[pos_name][layer_idx].append(state_vec)

    states = {}
    for pos_name in states_by_pos:
        states[pos_name] = {}
        for layer_idx in states_by_pos[pos_name]:
            states[pos_name][layer_idx] = np.stack(
                states_by_pos[pos_name][layer_idx], axis=0
            ).astype(np.float32)

    n = len(metadata)
    sample_pos = list(states.keys())[0]
    sample_layer = list(states[sample_pos].keys())[0]
    print(f"  Loaded {n} problems, {len(states)} positions, "
          f"{len(states[sample_pos])} layers, "
          f"shape={states[sample_pos][sample_layer].shape}", flush=True)

    return states, metadata


def make_targets(metadata):
    A = np.array([m["fact_value"] for m in metadata], dtype=np.float32)
    Y = np.array([m["x"] for m in metadata], dtype=np.float32)
    answer = np.array([m["answer"] for m in metadata], dtype=np.float32)

    correct_raw = [m["model_correct"] for m in metadata]
    correct = np.array(correct_raw, dtype=bool) if not any(c is None for c in correct_raw) else np.zeros(len(metadata), dtype=bool)

    rng = np.random.RandomState(42)
    A_shuffled = A.copy()
    rng.shuffle(A_shuffled)

    return {"A": A, "Y": Y, "A+Y": answer, "A_shuffled": A_shuffled, "model_correct": correct}


# ==============================================================================
# CV probe training
# ==============================================================================

def train_probe_cv(X, y, n_folds=5, seed=42):
    """Train probe with k-fold CV, returning out-of-sample predictions for all examples."""
    n = len(y)
    y_pred_all = np.full(n, np.nan)
    alphas = np.logspace(-2, 6, 20)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        std[std < 1e-8] = 1.0
        X_train_norm = (X_train - mean) / std
        X_test_norm = (X_test - mean) / std

        probe = RidgeCV(alphas=alphas, cv=5)
        probe.fit(X_train_norm, y_train)
        y_pred_all[test_idx] = probe.predict(X_test_norm)

    assert not np.any(np.isnan(y_pred_all)), "Some examples not predicted"

    return {
        "r2": r2_score(y, y_pred_all),
        "mae": mean_absolute_error(y, y_pred_all),
        "exact_match": float(np.mean(np.round(y_pred_all) == y)),
        "y_pred": y_pred_all,
        "y_true": y,
    }


def _fit_one(target_name, pos_name, layer_idx, X, y, n_folds):
    """Fit a single probe — called by joblib workers."""
    result = train_probe_cv(X, y, n_folds=n_folds)
    return target_name, pos_name, layer_idx, result


def run_probes_cv(states, targets, n_folds=5, target_names=None, n_jobs=4):
    """Train CV probes in parallel using joblib."""
    if target_names is None:
        target_names = ["A", "Y", "A+Y", "A_shuffled"]

    positions = sorted(states.keys())
    layers = sorted(states[positions[0]].keys())
    total = len(target_names) * len(positions) * len(layers)

    print(f"  Training {total} probes with {n_folds}-fold CV "
          f"({len(target_names)} targets x {len(positions)} positions x {len(layers)} layers) "
          f"using {n_jobs} workers",
          flush=True)

    jobs = []
    for target_name in target_names:
        y = targets[target_name]
        for pos_name in positions:
            for layer_idx in layers:
                X = states[pos_name][layer_idx]
                jobs.append(delayed(_fit_one)(
                    target_name, pos_name, layer_idx, X, y, n_folds
                ))

    t0 = time.time()
    outputs = Parallel(n_jobs=n_jobs, verbose=10)(jobs)
    elapsed = time.time() - t0
    print(f"  Done: {total} probes in {elapsed:.0f}s ({total/elapsed:.1f} probes/s)", flush=True)

    results = defaultdict(lambda: defaultdict(dict))
    for target_name, pos_name, layer_idx, result in outputs:
        results[target_name][pos_name][layer_idx] = result

    return results


# ==============================================================================
# Plotting (adapted for full-dataset predictions)
# ==============================================================================

def get_best_layer_preds(results, target, pos):
    layers = sorted(results[target][pos].keys())
    best_layer = max(layers, key=lambda l: results[target][pos][l]["r2"])
    r = results[target][pos][best_layer]
    return r["y_pred"], r["y_true"], best_layer


def plot_emergence(all_results, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    targets = ["A", "A+Y", "Y"]
    titles = ["Intermediate value A\n(fact retrieval result)", "Final answer A+Y",
              "Addend Y (control)"]

    for ax, target, title in zip(axes, targets, titles):
        for cond_name, results in all_results.items():
            if target not in results:
                continue
            filler_positions = sorted(
                [p for p in results[target] if p.startswith("filler_")],
                key=lambda p: float(p.split("_")[1])
            )
            if not filler_positions:
                for sp in ["question_end", "answer_prompt"]:
                    if sp in results[target]:
                        best_r2 = max(results[target][sp][l]["r2"] for l in results[target][sp])
                        ax.axhline(y=best_r2, linestyle="--", alpha=0.5, label=f"baseline ({sp})")
                continue

            fracs = [float(p.split("_")[1]) for p in filler_positions]
            r2_values = [max(results[target][pos][l]["r2"] for l in results[target][pos])
                         for pos in filler_positions]
            ax.plot(fracs, r2_values, marker="o", label=cond_name, linewidth=2)

        ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("Position in filler (fraction)")
        ax.set_ylabel("R² (best layer)")
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=-0.1)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/emergence.png", dpi=150, bbox_inches="tight")
    plt.savefig(f"{output_dir}/emergence.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: emergence", flush=True)


def plot_cross_condition(all_results, all_targets, all_metadata, output_dir):
    """The main cross-condition analysis with MAE and tolerance metrics."""
    if "baseline" not in all_targets or "dots_250" not in all_targets:
        print("Need both baseline and dots_250 for cross-condition analysis")
        return

    bl_correct = all_targets["baseline"]["model_correct"]
    dt_correct = all_targets["dots_250"]["model_correct"]

    categories = {
        "filler_helped\n(wrong→right)": ~bl_correct & dt_correct,
        "both_correct": bl_correct & dt_correct,
        "both_wrong": ~bl_correct & ~dt_correct,
    }
    cat_colors = {
        "filler_helped\n(wrong→right)": "#4CAF50",
        "both_correct": "#2196F3",
        "both_wrong": "#9E9E9E",
    }

    for cat_name, mask in categories.items():
        print(f"  {cat_name.replace(chr(10), ' ')}: n={mask.sum()}")

    dots_results = all_results["dots_250"]

    filler_positions = sorted(
        [p for p in dots_results["A"] if p.startswith("filler_")],
        key=lambda p: float(p.split("_")[1])
    )
    all_positions = ["question_end"] + filler_positions + ["answer_prompt"]
    all_positions = [p for p in all_positions if p in dots_results["A"]]
    pos_labels = [p.replace("filler_", "f").replace("question_end", "q_end").replace("answer_prompt", "ans")
                  for p in all_positions]

    # --- Plot: Per-example error trajectories ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, (cat_name, mask) in zip(axes, categories.items()):
        n_cat = mask.sum()
        if n_cat < 1:
            continue
        errors = np.zeros((n_cat, len(all_positions)))
        for j, pos in enumerate(all_positions):
            y_pred, y_true, _ = get_best_layer_preds(dots_results, "A", pos)
            errors[:, j] = np.abs(y_pred[mask] - y_true[mask])

        x = np.arange(len(all_positions))
        for i in range(n_cat):
            ax.plot(x, errors[i], color=cat_colors[cat_name], alpha=max(0.05, 0.15 * 27 / n_cat), linewidth=0.8)
        ax.plot(x, errors.mean(axis=0), color=cat_colors[cat_name],
                linewidth=3, marker="o", markersize=6, label=f"mean (n={n_cat})")
        ax.plot(x, np.median(errors, axis=0), color=cat_colors[cat_name],
                linewidth=2, linestyle="--", marker="s", markersize=4, label="median")

        ax.set_xticks(x)
        ax.set_xticklabels(pos_labels, rotation=45, ha="right")
        ax.set_title(cat_name, fontsize=13)
        ax.set_ylabel("Absolute probe error for A")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    fig.suptitle("Per-example probe error trajectories (A, dots_250, 5-fold CV predictions)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/error_trajectories.png", dpi=150, bbox_inches="tight")
    plt.savefig(f"{output_dir}/error_trajectories.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: error_trajectories", flush=True)

    # --- Plot: MAE + fraction within tolerance ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax = axes[0]
    for cat_name, mask in categories.items():
        if mask.sum() < 3:
            continue
        mae_vals = []
        for pos in all_positions:
            y_pred, y_true, _ = get_best_layer_preds(dots_results, "A", pos)
            mae_vals.append(np.mean(np.abs(y_pred[mask] - y_true[mask])))
        short = cat_name.split("\n")[0]
        ax.plot(range(len(all_positions)), mae_vals, marker="o",
                color=cat_colors[cat_name], linewidth=2, label=f"{short} (n={mask.sum()})")
    ax.set_xticks(range(len(all_positions)))
    ax.set_xticklabels(pos_labels, rotation=45, ha="right")
    ax.set_ylabel("MAE for A")
    ax.set_title("Mean Absolute Error")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    ax = axes[1]
    tol = 10
    for cat_name, mask in categories.items():
        if mask.sum() < 3:
            continue
        frac_vals = []
        for pos in all_positions:
            y_pred, y_true, _ = get_best_layer_preds(dots_results, "A", pos)
            frac_vals.append(np.mean(np.abs(y_pred[mask] - y_true[mask]) <= tol))
        short = cat_name.split("\n")[0]
        ax.plot(range(len(all_positions)), frac_vals, marker="o",
                color=cat_colors[cat_name], linewidth=2, label=f"{short} (n={mask.sum()})")
    ax.set_xticks(range(len(all_positions)))
    ax.set_xticklabels(pos_labels, rotation=45, ha="right")
    ax.set_ylabel(f"Fraction with |error| ≤ {tol}")
    ax.set_title(f"Probe accuracy (within ±{tol})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    fig.suptitle("Probe for A across filler positions (dots_250, 5-fold CV)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/mae_and_tolerance.png", dpi=150, bbox_inches="tight")
    plt.savefig(f"{output_dir}/mae_and_tolerance.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: mae_and_tolerance", flush=True)

    # --- Plot: Error reduction scatter ---
    if "question_end" in dots_results["A"] and "answer_prompt" in dots_results["A"]:
        pred_qe, true_qe, _ = get_best_layer_preds(dots_results, "A", "question_end")
        pred_ap, true_ap, _ = get_best_layer_preds(dots_results, "A", "answer_prompt")
        err_qe = np.abs(pred_qe - true_qe)
        err_ap = np.abs(pred_ap - true_ap)

        fig, ax = plt.subplots(figsize=(7, 7))
        for cat_name, mask in categories.items():
            if mask.sum() < 3:
                continue
            short = cat_name.split("\n")[0]
            ax.scatter(err_qe[mask], err_ap[mask], color=cat_colors[cat_name],
                       label=f"{short} (n={mask.sum()})", alpha=0.5, s=30,
                       edgecolors="white", linewidth=0.5)
        lim = max(err_qe.max(), err_ap.max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", alpha=0.3, label="no change")
        ax.set_xlabel("Probe |error| at question_end")
        ax.set_ylabel("Probe |error| at answer_prompt")
        ax.set_title("Error reduction: question_end → answer_prompt\n(below diagonal = improved during filler)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect("equal")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/error_reduction_scatter.png", dpi=150, bbox_inches="tight")
        plt.savefig(f"{output_dir}/error_reduction_scatter.pdf", bbox_inches="tight")
        plt.close()
        print("Saved: error_reduction_scatter", flush=True)

    # --- Plot: Box plots at key positions ---
    key_positions = [p for p in ["question_end", "filler_0.50", "filler_1.00", "answer_prompt"]
                     if p in dots_results["A"]]
    key_labels = [p.replace("filler_", "f").replace("question_end", "q_end").replace("answer_prompt", "ans")
                  for p in key_positions]

    fig, axes = plt.subplots(1, len(key_positions), figsize=(4 * len(key_positions), 5), sharey=True)
    for ax, pos, label in zip(axes, key_positions, key_labels):
        y_pred, y_true, best_l = get_best_layer_preds(dots_results, "A", pos)
        abs_errors = np.abs(y_pred - y_true)
        box_data, box_labels_list, box_colors = [], [], []
        for cat_name, mask in categories.items():
            if mask.sum() < 3:
                continue
            box_data.append(abs_errors[mask])
            short = cat_name.split("\n")[0]
            box_labels_list.append(f"{short}\n(n={mask.sum()})")
            box_colors.append(cat_colors[cat_name])
        bp = ax.boxplot(box_data, tick_labels=box_labels_list, patch_artist=True,
                        widths=0.6, showfliers=True, flierprops=dict(markersize=3))
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        ax.set_title(f"{label} (L{best_l})")
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("Absolute probe error for A")
    fig.suptitle("Probe error distributions by category (5-fold CV)", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/boxplots.png", dpi=150, bbox_inches="tight")
    plt.savefig(f"{output_dir}/boxplots.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: boxplots", flush=True)

    # --- Summary table ---
    print(f"\n{'='*90}")
    print("SUMMARY: Probe for A — MAE and fraction within ±10 (5-fold CV, all examples)")
    print(f"{'='*90}")
    header = f"{'Category':<25} {'n':>4}"
    for pos in key_positions:
        short = pos.replace("filler_", "f").replace("question_end", "q_end").replace("answer_prompt", "ans")
        header += f"  {short+' MAE':>10} {short+' ±10':>8}"
    print(header)
    print("-" * 90)
    for cat_name, mask in categories.items():
        if mask.sum() < 3:
            continue
        short_cat = cat_name.replace("\n", " ")
        row = f"{short_cat:<25} {mask.sum():>4}"
        for pos in key_positions:
            y_pred, y_true, _ = get_best_layer_preds(dots_results, "A", pos)
            errs = np.abs(y_pred[mask] - y_true[mask])
            row += f"  {errs.mean():>10.1f} {np.mean(errs <= 10):>7.0%}"
        print(row)
    print()


def print_summary_table(all_results, output_dir):
    lines = []
    header = (f"{'Condition':<20} {'Target':<12} {'Position':<15} "
              f"{'Best R²':<10} {'Layer':<8} {'MAE':<10} {'Exact%':<10}")
    lines.append(header)
    lines.append("-" * len(header))

    for cond_name, results in all_results.items():
        for target in ["A", "A+Y", "Y", "A_shuffled"]:
            if target not in results:
                continue
            report_positions = []
            if "question_end" in results[target]:
                report_positions.append("question_end")
            filler_positions = sorted(
                [p for p in results[target] if p.startswith("filler_")],
                key=lambda p: float(p.split("_")[1])
            )
            if filler_positions:
                report_positions.append(filler_positions[-1])
            if "answer_prompt" in results[target]:
                report_positions.append("answer_prompt")

            for pos in report_positions:
                best_layer = max(results[target][pos], key=lambda l: results[target][pos][l]["r2"])
                r = results[target][pos][best_layer]
                line = (f"{cond_name:<20} {target:<12} {pos:<15} "
                        f"{r['r2']:<10.4f} {str(best_layer):<8} "
                        f"{r['mae']:<10.2f} {r['exact_match']:<10.1%}")
                lines.append(line)
        lines.append("")

    summary = "\n".join(lines)
    print("\n" + summary)
    with open(f"{output_dir}/summary_table.txt", "w") as f:
        f.write(summary)


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extraction-dir", type=Path, default=Path("probing/extracted_states"))
    parser.add_argument("--output-dir", type=Path, default=Path("probing/probe_results/cv"))
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="Number of parallel workers (-1 = all CPUs)")
    parser.add_argument("--conditions", nargs="+", default=None)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.conditions:
        condition_names = args.conditions
    else:
        condition_names = sorted([
            d.name for d in args.extraction_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])
    print(f"Conditions: {condition_names}")
    print(f"Using {args.n_folds}-fold CV")

    # Process one condition at a time to stay within 16GB cgroup limit.
    # Load states → train probes → save results → free states before next condition.
    all_results = {}
    all_targets = {}
    all_metadata = {}

    for cond_name in condition_names:
        print(f"\nLoading {cond_name}...")
        states, metadata = load_condition_data(args.extraction_dir, cond_name)
        targets = make_targets(metadata)
        all_targets[cond_name] = targets
        all_metadata[cond_name] = metadata

        acc = np.mean(targets["model_correct"]) if targets["model_correct"].any() else 0
        print(f"  Model accuracy: {acc:.1%}")
        print(f"  A range: [{targets['A'].min():.0f}, {targets['A'].max():.0f}], "
              f"mean={targets['A'].mean():.1f}, std={targets['A'].std():.1f}")

        print(f"\n{'='*60}")
        print(f"Training CV probes for: {cond_name}")
        print(f"{'='*60}")

        results = run_probes_cv(
            states,
            targets,
            n_folds=args.n_folds,
            n_jobs=args.n_jobs,
        )

        all_results[cond_name] = {
            t: {p: dict(layers) for p, layers in positions.items()}
            for t, positions in results.items()
        }

        # Free states immediately — they're ~4.5GB per condition
        del states
        import gc; gc.collect()

    # Save results
    with open(output_dir / "all_probe_results_cv.pkl", "wb") as f:
        pickle.dump(all_results, f)

    compact = {}
    for cond in all_results:
        compact[cond] = {}
        for target in all_results[cond]:
            compact[cond][target] = {}
            for pos in all_results[cond][target]:
                compact[cond][target][pos] = {}
                for layer in all_results[cond][target][pos]:
                    r = all_results[cond][target][pos][layer]
                    compact[cond][target][pos][str(layer)] = {
                        "r2": float(r["r2"]),
                        "mae": float(r["mae"]),
                        "exact_match": float(r["exact_match"]),
                    }
    with open(output_dir / "probe_metrics_cv.json", "w") as f:
        json.dump(compact, f, indent=2)

    # Plots
    print("\n\nGenerating plots...", flush=True)
    plot_emergence(all_results, str(output_dir))
    print_summary_table(all_results, str(output_dir))
    plot_cross_condition(all_results, all_targets, all_metadata, str(output_dir))

    print(f"\nAll done! Results in {output_dir}/")


if __name__ == "__main__":
    main()
