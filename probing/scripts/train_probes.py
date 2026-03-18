"""
train_probes.py

Train linear probes on extracted hidden states to decode intermediate
values during filler token processing.

Usage:
    python probing/scripts/train_probes.py \
        --extraction-dir probing/extracted_states \
        --output-dir probing/probe_results
"""

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score


# ==============================================================================
# Data loading
# ==============================================================================

def load_condition_data(extraction_dir: Path, condition_name: str):
    """
    Load all extracted states for one condition.

    Returns:
        states: {position_name: {layer_idx: np.ndarray of shape (n_problems, 7168)}}
        metadata: list of dicts with fact_value, x, answer, model_correct, etc.
    """
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

    # Stack into arrays
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
          f"shape={states[sample_pos][sample_layer].shape}")

    return states, metadata


def make_targets(metadata: list) -> dict:
    """Create target arrays for probing."""
    A = np.array([m["fact_value"] for m in metadata], dtype=np.float32)
    Y = np.array([m["x"] for m in metadata], dtype=np.float32)
    answer = np.array([m["answer"] for m in metadata], dtype=np.float32)

    correct_raw = [m["model_correct"] for m in metadata]
    if any(c is None for c in correct_raw):
        correct = np.zeros(len(metadata), dtype=bool)
    else:
        correct = np.array(correct_raw, dtype=bool)

    rng = np.random.RandomState(42)
    A_shuffled = A.copy()
    rng.shuffle(A_shuffled)

    return {
        "A": A,
        "Y": Y,
        "A+Y": answer,
        "A_shuffled": A_shuffled,
        "model_correct": correct,
    }


# ==============================================================================
# Probe training
# ==============================================================================

def train_probe(X_train, y_train, X_test, y_test):
    """Train a ridge regression probe and evaluate."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std < 1e-8] = 1.0
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std

    alphas = np.logspace(-2, 6, 20)
    probe = RidgeCV(alphas=alphas, cv=5)
    probe.fit(X_train_norm, y_train)

    y_pred = probe.predict(X_test_norm)

    return {
        "r2": r2_score(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "exact_match": float(np.mean(np.round(y_pred) == y_test)),
        "best_alpha": float(probe.alpha_),
        "y_pred": y_pred,
        "y_test": y_test,
    }


def run_probes_for_condition(
    states: dict,
    targets: dict,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    target_names: list = None,
):
    """
    Train probes for all (position, layer, target) combinations.

    Returns:
        results: {target_name: {position_name: {layer_idx: probe_result}}}
    """
    if target_names is None:
        target_names = ["A", "Y", "A+Y", "A_shuffled"]

    results = defaultdict(lambda: defaultdict(dict))

    positions = sorted(states.keys())
    layers = sorted(states[positions[0]].keys())
    total = len(target_names) * len(positions) * len(layers)

    print(f"  Training {total} probes "
          f"({len(target_names)} targets x {len(positions)} positions x {len(layers)} layers)")

    count = 0
    for target_name in target_names:
        y = targets[target_name]
        y_train, y_test = y[train_idx], y[test_idx]

        for pos_name in positions:
            for layer_idx in layers:
                X = states[pos_name][layer_idx]
                X_train, X_test = X[train_idx], X[test_idx]

                result = train_probe(X_train, y_train, X_test, y_test)
                results[target_name][pos_name][layer_idx] = result

                count += 1
                if count % 200 == 0:
                    print(f"    {count}/{total} probes done")

    return results


# ==============================================================================
# Analysis 1: A emergence during filler
# ==============================================================================

def plot_emergence(all_results: dict, output_dir: str):
    """Plot R^2(A) and R^2(A+Y) vs filler position for each condition."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    targets = ["A", "A+Y", "Y"]
    titles = [
        "Intermediate value A\n(fact retrieval result)",
        "Final answer A+Y",
        "Addend Y (control -- explicit in prompt)",
    ]

    for ax, target, title in zip(axes, targets, titles):
        for cond_name, results in all_results.items():
            if target not in results:
                continue

            filler_positions = sorted(
                [p for p in results[target].keys() if p.startswith("filler_")],
                key=lambda p: float(p.split("_")[1])
            )

            if not filler_positions:
                for special_pos in ["question_end", "answer_prompt"]:
                    if special_pos in results[target]:
                        best_r2 = max(
                            results[target][special_pos][l]["r2"]
                            for l in results[target][special_pos]
                        )
                        ax.axhline(
                            y=best_r2, linestyle="--", alpha=0.5,
                            label=f"baseline ({special_pos})"
                        )
                continue

            fracs = [float(p.split("_")[1]) for p in filler_positions]
            r2_values = [
                max(results[target][pos][l]["r2"]
                    for l in results[target][pos])
                for pos in filler_positions
            ]
            ax.plot(fracs, r2_values, marker="o", label=cond_name, linewidth=2)

        ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5, label="chance")
        ax.set_xlabel("Position in filler (fraction)", fontsize=12)
        ax.set_ylabel("R^2 (best layer)", fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=-0.1)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/emergence.png", dpi=150, bbox_inches="tight")
    plt.savefig(f"{output_dir}/emergence.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: emergence plot")


# ==============================================================================
# Analysis 2: Layer-wise localization
# ==============================================================================

def plot_layerwise(all_results: dict, output_dir: str):
    """Plot R^2(A) across layers at the last filler position."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, target, title in [
        (axes[0], "A", "Intermediate value A"),
        (axes[1], "A+Y", "Final answer A+Y"),
    ]:
        for cond_name, results in all_results.items():
            if target not in results:
                continue

            filler_positions = sorted(
                [p for p in results[target].keys() if p.startswith("filler_")],
                key=lambda p: float(p.split("_")[1])
            )
            pos = filler_positions[-1] if filler_positions else "answer_prompt"

            if pos not in results[target]:
                continue

            layers = sorted(results[target][pos].keys())
            r2_values = [results[target][pos][l]["r2"] for l in layers]
            ax.plot(layers, r2_values, label=cond_name, alpha=0.8, linewidth=1.5)

        ax.set_xlabel("Layer", fontsize=12)
        ax.set_ylabel("R^2", fontsize=12)
        ax.set_title(f"Layer-wise probe: {title}\n(at last filler position)", fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/layerwise.png", dpi=150, bbox_inches="tight")
    plt.savefig(f"{output_dir}/layerwise.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: layerwise plot")


# ==============================================================================
# Analysis 3: Correct vs incorrect examples
# ==============================================================================

def plot_correct_vs_incorrect(all_results: dict, all_targets: dict,
                              test_idx: np.ndarray, output_dir: str):
    """Compare probe MAE on examples the model gets right vs wrong."""
    filler_conds = {k: v for k, v in all_results.items()
                    if any(p.startswith("filler_") for t in v for p in v[t])}

    if not filler_conds:
        return

    fig, axes = plt.subplots(1, len(filler_conds), figsize=(5 * len(filler_conds), 5))
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    for ax, (cond_name, results) in zip(axes, filler_conds.items()):
        if "A" not in results:
            continue

        filler_positions = sorted(
            [p for p in results["A"].keys() if p.startswith("filler_")],
            key=lambda p: float(p.split("_")[1])
        )
        if not filler_positions:
            continue

        fracs = [float(p.split("_")[1]) for p in filler_positions]

        # Correctness mask for TEST examples only
        full_correct = all_targets[cond_name]["model_correct"]
        test_correct = full_correct[test_idx]

        for subset_name, mask in [("correct", test_correct), ("incorrect", ~test_correct)]:
            if mask.sum() < 5:
                continue

            mae_values = []
            for pos in filler_positions:
                best_layer = max(
                    results["A"][pos].keys(),
                    key=lambda l: results["A"][pos][l]["r2"]
                )
                y_pred = results["A"][pos][best_layer]["y_pred"]
                y_test = results["A"][pos][best_layer]["y_test"]
                subset_mae = mean_absolute_error(y_test[mask], y_pred[mask])
                mae_values.append(subset_mae)

            ax.plot(fracs, mae_values, marker="o", label=subset_name)

        ax.set_xlabel("Position in filler")
        ax.set_ylabel("MAE for A (best layer)")
        ax.set_title(cond_name)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/correct_vs_incorrect.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: correct vs incorrect plot")


# ==============================================================================
# Analysis 4: A vs A+Y timing
# ==============================================================================

def plot_timing(all_results: dict, output_dir: str):
    """Plot A and A+Y emergence on the same axes to compare timing."""
    filler_conds = {k: v for k, v in all_results.items()
                    if any(p.startswith("filler_") for t in v for p in v[t])}

    if not filler_conds:
        return

    fig, axes = plt.subplots(1, len(filler_conds), figsize=(6 * len(filler_conds), 5))
    if len(filler_conds) == 1:
        axes = [axes]

    colors = {"A": "#2196F3", "A+Y": "#F44336", "Y": "#4CAF50", "A_shuffled": "#9E9E9E"}
    labels = {"A": "Intermediate A", "A+Y": "Final answer A+Y",
              "Y": "Addend Y (control)", "A_shuffled": "Shuffled A (chance)"}

    for ax, (cond_name, results) in zip(axes, filler_conds.items()):
        for target in ["A", "A+Y", "Y", "A_shuffled"]:
            if target not in results:
                continue

            filler_positions = sorted(
                [p for p in results[target].keys() if p.startswith("filler_")],
                key=lambda p: float(p.split("_")[1])
            )
            if not filler_positions:
                continue

            fracs = [float(p.split("_")[1]) for p in filler_positions]
            r2_values = [
                max(results[target][pos][l]["r2"] for l in results[target][pos])
                for pos in filler_positions
            ]

            ax.plot(fracs, r2_values, marker="o", color=colors.get(target, "black"),
                    label=labels.get(target, target), linewidth=2)

        ax.set_xlabel("Position in filler (fraction)", fontsize=12)
        ax.set_ylabel("R^2 (best layer)", fontsize=12)
        ax.set_title(cond_name, fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=-0.1)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/timing.png", dpi=150, bbox_inches="tight")
    plt.savefig(f"{output_dir}/timing.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: timing plot")


# ==============================================================================
# Summary table
# ==============================================================================

def print_summary_table(all_results: dict, output_dir: str):
    """Print a summary table of best probe R^2."""
    lines = []
    header = (f"{'Condition':<20} {'Target':<12} {'Position':<15} "
              f"{'Best R^2':<10} {'Layer':<8} {'MAE':<10} {'Exact%':<10}")
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
                [p for p in results[target].keys() if p.startswith("filler_")],
                key=lambda p: float(p.split("_")[1])
            )
            if filler_positions:
                report_positions.append(filler_positions[-1])

            if "answer_prompt" in results[target]:
                report_positions.append("answer_prompt")

            for pos in report_positions:
                if pos not in results[target]:
                    continue
                best_layer = max(
                    results[target][pos].keys(),
                    key=lambda l: results[target][pos][l]["r2"]
                )
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
    parser.add_argument("--output-dir", type=Path, default=Path("probing/probe_results"))
    parser.add_argument("--train-fraction", type=float, default=0.75)
    parser.add_argument("--conditions", nargs="+", default=None,
                        help="Which conditions to probe. Default: all found.")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover conditions
    if args.conditions:
        condition_names = args.conditions
    else:
        condition_names = sorted([
            d.name for d in args.extraction_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])
    print(f"Conditions: {condition_names}")

    # Load conditions one at a time to manage memory, then keep only what's needed
    all_states = {}
    all_targets = {}
    n_problems = None

    for cond_name in condition_names:
        print(f"\nLoading {cond_name}...")
        states, metadata = load_condition_data(args.extraction_dir, cond_name)
        targets = make_targets(metadata)

        if n_problems is None:
            n_problems = len(metadata)
        else:
            assert len(metadata) == n_problems, (
                f"Condition {cond_name} has {len(metadata)} problems, "
                f"expected {n_problems}"
            )

        all_states[cond_name] = states
        all_targets[cond_name] = targets

        acc = np.mean(targets["model_correct"]) if targets["model_correct"].any() else 0
        print(f"  Model accuracy: {acc:.1%}")
        print(f"  A range: [{targets['A'].min():.0f}, {targets['A'].max():.0f}], "
              f"mean={targets['A'].mean():.1f}, std={targets['A'].std():.1f}")

    # Fixed train/test split (same across all conditions)
    rng = np.random.RandomState(42)
    indices = rng.permutation(n_problems)
    split = int(args.train_fraction * n_problems)
    train_idx = indices[:split]
    test_idx = indices[split:]
    print(f"\nTrain/test split: {len(train_idx)}/{len(test_idx)}")

    # Train probes for each condition
    all_results = {}
    for cond_name in condition_names:
        print(f"\n{'='*60}")
        print(f"Training probes for: {cond_name}")
        print(f"{'='*60}")

        results = run_probes_for_condition(
            all_states[cond_name],
            all_targets[cond_name],
            train_idx,
            test_idx,
        )

        all_results[cond_name] = {
            t: {p: dict(layers) for p, layers in positions.items()}
            for t, positions in results.items()
        }

        # Free states for this condition after probing (keep targets for analysis 3)
        del all_states[cond_name]

    # Save raw results
    with open(output_dir / "all_probe_results.pkl", "wb") as f:
        pickle.dump(all_results, f)

    # Save compact metrics (no predictions)
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
    with open(output_dir / "probe_metrics.json", "w") as f:
        json.dump(compact, f, indent=2)

    # Plots
    print("\n\nGenerating plots...")
    plot_emergence(all_results, str(output_dir))
    plot_layerwise(all_results, str(output_dir))
    plot_timing(all_results, str(output_dir))
    plot_correct_vs_incorrect(all_results, all_targets, test_idx, str(output_dir))
    print_summary_table(all_results, str(output_dir))

    print(f"\nAll done! Results in {output_dir}/")


if __name__ == "__main__":
    main()
