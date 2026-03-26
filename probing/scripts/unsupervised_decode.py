"""
unsupervised_decode.py

Unsupervised discovery of intermediate values from filler token activations.

Goal: Given only (filler activations, model output) pairs, discover intermediate
computational values (like A) that the model uses internally but doesn't output.

Ground truth (A, x) is used ONLY for final evaluation, never during discovery.

Usage:
    uv run --project /workspace/filler-token-reasoning/probing \
        python probing/scripts/unsupervised_decode.py \
        --extraction-dir probing/extracted_states \
        --output-dir probing/unsupervised_results
"""

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA, FastICA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==============================================================================
# Data loading
# ==============================================================================

def load_dataset(extraction_dir: Path, condition: str, positions: list):
    """Load extracted hidden states and metadata for all examples.

    Returns:
        examples: list of dicts with keys: problem_idx, fact_value, x, answer,
            model_answer, model_correct, states: {pos: {layer: ndarray}}
    """
    cond_dir = extraction_dir / condition
    files = sorted(cond_dir.glob("prob_*.pkl"))
    examples = []
    for fpath in files:
        with open(fpath, "rb") as f:
            data = pickle.load(f)
        # Only include examples where model answered (non-None)
        if data.get("model_answer") is None:
            continue
        ex = {
            "problem_idx": data["problem_idx"],
            "fact_value": data["fact_value"],  # A
            "x": data["x"],                    # Y
            "answer": data["answer"],           # ground truth A+x
            "model_answer": data["model_answer"],
            "model_correct": data.get("model_correct", False),
            "states": {},
        }
        for pos in positions:
            if pos in data["states"]:
                ex["states"][pos] = data["states"][pos]
        examples.append(ex)
    return examples


def build_activation_matrix(examples, position, layer):
    """Stack hidden states into (n_examples, hidden_dim) matrix."""
    vecs = []
    valid_indices = []
    for i, ex in enumerate(examples):
        if position in ex["states"] and layer in ex["states"][position]:
            vecs.append(ex["states"][position][layer].astype(np.float32))
            valid_indices.append(i)
    if not vecs:
        return None, []
    return np.stack(vecs), valid_indices


# ==============================================================================
# Level 1: Self-supervised probing (predict model output from activations)
# ==============================================================================

def self_supervised_probe_sweep(examples, positions, layers, n_folds=5):
    """For each (position, layer), train a Ridge probe to predict model output.

    Uses cross-validation. No ground truth — target is model's own answer.

    Returns:
        results: {(pos, layer): {"mae": float, "r2": float, "corr": float}}
    """
    model_outputs = np.array([ex["model_answer"] for ex in examples], dtype=np.float32)
    results = {}

    for pos in positions:
        for layer in layers:
            X, valid_idx = build_activation_matrix(examples, pos, layer)
            if X is None or len(valid_idx) < 50:
                continue
            y = model_outputs[valid_idx]

            # Cross-validated Ridge
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            preds = np.zeros_like(y)
            for train_idx, test_idx in kf.split(X):
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X[train_idx])
                X_test = scaler.transform(X[test_idx])
                model = Ridge(alpha=1.0)
                model.fit(X_train, y[train_idx])
                preds[test_idx] = model.predict(X_test)

            mae = np.mean(np.abs(preds - y))
            ss_res = np.sum((y - preds) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            corr = np.corrcoef(preds, y)[0, 1] if np.std(y) > 0 else 0

            results[(pos, layer)] = {"mae": mae, "r2": r2, "corr": corr}

    return results


# ==============================================================================
# Level 2: PCA factor discovery
# ==============================================================================

def pca_analysis(examples, position, layer, n_components=20):
    """Run PCA on activations at (position, layer).

    Returns:
        pca: fitted PCA object
        projections: (n_examples, n_components) — examples projected onto PCs
        valid_indices: which examples were used
    """
    X, valid_idx = build_activation_matrix(examples, position, layer)
    if X is None:
        return None, None, []

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components)
    projections = pca.fit_transform(X_scaled)

    return pca, projections, valid_idx


# ==============================================================================
# Level 3: Intermediate value discovery
# ==============================================================================

def discover_residual_directions(examples, position, layer, n_components=10, n_folds=5):
    """
    Method A: Find directions that encode information BEYOND the model's output.

    1. Train probe: activations -> model output
    2. Subtract the output-predictive component
    3. PCA on residuals -> directions encoding intermediate values

    Returns:
        residual_pcs: (n_examples, n_components) projections onto residual PCs
        output_preds: probe predictions of model output
        valid_indices: which examples were used
        probe_direction: the learned weight vector (for analysis)
    """
    X, valid_idx = build_activation_matrix(examples, position, layer)
    if X is None:
        return None, None, [], None

    y = np.array([examples[i]["model_answer"] for i in valid_idx], dtype=np.float32)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train probe on full data to get the output-predictive direction
    probe = Ridge(alpha=1.0)
    probe.fit(X_scaled, y)
    output_preds = probe.predict(X_scaled)
    probe_direction = probe.coef_

    # Subtract the output-predictive component
    # Project out the direction that predicts the output
    proj_coef = probe.coef_ / (np.linalg.norm(probe.coef_) + 1e-8)
    output_component = X_scaled @ proj_coef[:, None] * proj_coef[None, :]
    X_residual = X_scaled - output_component

    # PCA on residuals
    pca = PCA(n_components=n_components)
    residual_pcs = pca.fit_transform(X_residual)

    return residual_pcs, output_preds, valid_idx, probe_direction


def discover_y_matched_direction(examples, position, layer):
    """
    Method B: Use Y-matching to find the A-encoding direction.

    Group examples by x (visible in prompt). Within each group, variation
    is driven by A. Find the direction of maximal within-group variation.

    Returns:
        direction: (hidden_dim,) unit vector encoding A
        projections: (n_examples,) scalar projection of each example onto direction
        valid_indices: which examples were used
    """
    X, valid_idx = build_activation_matrix(examples, position, layer)
    if X is None:
        return None, None, []

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Group by x value
    by_x = defaultdict(list)
    for i, vi in enumerate(valid_idx):
        by_x[examples[vi]["x"]].append(i)

    # Compute within-group covariance
    within_cov = np.zeros((X_scaled.shape[1], X_scaled.shape[1]))
    n_total = 0
    for x_val, indices in by_x.items():
        if len(indices) < 2:
            continue
        group = X_scaled[indices]
        group_centered = group - group.mean(axis=0)
        within_cov += group_centered.T @ group_centered
        n_total += len(indices)

    if n_total == 0:
        return None, None, []

    within_cov /= n_total

    # Top eigenvector of within-group covariance = direction of maximal
    # within-group variation = A-encoding direction
    eigenvalues, eigenvectors = np.linalg.eigh(within_cov)
    # eigh returns ascending order, take last (largest)
    direction = eigenvectors[:, -1]

    projections = X_scaled @ direction

    return direction, projections, valid_idx


def discover_ica_components(examples, position, layer, n_components=5):
    """
    Method C: ICA to find statistically independent sources.

    For addition task, A and x are independent variables, so ICA should
    separate them.

    Returns:
        ica: fitted ICA object
        sources: (n_examples, n_components)
        valid_indices: which examples were used
    """
    X, valid_idx = build_activation_matrix(examples, position, layer)
    if X is None:
        return None, None, []

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Reduce dimensionality first (ICA on 7168 dims is slow)
    pca = PCA(n_components=50)
    X_reduced = pca.fit_transform(X_scaled)

    ica = FastICA(n_components=n_components, random_state=42, max_iter=500)
    sources = ica.fit_transform(X_reduced)

    return ica, sources, valid_idx


# ==============================================================================
# Level 4: Evaluation (uses ground truth)
# ==============================================================================

def evaluate_directions(projections, valid_indices, examples, label=""):
    """Correlate discovered projections with ground truth A, x, A+x.

    projections: (n_examples,) or (n_examples, n_components)
    """
    A = np.array([examples[i]["fact_value"] for i in valid_indices], dtype=np.float32)
    x = np.array([examples[i]["x"] for i in valid_indices], dtype=np.float32)
    Ax = np.array([examples[i]["answer"] for i in valid_indices], dtype=np.float32)
    model_out = np.array([examples[i]["model_answer"] for i in valid_indices], dtype=np.float32)

    if projections.ndim == 1:
        projections = projections[:, None]

    results = []
    for comp_idx in range(projections.shape[1]):
        p = projections[:, comp_idx]
        if np.std(p) < 1e-8:
            continue

        corr_A = np.corrcoef(p, A)[0, 1]
        corr_x = np.corrcoef(p, x)[0, 1]
        corr_Ax = np.corrcoef(p, Ax)[0, 1]
        corr_out = np.corrcoef(p, model_out)[0, 1]

        # MAE as linear predictor of A
        slope_A = np.cov(p, A)[0, 1] / (np.var(p) + 1e-8)
        intercept_A = np.mean(A) - slope_A * np.mean(p)
        pred_A = slope_A * p + intercept_A
        mae_A = np.mean(np.abs(pred_A - A))

        results.append({
            "component": comp_idx,
            "corr_A": float(corr_A),
            "corr_x": float(corr_x),
            "corr_Ax": float(corr_Ax),
            "corr_output": float(corr_out),
            "mae_A": float(mae_A),
            "abs_corr_A": float(abs(corr_A)),
        })

    return results


# ==============================================================================
# Plotting
# ==============================================================================

def plot_probe_sweep(probe_results, output_dir):
    """Heatmap of self-supervised probe performance across (position, layer)."""
    positions = sorted(set(pos for pos, _ in probe_results.keys()))
    layers = sorted(set(layer for _, layer in probe_results.keys()))

    # Build matrix
    mae_matrix = np.full((len(layers), len(positions)), np.nan)
    for j, pos in enumerate(positions):
        for i, layer in enumerate(layers):
            if (pos, layer) in probe_results:
                mae_matrix[i, j] = probe_results[(pos, layer)]["mae"]

    fig, ax = plt.subplots(figsize=(max(8, len(positions) * 1.5), 10))
    im = ax.imshow(mae_matrix, aspect="auto", cmap="RdYlGn_r",
                   vmin=np.nanmin(mae_matrix), vmax=np.nanpercentile(mae_matrix, 90))

    # Labels
    pos_labels = []
    for p in positions:
        if p == "question_end": pos_labels.append("q_end")
        elif p == "pre_filler": pos_labels.append("pre_fill")
        elif p == "answer_prompt": pos_labels.append("ans")
        elif p == "pre_answer": pos_labels.append("pre_ans")
        elif p.startswith("filler_k"): pos_labels.append(f"k={p.split('k')[1]}")
        else: pos_labels.append(p)

    ax.set_xticks(range(len(positions)))
    ax.set_xticklabels(pos_labels, rotation=45, ha="right")
    layer_labels = [str(l) if l % 5 == 0 else "" for l in layers]
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layer_labels, fontsize=7)
    ax.set_ylabel("Layer")
    ax.set_title("Self-supervised probe MAE\n(predicting model output from activations)")
    plt.colorbar(im, ax=ax, label="MAE")

    plt.tight_layout()
    plt.savefig(output_dir / "probe_sweep_heatmap.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "probe_sweep_heatmap.pdf", bbox_inches="tight")
    plt.close()


def plot_component_correlations(eval_results, method_name, output_dir):
    """Bar chart of correlations with A, x, A+x for each component."""
    if not eval_results:
        return

    n = len(eval_results)
    fig, ax = plt.subplots(figsize=(max(8, n * 1.2), 5))

    x_pos = np.arange(n)
    width = 0.25

    corr_A = [r["corr_A"] for r in eval_results]
    corr_x = [r["corr_x"] for r in eval_results]
    corr_Ax = [r["corr_Ax"] for r in eval_results]

    ax.bar(x_pos - width, corr_A, width, label="corr(A)", color="#4CAF50", alpha=0.8)
    ax.bar(x_pos, corr_x, width, label="corr(x)", color="#2196F3", alpha=0.8)
    ax.bar(x_pos + width, corr_Ax, width, label="corr(A+x)", color="#FF9800", alpha=0.8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"PC{r['component']}" for r in eval_results])
    ax.set_ylabel("Correlation")
    ax.set_title(f"{method_name}: component correlations with ground truth")
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fname = method_name.lower().replace(" ", "_")
    plt.savefig(output_dir / f"{fname}_correlations.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / f"{fname}_correlations.pdf", bbox_inches="tight")
    plt.close()


def plot_scatter_best_component(projections, valid_indices, examples,
                                 best_comp, variable_name, output_dir, method_name):
    """Scatter plot: best discovered component vs ground truth variable."""
    gt = np.array([examples[i]["fact_value"] if variable_name == "A"
                    else examples[i]["x"] if variable_name == "x"
                    else examples[i]["answer"]
                    for i in valid_indices], dtype=np.float32)

    if projections.ndim == 1:
        p = projections
    else:
        p = projections[:, best_comp]

    corr = np.corrcoef(p, gt)[0, 1]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(gt, p, alpha=0.3, s=15, color="#4CAF50")

    # Best fit line
    slope = np.cov(p, gt)[0, 1] / (np.var(gt) + 1e-8)
    intercept = np.mean(p) - slope * np.mean(gt)
    x_line = np.array([gt.min(), gt.max()])
    ax.plot(x_line, slope * x_line + intercept, "r--", linewidth=2, alpha=0.7)

    ax.set_xlabel(f"Ground truth {variable_name}", fontsize=12)
    ax.set_ylabel(f"Discovered component", fontsize=12)
    ax.set_title(f"{method_name}: component vs {variable_name} (r={corr:.3f})", fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f"{method_name.lower().replace(' ', '_')}_{variable_name}_scatter"
    plt.savefig(output_dir / f"{fname}.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / f"{fname}.pdf", bbox_inches="tight")
    plt.close()


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Unsupervised intermediate value discovery")
    parser.add_argument("--extraction-dir", type=Path,
                        default=Path("probing/extracted_states"))
    parser.add_argument("--condition", default="dots_50")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("probing/unsupervised_results"))
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--correct-only", action="store_true",
                        help="Only use examples the model got correct")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Positions to analyze
    positions = ["question_end", "filler_k1", "filler_k5", "filler_k10",
                 "filler_k25", "filler_k50", "answer_prompt", "pre_answer"]

    print("Loading data...")
    examples = load_dataset(args.extraction_dir, args.condition, positions)
    if args.max_examples:
        examples = examples[:args.max_examples]
    if args.correct_only:
        examples = [e for e in examples if e["model_correct"]]
    print(f"  {len(examples)} examples loaded")

    # Available positions (may vary by condition)
    available_positions = set()
    for ex in examples[:5]:
        available_positions.update(ex["states"].keys())
    positions = [p for p in positions if p in available_positions]
    print(f"  Positions: {positions}")

    layers = list(range(61))

    # ==================================================================
    # Level 1: Self-supervised probe sweep
    # ==================================================================
    print("\n" + "=" * 60)
    print("Level 1: Self-supervised probing (model output from activations)")
    print("=" * 60)

    probe_results = self_supervised_probe_sweep(examples, positions, layers)

    # Find best (position, layer)
    best_key = min(probe_results.keys(), key=lambda k: probe_results[k]["mae"])
    best = probe_results[best_key]
    print(f"\nBest: {best_key[0]} L{best_key[1]}: MAE={best['mae']:.1f}, "
          f"R²={best['r2']:.3f}, corr={best['corr']:.3f}")

    # Top 5 per position
    for pos in positions:
        pos_results = {k: v for k, v in probe_results.items() if k[0] == pos}
        if not pos_results:
            continue
        best_k = min(pos_results.keys(), key=lambda k: pos_results[k]["mae"])
        b = pos_results[best_k]
        print(f"  {pos:<16} best L{best_k[1]}: MAE={b['mae']:.1f}, R²={b['r2']:.3f}")

    plot_probe_sweep(probe_results, args.output_dir)
    print("Saved probe sweep heatmap")

    # ==================================================================
    # Level 2: PCA at best positions
    # ==================================================================
    print("\n" + "=" * 60)
    print("Level 2: PCA factor discovery")
    print("=" * 60)

    # Run PCA at pre_answer (most likely to work) and best filler position
    filler_positions = [p for p in positions if p.startswith("filler_")]
    best_filler_key = None
    if filler_positions:
        filler_results = {k: v for k, v in probe_results.items() if k[0] in filler_positions}
        if filler_results:
            best_filler_key = min(filler_results.keys(), key=lambda k: filler_results[k]["mae"])

    analysis_positions = []
    if "pre_answer" in positions:
        # Find best layer for pre_answer
        pa_results = {k: v for k, v in probe_results.items() if k[0] == "pre_answer"}
        if pa_results:
            best_pa_layer = min(pa_results.keys(), key=lambda k: pa_results[k]["mae"])[1]
            analysis_positions.append(("pre_answer", best_pa_layer))

    if best_filler_key:
        analysis_positions.append(best_filler_key)

    if "answer_prompt" in positions:
        ap_results = {k: v for k, v in probe_results.items() if k[0] == "answer_prompt"}
        if ap_results:
            best_ap_layer = min(ap_results.keys(), key=lambda k: ap_results[k]["mae"])[1]
            analysis_positions.append(("answer_prompt", best_ap_layer))

    for pos, layer in analysis_positions:
        print(f"\n--- PCA at {pos} L{layer} ---")
        pca, projections, valid_idx = pca_analysis(examples, pos, layer, n_components=10)
        if pca is None:
            print("  Skipped (no data)")
            continue

        print(f"  Explained variance: {pca.explained_variance_ratio_[:5].round(3)}")

        eval_res = evaluate_directions(projections, valid_idx, examples)
        for r in eval_res[:5]:
            print(f"  PC{r['component']}: corr(A)={r['corr_A']:.3f}, "
                  f"corr(x)={r['corr_x']:.3f}, corr(A+x)={r['corr_Ax']:.3f}, "
                  f"MAE(A)={r['mae_A']:.1f}")

        plot_component_correlations(eval_res[:10], f"PCA {pos} L{layer}", args.output_dir)

        # Scatter of best A-correlated component
        best_A = max(eval_res, key=lambda r: r["abs_corr_A"])
        plot_scatter_best_component(
            projections, valid_idx, examples,
            best_A["component"], "A", args.output_dir,
            f"PCA {pos} L{layer}"
        )

    # ==================================================================
    # Level 3: Intermediate value discovery
    # ==================================================================
    print("\n" + "=" * 60)
    print("Level 3: Intermediate value discovery")
    print("=" * 60)

    for pos, layer in analysis_positions:
        print(f"\n--- Method A (residual directions) at {pos} L{layer} ---")
        residual_pcs, output_preds, valid_idx, probe_dir = \
            discover_residual_directions(examples, pos, layer, n_components=10)
        if residual_pcs is None:
            print("  Skipped")
            continue

        eval_res = evaluate_directions(residual_pcs, valid_idx, examples)
        for r in eval_res[:5]:
            print(f"  Residual PC{r['component']}: corr(A)={r['corr_A']:.3f}, "
                  f"corr(x)={r['corr_x']:.3f}, corr(A+x)={r['corr_Ax']:.3f}")

        plot_component_correlations(eval_res[:10],
                                     f"Residual {pos} L{layer}", args.output_dir)

        best_A = max(eval_res, key=lambda r: r["abs_corr_A"])
        if abs(best_A["corr_A"]) > 0.1:
            plot_scatter_best_component(
                residual_pcs, valid_idx, examples,
                best_A["component"], "A", args.output_dir,
                f"Residual {pos} L{layer}"
            )

        print(f"\n--- Method B (Y-matched direction) at {pos} L{layer} ---")
        direction, projections_1d, valid_idx = \
            discover_y_matched_direction(examples, pos, layer)
        if direction is None:
            print("  Skipped")
            continue

        eval_res = evaluate_directions(projections_1d, valid_idx, examples)
        r = eval_res[0]
        print(f"  Y-matched direction: corr(A)={r['corr_A']:.3f}, "
              f"corr(x)={r['corr_x']:.3f}, corr(A+x)={r['corr_Ax']:.3f}, "
              f"MAE(A)={r['mae_A']:.1f}")

        if abs(r["corr_A"]) > 0.1:
            plot_scatter_best_component(
                projections_1d, valid_idx, examples,
                0, "A", args.output_dir,
                f"Y-matched {pos} L{layer}"
            )

        print(f"\n--- Method C (ICA) at {pos} L{layer} ---")
        ica, sources, valid_idx = discover_ica_components(examples, pos, layer)
        if sources is None:
            print("  Skipped")
            continue

        eval_res = evaluate_directions(sources, valid_idx, examples)
        for r in sorted(eval_res, key=lambda r: -r["abs_corr_A"])[:3]:
            print(f"  IC{r['component']}: corr(A)={r['corr_A']:.3f}, "
                  f"corr(x)={r['corr_x']:.3f}, corr(A+x)={r['corr_Ax']:.3f}")

        plot_component_correlations(eval_res,
                                     f"ICA {pos} L{layer}", args.output_dir)

    # ==================================================================
    # Save all results
    # ==================================================================
    summary = {
        "condition": args.condition,
        "n_examples": len(examples),
        "correct_only": args.correct_only,
        "probe_sweep": {
            f"{pos}_L{layer}": vals
            for (pos, layer), vals in probe_results.items()
        },
    }
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            return super().default(obj)

    with open(args.output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, cls=NpEncoder)

    print(f"\nResults saved to {args.output_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
