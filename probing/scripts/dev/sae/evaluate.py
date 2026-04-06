"""Evaluation for paired-view SAE.

Unsupervised metrics first, post-hoc ground-truth evaluation second.

Usage:
    uv run --project /workspace/filler-token-reasoning/probing \
        python probing/scripts/sae/evaluate.py \
        --model-dir probing/sae_results/answer_prompt_L58

    # Or called from train.py with --evaluate flag.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats as scipy_stats
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_predict

from configs import SAEConfig
from data_loader import load_paired_views, PairedViewDataset
from model import PairedViewSAE


def encode_all(model: PairedViewSAE, dataset: PairedViewDataset):
    """Encode all examples through both views. Returns numpy arrays."""
    model.eval()
    with torch.no_grad():
        z_shared_a, z_view_a = model.encode(dataset.view_a)
        z_shared_b, z_view_b = model.encode(dataset.view_b)
        x_hat_a = model.decode(z_shared_a, z_view_a)
        x_hat_b = model.decode(z_shared_b, z_view_b)

    return {
        "z_shared_a": z_shared_a.numpy(),
        "z_shared_b": z_shared_b.numpy(),
        "z_view_a": z_view_a.numpy(),
        "z_view_b": z_view_b.numpy(),
        "x_hat_a": x_hat_a.numpy(),
        "x_hat_b": x_hat_b.numpy(),
    }


# ==============================================================================
# Unsupervised metrics
# ==============================================================================

def eval_cross_view_stability(z_shared_a, z_shared_b, correct_mask):
    """Cosine similarity of z_shared across views, per example."""
    from numpy.linalg import norm
    cos = np.array([
        np.dot(a, b) / (norm(a) * norm(b) + 1e-8)
        for a, b in zip(z_shared_a, z_shared_b)
    ])
    return {
        "mean_all": float(np.mean(cos)),
        "mean_correct": float(np.mean(cos[correct_mask])),
        "std_all": float(np.std(cos)),
        "median_all": float(np.median(cos)),
    }


def eval_problem_separation(z_shared_a):
    """Pairwise cosine similarity across different problems (sample)."""
    from numpy.linalg import norm
    n = len(z_shared_a)
    # Sample 1000 pairs to keep it fast
    rng = np.random.RandomState(42)
    n_pairs = min(1000, n * (n - 1) // 2)
    i_idx = rng.randint(0, n, n_pairs)
    j_idx = rng.randint(0, n, n_pairs)
    # Exclude same-problem pairs
    mask = i_idx != j_idx
    i_idx, j_idx = i_idx[mask], j_idx[mask]

    norms = np.linalg.norm(z_shared_a, axis=1, keepdims=False)
    cos = np.array([
        np.dot(z_shared_a[i], z_shared_a[j]) / (norms[i] * norms[j] + 1e-8)
        for i, j in zip(i_idx, j_idx)
    ])
    return {
        "mean_pairwise_cos": float(np.mean(cos)),
        "std_pairwise_cos": float(np.std(cos)),
    }


def eval_sparsity(z_shared_a, z_shared_b, z_view_a, z_view_b, threshold=0.01):
    """L0 sparsity statistics."""
    z_shared = np.concatenate([z_shared_a, z_shared_b], axis=0)
    z_view = np.concatenate([z_view_a, z_view_b], axis=0)

    l0_shared = (np.abs(z_shared) > threshold).sum(axis=1)
    l0_view = (np.abs(z_view) > threshold).sum(axis=1)

    return {
        "l0_shared_mean": float(np.mean(l0_shared)),
        "l0_shared_std": float(np.std(l0_shared)),
        "l0_shared_median": float(np.median(l0_shared)),
        "l0_view_mean": float(np.mean(l0_view)),
        "l0_view_std": float(np.std(l0_view)),
        "dim_shared": z_shared.shape[1],
        "dim_view": z_view.shape[1],
    }


def eval_category_enrichment(z_shared_a, categories):
    """Per-latent mean activation by category. Tests which latents differ."""
    cats = ["filler_helped", "both_correct", "both_wrong"]
    results = {}

    for j in range(z_shared_a.shape[1]):
        vals_by_cat = {}
        for cat in cats:
            mask = categories == cat
            if mask.sum() > 0:
                vals_by_cat[cat] = z_shared_a[mask, j]

        if len(vals_by_cat) >= 2:
            groups = [v for v in vals_by_cat.values()]
            stat, pval = scipy_stats.kruskal(*groups)

            # Effect size: difference between filler_helped and both_correct
            fh = vals_by_cat.get("filler_helped", np.array([]))
            bc = vals_by_cat.get("both_correct", np.array([]))
            diff = float(np.mean(fh) - np.mean(bc)) if len(fh) > 0 and len(bc) > 0 else 0.0

            results[f"latent_{j}"] = {
                "kruskal_stat": float(stat),
                "kruskal_p": float(pval),
                "fh_minus_bc": diff,
                "means": {cat: float(np.mean(v)) for cat, v in vals_by_cat.items()},
            }

    # Find most discriminative latents
    significant = {k: v for k, v in results.items() if v["kruskal_p"] < 0.05}
    return {
        "n_significant_p05": len(significant),
        "top_latents": dict(sorted(
            results.items(), key=lambda x: x[1]["kruskal_p"]
        )[:10]),
    }


def eval_view_specificity(z_view_a, z_view_b):
    """Can z_view predict which condition the input came from?"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    X = np.concatenate([z_view_a, z_view_b], axis=0)
    y = np.array([0] * len(z_view_a) + [1] * len(z_view_b))

    clf = LogisticRegression(max_iter=1000)
    scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")

    return {
        "view_classification_accuracy": float(np.mean(scores)),
        "view_classification_std": float(np.std(scores)),
    }


# ==============================================================================
# Post-hoc evaluation (uses ground truth)
# ==============================================================================

def eval_latent_correlations(z_shared_a, ground_truth):
    """Per-latent Pearson correlation with A, Y, A+Y."""
    targets = {"A": ground_truth["A"], "Y": ground_truth["Y"], "A+Y": ground_truth["A+Y"]}
    results = {}

    for target_name, target_vals in targets.items():
        correlations = []
        for j in range(z_shared_a.shape[1]):
            r, p = scipy_stats.pearsonr(z_shared_a[:, j], target_vals)
            correlations.append({"latent": j, "r": float(r), "p": float(p)})

        correlations.sort(key=lambda x: abs(x["r"]), reverse=True)
        results[target_name] = {
            "top5": correlations[:5],
            "max_abs_r": float(max(abs(c["r"]) for c in correlations)),
        }

    return results


def eval_ridge_from_z_shared(z_shared_a, ground_truth):
    """Ridge regression from z_shared to predict A, Y, A+Y."""
    targets = {"A": ground_truth["A"], "Y": ground_truth["Y"], "A+Y": ground_truth["A+Y"]}
    results = {}

    alphas = np.logspace(-2, 6, 20)

    for target_name, y in targets.items():
        ridge = RidgeCV(alphas=alphas, cv=5)
        y_pred = cross_val_predict(ridge, z_shared_a, y, cv=5)

        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        corr, _ = scipy_stats.pearsonr(y, y_pred)

        results[target_name] = {
            "r2": float(r2),
            "mae": float(mae),
            "corr": float(corr),
        }

    return results


def eval_probe_similarity(model, probe_weights_dir=None):
    """Cosine similarity between SAE shared_head weights and supervised probe direction."""
    # Extract SAE shared_head weight: (dim_shared, encoder_hidden)
    # The effective direction from input to each latent is:
    # encoder_trunk.weight^T @ shared_head.weight^T (chain rule)
    # For a rough comparison, we use the shared_head weights directly.
    # A more precise comparison would compose encoder_trunk + shared_head.

    W_trunk = model.encoder_trunk[0].weight.detach().numpy()  # (encoder_hidden, input_dim)
    W_shared = model.shared_head.weight.detach().numpy()       # (dim_shared, encoder_hidden)
    # Effective input→shared directions: (dim_shared, input_dim)
    W_effective = W_shared @ W_trunk

    return {
        "effective_directions_shape": list(W_effective.shape),
        "effective_directions_norms": [
            float(np.linalg.norm(W_effective[j]))
            for j in range(min(10, W_effective.shape[0]))
        ],
    }


# ==============================================================================
# Visualization
# ==============================================================================

def plot_results(z_shared_a, ground_truth, output_dir: Path):
    """Generate evaluation plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    categories = ground_truth["category"]
    A = ground_truth["A"]
    AY = ground_truth["A+Y"]

    # Find top latent correlated with A
    corrs_A = [abs(scipy_stats.pearsonr(z_shared_a[:, j], A)[0])
               for j in range(z_shared_a.shape[1])]
    top_A = int(np.argmax(corrs_A))

    corrs_AY = [abs(scipy_stats.pearsonr(z_shared_a[:, j], AY)[0])
                for j in range(z_shared_a.shape[1])]
    top_AY = int(np.argmax(corrs_AY))

    cat_colors = {
        "filler_helped": "#4CAF50",
        "both_correct": "#2196F3",
        "both_wrong": "#9E9E9E",
        "filler_hurt": "#F44336",
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Top A-correlated latent vs A
    ax = axes[0]
    for cat in ["filler_helped", "both_correct", "both_wrong"]:
        mask = categories == cat
        if mask.sum() > 0:
            ax.scatter(A[mask], z_shared_a[mask, top_A], alpha=0.4, s=20,
                       color=cat_colors.get(cat, "gray"), label=cat)
    ax.set_xlabel("A (fact value)")
    ax.set_ylabel(f"z_shared[{top_A}]")
    ax.set_title(f"Top A-correlated latent (r={corrs_A[top_A]:.3f})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. Top A+Y-correlated latent vs A+Y
    ax = axes[1]
    for cat in ["filler_helped", "both_correct", "both_wrong"]:
        mask = categories == cat
        if mask.sum() > 0:
            ax.scatter(AY[mask], z_shared_a[mask, top_AY], alpha=0.4, s=20,
                       color=cat_colors.get(cat, "gray"), label=cat)
    ax.set_xlabel("A+Y (answer)")
    ax.set_ylabel(f"z_shared[{top_AY}]")
    ax.set_title(f"Top A+Y-correlated latent (r={corrs_AY[top_AY]:.3f})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Correlation heatmap (top 20 latents × 3 targets)
    ax = axes[2]
    targets = {"A": A, "Y": ground_truth["Y"], "A+Y": AY}
    n_show = min(20, z_shared_a.shape[1])
    corr_matrix = np.zeros((n_show, 3))
    for j in range(n_show):
        for ti, (tn, tv) in enumerate(targets.items()):
            corr_matrix[j, ti] = scipy_stats.pearsonr(z_shared_a[:, j], tv)[0]

    im = ax.imshow(corr_matrix.T, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_yticks(range(3))
    ax.set_yticklabels(list(targets.keys()))
    ax.set_xlabel("Latent index")
    ax.set_title("Per-latent correlation with targets")
    plt.colorbar(im, ax=ax, label="Pearson r")

    plt.tight_layout()
    plt.savefig(output_dir / "sae_evaluation.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "sae_evaluation.pdf", bbox_inches="tight")
    plt.close()
    print(f"  Plots saved to {output_dir}/")


# ==============================================================================
# Main evaluation function
# ==============================================================================

def run_evaluation(model: PairedViewSAE, dataset: PairedViewDataset, output_dir: Path):
    """Run full evaluation pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nRunning evaluation...")
    encoded = encode_all(model, dataset)
    gt = dataset.get_ground_truth()

    results = {}

    # --- Unsupervised metrics ---
    print("  Unsupervised metrics:")

    stability = eval_cross_view_stability(
        encoded["z_shared_a"], encoded["z_shared_b"], gt["correct_mask"]
    )
    results["cross_view_stability"] = stability
    print(f"    Cross-view cosine: mean={stability['mean_all']:.3f}, "
          f"correct={stability['mean_correct']:.3f}")

    separation = eval_problem_separation(encoded["z_shared_a"])
    results["problem_separation"] = separation
    print(f"    Pairwise cosine (different problems): {separation['mean_pairwise_cos']:.3f}")

    sparsity = eval_sparsity(
        encoded["z_shared_a"], encoded["z_shared_b"],
        encoded["z_view_a"], encoded["z_view_b"],
    )
    results["sparsity"] = sparsity
    print(f"    L0 shared: {sparsity['l0_shared_mean']:.1f}/{sparsity['dim_shared']} "
          f"(target: 5-15)")
    print(f"    L0 view: {sparsity['l0_view_mean']:.1f}/{sparsity['dim_view']}")

    enrichment = eval_category_enrichment(encoded["z_shared_a"], gt["category"])
    results["category_enrichment"] = enrichment
    print(f"    Latents significant at p<0.05: {enrichment['n_significant_p05']}")

    view_spec = eval_view_specificity(encoded["z_view_a"], encoded["z_view_b"])
    results["view_specificity"] = view_spec
    print(f"    z_view classifies condition: {view_spec['view_classification_accuracy']:.1%}")

    # --- Post-hoc evaluation ---
    print("  Post-hoc evaluation (ground truth):")

    correlations = eval_latent_correlations(encoded["z_shared_a"], gt)
    results["latent_correlations"] = correlations
    for target in ["A", "Y", "A+Y"]:
        top = correlations[target]["top5"][0]
        print(f"    Best latent for {target}: latent_{top['latent']} "
              f"r={top['r']:.3f} (max |r|={correlations[target]['max_abs_r']:.3f})")

    ridge = eval_ridge_from_z_shared(encoded["z_shared_a"], gt)
    results["ridge_from_z_shared"] = ridge
    for target in ["A", "Y", "A+Y"]:
        r = ridge[target]
        print(f"    Ridge z_shared→{target}: R²={r['r2']:.3f}, MAE={r['mae']:.1f}")

    probe_sim = eval_probe_similarity(model)
    results["probe_similarity"] = probe_sim

    # --- Plots ---
    plot_results(encoded["z_shared_a"], gt, output_dir)

    # Save results
    # Convert numpy types for JSON serialization
    def make_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj

    with open(output_dir / "evaluation.json", "w") as f:
        json.dump(make_serializable(results), f, indent=2)

    print(f"  Evaluation saved to {output_dir}/evaluation.json")
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate paired-view SAE")
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="Directory containing model.pt and config.json")
    args = parser.parse_args()

    # Load config
    with open(args.model_dir / "config.json") as f:
        config_dict = json.load(f)
    config = SAEConfig(**{k: v for k, v in config_dict.items()
                          if k in SAEConfig.__dataclass_fields__})

    # Load model
    checkpoint = torch.load(args.model_dir / "model.pt", map_location="cpu")
    model = PairedViewSAE(
        input_dim=config.input_dim,
        encoder_hidden=config.encoder_hidden,
        dim_shared=config.dim_shared,
        dim_view=config.dim_view,
        topk_shared=getattr(config, "topk_shared", 0),
        topk_view=getattr(config, "topk_view", 0),
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load data
    view_a, view_b, metadata = load_paired_views(
        Path(config.extraction_dir), config.conditions,
        config.position, config.layer,
        categories_file=Path(config.categories_file),
    )
    include_cats = config.include_categories if config.include_categories else None
    dataset = PairedViewDataset(
        view_a, view_b, metadata, config.conditions,
        normalize=True, stats=checkpoint.get("stats"),
        include_categories=include_cats,
    )

    run_evaluation(model, dataset, args.model_dir)


if __name__ == "__main__":
    main()
