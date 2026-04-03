"""
variance_ratio_layers.py

For each layer, compute the ratio of between-example variance to
within-example (across filler positions) variance. The layer where
between-example variance dominates is where the model encodes
problem-specific information — a fully unsupervised layer selection.

Then run PCA at the selected layer on averaged-filler representations,
and correlate top PCs with A, Y, A+Y post-hoc.

Usage:
    uv run --project /workspace/filler-token-reasoning/probing \
        python probing/scripts/variance_ratio_layers.py \
        --condition dots_100 \
        --output-dir probing/results/variance_ratio
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extraction-dir", type=Path, default=Path("probing/extracted_states"))
    parser.add_argument("--condition", type=str, default="dots_100")
    parser.add_argument("--output-dir", type=Path, default=Path("probing/results/variance_ratio"))
    parser.add_argument("--n-components", type=int, default=20)
    parser.add_argument("--filter-categories", nargs="+",
                        default=["both_correct", "filler_helped"])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    cond_dir = args.extraction_dir / args.condition
    files = sorted(cond_dir.glob("prob_*.pkl"))

    # Load baseline for categories
    bl_dir = args.extraction_dir / "baseline"
    bl_files = sorted(bl_dir.glob("prob_*.pkl"))

    categories = {}
    for bf, cf in zip(bl_files, files):
        with open(bf, "rb") as f:
            bd = pickle.load(f)
        with open(cf, "rb") as f:
            cd = pickle.load(f)
        idx = bd["problem_idx"]
        bc, fc = bd.get("model_correct", False), cd.get("model_correct", False)
        if bc and fc: categories[idx] = "both_correct"
        elif not bc and fc: categories[idx] = "filler_helped"
        elif bc and not fc: categories[idx] = "filler_hurt"
        else: categories[idx] = "both_wrong"

    # Load all data
    all_data = []
    metadata = []
    for f in files:
        with open(f, "rb") as fp:
            d = pickle.load(fp)
        all_data.append(d)
        metadata.append({
            "problem_idx": d["problem_idx"],
            "fact_value": d["fact_value"],
            "x": d["x"],
            "answer": d["answer"],
        })

    positions = list(all_data[0]["states"].keys())
    layers = sorted(all_data[0]["states"][positions[0]].keys())
    filler_positions = [p for p in positions if p.startswith("filler_k") or p == "pre_filler"]

    # Filter
    keep = [i for i, m in enumerate(metadata)
            if categories.get(m["problem_idx"]) in args.filter_categories]
    all_data = [all_data[i] for i in keep]
    metadata = [metadata[i] for i in keep]
    n = len(metadata)
    print(f"Examples: {n} ({args.filter_categories})")
    print(f"Filler positions: {filler_positions}")
    print(f"Layers: {len(layers)}")

    A = np.array([m["fact_value"] for m in metadata], dtype=np.float32)
    Y = np.array([m["x"] for m in metadata], dtype=np.float32)
    AY = np.array([m["answer"] for m in metadata], dtype=np.float32)

    n_filler = len(filler_positions)

    # For each layer, compute between-example and within-example variance
    # Shape per layer: (n_examples, n_filler_positions, 7168)
    # Between-example variance: variance of the mean-across-positions representation
    # Within-example variance: mean variance across positions within each example

    print(f"\n{'='*70}")
    print(f"  Variance ratio by layer")
    print(f"{'='*70}")
    print(f"  {'Layer':>6} {'Between':>10} {'Within':>10} {'Ratio':>8} {'Top PC1 |r| with A':>20}")
    print(f"  {'-'*58}")

    layer_ratios = {}
    layer_avg_data = {}

    for layer in tqdm(layers, desc="Computing variance ratios"):
        # Build (n_examples, n_filler, dim) tensor
        example_vecs = []
        for d in all_data:
            vecs = []
            for p in filler_positions:
                if p in d["states"] and layer in d["states"][p]:
                    vecs.append(d["states"][p][layer].astype(np.float32))
            if len(vecs) == n_filler:
                example_vecs.append(np.stack(vecs))

        if len(example_vecs) != n:
            continue

        X = np.stack(example_vecs)  # (n, n_filler, dim)

        # Mean across filler positions per example
        X_mean = X.mean(axis=1)  # (n, dim)
        layer_avg_data[layer] = X_mean

        # Between-example variance: variance of X_mean across examples
        between_var = np.var(X_mean, axis=0).mean()

        # Within-example variance: for each example, variance across filler positions
        # Then average across examples
        within_var = np.mean([np.var(X[i], axis=0).mean() for i in range(n)])

        ratio = between_var / (within_var + 1e-10)
        layer_ratios[layer] = {
            "between": float(between_var),
            "within": float(within_var),
            "ratio": float(ratio),
        }

        # Quick PCA on averaged representation to get PC1 correlation with A
        pca = PCA(n_components=min(5, n - 1))
        Z = pca.fit_transform(X_mean)
        r_a = abs(stats.pearsonr(Z[:, 0], A)[0])

        layer_ratios[layer]["pc1_r_A"] = float(r_a)

        if layer % 5 == 0 or ratio > 1.5:
            print(f"  L{layer:>4} {between_var:>10.2f} {within_var:>10.2f} {ratio:>8.2f} {r_a:>20.3f}")

    # Find best layer by ratio
    best_layer = max(layer_ratios.keys(), key=lambda l: layer_ratios[l]["ratio"])
    print(f"\n  Best layer by variance ratio: L{best_layer} "
          f"(ratio={layer_ratios[best_layer]['ratio']:.2f})")

    # Also find best by PC1 correlation with A
    best_pc1_layer = max(layer_ratios.keys(), key=lambda l: layer_ratios[l]["pc1_r_A"])
    print(f"  Best layer by PC1 |r| with A: L{best_pc1_layer} "
          f"(|r|={layer_ratios[best_pc1_layer]['pc1_r_A']:.3f})")

    # Full PCA at best ratio layer
    print(f"\n{'='*70}")
    print(f"  PCA at L{best_layer} (best variance ratio) on averaged filler")
    print(f"{'='*70}")

    X_best = layer_avg_data[best_layer]
    pca = PCA(n_components=min(args.n_components, n - 1))
    Z = pca.fit_transform(X_best)

    print(f"  Explained variance (top 10): {pca.explained_variance_ratio_[:10].round(3).tolist()}")
    print(f"  Cumulative: {np.cumsum(pca.explained_variance_ratio_[:10]).round(3).tolist()}")

    print(f"\n  {'PC':>4} {'Var%':>6} {'r(A)':>7} {'r(Y)':>7} {'r(A+Y)':>8}")
    print(f"  {'-'*36}")
    for pc in range(min(args.n_components, Z.shape[1])):
        var_pct = pca.explained_variance_ratio_[pc] * 100
        r_a = stats.pearsonr(Z[:, pc], A)[0]
        r_y = stats.pearsonr(Z[:, pc], Y)[0]
        r_ay = stats.pearsonr(Z[:, pc], AY)[0]
        print(f"  PC{pc:>2} {var_pct:>5.1f}% {r_a:>7.3f} {r_y:>7.3f} {r_ay:>8.3f}")

    # Also do PCA at best PC1 layer if different
    if best_pc1_layer != best_layer:
        print(f"\n{'='*70}")
        print(f"  PCA at L{best_pc1_layer} (best PC1 |r| with A) on averaged filler")
        print(f"{'='*70}")

        X_pc1 = layer_avg_data[best_pc1_layer]
        pca2 = PCA(n_components=min(args.n_components, n - 1))
        Z2 = pca2.fit_transform(X_pc1)

        print(f"\n  {'PC':>4} {'Var%':>6} {'r(A)':>7} {'r(Y)':>7} {'r(A+Y)':>8}")
        print(f"  {'-'*36}")
        for pc in range(min(args.n_components, Z2.shape[1])):
            var_pct = pca2.explained_variance_ratio_[pc] * 100
            r_a = stats.pearsonr(Z2[:, pc], A)[0]
            r_y = stats.pearsonr(Z2[:, pc], Y)[0]
            r_ay = stats.pearsonr(Z2[:, pc], AY)[0]
            print(f"  PC{pc:>2} {var_pct:>5.1f}% {r_a:>7.3f} {r_y:>7.3f} {r_ay:>8.3f}")

    # Plot variance ratio across layers
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        all_layers = sorted(layer_ratios.keys())
        ratios = [layer_ratios[l]["ratio"] for l in all_layers]
        pc1_rs = [layer_ratios[l]["pc1_r_A"] for l in all_layers]

        ax = axes[0]
        ax.plot(all_layers, ratios, "o-", color="#2E7D32", markersize=4, linewidth=1.5)
        ax.axvline(x=best_layer, color="red", ls="--", alpha=0.5, label=f"Best: L{best_layer}")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Between / Within variance ratio")
        ax.set_title("Variance ratio (higher = more problem-specific)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(all_layers, pc1_rs, "o-", color="#1565C0", markersize=4, linewidth=1.5)
        ax.axvline(x=best_pc1_layer, color="red", ls="--", alpha=0.5, label=f"Best: L{best_pc1_layer}")
        ax.set_xlabel("Layer")
        ax.set_ylabel("|r| between PC1 and A")
        ax.set_title("PC1 correlation with A (unsupervised)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(args.output_dir / "variance_ratio.png", dpi=150, bbox_inches="tight")
        plt.savefig(args.output_dir / "variance_ratio.pdf", bbox_inches="tight")
        plt.close()
        print(f"\nPlot saved to {args.output_dir}/")
    except ImportError:
        pass

    # Save
    with open(args.output_dir / "variance_ratio_results.json", "w") as f:
        json.dump({
            "best_ratio_layer": int(best_layer),
            "best_pc1_layer": int(best_pc1_layer),
            "layer_ratios": {str(k): v for k, v in layer_ratios.items()},
        }, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
