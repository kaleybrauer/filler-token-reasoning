"""Plot HDBSCAN clustering results from cross-example consistency with MDS embedding.

Reads results from cross_example_consistency_*.npz and produces an MDS plot
highlighting the A1 and A2 clusters (highest intra-cluster AMI among variable-dominant
clusters).

Usage:
    python plotting/plot_hdbscan_mds.py --condition dots_10
    python plotting/plot_hdbscan_mds.py --condition dots_50 --min-cluster-size 10
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.manifold import MDS

plt.rcParams.update({"font.size": 20})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", type=str, default="dots_10")
    parser.add_argument("--results-dir", type=Path,
                        default=Path("results/cross_example_consistency"))
    parser.add_argument("--min-cluster-size", type=int, default=10)
    parser.add_argument("--show-all", action="store_true",
                        help="Color all clusters distinctly instead of only A1/A2")
    parser.add_argument("--top-n", type=int, default=3,
                        help="Highlight top N clusters by intra-cluster AMI "
                             "(default: 3). Ignored if --show-all.")
    args = parser.parse_args()

    npz_file = args.results_dir / f"cross_example_consistency_{args.condition}.npz"
    data = np.load(npz_file, allow_pickle=True)
    ami_matrix = data["ami_matrix"]
    settings = data["settings"]
    predictions = data["predictions"]
    A1 = data["A1"]
    A2 = data["A2"]

    ami_clipped = np.clip(ami_matrix, 0, None)
    distance = 1 - ami_clipped
    np.fill_diagonal(distance, 0)

    clusterer = HDBSCAN(min_cluster_size=args.min_cluster_size, metric="precomputed")
    labels = clusterer.fit_predict(distance)
    n_clusters = len(set(labels) - {-1})
    n_noise = (labels == -1).sum()
    print(f"{n_clusters} clusters, {n_noise} noise")

    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42,
              n_init=4, max_iter=300)
    embedding = mds.fit_transform(distance)

    # Per-point intra-cluster AMI for sizing
    intra_ami_per_point = np.zeros(len(labels))
    for c in sorted(set(labels) - {-1}):
        members = np.where(labels == c)[0]
        for m in members:
            other_members = [o for o in members if o != m]
            if other_members:
                intra_ami_per_point[m] = ami_matrix[m, other_members].mean()
    intra_norm = intra_ami_per_point / (intra_ami_per_point.max() + 1e-8)

    # Cluster info
    cluster_info = {}
    for c in sorted(set(labels) - {-1}):
        members = np.where(labels == c)[0]
        intra = ami_matrix[np.ix_(members, members)]
        np.fill_diagonal(intra, 0)
        mean_intra = intra.sum() / (len(members) * (len(members) - 1))
        best_local = np.argmax(intra.mean(axis=1))
        rep = members[best_local]
        a1 = np.mean(predictions[rep] == A1)
        a2 = np.mean(predictions[rep] == A2)
        cluster_info[c] = {"rep": rep, "a1": a1, "a2": a2,
                            "n": len(members), "mean_ami": mean_intra}

    # Rank all clusters by intra-cluster AMI, take top N
    ranked = sorted(cluster_info.items(), key=lambda x: -x[1]["mean_ami"])
    top_n_clusters = [c for c, _ in ranked[:args.top_n]]

    def describe(info):
        a1, a2 = info["a1"], info["a2"]
        parts = []
        if a1 >= 0.10:
            parts.append(f"A₁ {a1:.0%}")
        if a2 >= 0.10:
            parts.append(f"A₂ {a2:.0%}")
        return ", ".join(parts) if parts else "—"

    # Distinct colors for each top-N cluster
    top_n_colors = ["#228B22", "#1f77b4", "#D4A03C", "#9467bd", "#e377c2", "#17becf"]

    fig, ax = plt.subplots(figsize=(11, 8))

    noise_mask = labels == -1
    ax.scatter(embedding[noise_mask, 0], embedding[noise_mask, 1],
               c="#eeeeee", s=15, alpha=0.3, edgecolor="none",
               label=f"Noise (n={noise_mask.sum()})")

    if args.show_all:
        cmap = plt.colormaps["tab10"]
        # Sort cluster IDs by mean intra-cluster AMI (descending)
        clusters_by_ami = sorted(set(labels) - {-1},
                                  key=lambda c: -cluster_info[c]["mean_ami"])
        cluster_color = {c: cmap(i % 10) for i, c in enumerate(clusters_by_ami)}
        for c in clusters_by_ami:
            mask = labels == c
            info = cluster_info[c]
            decode = describe(info)
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                       c=[cluster_color[c]], s=15 + intra_norm[mask] * 200,
                       alpha=0.6, edgecolor="none",
                       label=f"C{c} AMI={info['mean_ami']:.2f} {decode} (n={info['n']})")
        # Annotate top-N representatives
        offsets = [(12, 12), (12, -20), (-100, 12), (-100, -20)]
        for rank, c in enumerate(top_n_clusters):
            info = cluster_info[c]
            label_str = describe(info)
            rep = info["rep"]
            pos, layer = settings[rep]
            rep_size = 15 + intra_norm[rep] * 200
            ax.scatter(embedding[rep, 0], embedding[rep, 1],
                       c=[cluster_color[c]], s=rep_size, edgecolor="black",
                       linewidth=2, zorder=5)
            xytext = offsets[rank % len(offsets)]
            ax.annotate(f"{pos} L{layer}\n({label_str})",
                        (embedding[rep, 0], embedding[rep, 1]),
                        textcoords="offset points", xytext=xytext,
                        fontsize=12, fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
    else:
        # Gray out non-top-N clusters
        other_mask = np.array([l != -1 and l not in top_n_clusters for l in labels])
        if other_mask.any():
            ax.scatter(embedding[other_mask, 0], embedding[other_mask, 1],
                       c="#cccccc", s=15 + intra_norm[other_mask] * 200,
                       alpha=0.5, edgecolor="none",
                       label=f"Other clusters (n={other_mask.sum()})")

        # Highlight top N with distinct colors
        offsets = [(12, 12), (12, -20), (-100, 12), (-100, -20)]
        for rank, c in enumerate(top_n_clusters):
            info = cluster_info[c]
            label_str = describe(info)
            color = top_n_colors[rank % len(top_n_colors)]
            mask = labels == c
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                       c=color, s=15 + intra_norm[mask] * 200,
                       alpha=0.7, edgecolor="none",
                       label=f"#{rank+1} {label_str} AMI={info['mean_ami']:.2f} (n={info['n']})")
            rep = info["rep"]
            pos, layer = settings[rep]
            rep_size = 15 + intra_norm[rep] * 200
            ax.scatter(embedding[rep, 0], embedding[rep, 1],
                       c=color, s=rep_size, edgecolor="black", linewidth=2, zorder=5)
            xytext = offsets[rank % len(offsets)]
            ax.annotate(f"{pos} L{layer}\n({label_str})",
                        (embedding[rep, 0], embedding[rep, 1]),
                        textcoords="offset points", xytext=xytext,
                        fontsize=12, fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

    ax.set_xlabel("MDS Dimension 1")
    ax.set_ylabel("MDS Dimension 2")
    ax.legend(fontsize=10 if args.show_all else 12,
              loc="lower right", framealpha=0.9,
              ncol=2 if args.show_all else 1)

    plt.tight_layout()
    suffix = "_all" if args.show_all else ""
    for ext in ["png", "pdf"]:
        outpath = args.results_dir / f"consistency_hdbscan_{args.condition}{suffix}.{ext}"
        fig.savefig(outpath, dpi=200, bbox_inches="tight")
        print(f"Saved {outpath}")
    plt.close()


if __name__ == "__main__":
    main()
