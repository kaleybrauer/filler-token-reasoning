"""Plot unsupervised variable discovery results.

Shows consistency × diversity scatter with:
- Threshold lines marking the "interesting region"
- Settings inside the region colored by discovered variable cluster
- Settings outside grayed out
- Post-hoc A1/A2/Σ labels on each discovered variable

Usage:
    python plotting/plot_variable_discovery.py --condition dots_10
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

try:
    from adjustText import adjust_text
    HAS_ADJUSTTEXT = True
except ImportError:
    HAS_ADJUSTTEXT = False


def connected_components(n, edges):
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    for a, b in edges:
        union(a, b)
    roots = {}
    for i in range(n):
        r = find(i)
        roots.setdefault(r, []).append(i)
    return list(roots.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", type=str, default="dots_10")
    parser.add_argument("--npz", type=Path, default=None)
    parser.add_argument("--tol", type=int, default=5)
    parser.add_argument("--min-diversity", type=float, default=0.25)
    parser.add_argument("--min-consistency", type=float, default=0.85)
    parser.add_argument("--agreement-threshold", type=float, default=0.40)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--include-post-filler", action="store_true",
                        help="Include positions after filler_end (e.g., answer_prompt)")
    parser.add_argument("--extraction-dir", type=Path,
                        default=Path("data/extracted_states_2fact_allpos"))
    args = parser.parse_args()

    if args.npz is None:
        args.npz = Path(f"results/cross_example_consistency/cross_example_consistency_{args.condition}.npz")

    data = np.load(args.npz, allow_pickle=True)
    settings = data["settings"]
    predictions = data["predictions"]
    A1 = data["A1"]
    A2 = data["A2"]
    A1A2 = data["A1A2"]

    # Get filler_end_offset to optionally exclude post-filler positions
    filler_end = None
    if not args.include_post_filler:
        import pickle
        ex_file = sorted((args.extraction_dir / args.condition).glob("prob_*.pkl"))[0]
        with open(ex_file, "rb") as f:
            d = pickle.load(f)
        filler_end = d["boundaries"]["filler_end_offset"]
        print(f"Filtering to filler region (pos ≤ {filler_end})")

    pos_layer_idx = {}
    for i, s in enumerate(settings):
        pos, layer = str(s[0]), int(s[1])
        if filler_end is not None and pos.startswith("pos_"):
            pos_num = int(pos.split("_")[1])
            if pos_num > filler_end:
                continue
        pos_layer_idx.setdefault(pos, {})[layer] = i

    # Per-position best layer
    best_per_pos = {}
    for pos in pos_layer_idx:
        layers = sorted(pos_layer_idx[pos].keys())
        scores = {}
        for layer in layers:
            idx = pos_layer_idx[pos][layer]
            preds = predictions[idx]
            agrees = []
            for dl in [-1, 1]:
                if layer + dl in pos_layer_idx[pos]:
                    npreds = predictions[pos_layer_idx[pos][layer + dl]]
                    agrees.append(np.mean(np.abs(preds - npreds) <= args.tol))
            if agrees:
                scores[layer] = np.mean(agrees)
        if not scores:
            continue
        best_layer = max(scores, key=scores.get)
        best_idx = pos_layer_idx[pos][best_layer]
        preds = predictions[best_idx]
        diversity = len(np.unique(preds)) / len(preds)
        best_per_pos[pos] = {
            "layer": best_layer,
            "consistency": scores[best_layer],
            "diversity": diversity,
            "idx": best_idx,
        }

    # Interesting region
    candidates = [(pos, info) for pos, info in best_per_pos.items()
                  if info["consistency"] >= args.min_consistency
                  and info["diversity"] >= args.min_diversity]
    candidate_set = {pos for pos, _ in candidates}

    # Pairwise agreement → cluster
    n = len(candidates)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            p1 = predictions[candidates[i][1]["idx"]]
            p2 = predictions[candidates[j][1]["idx"]]
            if np.mean(p1 == p2) >= args.agreement_threshold:
                edges.append((i, j))
    components = connected_components(n, edges)

    # Assign colors per cluster
    cmap_colors = ["#228B22", "#1f77b4", "#d62728", "#D4A03C", "#9467bd",
                   "#17becf", "#e377c2", "#8c564b", "#bcbd22", "#7f7f7f"]
    pos_to_cluster = {}
    for c_idx, members in enumerate(components):
        for m in members:
            pos_to_cluster[candidates[m][0]] = c_idx

    plt.rcParams.update({"font.size": 20})
    fig, ax = plt.subplots(figsize=(12, 8.5))

    # Plot threshold lines
    ax.axvline(args.min_consistency, color="gray", linestyle=":", alpha=0.5)
    ax.axhline(args.min_diversity, color="gray", linestyle=":", alpha=0.5)

    texts = []
    # Plot all points
    for pos, info in best_per_pos.items():
        c = info["consistency"]
        d = info["diversity"]
        pos_num = int(pos.split("_")[1])
        is_candidate = pos in candidate_set

        if is_candidate:
            cluster = pos_to_cluster[pos]
            color = cmap_colors[cluster % len(cmap_colors)]
            alpha = 0.85
            size = 700
            edge = "black"
            # Compute post-hoc for annotation
            preds = predictions[info["idx"]]
            a1 = np.mean(preds == A1)
            a2 = np.mean(preds == A2)
            sum_v = np.mean(preds == A1A2)
            parts = []
            if a1 >= 0.10: parts.append(f"A₁={a1:.0%}")
            if a2 >= 0.10: parts.append(f"A₂={a2:.0%}")
            if sum_v >= 0.10: parts.append(f"Σ={sum_v:.0%}")
            post_hoc = ", ".join(parts) if parts else ""
            text_str = f"pos {pos_num}"
            if post_hoc:
                text_str += f"\n{post_hoc}"
            t = ax.text(c, d, text_str, fontsize=14, fontweight="bold",
                        color=color)
            texts.append(t)
        else:
            color = "#cccccc"
            alpha = 0.35
            size = 250
            edge = "none"
            ax.annotate(f"{pos_num}", (c, d),
                        textcoords="offset points", xytext=(6, 6),
                        fontsize=10, color="#888888")

        ax.scatter(c, d, c=color, s=size, alpha=alpha,
                   edgecolor=edge, linewidth=1.5, zorder=3 if is_candidate else 2)

    # Expand limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_pad = (xlim[1] - xlim[0]) * 0.08
    y_pad = (ylim[1] - ylim[0]) * 0.10
    ax.set_xlim(xlim[0] - x_pad, xlim[1] + x_pad)
    ax.set_ylim(ylim[0] - y_pad, ylim[1] + y_pad)

    if HAS_ADJUSTTEXT and texts:
        adjust_text(texts, ax=ax,
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.5, alpha=0.5),
                    expand_points=(1.4, 1.8))

    # Legend: one entry per discovered variable
    legend_elements = []
    for c_idx, members in enumerate(components):
        # Representative: highest consistency
        rep_i = max(members, key=lambda i: candidates[i][1]["consistency"])
        rep_pos = candidates[rep_i][0]
        member_str = ", ".join(sorted([candidates[m][0].replace("pos_", "") for m in members]))
        color = cmap_colors[c_idx % len(cmap_colors)]
        legend_elements.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
                   markersize=22, label=f"Var {c_idx+1}: pos {{{member_str}}}")
        )
    legend_elements.append(
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#cccccc",
               markersize=16, label="Outside interesting region")
    )
    ax.legend(handles=legend_elements, fontsize=16, loc="lower right", framealpha=0.9)

    ax.set_xlabel("Neighbor-layer consistency (best layer)", fontsize=22)
    ax.set_ylabel("Prediction diversity (unique / total)", fontsize=22)
    ax.set_title(f"{args.condition}: unsupervised variable discovery", fontsize=24)
    ax.tick_params(labelsize=16)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        outpath = args.output_dir / f"variable_discovery_{args.condition}.{ext}"
        fig.savefig(outpath, dpi=200, bbox_inches="tight")
        print(f"Saved {outpath}")
    plt.close()

    print(f"\n{len(components)} discovered variables:")
    for c_idx, members in enumerate(components):
        names = [candidates[m][0] for m in members]
        print(f"  Variable {c_idx + 1}: {names}")


if __name__ == "__main__":
    main()
