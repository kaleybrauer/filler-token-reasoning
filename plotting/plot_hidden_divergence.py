"""
plot_hidden_divergence.py

Plot cosine similarity and relative L2 distance between baseline and dots_50
hidden states at pre_answer, layer by layer. Shows where the filler effect
hits the representation.

Usage:
    uv run --project /workspace/filler-token-reasoning/probing \
        python probing/scripts/plot_hidden_divergence.py \
        --output-dir probing/plots
"""

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extraction-dir", type=Path,
                        default=Path("probing/extracted_states"))
    parser.add_argument("--categories-file", type=Path,
                        default=Path("probing/logit_lens_results/all_dots50/per_example.json"))
    parser.add_argument("--position", default="pre_answer")
    parser.add_argument("--output-dir", type=Path, default=Path("probing/plots"))
    parser.add_argument("--no-l2", action="store_true", help="Only plot cosine similarity, skip L2")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    bl_dir = args.extraction_dir / "baseline"
    d50_dir = args.extraction_dir / "dots_50"

    with open(args.categories_file) as f:
        cats = {e["problem_idx"]: e["category"] for e in json.load(f)}

    pos = args.position
    layers = list(range(61))

    by_cat = defaultdict(lambda: {"cosine": [[] for _ in range(61)], "l2": [[] for _ in range(61)]})

    n_loaded = 0
    for i in range(500):
        bl_path = bl_dir / f"prob_{i:04d}.pkl"
        d50_path = d50_dir / f"prob_{i:04d}.pkl"
        if not bl_path.exists() or not d50_path.exists():
            continue
        bl = pickle.load(open(bl_path, "rb"))
        d50 = pickle.load(open(d50_path, "rb"))
        if pos not in bl["states"] or pos not in d50["states"]:
            continue

        cat = cats.get(i, "unknown")
        n_loaded += 1

        for l in layers:
            v_bl = bl["states"][pos][l].astype(np.float32)
            v_d50 = d50["states"][pos][l].astype(np.float32)
            cos = np.dot(v_bl, v_d50) / (np.linalg.norm(v_bl) * np.linalg.norm(v_d50) + 1e-8)
            avg_norm = (np.linalg.norm(v_bl) + np.linalg.norm(v_d50)) / 2
            by_cat[cat]["cosine"][l].append(cos)
            by_cat[cat]["l2"][l].append(np.linalg.norm(v_d50 - v_bl) / avg_norm)
            by_cat["all"]["cosine"][l].append(cos)
            by_cat["all"]["l2"][l].append(np.linalg.norm(v_d50 - v_bl) / avg_norm)

    print(f"Loaded {n_loaded} examples")

    cat_style = {
        "all":           {"color": "black",   "ls": "-",  "lw": 2.5, "label": "All (n=500)"},
        "filler_helped": {"color": "#4CAF50", "ls": "-",  "lw": 2,   "label": "filler_helped (n=106)"},
        "both_correct":  {"color": "#2196F3", "ls": "--", "lw": 1.5, "label": "both_correct (n=270)"},
        "both_wrong":    {"color": "#F44336", "ls": "--", "lw": 1.5, "label": "both_wrong (n=108)"},
    }

    if args.no_l2:
        fig, ax1 = plt.subplots(figsize=(12, 4.5))
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Top (or only): cosine similarity
    for cat in ["all", "filler_helped", "both_correct", "both_wrong"]:
        s = cat_style[cat]
        medians = [np.median(by_cat[cat]["cosine"][l]) for l in layers]
        ax1.plot(layers, medians, color=s["color"], ls=s["ls"], lw=s["lw"], label=s["label"])
        if cat == "filler_helped":
            q25 = [np.percentile(by_cat[cat]["cosine"][l], 25) for l in layers]
            q75 = [np.percentile(by_cat[cat]["cosine"][l], 75) for l in layers]
            ax1.fill_between(layers, q25, q75, color=s["color"], alpha=0.1)

    ax1.set_ylabel("Cosine similarity", fontsize=12)
    ax1.set_title(f"Hidden state divergence: baseline vs dots_50 at {pos}", fontsize=13)
    ax1.legend(fontsize=9, loc="lower left")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.75, 1.01)
    ax1.set_xlim(0, 60)
    ax1.axhline(1.0, color="gray", ls=":", alpha=0.3)
    if args.no_l2:
        ax1.set_xlabel("Layer", fontsize=12)

    # Bottom: relative L2
    if not args.no_l2:
        for cat in ["all", "filler_helped", "both_correct", "both_wrong"]:
            s = cat_style[cat]
            medians = [np.median(by_cat[cat]["l2"][l]) for l in layers]
            ax2.plot(layers, medians, color=s["color"], ls=s["ls"], lw=s["lw"], label=s["label"])
            if cat == "filler_helped":
                q25 = [np.percentile(by_cat[cat]["l2"][l], 25) for l in layers]
                q75 = [np.percentile(by_cat[cat]["l2"][l], 75) for l in layers]
                ax2.fill_between(layers, q25, q75, color=s["color"], alpha=0.1)

        ax2.set_xlabel("Layer", fontsize=12)
        ax2.set_ylabel("Relative L2 distance", fontsize=12)
        ax2.legend(fontsize=9, loc="upper left")
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 60)

    plt.tight_layout()
    plt.savefig(args.output_dir / "hidden_state_divergence.png", dpi=200, bbox_inches="tight")
    plt.savefig(args.output_dir / "hidden_state_divergence.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved to {args.output_dir}/hidden_state_divergence.png")


if __name__ == "__main__":
    main()
