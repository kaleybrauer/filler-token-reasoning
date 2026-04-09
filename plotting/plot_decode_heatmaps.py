"""Heatmaps of unsupervised decode accuracy by layer and condition."""

import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 28})

results_dir = Path("results/unsupervised_decode")
output_dir = results_dir


def sort_key(name):
    ftype = name.rsplit("_", 1)[0]
    length = int(name.rsplit("_", 1)[1])
    order = {"dots": 0, "alphabet": 1, "counting": 2}
    return (order.get(ftype, 99), length)


# Load all decode results, filtering to most common best_layer
all_cond_data = {}
for f in sorted(results_dir.glob("decode_*.json")):
    cond = f.stem.replace("decode_", "")
    with open(f) as fp:
        all_cond_data[cond] = json.load(fp)

layer_counts = Counter(d["best_layer"] for d in all_cond_data.values())
dominant_layer = layer_counts.most_common(1)[0][0]
cond_data = {c: d for c, d in all_cond_data.items() if d["best_layer"] == dominant_layer}
skipped = [c for c in all_cond_data if c not in cond_data]
if skipped:
    print(f"Skipping {skipped} (best_layer != L{dominant_layer})")

cond_order = sorted(cond_data.keys(), key=sort_key)
print(f"Conditions: {cond_order}")

# Get layers from first condition
layers = sorted(int(l) for l in cond_data[cond_order[0]]["per_layer"].keys())

# --- Averaged filler heatmaps (all 6 conditions) ---
for metric, cmap, label, vmin, vmax, fname in [
    ("frac_within_5", "RdYlGn", "% within ±5 of A", 0, 100, "heatmap_avg_frac5_all"),
    ("frac_exact", "RdYlGn", "% exact match to A", 0, 100, "heatmap_avg_exact_all"),
    ("median_ae", "RdYlGn_r", "Median AE", 0, 60, "heatmap_avg_medae_all"),
]:
    n_conds = len(cond_order)
    fig, axes = plt.subplots(1, n_conds, figsize=(max(6, n_conds * 1.5 + 2), 11),
                              sharey=True)
    if n_conds == 1:
        axes = [axes]

    for ax, cond in zip(axes, cond_order):
        data = cond_data[cond]["per_layer"]
        vals = []
        for layer in layers:
            v = data[str(layer)]["eval"][metric]
            if "frac" in metric:
                v *= 100
            vals.append(v)

        matrix = np.array(vals).reshape(-1, 1)
        im = ax.imshow(matrix, aspect=0.15, cmap=cmap, vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        ax.set_xticks([0])
        ax.set_xticklabels(["avg filler"], fontsize=12)
        layer_labels = [str(l) if l % 5 == 0 else "" for l in layers]
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels(layer_labels, fontsize=10)
        ax.set_title(cond, fontsize=14, fontweight="bold")
        if ax == axes[0]:
            ax.set_ylabel("Layer", fontsize=14)

    fig.suptitle(f"Unsupervised decode (avg filler): {label}", fontsize=16)
    fig.subplots_adjust(right=0.82, wspace=0.08, top=0.93)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.025, 0.65])
    fig.colorbar(im, cax=cbar_ax, label=label)

    for ext in ["png", "pdf"]:
        outpath = output_dir / f"{fname}.{ext}"
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {fname}")

# --- Per-position × layer heatmaps (3-panel: dots_10, dots_100, counting_25) ---
per_pos_dir = Path("results/unsupervised_decode_per_pos")

def pos_sort_key(p):
    if p == "question_end": return -2
    if p == "pre_filler": return -1
    if p == "answer_prompt": return 9999
    if p.startswith("filler_k"):
        return int(p.split("k")[1])
    return 0

def pos_label(p):
    if p == "pre_filler": return "filler_label"
    if p == "answer_prompt": return "ans_prompt"
    if p == "question_end": return "q_end"
    if p.startswith("filler_k"): return f"k{p.split('k')[1]}"
    return p[:8]

# Load per-position data, filter to consistent conditions
pp_data = {}
for f in sorted(per_pos_dir.glob("per_position_*.json")):
    cond = f.stem.replace("per_position_", "")
    if cond in cond_data:
        with open(f) as fp:
            pp_data[cond] = json.load(fp)

pp_order = sorted(pp_data.keys(), key=sort_key)

for actual_metric, cmap, label, vmin, vmax, fname in [
    ("frac_within_5", "RdYlGn", "% within ±5 of A", 0, 100, "heatmap_frac5_all"),
    ("median_ae", "RdYlGn_r", "Median AE", 0, 60, "heatmap_medae_all"),
]:
    conds_to_plot = pp_order
    if not conds_to_plot:
        continue

    n = len(conds_to_plot)
    nrows = 2 if n > 3 else 1
    ncols = (n + nrows - 1) // nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 10 * nrows),
                              squeeze=False)
    axes_flat = axes.flatten()

    for idx, cond in enumerate(conds_to_plot):
        ax = axes_flat[idx]
        data = pp_data[cond]
        positions = sorted([p for p in data.get("_positions", [])
                            if p != "answer_prompt"], key=pos_sort_key)
        pp_layers = data.get("_layers", [])
        if not positions or not pp_layers:
            continue

        matrix = np.full((len(pp_layers), len(positions)), np.nan)
        for j, pos in enumerate(positions):
            if pos not in data or pos.startswith("_"):
                continue
            for i, layer in enumerate(pp_layers):
                lkey = str(layer)
                if lkey in data[pos]:
                    val = data[pos][lkey].get(actual_metric, np.nan)
                    if val is not None and "frac" in actual_metric:
                        val *= 100
                    matrix[i, j] = val if val is not None else np.nan

        im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                       interpolation="nearest")

        p_labels = [pos_label(p) for p in positions]
        ax.set_xticks(range(len(positions)))
        ax.set_xticklabels(p_labels, rotation=45, ha="right")
        layer_labels = [str(l) if l % 5 == 0 else "" for l in pp_layers]
        ax.set_yticks(range(len(pp_layers)))
        ax.set_yticklabels(layer_labels)
        ax.set_title(cond, fontweight="bold")
        if idx % ncols == 0:
            ax.set_ylabel("Layer")

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.subplots_adjust(right=0.88, wspace=0.2, hspace=0.35, top=0.94)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.65])
    fig.colorbar(im, cax=cbar_ax, label=label)
    for ext in ["png", "pdf"]:
        outpath = output_dir / f"{fname}.{ext}"
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {fname}")

print("Done.")
