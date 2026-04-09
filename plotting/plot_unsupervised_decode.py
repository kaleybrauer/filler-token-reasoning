"""Plot unsupervised decode results: per-layer line plots and per-position heatmaps."""

import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

output_dir = Path("results/unsupervised_decode_plots")
output_dir.mkdir(parents=True, exist_ok=True)

# --- 1. Per-layer averaged filler line plots ---
# Load v2 results (0-999 range)
avg_dir = Path("results/unsupervised_decode_v2")
conditions_avg = {}
for f in sorted(avg_dir.glob("unsupervised_decode_*.json")):
    cond = f.stem.replace("unsupervised_decode_", "")
    with open(f) as fp:
        data = json.load(fp)
    layers = []
    frac5 = []
    medae = []
    for k, v in data.items():
        if k.startswith("_") or k in ("best_layer", "method", "condition", "n_examples"):
            continue
        try:
            layer = int(k)
        except ValueError:
            continue
        layers.append(layer)
        frac5.append(v["frac_within_5"] * 100)
        medae.append(v["median_ae"])
    order = np.argsort(layers)
    conditions_avg[cond] = {
        "layers": np.array(layers)[order],
        "frac5": np.array(frac5)[order],
        "medae": np.array(medae)[order],
    }

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

ax = axes[0]
for cond, d in conditions_avg.items():
    ax.plot(d["layers"], d["frac5"], ".-", markersize=5, label=cond)
ax.set_ylabel("% within ±5")
ax.set_title("Unsupervised decode — avg filler → RMSNorm → lm_head")
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.set_ylim(0, 100)

ax = axes[1]
for cond, d in conditions_avg.items():
    ax.plot(d["layers"], np.clip(d["medae"], 0, 80), ".-", markersize=5, label=cond)
ax.set_ylabel("Median AE")
ax.set_xlabel("Layer")
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.set_ylim(0, 80)

plt.tight_layout()
plt.savefig(output_dir / "avg_filler_by_layer.png", dpi=150, bbox_inches="tight")
plt.savefig(output_dir / "avg_filler_by_layer.pdf", bbox_inches="tight")
plt.close()
print(f"Saved avg_filler_by_layer.png")


# --- 2. Per-position heatmaps (combined 3-panel) ---
per_pos_dir = Path("results/unsupervised_decode_per_pos")

def pos_sort_key(p):
    if p == "pre_filler": return -1
    if p == "answer_prompt": return 9999
    if p == "question_end": return -2
    if p.startswith("filler_k"):
        return int(p.split("k")[1])
    return 0

def pos_label(p):
    if p == "pre_filler": return "filler_label"
    if p == "answer_prompt": return "ans_prompt"
    if p == "question_end": return "q_end"
    if p.startswith("filler_k"): return f"k{p.split('k')[1]}"
    return p[:8]

# Load all conditions
cond_data = {}
cond_order = ["dots_100", "dots_10", "counting_25"]
for f in sorted(per_pos_dir.glob("per_position_*.json")):
    cond = f.stem.replace("per_position_", "")
    with open(f) as fp:
        data = json.load(fp)
    cond_data[cond] = data

for metric, cmap, label, vmin, vmax in [
    ("frac_within_5", "RdYlGn", "% within ±5", 0, 100),
    ("median_ae", "RdYlGn_r", "Median AE", 0, 60),
]:
    metric_short = "frac5" if "frac" in metric else "medae"
    conds_to_plot = [c for c in cond_order if c in cond_data]
    fig, axes = plt.subplots(1, len(conds_to_plot), figsize=(5 * len(conds_to_plot) + 2, 10))
    if len(conds_to_plot) == 1:
        axes = [axes]

    for ax, cond in zip(axes, conds_to_plot):
        data = cond_data[cond]
        positions = sorted(data.get("_positions", []), key=pos_sort_key)
        layers = data.get("_layers", [])
        if not positions or not layers:
            continue

        matrix = np.full((len(layers), len(positions)), np.nan)
        for j, pos in enumerate(positions):
            if pos not in data or pos.startswith("_"):
                continue
            for i, layer in enumerate(layers):
                lkey = str(layer)
                if lkey in data[pos]:
                    val = data[pos][lkey][metric]
                    if "frac" in metric:
                        val *= 100
                    matrix[i, j] = val

        im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                       interpolation="nearest")

        p_labels = [pos_label(p) for p in positions]
        ax.set_xticks(range(len(positions)))
        ax.set_xticklabels(p_labels, rotation=45, ha="right", fontsize=10)
        layer_labels = [str(l) if l % 5 == 0 else "" for l in layers]
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels(layer_labels, fontsize=7)
        ax.set_title(cond, fontsize=12)
        if ax == axes[0]:
            ax.set_ylabel("Layer")

    fig.suptitle(f"Unsupervised decode: {label}", fontsize=14)
    fig.subplots_adjust(right=0.88, wspace=0.15, top=0.93)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.65])
    fig.colorbar(im, cax=cbar_ax, label=label)
    plt.savefig(output_dir / f"heatmap_{metric_short}_all.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / f"heatmap_{metric_short}_all.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved heatmap_{metric_short}_all.png")

# --- 3. Averaged filler heatmaps (1 column per condition) ---
avg_dir2 = Path("results/unsupervised_decode_v2")

avg_cond_data = {}
for f in sorted(avg_dir2.glob("unsupervised_decode_*.json")):
    cond = f.stem.replace("unsupervised_decode_", "")
    with open(f) as fp:
        avg_cond_data[cond] = json.load(fp)

for metric, cmap, label, vmin, vmax in [
    ("frac_within_5", "RdYlGn", "% within ±5", 0, 100),
    ("median_ae", "RdYlGn_r", "Median AE", 0, 60),
]:
    metric_short = "frac5" if "frac" in metric else "medae"
    conds_to_plot = [c for c in cond_order if c in avg_cond_data]

    fig, axes = plt.subplots(1, len(conds_to_plot), figsize=(5, 10),
                              sharey=True)
    if len(conds_to_plot) == 1:
        axes = [axes]

    for ax, cond in zip(axes, conds_to_plot):
        data = avg_cond_data[cond]
        layers = []
        vals = []
        for k, v in data.items():
            if k.startswith("_") or k in ("best_layer", "method", "condition", "n_examples"):
                continue
            try:
                layer = int(k)
            except ValueError:
                continue
            layers.append(layer)
            val = v[metric]
            if "frac" in metric:
                val *= 100
            vals.append(val)

        order = np.argsort(layers)
        layers = np.array(layers)[order]
        vals = np.array(vals)[order]

        # Make a 2D column for heatmap display
        matrix = vals.reshape(-1, 1)
        im = ax.imshow(matrix, aspect=0.15, cmap=cmap, vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        ax.set_xticks([0])
        ax.set_xticklabels(["avg_filler"], fontsize=10)
        layer_labels = [str(l) if l % 5 == 0 else "" for l in layers]
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels(layer_labels, fontsize=7)
        ax.set_title(cond, fontsize=12)
        if ax == axes[0]:
            ax.set_ylabel("Layer")

    fig.suptitle(f"Unsupervised decode (avg filler): {label}", fontsize=14)
    fig.subplots_adjust(right=0.78, wspace=0.08, top=0.93)
    cbar_ax = fig.add_axes([0.82, 0.15, 0.03, 0.65])
    fig.colorbar(im, cax=cbar_ax, label=label)

    plt.savefig(output_dir / f"heatmap_avg_{metric_short}_all.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / f"heatmap_avg_{metric_short}_all.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved heatmap_avg_{metric_short}_all.png")

# --- 4. Averaged filler: MedAE/MAE + tolerance curves ---
# Need frac_within_0 (exactly correct) — reload and compute from predictions
# First check if predictions are saved in results
avg_dir2 = Path("results/unsupervised_decode_v2")

# Load all conditions
avg_line_data = {}
for f in sorted(avg_dir2.glob("unsupervised_decode_*.json")):
    cond = f.stem.replace("unsupervised_decode_", "")
    with open(f) as fp:
        data = json.load(fp)
    layers = []
    mae_vals = []
    medae_vals = []
    frac0_vals = []
    frac5_vals = []
    frac10_vals = []
    for k, v in data.items():
        if k.startswith("_") or k in ("best_layer", "method", "condition", "n_examples"):
            continue
        try:
            layer = int(k)
        except ValueError:
            continue
        layers.append(layer)
        mae_vals.append(v["mae"])
        medae_vals.append(v["median_ae"])
        frac5_vals.append(v["frac_within_5"] * 100)
        frac10_vals.append(v["frac_within_10"] * 100)
        # Exact match: median_ae == 0 means >50% exact, but we need the actual fraction
        # Check if frac_within_0 exists, otherwise approximate from median
        if "frac_within_0" in v:
            frac0_vals.append(v["frac_within_0"] * 100)
        else:
            # Not available — will need to recompute or skip
            frac0_vals.append(None)

    order = np.argsort(layers)
    avg_line_data[cond] = {
        "layers": np.array(layers)[order],
        "mae": np.array(mae_vals)[order],
        "medae": np.array(medae_vals)[order],
        "frac5": np.array(frac5_vals)[order],
        "frac10": np.array(frac10_vals)[order],
        "frac0": [frac0_vals[i] for i in order],
    }

# Check if we have frac0
has_frac0 = all(v is not None for cond in avg_line_data for v in avg_line_data[cond]["frac0"])

cond_colors = {"dots_100": "tab:green", "dots_10": "tab:orange", "counting_25": "tab:blue"}
cond_order_plot = [c for c in cond_order if c in avg_line_data]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# Left: MAE and MedAE
for cond in cond_order_plot:
    d = avg_line_data[cond]
    color = cond_colors.get(cond, None)
    ax1.plot(d["layers"], np.clip(d["mae"], 0, 80), "-", color=color, linewidth=1.5,
             label=f"{cond} MAE", alpha=0.7)
    ax1.plot(d["layers"], np.clip(d["medae"], 0, 80), "--", color=color, linewidth=2,
             label=f"{cond} MedAE")

ax1.axhline(y=34.5, color="gray", linestyle="-", alpha=0.5, linewidth=1)
ax1.axhline(y=30.0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
ax1.text(1, 35.5, "shuffle MAE", fontsize=8, color="gray")
ax1.text(1, 27, "shuffle MedAE", fontsize=8, color="gray")
ax1.set_xlabel("Layer", fontsize=12)
ax1.set_ylabel("Error", fontsize=12)
ax1.set_title("MAE and Median AE", fontsize=13)
ax1.legend(fontsize=9, ncol=2)
ax1.grid(alpha=0.3)
ax1.set_ylim(0, 80)

# Right: tolerance curves
for cond in cond_order_plot:
    d = avg_line_data[cond]
    color = cond_colors.get(cond, None)
    if has_frac0:
        frac0 = np.array(d["frac0"], dtype=float)
        ax2.plot(d["layers"], frac0, ":", color=color, linewidth=1.5,
                 label=f"{cond} exact", alpha=0.7)
    ax2.plot(d["layers"], d["frac5"], "--", color=color, linewidth=2,
             label=f"{cond} ±5")
    ax2.plot(d["layers"], d["frac10"], "-", color=color, linewidth=1.5,
             label=f"{cond} ±10", alpha=0.7)

ax2.axhline(y=1.4, color="gray", linestyle=":", alpha=0.5, linewidth=1)
ax2.axhline(y=10.5, color="gray", linestyle="--", alpha=0.5, linewidth=1)
ax2.axhline(y=19.7, color="gray", linestyle="-", alpha=0.5, linewidth=1)
ax2.text(1, 2.5, "shuffle exact", fontsize=8, color="gray")
ax2.text(1, 11.5, "shuffle ±5", fontsize=8, color="gray")
ax2.text(1, 20.7, "shuffle ±10", fontsize=8, color="gray")
ax2.set_xlabel("Layer", fontsize=12)
ax2.set_ylabel("% of examples", fontsize=12)
ax2.set_title("Fraction within tolerance", fontsize=13)
ax2.legend(fontsize=8, ncol=3, loc="upper left")
ax2.grid(alpha=0.3)
ax2.set_ylim(0, 100)

fig.suptitle("Unsupervised decode — avg filler → RMSNorm → lm_head", fontsize=14)
plt.tight_layout()
plt.savefig(output_dir / "avg_filler_error_and_tolerance.png", dpi=150, bbox_inches="tight")
plt.savefig(output_dir / "avg_filler_error_and_tolerance.pdf", bbox_inches="tight")
plt.close()
print(f"Saved avg_filler_error_and_tolerance.png")

print(f"\nAll plots saved to {output_dir}/")
