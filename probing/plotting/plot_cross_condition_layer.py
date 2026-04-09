"""Plot cross-condition layer selection: single panel with cons-div score and per-condition exact match."""

import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 28})

results_dir = Path("probing/results/unsupervised_decode")


def sort_key(name):
    ftype = name.rsplit("_", 1)[0]
    length = int(name.rsplit("_", 1)[1])
    order = {"dots": 0, "alphabet": 1, "counting": 2}
    return (order.get(ftype, 99), length)


# Load all decode results, filter to dominant best_layer
all_data = {}
for f in sorted(results_dir.glob("decode_*.json")):
    cond = f.stem.replace("decode_", "")
    with open(f) as fp:
        all_data[cond] = json.load(fp)

layer_counts = Counter(d["best_layer"] for d in all_data.values())
dominant_layer = layer_counts.most_common(1)[0][0]
cond_data = {c: d for c, d in all_data.items() if d["best_layer"] == dominant_layer}
skipped = [c for c in all_data if c not in cond_data]
if skipped:
    print(f"Skipping {skipped} (best_layer != L{dominant_layer})")

conditions = sorted(cond_data.keys(), key=sort_key)
print(f"Conditions: {conditions}")

# Get layers and scores from first condition (shared across all)
first = cond_data[conditions[0]]
layers = sorted(int(l) for l in first["layer_scores"].keys())
cc_scores = [first["layer_scores"][str(l)]["score"] for l in layers]
best_layer = first["best_layer"]

# Colors
colors = ["#2176AE", "#4A90D9", "#57B894", "#8B5CF6", "#A78BFA", "#C084FC"]

fig, ax = plt.subplots(figsize=(9, 5.5))

for i, cond in enumerate(conditions):
    per_layer = cond_data[cond]["per_layer"]
    exact = [per_layer[str(l)]["eval"]["frac_exact"] * 100 for l in layers]
    ax.plot(layers, exact, color=colors[i % len(colors)], linewidth=2,
            label=cond.replace("_", " "), zorder=3)

ax.set_xlabel("Layer", fontsize=15)
ax.set_ylabel("Exact match to A (%)", fontsize=15)
ax.set_xlim(layers[0], layers[-1])
ax.set_ylim(0, 100)
ax.tick_params(labelsize=12)

# Right axis: cons-div score
ax2 = ax.twinx()
ax2.plot(layers, cc_scores, color="#F0A500", linewidth=2, alpha=0.85,
         label="Cons-div score", zorder=2)
ax2.set_ylabel("Cons-div score", fontsize=15, color="#F0A500")
ax2.tick_params(axis="y", labelcolor="#F0A500", labelsize=12)
ax2.set_ylim(0, max(cc_scores) * 1.5)

# Vertical line at best layer
ax.axvline(best_layer, color="#F0A500", linestyle="--", linewidth=2, alpha=0.8,
           label=f"Selected: L{best_layer}")

# Combined legend
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=11, loc="upper left",
          framealpha=0.9)

plt.tight_layout()

for ext in ["png", "pdf"]:
    outpath = results_dir / f"cross_condition_layer_overlay.{ext}"
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    print(f"Saved {outpath}")

plt.close()
