"""Plot cross-condition layer selection: single panel with cons-div score and per-condition exact match."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

results_dir = Path("probing/results/unsupervised_decode")

# Load cross-condition results
with open(results_dir / "cross_condition_layer_select.json") as f:
    cc = json.load(f)

# Load per-condition decode results
conditions = cc["conditions"]
cond_data = {}
for cond in conditions:
    with open(results_dir / f"decode_{cond}.json") as f:
        cond_data[cond] = json.load(f)

cc_best = cc["best_layer"]
layers = sorted(int(l) for l in cc["layer_scores"].keys())
cc_scores = [cc["layer_scores"][str(l)]["score"] for l in layers]

# Condition colors and labels
cond_style = {
    "dots_10":     {"color": "#2176AE", "label": "dots 10"},
    "dots_100":    {"color": "#57B894", "label": "dots 100"},
    "counting_25": {"color": "#8B5CF6", "label": "counting 25"},
}

fig, ax = plt.subplots(figsize=(7, 4.5))

# Per-condition exact match curves (left axis)
for cond in conditions:
    per_layer = cond_data[cond]["per_layer"]
    exact = [per_layer[str(l)]["eval"]["frac_exact"] * 100 for l in layers]
    style = cond_style[cond]
    ax.plot(layers, exact, color=style["color"], linewidth=1.8,
            label=style["label"], zorder=3)

ax.set_xlabel("Layer", fontsize=12)
ax.set_ylabel("Exact match to A (%)", fontsize=12)
ax.set_xlim(layers[0], layers[-1])
ax.set_ylim(0, 100)

# Right axis: cons-div score
ax2 = ax.twinx()
ax2.plot(layers, cc_scores, color="#F0A500", linewidth=1.8, alpha=0.85,
         label="Cons-div score", zorder=2)
ax2.set_ylabel("Cons-div score", fontsize=12, color="#F0A500")
ax2.tick_params(axis="y", labelcolor="#F0A500")
ax2.set_ylim(0, max(cc_scores) * 1.5)

# Vertical line at best layer
ax.axvline(cc_best, color="#F0A500", linestyle="--", linewidth=1.5, alpha=0.8,
           label=f"Selected: L{cc_best}")

# Combined legend
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left",
          framealpha=0.9)

ax.set_title("Cross-condition unsupervised layer selection", fontsize=13)
plt.tight_layout()

for ext in ["png", "pdf"]:
    outpath = results_dir / f"cross_condition_layer_overlay.{ext}"
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    print(f"Saved {outpath}")

plt.close()
