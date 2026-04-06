"""Stacked bar chart: what does the unsupervised decode pipeline output at its best layer?

Categories: Exact A, ±5 of A, Exact A+X, ±5 of A+X, ±6-20 of A+X, Other
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

results_dir = Path("probing/results/unsupervised_decode")


def categorize_predictions(predictions, metadata):
    """Classify each prediction into decode categories."""
    cats = {"exact_A": 0, "near_A": 0, "exact_AX": 0,
            "near_AX": 0, "far_AX": 0, "other": 0}
    n = len(predictions)

    for pred, m in zip(predictions, metadata):
        A = m["fact_value"]
        X = m["x"]
        AX = A + X
        err_A = abs(pred - A)
        err_AX = abs(pred - AX)

        if err_A == 0:
            cats["exact_A"] += 1
        elif err_A <= 5:
            cats["near_A"] += 1
        elif err_AX == 0:
            cats["exact_AX"] += 1
        elif err_AX <= 5:
            cats["near_AX"] += 1
        elif err_AX <= 20:
            cats["far_AX"] += 1
        else:
            cats["other"] += 1

    return {k: v / n * 100 for k, v in cats.items()}


# Load all available decode results
conditions = []
for f in sorted(results_dir.glob("decode_*.json")):
    cond = f.stem.replace("decode_", "")
    with open(f) as fp:
        d = json.load(fp)
    preds = d["best_decode"]["predictions"]
    meta = d["metadata"]
    layer = d["best_layer"]
    cats = categorize_predictions(preds, meta)
    conditions.append({"name": cond, "layer": layer, "cats": cats, "n": len(preds)})

# Sort: dots by length, alphabet by length, counting by length (last)
def sort_key(c):
    name = c["name"]
    ftype = name.rsplit("_", 1)[0]
    length = int(name.rsplit("_", 1)[1])
    order = {"dots": 0, "alphabet": 1, "counting": 2}
    return (order.get(ftype, 99), length)

conditions.sort(key=sort_key)

# Plot
cat_names = ["exact_A", "near_A", "exact_AX", "near_AX", "far_AX", "other"]
cat_labels = ["Exact A", "±5 of A", "Exact A+X", "±5 of A+X", "±6-20 of A+X", "Other"]
cat_colors = ["#228B22", "#98df8a", "#1f77b4", "#6baed6", "#c6dbef", "#d62728"]

n_conds = len(conditions)
fig, ax = plt.subplots(figsize=(max(7, n_conds * 1.5 + 2), 5.5))

x = np.arange(n_conds)
bar_width = 0.6

bottoms = np.zeros(n_conds)
for cat_key, label, color in zip(cat_names, cat_labels, cat_colors):
    vals = [c["cats"][cat_key] for c in conditions]
    bars = ax.bar(x, vals, bar_width, bottom=bottoms, color=color, label=label,
                  edgecolor="white", linewidth=0.5)
    # Add percentage labels for segments > 4%
    for i, (v, b) in enumerate(zip(vals, bottoms)):
        if v > 4:
            ax.text(x[i], b + v / 2, f"{v:.0f}%", ha="center", va="center",
                    fontsize=8, fontweight="bold", color="white")
    bottoms += vals

ax.set_xticks(x)
ax.set_xticklabels([c["name"] for c in conditions], fontsize=10, rotation=30, ha="right")
ax.set_ylabel("% of examples", fontsize=12)
ax.set_ylim(0, 105)
ax.set_title("Unsupervised decode: what does the pipeline output?", fontsize=13)
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9, framealpha=0.9)

plt.tight_layout()
for ext in ["png", "pdf"]:
    outpath = results_dir / f"decode_breakdown_stacked.{ext}"
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    print(f"Saved {outpath}")
plt.close()
