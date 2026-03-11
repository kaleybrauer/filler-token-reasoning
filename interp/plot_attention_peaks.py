"""
Visualize attention peaks relative to intermediate values a1 and a2.

Plot 1: Per-example attention profiles with a1/a2 markers
Plot 2: Scatter of peak positions vs nearest intermediate value
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from transformers import AutoTokenizer
import os

# ── Config ──
results_dir = "results/attention_analysis_counting_scrambled"  # adjust
outdir = results_dir  # save plots alongside data
filler_len = 128  # adjust to match your --filler-len
n_examples = 1000  # will stop at FileNotFoundError, effectively uses all available
filler_type = "scrambled_counting"  # "counting" or "scrambled_counting"

# ── Build filler position → number mapping ──
import random as _random
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-72B-Instruct")
if filler_type == "scrambled_counting":
    nums = list(range(1, filler_len + 1))
    _random.Random(42).shuffle(nums)
    filler_text = " ".join(str(n) for n in nums)
else:
    filler_text = " ".join(str(i) for i in range(1, filler_len + 1))
filler_ids = tokenizer.encode(filler_text, add_special_tokens=False)

# Map each token position to the number it belongs to
# E.g., tokens for "1", " ", "2", " ", ... "1", "2", "8" (=128)
pos_to_number = []
current_num_str = ""
for i, tid in enumerate(filler_ids):
    tok = tokenizer.decode([tid])
    if tok.strip().isdigit():
        current_num_str += tok.strip()
        pos_to_number.append(int(current_num_str) if current_num_str else 0)
    else:
        # Space token — belongs to the preceding number
        if current_num_str:
            pos_to_number.append(int(current_num_str))
            current_num_str = ""
        else:
            pos_to_number.append(0)
# Handle last number if no trailing space
if current_num_str:
    pass  # already appended

pos_to_number = np.array(pos_to_number[:len(filler_ids)])

# ── Plot 1: Individual example attention profiles ──
# Show top-attending heads (excluding L1H50 which is structural)
# averaged together, with a1/a2 markers

fig, axes = plt.subplots(3, 2, figsize=(16, 12))
axes = axes.flatten()

plot_idx = 0
for group in ["filler_helped", "filler_didnt_help"]:
    for i in range(3):  # 3 examples per group
        path = f"{results_dir}/attn_{group}_{i}.npz"
        try:
            data = np.load(path)
        except FileNotFoundError:
            continue

        attn = data["attention"]  # [80, 64, seq_len]
        fs = int(data["filler_start"])
        fe = int(data["filler_end"])
        a1 = int(data["a1"])
        a2 = int(data["a2"])
        answer = int(data["answer"])

        filler_attn = attn[:, :, fs:fe]  # [80, 64, filler_tokens]

        # Average across all heads in layers 55-72 (where content-dependent attention lives)
        # Exclude layer 1 which is structural
        layer_range = slice(55, 73)
        avg_attn = filler_attn[layer_range].mean(axis=(0, 1))  # [filler_tokens]

        # Truncate to actual filler_ids length if needed
        plot_len = min(len(avg_attn), len(pos_to_number))
        x_numbers = pos_to_number[:plot_len]
        y_attn = avg_attn[:plot_len]

        ax = axes[plot_idx]
        ax.plot(range(plot_len), y_attn, linewidth=0.8, color="steelblue", alpha=0.8)
        ax.fill_between(range(plot_len), y_attn, alpha=0.2, color="steelblue")

        # Mark a1 and a2 positions on the number line
        # Find token positions closest to a1 and a2
        a1_positions = np.where(x_numbers == a1)[0]
        a2_positions = np.where(x_numbers == a2)[0]

        if len(a1_positions) > 0:
            for p in a1_positions:
                ax.axvline(p, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
        if len(a2_positions) > 0:
            for p in a2_positions:
                ax.axvline(p, color="orange", linestyle="--", linewidth=1.5, alpha=0.7)

        # Also mark a1+a2 = answer
        ans_positions = np.where(x_numbers == answer)[0]
        if len(ans_positions) > 0 and answer <= filler_len:
            for p in ans_positions:
                ax.axvline(p, color="green", linestyle=":", linewidth=1.5, alpha=0.7)

        label = "HELPED" if "helped" in group else "DIDN'T HELP"
        color = "#2d8a4e" if "helped" in group else "#c44e52"
        ax.set_title(f"{label}: a1={a1}, a2={a2}, sum={answer}", fontsize=11,
                     fontweight="bold", color=color)
        ax.set_xlabel("Filler token position", fontsize=9)
        ax.set_ylabel("Attention (L55-72 avg)", fontsize=9)
        ax.tick_params(labelsize=8)

        # Add number-line ticks at major positions
        tick_positions = []
        tick_labels = []
        for num in range(0, filler_len + 1, 20):
            positions = np.where(x_numbers == num)[0]
            if len(positions) > 0:
                tick_positions.append(positions[0])
                tick_labels.append(str(num))
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=7)

        plot_idx += 1

# Legend
legend_elements = [
    Line2D([0], [0], color="red", linestyle="--", linewidth=1.5, label="a1"),
    Line2D([0], [0], color="orange", linestyle="--", linewidth=1.5, label="a2"),
    Line2D([0], [0], color="green", linestyle=":", linewidth=1.5, label="a1+a2 (if ≤ N)"),
]
fig.legend(handles=legend_elements, loc="upper center", ncol=3, fontsize=11,
           bbox_to_anchor=(0.5, 1.02))

plt.suptitle("Last-Token Attention to Filler Region (Layers 55-72 Average)",
             fontsize=14, fontweight="bold", y=1.05)
plt.tight_layout()
fname = os.path.join(outdir, "attention_profiles_with_markers.png")
plt.savefig(fname, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {fname}")


# ── Plot 2: Scatter of peak token positions vs filler token positions of a1/a2 ──
# Distance is measured in FILLER TOKEN POSITIONS, not numerical value.
# This is correct for both ordered and scrambled filler: we ask whether the
# attention peak lands near the position where a1/a2 actually sits in the filler.

all_peaks = []

for group in ["filler_helped", "filler_didnt_help"]:
    for i in range(n_examples):
        path = f"{results_dir}/attn_{group}_{i}.npz"
        try:
            data = np.load(path)
        except FileNotFoundError:
            break

        attn = data["attention"]
        fs = int(data["filler_start"])
        fe = int(data["filler_end"])
        a1 = int(data["a1"])
        a2 = int(data["a2"])
        answer = int(data["answer"])

        filler_attn = attn[:, :, fs:fe]

        # Filler token positions where a1, a2, answer appear in this filler layout
        plot_len = min(filler_attn.shape[2], len(pos_to_number))
        x_numbers = pos_to_number[:plot_len]

        a1_tok_positions = np.where(x_numbers == a1)[0]
        a2_tok_positions = np.where(x_numbers == a2)[0]
        ans_tok_positions = np.where(x_numbers == answer)[0]

        # Use the first token of each number's run as its representative position
        candidates = []
        if len(a1_tok_positions) > 0:
            candidates.append((a1_tok_positions[0], a1, "a1"))
        if len(a2_tok_positions) > 0:
            candidates.append((a2_tok_positions[0], a2, "a2"))
        if len(ans_tok_positions) > 0:
            candidates.append((ans_tok_positions[0], answer, "sum"))

        if not candidates:
            continue

        # Get per-head total filler attention and find top heads (all layers)
        filler_per_head = filler_attn.sum(axis=2)  # [80, 64]
        top_flat = np.argsort(filler_per_head.ravel())[-5:][::-1]

        for idx in top_flat:
            layer = idx // 64
            head = idx % 64
            profile = filler_attn[layer, head, :plot_len]
            peak_pos = int(np.argmax(profile))

            # Distance in filler token positions to each intermediate value's location
            tok_dists = [abs(peak_pos - cand_pos) for cand_pos, _, _ in candidates]
            best = int(np.argmin(tok_dists))
            nearest_tok_pos = candidates[best][0]
            nearest_label = candidates[best][2]
            dist = tok_dists[best]

            all_peaks.append({
                "peak_pos": peak_pos,
                "nearest_tok_pos": nearest_tok_pos,
                "nearest_label": nearest_label,
                "dist": dist,  # in token positions
                "group": "helped" if "helped" in group and "didnt" not in group else "didn't help",
                "a1": a1, "a2": a2, "answer": answer,
            })

# Scatter plot: peak token position vs token position of nearest intermediate
fig, ax = plt.subplots(figsize=(8, 8))

for label, color, marker in [("helped", "#2d8a4e", "o"), ("didn't help", "#c44e52", "x")]:
    points = [p for p in all_peaks if p["group"] == label]
    x = [p["nearest_tok_pos"] for p in points]
    y = [p["peak_pos"] for p in points]
    ax.scatter(x, y, c=color, marker=marker, alpha=0.5, s=40, label=label)

max_pos = max(
    max(p["nearest_tok_pos"] for p in all_peaks),
    max(p["peak_pos"] for p in all_peaks),
)
ax.plot([0, max_pos], [0, max_pos], "k--", linewidth=1, alpha=0.3, label="perfect match")
ax.fill_between([0, max_pos], [0, max_pos - 5], [5, max_pos + 5],
                alpha=0.1, color="gray", label="±5 token band")

ax.set_xlabel("Filler token position of nearest intermediate (a1, a2, or sum)", fontsize=11)
ax.set_ylabel("Filler token position of attention peak", fontsize=11)
ax.set_title("Attention Peak Position vs Intermediate Value Position\n(Top 5 heads per example, all layers; distance in token positions)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.set_aspect("equal")
ax.grid(True, alpha=0.2)

fname = os.path.join(outdir, "peak_vs_intermediate_scatter.png")
plt.savefig(fname, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {fname}")

# Print summary stats (distance in filler token positions)
dists = [p["dist"] for p in all_peaks]
print(f"\nPeak-to-nearest-value distance stats ({len(all_peaks)} peaks, in TOKEN POSITIONS):")
print(f"  Mean: {np.mean(dists):.1f}")
print(f"  Median: {np.median(dists):.1f}")
print(f"  Within 3 tok: {sum(1 for d in dists if d <= 3) / len(dists) * 100:.0f}%")
print(f"  Within 5 tok: {sum(1 for d in dists if d <= 5) / len(dists) * 100:.0f}%")

# Breakdown by group
for g in ["helped", "didn't help"]:
    g_dists = [p["dist"] for p in all_peaks if p["group"] == g]
    if g_dists:
        print(f"  {g}: mean={np.mean(g_dists):.1f}, within 5 tok: {sum(1 for d in g_dists if d <= 5) / len(g_dists) * 100:.0f}%")

# Breakdown by nearest label
for nl in ["a1", "a2", "sum"]:
    nl_peaks = [p for p in all_peaks if p["nearest_label"] == nl]
    if nl_peaks:
        print(f"  Nearest to {nl}: {len(nl_peaks)} peaks, mean dist={np.mean([p['dist'] for p in nl_peaks]):.1f}")