#!/usr/bin/env python3
"""
Two-panel plot of filler-token accuracy.

Left panel  — DeepSeek v3, Task 1: multiple filler types
Right panel — Task 2 (dots only): DeepSeek v3 + Kimi K2
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

# ── Colors ────────────────────────────────────────────────────────────────────

COLOR_DOTS    = "#4878d0"   # blue
COLOR_COUNT   = "#6acc65"   # green
COLOR_ALPHA   = "#ee854a"   # orange
COLOR_DS      = "#9467bd"   # purple  (DeepSeek v3, task 2)
COLOR_DS_CNT  = "#c5b0d5"   # light purple  (DeepSeek v3 counting, task 2)
COLOR_KIMI    = "#17becf"   # teal    (Kimi K2)
COLOR_KIMI_CNT= "#9edae5"   # light teal    (Kimi K2 counting)

# ── Data — Task 1 (DeepSeek v3, multiple filler types) ───────────────────────

BASELINE_T1 = 54.0

data_t1 = {
    "k":          [0,          5,    10,   25,   50,   100,  250,  500,  1000],
    "dots":       [BASELINE_T1,62.1, 62.9, 66.8, 67.6, 69.4, 69.2, 72.0, 69.9],
    "counting":   [BASELINE_T1,64.0, 63.2, 69.4, 72.0, 71.9, 71.5, 71.0, 69.5],
    "alpha-mixed":[BASELINE_T1,61.6, 60.8, 67.5, 69.5, 70.5, 71.6, 72.8, 71.5],
    "c-scram":    [BASELINE_T1,59.5, 60.0, 63.2, 65.2, 62.9, 59.9, 57.6, 56.5],
    "a-scram":    [BASELINE_T1,54.5, 50.9, 57.8, 62.1, 59.1, 61.6, 60.0, 60.8],
}

series_t1 = [
    # (label,       key,           color,       ls,   lw,  zo, alpha)
    ("dots",        "dots",        COLOR_DOTS,  "-",  2.0, 3,  1.0),
    ("counting",    "counting",    COLOR_COUNT, "-",  2.0, 3,  1.0),
    ("alphabet",    "alpha-mixed", COLOR_ALPHA, "-",  2.0, 3,  1.0),
    ("c-scram",     "c-scram",     COLOR_COUNT, "--", 1.8, 2,  0.4),
    ("a-scram",     "a-scram",     COLOR_ALPHA, "--", 1.8, 2,  0.4),
]

# ── Data — Task 3 (dots only, two models) ────────────────────────────────────

BASELINE_DS_T3   = 61.1
BASELINE_KIMI_T3 = 68.4

data_ds_t3 = {
    "k":    [0,              3,    5,    10,   25,   50,   100,  250,  500,  1000],
    "dots": [BASELINE_DS_T3, 64.4, 67.0, 66.7, 68.1, 69.4, 69.9, 66.9, 67.2, 66.9],
}

data_kimi_t3 = {
    "k":    [0,               3,    5,    10,   25,   50,   100,  250,  500,  1000],
    "dots": [BASELINE_KIMI_T3,68.2, 70.5, 72.7, 71.7, 73.5, 73.5, 75.0, 71.4, 68.3],
}

# ── Data — Task 2 (dots only, two models) ────────────────────────────────────

BASELINE_DS   = 20.8
BASELINE_KIMI = 27.3

data_ds = {
    "k":      [0,           1,    5,    10,   25,   50,   100],
    "dots":   [BASELINE_DS, 21.6, 21.9, 23.1, 23.5, 24.1, 23.9],
}

data_ds_counting = {
    "k":      [0,           1,    5,    10,   25,   50,   100],
    "dots":   [BASELINE_DS, 21.7, 23.3, 22.6, 25.0, 22.3, 24.3],
}

data_kimi = {
    "k":      [0,             1,    3,    5,    10,   25,   50,   100],
    "dots":   [BASELINE_KIMI, 29.4, 30.2, 29.8, 31.7, 35.1, 33.7, 34.3],
}

data_kimi_counting = {
    "k":      [0,             1,    3,    5,    10,   25,   50,   100],
    "dots":   [BASELINE_KIMI, 29.6, 30.4, 30.9, 35.0, 35.2, 35.6, 35.6],
}

# ── Sample sizes ──────────────────────────────────────────────────────────────

N_T1 = 800
N_T2 = 1500
N_T3 = 637

def se(pcts, n):
    """±1 SE error bars for a list of percentage values given sample size n."""
    return [((p / 100) * (1 - p / 100) / n) ** 0.5 * 100 for p in pcts]


# ── Helper: build x positions (k=0 offset, rest uniformly spaced) ─────────────

def make_xpos(k_values, gap=-0.6):
    k_nonzero = [v for v in k_values if v > 0]
    pos = {kv: i + 1 for i, kv in enumerate(k_nonzero)}
    pos[0] = gap
    return pos, [pos[kv] for kv in k_values]


# ── Figure ────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(15, 3.5))
gs_outer = GridSpec(1, 3, figure=fig, wspace=0.38)
ax1 = fig.add_subplot(gs_outer[0, 0])
# Task 2 middle: two tightly-spaced subpanels with independent y-axes
gs_mid = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_outer[0, 1], wspace=0.08)
ax2a = fig.add_subplot(gs_mid[0, 0])   # DeepSeek v3
ax2b = fig.add_subplot(gs_mid[0, 1])   # Kimi K2
ax3 = fig.add_subplot(gs_outer[0, 2])

# ── Left panel: Task 1 ────────────────────────────────────────────────────────

k1 = data_t1["k"]
xpos1, xs1 = make_xpos(k1)

for label, key, color, ls, lw, zo, alpha in series_t1:
    ax1.errorbar(xs1, data_t1[key], yerr=se(data_t1[key], N_T1),
                 color=color, linestyle=ls, linewidth=lw,
                 marker="o", markersize=4.5, label=label, zorder=zo, alpha=alpha,
                 capsize=2.5, capthick=1.0, elinewidth=0.8)

ax1.set_xticks(xs1)
ax1.set_xticklabels([str(v) for v in k1], rotation=45, ha="right")
ax1.set_xlim(xs1[0] - 0.4, xs1[-1] + 0.4)

yvals_t1 = [v for key in ["dots","counting","alpha-mixed","c-scram","a-scram"]
            for v in data_t1[key]]
ax1.set_ylim(max(0, min(yvals_t1) - 3), max(yvals_t1) + 3)
ax1.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
ax1.set_xlabel("Number of Filler [k]", fontsize=12)
ax1.set_ylabel("Accuracy", fontsize=12)
ax1.legend(fontsize=9, framealpha=0.0, ncol=3,
           loc="lower center", bbox_to_anchor=(0.5, 1.01),
           borderaxespad=0, handlelength=1.6)
ax1.grid(axis="y", alpha=0.3, linewidth=0.8)
ax1.grid(axis="x", alpha=0.15, linewidth=0.6)
ax1.text(0.97, 0.03, "1-fact addition", transform=ax1.transAxes,
         ha="right", va="bottom", fontsize=11, color="gray")

# ── Middle panel: Task 2 (two subpanels, independent y-axes) ──────────────────

def setup_subpanel(ax, data, color, label, show_ylabel):
    _, xs = make_xpos(data["k"])
    ax.plot(xs, data["dots"], color=color, linestyle="-", linewidth=2.0,
            marker="o", markersize=4.5, label=label, zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(v) for v in data["k"]], fontsize=10, rotation=45, ha="right")
    ax.set_xlim(xs[0] - 0.5, xs[-1] + 0.5)
    yvals = data["dots"]
    ax.set_ylim(max(0, min(yvals) - 1), max(yvals) + 1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    if show_ylabel:
        ax.set_ylabel("Accuracy", fontsize=12)
    ax.legend(fontsize=10, framealpha=0.0, ncol=1,
              loc="lower center", bbox_to_anchor=(0.5, 1.01),
              borderaxespad=0, handlelength=1.6)
    ax.grid(axis="y", alpha=0.3, linewidth=0.8)
    ax.grid(axis="x", alpha=0.15, linewidth=0.6)

# ax2a: DeepSeek v3 — dots + counting, x-axis covers union of both k sets
all_k2a = sorted(set(data_ds["k"]) | set(data_ds_counting["k"]))
xpos2a, _ = make_xpos(all_k2a)

xs_dots = [xpos2a[kv] for kv in data_ds["k"]]
xs_cnt  = [xpos2a[kv] for kv in data_ds_counting["k"]]
ax2a.errorbar(xs_dots, data_ds["dots"],          yerr=se(data_ds["dots"], N_T2),
              color=COLOR_DS,    linestyle="-", linewidth=2.0,
              marker="o", markersize=4.5, label="dots",     zorder=3,
              capsize=2.5, capthick=1.0, elinewidth=0.8)
ax2a.errorbar(xs_cnt,  data_ds_counting["dots"], yerr=se(data_ds_counting["dots"], N_T2),
              color=COLOR_DS_CNT, linestyle="-", linewidth=2.0,
              marker="o", markersize=4.5, label="counting", zorder=3,
              capsize=2.5, capthick=1.0, elinewidth=0.8)

xs2a_all = [xpos2a[kv] for kv in all_k2a]
ax2a.set_xticks(xs2a_all)
ax2a.set_xticklabels([str(v) for v in all_k2a], fontsize=10, rotation=45, ha="right")
ax2a.set_xlim(xs2a_all[0] - 0.5, xs2a_all[-1] + 0.5)
yvals2a = data_ds["dots"] + data_ds_counting["dots"]
ax2a.set_ylim(max(0, min(yvals2a) - 1), max(yvals2a) + 1)
ax2a.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
ax2a.set_ylabel("Accuracy", fontsize=12)
ax2a.legend(fontsize=10, framealpha=0.0, ncol=1,
            loc="lower center", bbox_to_anchor=(0.5, 1.01),
            borderaxespad=0, handlelength=1.6)
ax2a.grid(axis="y", alpha=0.3, linewidth=0.8)
ax2a.grid(axis="x", alpha=0.15, linewidth=0.6)
ax2a.text(0.03, 0.97, "DeepSeek v3", transform=ax2a.transAxes,
          ha="left", va="top", fontsize=10, color="gray")
# ax2b: Kimi K2 — dots + counting (same k values for both)
xpos2b, _ = make_xpos(data_kimi["k"])
xs_kimi_dots = [xpos2b[kv] for kv in data_kimi["k"]]
xs_kimi_cnt  = [xpos2b[kv] for kv in data_kimi_counting["k"]]
ax2b.errorbar(xs_kimi_dots, data_kimi["dots"],          yerr=se(data_kimi["dots"], N_T2),
              color=COLOR_KIMI,     linestyle="-", linewidth=2.0,
              marker="o", markersize=4.5, label="dots",     zorder=3,
              capsize=2.5, capthick=1.0, elinewidth=0.8)
ax2b.errorbar(xs_kimi_cnt,  data_kimi_counting["dots"], yerr=se(data_kimi_counting["dots"], N_T2),
              color=COLOR_KIMI_CNT, linestyle="-", linewidth=2.0,
              marker="o", markersize=4.5, label="counting", zorder=3,
              capsize=2.5, capthick=1.0, elinewidth=0.8)

xs2b_all = [xpos2b[kv] for kv in data_kimi["k"]]
ax2b.set_xticks(xs2b_all)
ax2b.set_xticklabels([str(v) for v in data_kimi["k"]], fontsize=10, rotation=45, ha="right")
ax2b.set_xlim(xs2b_all[0] - 0.5, xs2b_all[-1] + 0.5)
yvals2b = data_kimi["dots"] + data_kimi_counting["dots"]
ax2b.set_ylim(max(0, min(yvals2b) - 1), max(yvals2b) + 1)
ax2b.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
ax2b.yaxis.tick_right()
ax2b.yaxis.set_label_position("right")
ax2b.legend(fontsize=10, framealpha=0.0, ncol=1,
            loc="lower center", bbox_to_anchor=(0.5, 1.01),
            borderaxespad=0, handlelength=1.6)
ax2b.grid(axis="y", alpha=0.3, linewidth=0.8)
ax2b.grid(axis="x", alpha=0.15, linewidth=0.6)
ax2b.text(0.03, 0.97, "Kimi K2", transform=ax2b.transAxes,
          ha="left", va="top", fontsize=10, color="gray")

# Single x-label centered between the two subpanels
ax2a.set_xlabel("Number of Filler [k]", fontsize=12)
ax2a.xaxis.set_label_coords(1.0, -0.17)

# Shared annotation centered across both subpanels
ax2b.text(0.97, 0.03, "2-fact addition", transform=ax2b.transAxes,
          ha="right", va="bottom", fontsize=11, color="gray")

# ── Right panel: Task 3 ───────────────────────────────────────────────────────

all_k3 = data_ds_t3["k"][:]
if data_kimi_t3 is not None:
    all_k3 = sorted(set(all_k3) | set(data_kimi_t3["k"]))
xpos3, _ = make_xpos(all_k3)

xs_ds3 = [xpos3[kv] for kv in data_ds_t3["k"]]
ax3.errorbar(xs_ds3, data_ds_t3["dots"], yerr=se(data_ds_t3["dots"], N_T3),
             color=COLOR_DS, linestyle="-", linewidth=2.0,
             marker="o", markersize=4.5, label="DeepSeek v3", zorder=3,
             capsize=2.5, capthick=1.0, elinewidth=0.8)

if data_kimi_t3 is not None:
    xs_kimi3 = [xpos3[kv] for kv in data_kimi_t3["k"]]
    ax3.errorbar(xs_kimi3, data_kimi_t3["dots"], yerr=se(data_kimi_t3["dots"], N_T3),
                 color=COLOR_KIMI, linestyle="-", linewidth=2.0,
                 marker="o", markersize=4.5, label="Kimi K2", zorder=3,
                 capsize=2.5, capthick=1.0, elinewidth=0.8)

xs3_all = [xpos3[kv] for kv in all_k3]
ax3.set_xticks(xs3_all)
ax3.set_xticklabels([str(v) for v in all_k3], fontsize=10, rotation=45, ha="right")
ax3.set_xlim(xs3_all[0] - 0.4, xs3_all[-1] + 0.4)

yvals_t3 = data_ds_t3["dots"][:]
if data_kimi_t3 is not None:
    yvals_t3 += data_kimi_t3["dots"]
ax3.set_ylim(max(0, min(yvals_t3) - 3), max(yvals_t3) + 3)
ax3.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
ax3.set_xlabel("Number of Filler [k]", fontsize=12)
ax3.set_ylabel("Accuracy", fontsize=12)
ncol3 = 2 if data_kimi_t3 is not None else 1
ax3.legend(fontsize=10, framealpha=0.0, ncol=ncol3,
           loc="lower center", bbox_to_anchor=(0.5, 1.01),
           borderaxespad=0, handlelength=1.6)
ax3.grid(axis="y", alpha=0.3, linewidth=0.8)
ax3.grid(axis="x", alpha=0.15, linewidth=0.6)
ax3.text(0.97, 0.03, "2-hop letter position", transform=ax3.transAxes,
         ha="right", va="bottom", fontsize=11, color="gray")

# ── Save ──────────────────────────────────────────────────────────────────────

plt.tight_layout()

for ext in ("pdf", "png"):
    outpath = f"plotting/filler_accuracy.{ext}"
    plt.savefig(outpath, bbox_inches="tight", dpi=150 if ext == "png" else None)
    print(f"Saved {outpath}")
