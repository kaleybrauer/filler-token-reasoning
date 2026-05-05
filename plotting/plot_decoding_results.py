"""
Figure 5: Decoding accuracy across tasks, models, and judges.

Reproduces the bar chart with five task-model conditions and three bars per
condition (supervised top-2 baseline, Haiku judge, Sonnet judge).

Usage:
    python figure5.py                    # saves figure5.pdf and figure5.png
    python figure5.py --show             # also displays interactively

Edit the DATA dict to change values; everything else is layout.
"""

import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import numpy as np


# ---------------------------------------------------------------------------
# DATA — edit values here
# ---------------------------------------------------------------------------

# Each group: (label, model, n, supervised_top2, haiku_top2, sonnet_top2)
# Use None for missing/inapplicable values.
GROUPS = [
    {
        "task": "1-fact addition",
        "task_desc": "retrieve fact A, add x",
        "model": "DeepSeek V3",
        "n": 3239,
        "supervised": 84.8, "supervised_min": 76.3, "supervised_max": 93.2,
        "haiku":      89.3, "haiku_min":      85.2, "haiku_max":      93.8,
        "sonnet":     94.3, "sonnet_min":     91.6, "sonnet_max":     96.6,
        "supervised_note": None,
    },
    {
        "task": "2-fact addition",
        "task_desc": r"retrieve facts A$_1$, A$_2$, add together",
        "model": "DeepSeek V3",
        "n": 1460,
        "supervised": 35.2, "supervised_min": 29.0, "supervised_max": 45.0,
        "haiku":      51.1, "haiku_min":      43.4, "haiku_max":      56.1,
        "sonnet":     82.2, "sonnet_min":     78.3, "sonnet_max":     86.6,
        "supervised_note": None,
    },
    {
        "task": "2-fact addition",
        "task_desc": r"retrieve facts A$_1$, A$_2$, add together",
        "model": "Kimi K2",
        "n": 2082,
        "supervised": 74.1, "supervised_min": 67.9, "supervised_max": 78.9,
        "haiku":      75.7, "haiku_min":      73.0, "haiku_max":      79.8,
        "sonnet":     90.0, "sonnet_min":     84.1, "sonnet_max":     94.7,
        "supervised_note": None,
    },
    {
        "task": "Letter position",
        "task_desc": "retrieve fact, identify Nth letter",
        "model": "DeepSeek V3",
        "n": 2702,
        "supervised": 66.9, "supervised_min": 53.2, "supervised_max": 81.1,
        "haiku":      85.9, "haiku_min":      84.8, "haiku_max":      87.4,
        "sonnet":     92.1, "sonnet_min":     90.8, "sonnet_max":     94.3,
        "supervised_note": "substring",  # marks † on the bar
    },
    {
        "task": "Letter position",
        "task_desc": "retrieve element/city, identify Nth letter",
        "model": "Kimi K2",
        "n": 2997,
        "supervised": 62.4, "supervised_min": 54.7, "supervised_max": 72.1,
        "haiku":      83.6, "haiku_min":      80.7, "haiku_max":      85.4,
        "sonnet":     90.0, "sonnet_min":     86.6, "sonnet_max":     92.3,
        "supervised_note": "substring",
    },
]

# Colors — matched to the SVG version
COLOR_SUPERVISED = "#b8a290"   # warm grey/tan
COLOR_HAIKU      = "#d4946f"   # light rust
COLOR_SONNET     = "#a64829"   # full rust
EDGE_SUPERVISED  = "#8a7568"
EDGE_HAIKU       = "#a66848"
EDGE_SONNET      = "#7a3320"

INK        = "#1a1a1a"
INK_SOFT   = "#5a5a5a"
INK_FAINT  = "#a39a90"
RULE       = "#e0d8c8"
GRID       = "#f0e8d8"
PAPER      = "#fefdf8"


# ---------------------------------------------------------------------------
# PLOT
# ---------------------------------------------------------------------------

def make_figure():
    # Layout constants
    n_groups = len(GROUPS)
    bar_width = 0.25
    group_gap = 1.0           # gap between group centers
    group_centers = np.arange(n_groups) * group_gap

    # Bar positions within a group (offsets from center)
    bar_offsets = np.array([-bar_width, 0, bar_width])

    fig, ax = plt.subplots(figsize=(14, 6.2))

    # --- bars ---
    min_keys = ["supervised_min", "haiku_min", "sonnet_min"]
    max_keys = ["supervised_max", "haiku_max", "sonnet_max"]

    for i, g in enumerate(GROUPS):
        center = group_centers[i]
        values = [g["supervised"], g["haiku"], g["sonnet"]]
        mins   = [g[k] for k in min_keys]
        maxs   = [g[k] for k in max_keys]
        colors = [COLOR_SUPERVISED, COLOR_HAIKU, COLOR_SONNET]
        edges  = [EDGE_SUPERVISED,  EDGE_HAIKU,  EDGE_SONNET]
        text_colors = [INK_SOFT, EDGE_HAIKU, EDGE_SONNET]

        for j, (v, vmin, vmax, c, e, tc) in enumerate(
                zip(values, mins, maxs, colors, edges, text_colors)):
            x = center + bar_offsets[j]
            if v is None:
                # hatched placeholder
                rect = Rectangle((x - bar_width/2, 0), bar_width, 100,
                                 facecolor="white", edgecolor=INK_FAINT,
                                 linewidth=0.6, linestyle="--", hatch="///",
                                 alpha=0.5)
                ax.add_patch(rect)
                # explanatory label inside
                if j == 0:
                    label_lines = ["non-numeric", "target;", "N/A"] \
                        if g["task"] == "Letter position" \
                        else ["not", "computed"]
                    for k, line in enumerate(label_lines):
                        ax.text(x, 50 - 5*(len(label_lines)-1)/2 + k*5, line,
                                ha="center", va="center", style="italic",
                                fontsize=9, color=INK_FAINT)
            else:
                ax.bar(x, v, width=bar_width, color=c, edgecolor=e,
                       linewidth=0.7)
                # error bar (min/max range), only if data available
                label_top = v  # default: label sits just above bar
                if vmin is not None and vmax is not None:
                    ax.errorbar(x, v, yerr=[[v - vmin], [vmax - v]],
                                fmt="none", color=e, linewidth=1.2,
                                capsize=3.5, capthick=1.2, zorder=5)
                    label_top = vmax
                # value label above bar (or above error bar cap)
                ax.text(x, label_top + 1.2, f"{v:.1f}", ha="center", va="bottom",
                        fontsize=11, color=tc, fontweight="medium",
                        family="monospace")

    # --- axes ---
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels(["0", "20", "40", "60", "80", "100"],
                       family="monospace", fontsize=11, color=INK_SOFT)
    ax.set_ylabel("% of examples with intermediates identified correctly",
                  fontsize=13, style="italic", color=INK_SOFT, labelpad=10)

    # x-axis: hide default ticks/labels (we'll add our own two-tier labels)
    ax.set_xticks([])
    ax.set_xlim(-0.6, n_groups - 0.4)

    # spines
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(INK)
        ax.spines[spine].set_linewidth(1)

    # gridlines (faint, behind bars)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

    # --- two-tier x labels ---
    # tier 1: model + n (just below x axis)
    for i, g in enumerate(GROUPS):
        center = group_centers[i]
        ax.text(center, -5, g["model"],
                ha="center", va="top", fontsize=13,
                fontweight="medium", color=INK)
        ax.text(center, -10, f"n = {g['n']:,}",
                ha="center", va="top", fontsize=11, style="italic",
                color=INK_FAINT, family="monospace")

    # tier 2: task spans (with bracket-like underlines)
    # Group consecutive groups with the same task
    tasks_grouped = []
    i = 0
    while i < n_groups:
        task = GROUPS[i]["task"]
        desc = GROUPS[i]["task_desc"]
        start = i
        while i < n_groups and GROUPS[i]["task"] == task:
            i += 1
        end = i - 1
        tasks_grouped.append((task, desc, start, end))

    # bracket geometry — placed in axes coordinates, below the n labels
    bracket_y = -16
    bracket_tick_h = 1.5
    label_y = -22
    desc_y  = -28

    for task, desc, start, end in tasks_grouped:
        x_start = group_centers[start] - bar_width * 1.7
        x_end   = group_centers[end]   + bar_width * 1.7
        # horizontal line
        ax.plot([x_start, x_end], [bracket_y, bracket_y],
                color=INK_SOFT, linewidth=1, clip_on=False)
        # end ticks
        ax.plot([x_start, x_start], [bracket_y, bracket_y + bracket_tick_h],
                color=INK_SOFT, linewidth=1, clip_on=False)
        ax.plot([x_end, x_end], [bracket_y, bracket_y + bracket_tick_h],
                color=INK_SOFT, linewidth=1, clip_on=False)
        # task name
        x_mid = (x_start + x_end) / 2
        ax.text(x_mid, label_y, task,
                ha="center", va="top", fontsize=14, fontweight="medium",
                color=INK, clip_on=False)
        ax.text(x_mid, desc_y, desc,
                ha="center", va="top", fontsize=11, style="italic",
                color=INK_SOFT, clip_on=False)

    # --- legend (custom, since we want subtitles per entry) ---
    legend_y = 1.06   # in axes-fraction space, above plot
    legend_entries = [
        (COLOR_SUPERVISED, EDGE_SUPERVISED,
         "target in top-2 recovered tokens"),
        (COLOR_HAIKU, EDGE_HAIKU,
         "Haiku judge correct (top-2)"),
        (COLOR_SONNET, EDGE_SONNET,
         "Sonnet judge correct (top-2)"),
    ]
    legend_x_starts = [0.16, 0.42, 0.68]  # axes-fraction
    for (color, edge, label), x in zip(legend_entries, legend_x_starts):
        # swatch
        ax.add_patch(mpatches.Rectangle(
            (x, legend_y), 0.018, 0.025,
            transform=ax.transAxes, facecolor=color, edgecolor=edge,
            linewidth=0.6, clip_on=False))
        ax.text(x + 0.025, legend_y + 0.022, label,
                transform=ax.transAxes, fontsize=12, color=INK,
                va="top", clip_on=False)



    # fig.text(0.5, 0.012,
    #          "† For letter-position, target-in-top-2 uses substring match against "
    #          "name-like tokens (alphabetic, length ≥ 2); see caption.",
    #          ha="center", va="bottom", fontsize=9.5, style="italic",
    #          color=INK_SOFT)

    # tight layout with manual margins (extra room at bottom for tier-2 labels + footer)
    plt.subplots_adjust(left=0.07, right=0.97, top=0.85, bottom=0.30)

    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true",
                        help="display the figure interactively")
    parser.add_argument("--out", default="plotting/figure5",
                        help="output filename (without extension)")
    args = parser.parse_args()

    fig = make_figure()
    # Use bbox_inches=None to respect the manual subplots_adjust margins
    fig.savefig(f"{args.out}.pdf")
    fig.savefig(f"{args.out}.png", dpi=200)
    print(f"Saved {args.out}.pdf and {args.out}.png")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()