"""
plot_kurt_paired.py — Figure 28 panel (b), excess kurtosis of each readout over the full vocabulary as percentiles
across activations, for the same ten prompts fitted to the final layer and to the penultimate layer, beside Sonnet
4.5's published panel. Each lens's line stops at its last non-identity layer; the grey markers at 100 % are the model's own
output logits (1st, 50th and 99th percentiles), which every lens equals at its target. Held-out English WikiText, 24 positions x 100 paragraphs (bilingual_depth_{target60_same,
target59}.json). The two V3 panels share a y-axis; values above it are printed at the top.

    python scripts/jlens/plot_kurt_paired.py
"""
from __future__ import annotations

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_bilingual import load
from plot_fig28 import KURT_PCTS, MUTED, OUT, viridis
from plot_hero_offset import GRAY
from plot_hero import panel, V3_YLIM

INK = "#1b2733"


def main():
    ref = load("reference_fig28_sonnet45.json")["panels"][1]
    fin, pen = load("bilingual_depth_target60_same.json"), load("bilingual_depth_target59.json")
    n = fin["lens_n_prompts"]
    assert n == pen["lens_n_prompts"], (n, pen["lens_n_prompts"])
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(17.5, 5.6))
    panel(a, "(a) Claude Sonnet 4.5, from the paper", [i / 24 * 100 for i in ref["series"][0]["x"]],
          [np.array(s["y"]) for s in ref["series"]], (-1.5, 19))
    for ax, d, title in ((b, fin, "(b) DeepSeek-V3, final-layer target"),
                         (c, pen, "(c) DeepSeek-V3, penultimate target")):
        rows = sorted(d["per_layer"], key=lambda r: r["layer"])
        # each lens's line stops below its target
        lens_rows = [r for r in rows if r["layer"] < 59]
        x = [r["depth_0_100"] for r in lens_rows]
        series = [np.array([r["kurtosis"]["full"][f"p{q}"] for r in lens_rows]) for q in KURT_PCTS]
        panel(ax, title, x, series, V3_YLIM, hatch=True)
        top = series[-1]
        over = [(xi, yi) for xi, yi in zip(x, top) if yi > V3_YLIM[1]]
        if over:                                                   # one label per run of off-scale points
            # group off-scale points into runs (split where the gap in depth exceeds 6 points), one label per run
            runs, cur = [], [over[0]]
            for pt in over[1:]:
                if pt[0] - cur[-1][0] < 6:
                    cur.append(pt)
                else:
                    runs.append(cur); cur = [pt]
            runs.append(cur)
            for run in runs:
                lo, hi = min(v for _, v in run), max(v for _, v in run)
                label = f"{hi:.0f}" if len(run) == 1 else f"{lo:.0f}–{hi:.0f}"
                ax.annotate(label, xy=(np.mean([v for v, _ in run]), V3_YLIM[1]), xytext=(0, 2), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7.2, color=MUTED, annotation_clip=False)
    # larger text for the post: scale every title, label and annotation, and the tick labels
    for ax in (a, b, c):
        for t in [ax.title, ax._left_title, ax._right_title, ax.xaxis.label, ax.yaxis.label, *ax.texts]:
            t.set_fontsize(t.get_fontsize() * 1.45)
        ax.tick_params(labelsize=13)
        lt = ax._left_title            # more room above the axes, so the off-scale labels clear the titles
        ax.set_title(lt.get_text(), loc="left", color=lt.get_color(), fontsize=lt.get_fontsize(),
                     fontweight=lt.get_fontweight(), pad=24)
    fig.tight_layout(w_pad=2.2)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"kurt_paired.{ext}", dpi=170, facecolor="white")
    print(f"wrote {OUT / 'kurt_paired.png'}")


if __name__ == "__main__":
    main()
