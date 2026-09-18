"""
plot_hero.py — the post's first figure: Figure 28 panel (b), excess kurtosis of each readout over the vocabulary as
percentiles across activations. (a) Sonnet 4.5 from the paper's spec file (not redistributed); (b) DeepSeek-V3's
released lens over the full vocabulary; (c) the same readouts restricted to Latin-script tokens on English text;
(d) restricted to Han tokens on Chinese Wikipedia. Released lens only (n = 100); the penultimate-target result is
Figure 3 (plot_penult.py). No title and no annotation text: the caption carries them. The workspace band and the
hatching are drawn lighter than the other figures' so the percentile lines stay readable on top of them.

    python scripts/jlens/plot_hero.py
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_bilingual import load
from plot_fig28 import AXIS, KURT_PCTS, MUTED, OUT, PAPER_BAND, style, viridis

V3_YLIM = (-1.7, 6.0)
UNSTABLE = 30          # % depth below which the two half-fit lenses disagree (p99 51 vs 17 at layer 5)
BAND_LIGHT = "#f1f4f8"  # Sonnet's workspace band, lighter than plot_fig28.BAND_C
HATCH = "#e3e8ee"
BELOW_ZERO = "#f8f9fb"
NO_BAND = (0, 0)


def panel(ax, title, x, series, ylim, hatch=False):
    style(ax, title, "Excess kurtosis →", NO_BAND)
    ax.axvspan(*PAPER_BAND, color=BAND_LIGHT, zorder=0, lw=0)
    ax.set_xlabel("Depth (% of layers) →", fontsize=9, color=MUTED)
    ax.set_ylim(*ylim)
    ax.axhline(0, color=AXIS, lw=0.9, zorder=1)
    ax.axhspan(ylim[0], 0, color=BELOW_ZERO, zorder=0, lw=0)
    if hatch:
        ax.axvspan(0, UNSTABLE, facecolor="none", edgecolor=HATCH, hatch="///", lw=0, zorder=0)
    cols = viridis(len(series))
    for y, q, c in zip(series, KURT_PCTS, cols):
        ax.plot(x, np.clip(y, *ylim), color=c, lw=2.6 if q == 50 else 1.3, solid_capstyle="round", zorder=3)
    for q, c, dy in ((99, cols[-1], 6), (50, cols[3], 0), (1, cols[0], -6)):
        i = KURT_PCTS.index(q)
        ax.annotate({99: "99th", 50: "median", 1: "1st"}[q], xy=(x[-1], float(np.clip(series[i][-1], *ylim))),
                    xytext=(4, dy), textcoords="offset points", fontsize=7.8, color=c, va="center", annotation_clip=False)


def main():
    ref, en, zh = load("reference_fig28_sonnet45.json"), load("bilingual_depth_shipped.json"), load("bilingual_depth_shipped_wikizh.json")
    fig, (a, b, c, d) = plt.subplots(1, 4, figsize=(17.5, 4.6))
    p = ref["panels"][1]
    panel(a, "(a) Claude Sonnet 4.5 · from the paper", [i / 24 * 100 for i in p["series"][0]["x"]],
          [np.array(s["y"]) for s in p["series"]], (-1.5, 19))
    for ax, rows, var, title in ((b, en["per_layer"], "full", "(b) DeepSeek-V3 · full vocabulary"),
                                 (c, en["per_layer"], "latin", "(c) DeepSeek-V3 · English text"),
                                 (d, zh["per_layer"], "han", "(d) DeepSeek-V3 · Chinese text")):
        rows = sorted(rows, key=lambda r: r["layer"])
        x = [r["depth_0_100"] for r in rows]
        panel(ax, title, x, [np.array([r["kurtosis"][var][f"p{q}"] for r in rows]) for q in KURT_PCTS], V3_YLIM, hatch=True)
    fig.tight_layout(w_pad=1.6)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"hero_kurtosis.{ext}", dpi=170, facecolor="white")
    print(f"wrote {OUT / 'hero_kurtosis.png'}")


if __name__ == "__main__":
    main()
