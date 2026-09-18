"""
plot_hero_offset.py — the post's main figure: a final-layer-target J-lens carries the last block's script shift into
every layer as a constant, and a penultimate target leaves it out.

  (a) Han-minus-Latin readout offset (sd units, median over activations) across depth: the released J-lens on English
      and on Chinese text, the logit lens on both; the model's own output at 100 % depth points opposite ways.
  (b) Script share of readout variance (eta^2, median) by layer: logit lens vs released J-lens, English text. The logit
      lens is flat until the last block.
  (c) The same statistic, log axis: released lens, three prompts at the final target, the same three at the
      penultimate target, logit lens (PENULT_RUNBOOK.md). Inputs: bilingual_depth_*.json, penult_compare.json.
No title and no annotation text; the caption carries them.

    python scripts/jlens/plot_hero_offset.py
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_bilingual import load
from plot_fig28 import AXIS, MUTED, OUT, PAPER_BAND, style

BLUE, ORANGE, GRAY, RED = "#2a78d6", "#eb6834", "#8a97a5", "#c0392b"
BAND_LIGHT = "#f1f4f8"
NO_BAND = (0, 0)
DASH = (0, (4, 2))
DOT = (0, (2, 2))


def rows(name):
    return sorted(load(f"bilingual_depth_{name}.json")["per_layer"], key=lambda r: r["layer"])


def frame(ax, title, ylabel):
    style(ax, title, ylabel, NO_BAND)
    ax.axvspan(*PAPER_BAND, color=BAND_LIGHT, zorder=0, lw=0)
    ax.set_xlabel("Depth (% of layers) →", fontsize=9, color=MUTED)


def series(name, key, sub="p50", below_output=True):
    rs = [r for r in rows(name) if (r["layer"] < 60 or not below_output)]
    return [r["depth_0_100"] for r in rs], [r[key][sub] for r in rs]


def endlabel(ax, x, y, text, color, dy=0):
    ax.annotate(text, xy=(x, y), xytext=(4, dy), textcoords="offset points", fontsize=7.8, color=color,
                va="center", ha="left", annotation_clip=False)


def main():
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(15.5, 4.7))

    # (a) the offset is a constant that ignores the input language
    frame(a, "(a) Han − Latin readout offset", "median over activations, sd units →")
    a.axhline(0, color=AXIS, lw=0.9, zorder=1)
    for name, col, ls, lab in (("shipped", BLUE, "-", "J-lens, English"), ("shipped_wikizh", ORANGE, "-", "J-lens, Chinese"),
                               ("logit", BLUE, DOT, "logit lens, English"), ("logit_wikizh", ORANGE, DOT, "logit lens, Chinese")):
        x, y = series(name, "han_minus_latin_mean_logit_sd")
        a.plot(x, y, color=col, ls=ls, lw=2.0 if ls == "-" else 1.4, zorder=3)
    for name, col, dy in (("shipped", BLUE, -7), ("shipped_wikizh", ORANGE, 7)):        # the model's own output
        x, y = series(name, "han_minus_latin_mean_logit_sd", below_output=False)
        a.plot([x[-1]], [y[-1]], marker="D", ms=6, color=col, mec="white", mew=0.8, zorder=5)
        endlabel(a, x[-1], y[-1], f"model's output, {'English' if name == 'shipped' else 'Chinese'}: {y[-1]:+.1f}", col, dy)
    xe, ye = series("shipped", "han_minus_latin_mean_logit_sd")
    xz, yz = series("shipped_wikizh", "han_minus_latin_mean_logit_sd")
    a.text(xe[-1] - 1, ye[-1] - 0.12, "J-lens, English", fontsize=7.8, color=BLUE, ha="right", va="top")
    a.text(xz[-1] - 1, yz[-1] + 0.12, "J-lens, Chinese", fontsize=7.8, color=ORANGE, ha="right", va="bottom")
    xl, yl = series("logit", "han_minus_latin_mean_logit_sd")
    a.text(50, 0.12, "logit lens, both languages", fontsize=7.8, color=GRAY, ha="center", va="bottom")
    a.set_ylim(-1.9, 1.9)
    a.set_xlim(0, 100)

    # (b) the last block writes it
    frame(b, "(b) Script share of readout variance (η²)", "median over activations →")
    x, y = series("shipped", "eta2_script3")
    b.plot(x, y, color=BLUE, lw=2.0, zorder=3)
    x, y = series("logit", "eta2_script3", below_output=False)
    b.plot(x, y, color=GRAY, ls=DOT, lw=1.6, zorder=3)
    b.plot([x[-1]], [y[-1]], marker="D", ms=6, color=GRAY, mec="white", mew=0.8, zorder=5)
    b.text(30, 0.56, "J-lens, final-layer target (released lens)", fontsize=7.8, color=BLUE, ha="left", va="bottom")
    b.text(60, 0.045, "logit lens", fontsize=7.8, color=GRAY, ha="left", va="bottom")
    endlabel(b, x[-1], y[-1], "the model's\nown output", GRAY, 0)
    b.set_ylim(0, 0.62)
    b.set_xlim(0, 100)

    # (c) leave the last block out of the Jacobian and it is gone
    frame(c, "(c) Same statistic, two target layers (log axis)", "median over activations, log scale →")
    for name, col, ls, lw in (("shipped", BLUE, "-", 2.0), ("target60_same", BLUE, DASH, 1.4), ("target59", ORANGE, "-", 2.2), ("logit", GRAY, DOT, 1.4)):
        x, y = series(name, "eta2_script3")
        c.plot(x, y, color=col, ls=ls, lw=lw, zorder=3)
    c.set_yscale("log")
    c.set_ylim(1e-3, 1.0)
    c.set_xlim(0, 100)
    c.text(2, 0.62, "final-layer target: released lens (solid), 3 prompts (dashed)", fontsize=7.8, color=BLUE, va="bottom")
    c.text(2, 0.036, "penultimate-layer target, the same 3 prompts", fontsize=7.8, color=ORANGE, va="bottom")
    c.text(2, 0.0034, "logit lens", fontsize=7.8, color=GRAY, va="bottom")

    fig.tight_layout(w_pad=2.0)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"hero_offset.{ext}", dpi=170, facecolor="white")
    print(f"wrote {OUT / 'hero_offset.png'}")


if __name__ == "__main__":
    main()
