"""
plot_bilingual.py — V3's Figure 28 panel (b) against Sonnet 4.5's, taken apart.

Top row, kurtosis percentiles across activations (the panel (b) statistic):
  (a) Sonnet 4.5, full vocabulary (the paper's panel, redrawn from its spec file)
  (b) V3, full vocabulary (our panel (b)), on Sonnet's y-scale
  (c) V3, Latin-script tokens only      (d) V3, Han tokens only
Bottom row, what the readouts carry:
  (e) share of each readout's variance explained by script-class means (eta^2, median)
  (f) other-script tokens that stand out individually once class means are removed
  (g) activations whose top-10 Latin and top-10 Han readout tokens contain a dictionary
      translation pair, against a different-prompt null
  (h) the early p99 spikes under each half-fit lens and the logit lens

Inputs: bilingual_diagnostics.py (spikes, depth; WikiText and the zh/en Wikipedia states),
translation_pairs.py, and the paper's spec file (reference_fig28_sonnet45.json, not
redistributed). Series whose inputs are missing are skipped.

    python scripts/jlens/plot_bilingual.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_fig28 import INK, MUTED, OUT, PAPER_BAND, draw, style

REPO = Path(__file__).resolve().parents[2]
J = REPO / "outputs/jlens"
KURT_PCTS = (1, 10, 25, 50, 75, 90, 99)
CORPORA = {  # tag -> (label, colour, input script, other script)
    "":        ("WikiText (English)",       "#2f6690", "latin", "han"),
    "_wikien": ("Wikipedia, English",       "#3a9e8f", "latin", "han"),
    "_wikizh": ("Wikipedia, Chinese",       "#d1495b", "han", "latin"),
}
NO_BAND = (0, 0)


def load(name):
    p = J / name
    return json.loads(p.read_text()) if p.exists() else None


def sonnet_edges(ax):
    for x in PAPER_BAND:
        ax.axvline(x, color="#9aa7b4", lw=0.8, ls=(0, (3, 3)), zorder=1)


def kurt_panel(ax, rows, variant, title, ylim):
    x = [r["depth_0_100"] for r in rows]
    series = [[r["kurtosis"][variant][f"p{q}"] for r in rows] for q in KURT_PCTS]
    style(ax, title, "Excess kurtosis →", NO_BAND)
    sonnet_edges(ax)
    ax.set_ylim(*ylim)
    draw(ax, x, [np.clip(s, *ylim) for s in series], [str(q) for q in KURT_PCTS], "percentile")
    for s, q in zip(series, KURT_PCTS):           # say where a curve leaves the axes
        for xi, yi in zip(x, s):
            if yi > ylim[1] and q == 99:
                ax.annotate(f"p99 {yi:.0f}", xy=(xi, ylim[1]), xytext=(5, -3), textcoords="offset points",
                            fontsize=7.5, color=MUTED, ha="left", va="top")


def main():
    ref = load("reference_fig28_sonnet45.json")
    wt = load("bilingual_depth_shipped.json")
    fig, axes = plt.subplots(2, 4, figsize=(17.5, 8.2))
    (a, b, c, d), (e, f, g, h) = axes

    if ref:
        p = ref["panels"][1]
        x = [i / 24 * 100 for i in p["series"][0]["x"]]
        style(a, "(a) Sonnet 4.5 · full vocabulary", "Excess kurtosis →", PAPER_BAND)
        a.set_ylim(-1.5, 19)
        draw(a, x, [s["y"] for s in p["series"]], [s["label"] for s in p["series"]], "percentile")
    kurt_panel(b, wt["per_layer"], "full", "(b) DeepSeek-V3 · full vocabulary", (-1.5, 19))
    kurt_panel(c, wt["per_layer"], "latin", "(c) V3 · Latin-script tokens only", (-1.0, 4.5))
    kurt_panel(d, wt["per_layer"], "han", "(d) V3 · Han tokens only", (-1.0, 4.5))
    b.text(0.40, 0.97, "held-out WikiText, shipped lens", transform=b.transAxes, fontsize=7.8,
           color=MUTED, va="top")

    # (e) script offset share, (f) tokens that stand out within their own script
    dashed = (0, (2, 2))
    style(e, "(e) Script offset: share of readout variance", "η² (median) →", NO_BAND)
    style(f, "(f) Tokens standing out within their script", "Share of top-100, offset removed →", NO_BAND)
    for ax in (e, f):
        sonnet_edges(ax)
    for tag, (label, col, _, _) in CORPORA.items():
        for lens, ls in (("shipped", "-"), ("logit", dashed)):
            dep = load(f"bilingual_depth_{lens}{tag}.json")
            if not dep:
                continue
            rows = dep["per_layer"]
            x = [r["depth_0_100"] for r in rows]
            e.plot(x, [r["eta2_script3"]["p50"] for r in rows], color=col, ls=ls, lw=1.8)
            if lens == "shipped":
                comp = [r["topk_scripts"]["offset_removed_top_ids@100"] for r in rows]
                f.plot(x, [100 * c["frac_han"] for c in comp], color=col, lw=1.8)
                f.plot(x, [100 * c["frac_latin"] for c in comp], color=col, lw=1.4, ls=dashed)
    e.set_ylim(-0.02, 0.72)
    e.plot([], [], color=MUTED, lw=1.8, label="J-lens")
    e.plot([], [], color=MUTED, lw=1.8, ls=dashed, label="logit lens")
    e.legend(fontsize=7.5, frameon=False, loc="upper right", ncol=2)
    f.set_ylim(0, 100)
    f.yaxis.set_major_formatter(lambda v, p: f"{v:.0f}%")
    f.plot([], [], color=MUTED, lw=1.8, label="Han tokens")
    f.plot([], [], color=MUTED, lw=1.4, ls=dashed, label="Latin tokens")
    f.legend(fontsize=7.5, frameon=False, loc="upper left")

    # (g) translation pairs
    style(g, "(g) Latin and Han readouts are translations", "Activations with a pair →", NO_BAND)
    sonnet_edges(g)
    tp = load("translation_pairs.json") or {}
    for tag, (label, col, _, _) in CORPORA.items():
        for lens, ls, mk in (("shipped", "-", "o"), ("logit", dashed, "s")):
            rows = tp.get(f"bilingual_depth_{lens}{tag}_topk.npz")
            if not rows:
                continue
            Ls = sorted(int(k) for k in rows)
            x = [L / 60 * 100 for L in Ls]
            g.plot(x, [100 * rows[str(L)]["hit_rate"] for L in Ls], color=col, ls=ls, lw=1.6,
                   marker=mk, ms=3.5)
            g.plot(x, [100 * rows[str(L)]["null_hit_rate"] for L in Ls], color=col, ls=ls, lw=0.9,
                   alpha=0.45)
    g.plot([], [], color=MUTED, lw=1.6, marker="o", ms=3.5, label="J-lens")
    g.plot([], [], color=MUTED, lw=1.6, ls=dashed, marker="s", ms=3.5, label="logit lens")
    g.plot([], [], color=MUTED, lw=0.9, alpha=0.6, label="null: another prompt's Han list")
    g.set_ylim(0, 100)
    g.yaxis.set_major_formatter(lambda v, p: f"{v:.0f}%")
    g.legend(fontsize=7.5, frameon=False, loc="lower right", bbox_to_anchor=(0.9, 0.08))
    handles = [plt.Line2D([], [], color=col, lw=2.2) for _, col, _, _ in CORPORA.values()]
    fig.legend(handles, [lab for lab, *_ in CORPORA.values()], loc="lower center", ncol=3,
               frameon=False, fontsize=8.5, bbox_to_anchor=(0.4, 0.0))

    # (h) spikes by lens
    sp = load("bilingual_spikes.json")
    style(h, "(h) Early spikes come from one half of the fit", "p99 excess kurtosis →", NO_BAND)
    h.set_xlim(-0.6, 1.6)
    h.set_xlabel("")
    if sp:
        lenses = [("shipped", "shipped (100)", "#4b5d6e"), ("n50", "half A (50)", "#8fb3cf"),
                  ("n50b", "half B (50)", "#d1495b"), ("logit", "logit lens", "#c3ced9")]
        for i, L in enumerate(sorted(sp["layers"], key=int)):
            k = sp["layers"][L]["kurtosis_pcts"]
            for j, (key, lab, col) in enumerate(lenses):
                h.bar(i + (j - 1.5) * 0.2, k[key]["p99"], width=0.19, color=col,
                      label=lab if i == 0 else None)
        h.set_xticks(range(len(sp["layers"])))
        h.set_xticklabels([f"layer {L} ({sp['layers'][L]['depth_0_100']:.0f}% depth)"
                           for L in sorted(sp["layers"], key=int)], fontsize=8.5, color=MUTED)
        h.legend(fontsize=7.5, frameon=False, loc="upper right")

    fig.text(0.005, 0.035, "Dashed verticals: Sonnet 4.5's workspace band (37.5-91.7% depth). V3: shipped "
             "T=40 lens, 2,400 held-out activations per layer; (b)-(d) on WikiText. Bottom row colours "
             "by corpus (both Wikipedia sets from one dump):", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 1), w_pad=2.2, h_pad=2.0)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"bilingual_panel_b.{ext}", dpi=170)
    print(f"wrote {OUT / 'bilingual_panel_b.png'}")


if __name__ == "__main__":
    main()
