"""
plot_anatomy.py — the Jacobian's own anatomy across depth (candidate panel for the main figure): for the released
lens (final target, 100 prompts), the same 10 prompts at the final target, and the same 10 at the penultimate target,
(top) the share of ||J_L||_F^2 held by J_L's top singular direction, log axis, and (bottom) how well that direction
separates Chinese-character from Latin-alphabet unembedding rows (AUC; 0.5 = not at all). Grey reference: the last
block's own Jacobian J_{59->60}. Input: audit_v2/lead/jacobian_anatomy.json (jacobian_anatomy.py).

    python scripts/jlens/plot_anatomy.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_fig28 import AXIS, MUTED, OUT, PAPER_BAND, style
from plot_hero_offset import BLUE, ORANGE, GRAY, DASH, BAND_LIGHT

SRC = Path("/workspace/jlens_blogpost/audit_v2/lead/jacobian_anatomy.json")
NO_BAND = (0, 0)
LENSES = (("released", BLUE, "-", 2.0), ("target60_same", BLUE, DASH, 1.4), ("target59", ORANGE, "-", 2.2))


def main():
    d = json.loads(SRC.read_text())
    fig, (a, b) = plt.subplots(2, 1, figsize=(5.2, 6.2), sharex=True, gridspec_kw={"height_ratios": [1, 1]})
    for ax, title, ylabel in ((a, "Share of ‖J‖² in the top direction", "share, log scale →"),
                              (b, "Does that direction separate the scripts?", "AUC, Chinese vs Latin rows →")):
        style(ax, title, ylabel, NO_BAND)
        ax.axvspan(*PAPER_BAND, color=BAND_LIGHT, zorder=0, lw=0)
        ax.set_xlim(0, 100)
    b.set_xlabel("Depth (% of layers) →", fontsize=9, color=MUTED)
    a.set_xlabel("")
    for name, col, ls, lw in LENSES:
        rows = d["lenses"][name]
        x = [100 * r["layer"] / 60 for r in rows]
        a.plot(x, [r["share_top1"] for r in rows], color=col, ls=ls, lw=lw, zorder=3)
        b.plot(x, [r["auc_script_top1"] for r in rows], color=col, ls=ls, lw=lw, zorder=3)
    lb = d["last_block"]
    a.axhline(lb["share_top1"], color=GRAY, ls=(0, (2, 2)), lw=1.4, zorder=2)
    b.axhline(lb["auc_script_top1"], color=GRAY, ls=(0, (2, 2)), lw=1.4, zorder=2)
    b.axhline(0.5, color=AXIS, lw=0.9, zorder=1)
    a.set_yscale("log"); a.set_ylim(1e-3, 1.0)
    b.set_ylim(0.45, 1.02)
    a.text(35, 0.55, "final-layer target: released lens (solid), 10 prompts (dashed)", fontsize=7.8, color=BLUE, va="bottom")
    a.text(60, 0.011, "penultimate-layer target, the same 10 prompts", fontsize=7.8, color=ORANGE, va="top")
    a.text(2, lb["share_top1"] * 1.15, "the last block's own Jacobian, J₅₉→₆₀", fontsize=7.8, color=GRAY, va="bottom")
    b.text(2, 0.905, "final-layer target", fontsize=7.8, color=BLUE, va="bottom")
    b.text(40, 0.62, "penultimate-layer target", fontsize=7.8, color=ORANGE, va="bottom")
    b.text(60, 0.515, "chance", fontsize=7.8, color=MUTED, va="bottom")
    fig.tight_layout(h_pad=1.5)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"anatomy.{ext}", dpi=170, facecolor="white")
    print(f"wrote {OUT / 'anatomy.png'}")


if __name__ == "__main__":
    main()
