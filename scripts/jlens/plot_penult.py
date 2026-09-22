"""
plot_penult.py — V3's script offset under a final-layer and a penultimate-layer target (PENULT_RUNBOOK.md).

  A  share of each readout's variance over the vocabulary explained by script class (eta^2, log axis): the shipped
     lens (target layer 60, 100 prompts), three prompts fitted to target 60, the SAME three prompts fitted to target
     59, and the logit lens, whose jump at the output is the model's own script gate
  B  full-vocabulary excess kurtosis, median over activations, same four readouts
  C  how much each lens amplifies the shipped lens's script axis, relative to a typical direction

    python scripts/jlens/plot_penult.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_fig28 import AXIS, INK, MUTED, OUT, style

REPO = Path(__file__).resolve().parents[2]
J = REPO / "outputs/jlens"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a97a5"
NO_BAND = (0, 0)
# hue = target layer (blue: final, orange: penultimate); grey = no Jacobian. Line style separates the two blue lenses.
LENSES = [("shipped", "final-layer target, 100 prompts (the released lens)", BLUE, "-", 2.0),
          ("target60_same", "final-layer target, 3 prompts", BLUE, (0, (4, 2)), 1.4),
          ("target59", "penultimate-layer target, the same 3 prompts", ORANGE, "-", 2.2),
          ("logit", "logit lens (no Jacobian)", GRAY, (0, (2, 2)), 1.4)]
KURT_CLIP = 2.6


def rows(name):
    return sorted(json.loads((J / f"bilingual_depth_{name}.json").read_text())["per_layer"], key=lambda r: r["layer"])


def main():
    cmp_ = json.loads((J / "penult_compare.json").read_text())
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(15, 5.0))
    style(a, "A  Script-class share of readout variance (η²)", "median η² over activations, log scale →", NO_BAND)
    style(b, "B  Full-vocabulary excess kurtosis", "median over activations →", NO_BAND)
    style(c, "C  Gain along the released lens's script axis", "× a typical direction, log scale →", NO_BAND)
    for ax in (a, b, c):
        ax.set_xlabel("Depth (% of blocks) →", fontsize=9, color=MUTED)

    for name, label, col, ls, lw in LENSES:
        rs = rows(name)
        if name != "logit":                       # at the output every lens is the model's own logits
            rs = [r for r in rs if r["layer"] < 59]
        x = [r["depth_0_100"] for r in rs]
        a.plot(x, [r["eta2_script3"]["p50"] for r in rs], color=col, ls=ls, lw=lw, label=label)
        k = [r["kurtosis"]["full"]["p50"] for r in rs]
        b.plot(x, [min(v, KURT_CLIP + 1) for v in k], color=col, ls=ls, lw=lw)
    a.set_yscale("log")
    a.set_ylim(1e-3, 1.0)
    a.legend(fontsize=7.8, frameon=False, loc="center left", bbox_to_anchor=(0.30, 0.60))   # the empty band between the targets
    out = rows("logit")[-1]["eta2_script3"]["p50"]
    a.annotate(f"the model's own output: {out:.3f}", xy=(100, out), xytext=(60, 0.78), fontsize=7.8, color=INK,
               arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.8))
    verdict = cmp_["per_layer"]
    for L in ("20", "30", "45"):
        e = verdict[L]["eta2"]
        a.plot([verdict[L]["depth_0_100"]] * 2, [e["target59"], e["target60_same"]], ls="none", marker="o", ms=4.5,
               mec="white", mew=0.8, color=INK, zorder=5)
        a.annotate(f"{e['target59']:.3f}", xy=(verdict[L]["depth_0_100"], e["target59"]), xytext=(0, -11),
                   textcoords="offset points", ha="center", fontsize=7.5, color=INK)
        a.annotate(f"{e['target60_same']:.2f}", xy=(verdict[L]["depth_0_100"], e["target60_same"]), xytext=(0, -12),
                   textcoords="offset points", ha="center", fontsize=7.5, color=INK)

    b.axhline(0, color=AXIS, lw=0.9)
    b.set_ylim(-1.25, KURT_CLIP)
    b.text(35, KURT_CLIP - 0.08, "3-prompt final-target lens runs off scale\nbelow 30% depth (median up to 33)",
           fontsize=7.5, color=MUTED, va="top")
    b.text(36, -1.12, "negative band: the script step", fontsize=7.8, color=INK)

    layers = sorted(cmp_["lens"], key=int)
    depth = [100 * int(L) / 60 for L in layers]
    for name, _label, col, ls, lw in LENSES[:3]:
        g = [cmp_["lens"][L][name]["gain_along_shipped_u1_over_typical"] for L in layers]
        c.plot(depth, g, color=col, ls=ls, lw=lw, marker="o", ms=4.5, mec="white", mew=0.8)
        c.annotate(f"{g[0]:.0f}×" if g[0] >= 10 else f"{g[0]:.1f}×", xy=(depth[0], g[0]), xytext=(6, 4),
                   textcoords="offset points", fontsize=7.8, color=INK)
        dy = {"shipped": -13, "target60_same": 7, "target59": 7}[name]          # the two final-target lenses end together
        c.annotate(f"{g[-1]:.1f}×", xy=(depth[-1], g[-1]), xytext=(-2, dy), textcoords="offset points", ha="right",
                   fontsize=7.8, color=INK)
    c.text(52, 40, "final-layer target", fontsize=8, color=INK)
    c.text(52, 3.1, "penultimate-layer target", fontsize=8, color=INK)
    c.axhline(1, color=AXIS, lw=0.9)
    c.text(50, 1.06, "a typical direction", fontsize=7.8, color=MUTED, ha="center", va="bottom")
    c.set_yscale("log")
    c.set_ylim(0.8, 90)
    c.set_yticks([1, 3, 10, 30])
    c.set_yticklabels(["1×", "3×", "10×", "30×"])

    # no figure title: the post's caption carries it
    fig.tight_layout(rect=[0, 0.0, 1, 0.99])
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"penult_target.{ext}", dpi=170, facecolor="white")
    print(f"wrote {OUT / 'penult_target.png'}")


if __name__ == "__main__":
    main()
