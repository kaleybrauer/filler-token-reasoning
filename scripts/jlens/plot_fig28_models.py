"""
plot_fig28_models.py — the Figure 28 signatures and the script-aware readout measures, for DeepSeek-V3 beside
Qwen3.5-397B and Qwen3.5-122B, over normalised depth. Everything here uses held-out WikiText activations.

  A  next-token top-8 accuracy of the J-lens readout        (workspace_readouts.py)
  B  readout kurtosis, median over activations, full vocabulary
  C  top-1 autocorrelation at offset 1 (delta log p vs the null)
  D  J-space dimensionality at 90% variance                 (workspace_signatures.py / survey_published.py)
  E  kurtosis within Latin tokens only (solid) against a size-matched random vocabulary subset (dashed)
     -- the vocabulary-matched version of B                 (bilingual_diagnostics.py depth)
  F  Han share of the top-100 readout tokens, against each model's Han share of the vocabulary (dotted)

V3's two 50-prompt half-fits give the ribbon in A-D; the Qwen lenses are single fits. Missing inputs are drawn
as "pending", so this can be re-run while the analyses are still going.

    python scripts/jlens/plot_fig28_models.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_fig28 import AXIS, GRID, INK, MUTED, OUT, style

REPO = Path(__file__).resolve().parents[2]
J, Q, QS = REPO / "outputs/jlens", REPO / "outputs/jlens/qwen35", REPO / "outputs/jlens_qwen35"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
NO_BAND = (0, 0)
MODELS = [  # label, colour, blocks, readouts, dimensionality, bilingual depth
    ("DeepSeek-V3", BLUE, 61, J / "workspace_readouts_shipped100.json", J / "workspace_signatures_shipped100.json",
     J / "bilingual_depth_shipped.json"),
    ("Qwen3.5-397B", ORANGE, 60, QS / "qwen35_397b_fp8/analysis/workspace_readouts_shipped.json",
     Q / "survey_qwen35_397b_fp8.json", QS / "qwen35_397b_fp8/analysis/bilingual_depth_shipped.json"),
    ("Qwen3.5-122B", AQUA, 48, QS / "qwen35_122b/analysis/workspace_readouts_shipped.json",
     Q / "survey_qwen35_122b.json", QS / "qwen35_122b/analysis/bilingual_depth_shipped.json"),
]
HALVES = [J / "workspace_readouts_n50.json", J / "workspace_readouts_n50b.json"]
SIG_HALVES = [J / "workspace_signatures_n50.json", J / "workspace_signatures_n50b.json"]


def load(p):
    return json.loads(p.read_text()) if p.exists() else None


def curve(src, key, scale=1.0):
    rows = sorted(src["per_layer"], key=lambda r: r["layer"])
    x = np.array([r["depth_0_100"] for r in rows], float)
    y = np.array([_get(r, key) for r in rows], float) * scale
    return x, y


def _get(row, key):
    cur = row
    for part in key.split("."):
        cur = cur[part]
    return cur


def ribbon(ax, paths, key, colour, scale=1.0):
    srcs = [load(p) for p in paths]
    if not all(srcs):
        return
    cs = [curve(s, key, scale) for s in srcs]
    g = np.union1d(cs[0][0], cs[1][0])
    y0, y1 = np.interp(g, *cs[0]), np.interp(g, *cs[1])
    ax.fill_between(g, np.minimum(y0, y1), np.maximum(y0, y1), color=colour, alpha=0.2, lw=0, zorder=1)


def main():
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9))
    (a, b, c), (d, e, f) = axes
    style(a, "A  Next-token top-8 accuracy", "Top-8 accuracy (%) →", NO_BAND)
    style(b, "B  Readout kurtosis (median, full vocabulary)", "Excess kurtosis →", NO_BAND)
    style(c, "C  Top-1 autocorrelation at offset 1", "Δ log p vs null →", NO_BAND)
    style(d, "D  J-space dimensionality", "Dimensions at 90% variance (%) →", NO_BAND)
    style(e, "E  Kurtosis within Latin tokens (p50)", "Excess kurtosis →", NO_BAND)
    style(f, "F  Han share of the top-100 readouts", "Share (%) →", NO_BAND)
    for ax in axes.ravel():
        ax.set_xlabel("Depth (% of blocks) →", fontsize=9, color=MUTED)

    pending = []
    for label, col, n_total, rd_p, sig_p, bil_p in MODELS:
        rd, sig, bil = load(rd_p), load(sig_p), load(bil_p)
        if rd:
            a.plot(*curve(rd, "nexttok_top8", 100), color=col, lw=2, label=label)
            b.plot(*curve(rd, "kurtosis_p50"), color=col, lw=2, label=label)
            c.plot(*curve(rd, "autocorr_d1"), color=col, lw=2, label=label)
        else:
            pending.append(f"{label} readouts")
        if sig:
            d.plot(*curve(sig, "eff_dim_900", 100), color=col, lw=2, label=label)
        if bil:
            e.plot(*curve(bil, "kurtosis.latin.p50"), color=col, lw=2, label=label)
            e.plot(*curve(bil, "kurtosis.rand_latin_size.p50"), color=col, lw=1, ls=(0, (2, 2)))
            f.plot(*curve(bil, "topk_scripts.top_ids@100.frac_han", 100), color=col, lw=2, label=label)
            f.axhline(100 * bil["vocab"]["shares"]["Han"], color=col, lw=1, ls=(0, (1, 2)))
        else:
            pending.append(f"{label} bilingual depth")
    ribbon(a, HALVES, "nexttok_top8", BLUE, 100)
    ribbon(b, HALVES, "kurtosis_p50", BLUE)
    ribbon(c, HALVES, "autocorr_d1", BLUE)
    ribbon(d, SIG_HALVES, "eff_dim_900", BLUE, 100)

    for ax in (b, c, e):
        ax.axhline(0, color=AXIS, lw=0.9)
    a.set_ylim(-3, 103)
    d.set_ylim(0, 100)
    for ax in axes.ravel():
        ax.legend(fontsize=8, frameon=False, loc="upper left")
    if pending:
        f.text(0.02, 0.02, "pending: " + ", ".join(sorted(set(pending))), transform=f.transAxes,
               fontsize=7.5, color=MUTED)

    fig.suptitle("Workspace signatures across models: DeepSeek-V3 vs Qwen3.5", fontsize=13.5, color=INK,
                 x=0.006, ha="left", y=0.985)
    fig.text(0.006, 0.945, "Held-out WikiText: 100 paragraphs x 112 positions (A-D) and 24 positions (E, F). "
             "Depth = layer / (blocks - 1). Lenses: V3's shipped T=40 fit on 100 prompts; the dallinmj Qwen3.5 "
             "lenses on 500 passages. Blue ribbon in A-D: V3's two 50-prompt half-fits.", fontsize=8.2, color=MUTED)
    fig.text(0.006, 0.012, "E: dashed = a random vocabulary subset of the same size as the Latin one, the control "
             "for kurtosis depending on how many tokens are measured. F: dotted = that model's Han share of the "
             "vocabulary (the base rate).", fontsize=8, color=MUTED)
    fig.tight_layout(rect=[0, 0.03, 1, 0.935])
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig28_models.{ext}", dpi=170, facecolor="white")
    print(f"wrote {OUT / 'fig28_models.png'}" + (f"  (pending: {sorted(set(pending))})" if pending else ""))


if __name__ == "__main__":
    main()
