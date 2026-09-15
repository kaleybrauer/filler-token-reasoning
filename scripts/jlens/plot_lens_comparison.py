"""
plot_lens_comparison.py — DeepSeek-V3 against Qwen3.5-397B and Qwen3.5-122B on measures of the lens itself
(no activations): what each model's J-lens looks like over normalized depth, and how far to trust it.

  A  J-space dimensionality at 90% variance (fraction of d_model); V3 half-fit range as a ribbon
  B  mean cosine between J-lens and logit-lens readout directions over the vocabulary
  C  how cleanly each lens's dominant output direction separates Han from Latin tokens
  D  agreement of two independent fits of the same model: V3's 50-prompt halves, Qwen3.5-397B's
     500- and 24-prompt lenses (mean per-token readout cosine)
  E  layer-by-layer linear CKA, one small matrix per model

Inputs are the JSON outputs of workspace_signatures.py (V3), survey_published.py (Qwen), script_axis.py,
lens_agreement.py and cka_layers.py; missing inputs are skipped.

    python scripts/jlens/plot_lens_comparison.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_fig28 import AXIS, INK, MUTED, OUT, style
from cka_layers import decay_only, three_blocks

REPO = Path(__file__).resolve().parents[2]
J, Q = REPO / "outputs/jlens", REPO / "outputs/jlens/qwen35"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#a3acb5"
MODELS = [  # label, colour, decoder blocks, per-layer dimensionality/cosine JSON, script-axis JSON, CKA JSON
    ("DeepSeek-V3 (100 prompts)", BLUE, 61, J / "workspace_signatures_shipped100.json", J / "script_axis.json", J / "cka_layers.json"),
    ("Qwen3.5-397B (500 prompts)", ORANGE, 60, Q / "survey_qwen35_397b_fp8.json", Q / "script_axis_qwen35_397b.json", Q / "cka_qwen35_397b.json"),
    ("Qwen3.5-122B (500 prompts)", AQUA, 48, Q / "survey_qwen35_122b.json", Q / "script_axis_qwen35_122b.json", Q / "cka_qwen35_122b.json"),
]
NO_BAND = (0, 0)
CKA_MIN = 0.3  # one colour scale for every matrix: Qwen's late layers fall to ~0.3 against early ones


def load(p):
    return json.loads(p.read_text()) if p.exists() else None


def curve(rows, key):
    rows = sorted(rows, key=lambda r: r["layer"])
    return np.array([r["depth_0_100"] for r in rows]), np.array([r[key] for r in rows], float)


def main():
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, left=0.05, right=0.97, top=0.92, bottom=0.13, wspace=0.28, hspace=0.42)
    a, b, c = (fig.add_subplot(gs[0, i]) for i in range(3))
    d = fig.add_subplot(gs[1, 0])
    style(a, "A  J-space dimensionality", "Fraction of dimensions at 90% variance →", NO_BAND)
    style(b, "B  J-lens vs logit-lens readout directions", "Mean cosine over the vocabulary →", NO_BAND)
    style(c, "C  Is the lens's dominant direction a script axis?", "Han-vs-Latin separation (AUC) →", NO_BAND)
    style(d, "D  Agreement of two independent fits", "Mean per-token readout cosine →", NO_BAND)
    for ax_ in (a, b, c, d):
        ax_.set_xlabel("Depth (% of blocks) →", fontsize=9, color=MUTED)
    for label, col, n_total, sig_p, axis_p, _ in MODELS:
        sig = load(sig_p)
        if sig:
            x, y = curve(sig["per_layer"], "eff_dim_900")
            a.plot(x, 100 * y, color=col, lw=2, label=label)
            x, y = curve(sig["per_layer"], "cos_to_logit")
            b.plot(x, y, color=col, lw=2, label=label)
        ax_js = load(axis_p)
        if ax_js:
            Ls = sorted(int(k) for k in ax_js["layers"])
            c.plot([100 * L / (n_total - 1) for L in Ls], [ax_js["layers"][str(L)]["script_separation"] for L in Ls],
                   color=col, lw=2, marker="o", ms=6, label=label)
    halves = [load(J / f"workspace_signatures_{h}.json") for h in ("n50", "n50b")]
    if all(halves):
        xs = [curve(h["per_layer"], "eff_dim_900") for h in halves]
        g = np.union1d(xs[0][0], xs[1][0])
        y0, y1 = np.interp(g, *xs[0]), np.interp(g, *xs[1])
        a.fill_between(g, 100 * np.minimum(y0, y1), 100 * np.maximum(y0, y1), color=BLUE, alpha=0.2, lw=0)
    n24 = load(Q / "survey_qwen35_397b_n24.json")
    if n24:
        x, y = curve(n24["per_layer"], "eff_dim_900")
        a.plot(x, 100 * y, color=ORANGE, lw=1, ls=(0, (2, 2)), label="Qwen3.5-397B (24 prompts)")
    n24_axis = load(Q / "script_axis_qwen35_397b_n24.json")
    if n24_axis:
        Ls = sorted(int(k) for k in n24_axis["layers"])
        c.plot([100 * L / 59 for L in Ls], [n24_axis["layers"][str(L)]["script_separation"] for L in Ls],
               color=ORANGE, lw=1, ls=(0, (2, 2)), marker="o", ms=4, label="Qwen3.5-397B (24 prompts)")
    a.yaxis.set_major_formatter(lambda v, p: f"{v:.0f}%")
    a.set_ylim(0, 100)
    c.set_ylim(0.45, 1.02)
    c.axhline(0.5, color=AXIS, lw=1)
    c.text(2, 0.515, "chance", fontsize=7.5, color=MUTED)
    for pair, col, label in ((J / "lens_agreement_v3_halves.json", BLUE, "DeepSeek-V3: 50- vs 50-prompt halves"),
                             (Q / "lens_agreement_397b_n500_vs_n24.json", ORANGE, "Qwen3.5-397B: 500- vs 24-prompt lenses")):
        ag = load(pair)
        if ag:
            x, y = curve(ag["per_layer"], "readout_cos_mean")
            d.plot(x, y, color=col, lw=2, label=label)
    d.set_ylim(0.4, 1.02)
    for ax_ in (a, b, d):
        ax_.legend(fontsize=7.5, frameon=False, loc="lower right")
    c.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.88), ncol=2)

    sub = gs[1, 1:].subgridspec(1, 3, wspace=0.3)
    heat, im = [], None
    for i, (label, col, n_total, sig_p, _, cka_p) in enumerate(MODELS):
        e = fig.add_subplot(sub[0, i])
        heat.append(e)
        ck = load(cka_p)
        if not ck:
            e.set_title(label.split(" (")[0], fontsize=9.5, color=INK, loc="left")
            e.text(0.5, 0.5, "pending", ha="center", va="center", color=MUTED, transform=e.transAxes)
            e.axis("off")
            continue
        M = np.array(ck["cka"])
        score, b1, b2 = three_blocks(M)
        e.set_title(f"{label.split(' (')[0]}\nblock score {score:.2f} (decay alone {three_blocks(decay_only(M))[0]:.2f})",
                    fontsize=9.5, color=INK, loc="left")
        depth = 100 * np.array(ck["layers"]) / (n_total - 1)
        h = (depth[1] - depth[0]) / 2
        im = e.imshow(M, vmin=CKA_MIN, vmax=1.0, cmap="Blues", origin="lower",
                      extent=(depth[0] - h, depth[-1] + h, depth[0] - h, depth[-1] + h))
        for edge in (depth[b1] - h, depth[b2] - h):   # the best 3-block segmentation's boundaries
            e.axvline(edge, color="white", lw=0.8, ls=(0, (3, 2)))
            e.axhline(edge, color="white", lw=0.8, ls=(0, (3, 2)))
        e.set_xlabel("depth (%) →", fontsize=8.5, color=MUTED)
        e.tick_params(labelsize=7.5, colors=MUTED)
        if i == 0:
            e.set_ylabel("depth (%) →", fontsize=8.5, color=MUTED)
    heat[0].text(0, 1.28, "E  Layer-by-layer linear CKA of the lens", transform=heat[0].transAxes,
                 fontsize=10.5, color=INK, va="bottom")
    if im is not None:
        fig.colorbar(im, ax=heat, shrink=0.7, label="linear CKA")
    fig.suptitle("The J-lens itself across models: DeepSeek-V3 vs Qwen3.5", fontsize=13, color=INK, x=0.05, ha="left")
    fig.text(0.05, 0.015, "Depth = layer / (blocks - 1). Lens-only measures: no activations. Qwen3.5 lenses: dallinmj (500 WikiText "
             "passages; 397B on FP8 weights with bf16 compute) and praxagent (24 WikiText prompts of up to 128 tokens, bf16 weights). "
             "V3: shipped T=40 lens; ribbon in A = the two 50-prompt half-fits.\n"
             "D: the two Qwen3.5-397B lenses differ in prompt count, passage selection and weight precision, while V3's halves "
             "differ only in prompts, so the curves are not a matched comparison.\n"
             "E: dashed lines = the 3-block segmentation maximising mean within-block minus between-block CKA (its block score); "
             "decay alone = that score for a matrix keeping only the mean CKA at each layer distance.",
             fontsize=8, color=MUTED)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"lens_comparison.{ext}", dpi=160)
    print(f"wrote {OUT / 'lens_comparison.png'}")


if __name__ == "__main__":
    main()
