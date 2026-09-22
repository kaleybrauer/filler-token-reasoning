"""
plot_script_gate.py — the script offset in V3's J-lens readouts is the lens carrying the output layer's language
gate, not a property of the residual states.

  A  share of each readout's variance across the vocabulary explained by script class (Latin / Han / other),
     J-lens (solid) vs logit lens (dashed), English WikiText, three models
  B  the output layer's own logits: mean Han logit minus mean Latin logit in units of the logit vector's sd,
     on English and on Chinese prose
  C  share of each layer's |J|_F^2 in its top output direction (the direction that separates Han from Latin
     tokens with the AUC given in the legend)

    python scripts/jlens/plot_script_gate.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_fig28 import AXIS, INK, MUTED, OUT, style

REPO = Path(__file__).resolve().parents[2]
J, Q, QS = REPO / "outputs/jlens", REPO / "outputs/jlens/qwen35", REPO / "outputs/jlens_qwen35"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a97a5"
NO_BAND = (0, 0)
MODELS = [  # label, colour, blocks, depth-analysis dir, top-direction share JSON, script-axis JSON
    ("DeepSeek-V3", BLUE, 61, J, J / "cka_layers_v3_every4_no_top1.json", J / "script_axis.json"),
    ("Qwen3.5-397B", ORANGE, 60, QS / "qwen35_397b_fp8/analysis", Q / "cka_qwen35_397b_n500_every3_no_top1.json", Q / "script_axis_qwen35_397b.json"),
    ("Qwen3.5-122B", AQUA, 48, QS / "qwen35_122b/analysis", Q / "cka_qwen35_122b_every3_no_top1.json", Q / "script_axis_qwen35_122b.json"),
]


def rows(p):
    return sorted(json.loads(p.read_text())["per_layer"], key=lambda r: r["layer"])


def main():
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(18.5, 5.9))
    style(a, "A  Three models, J-lens and logit lens", "Language share of readout variance", NO_BAND)
    style(c, "C  Share of ‖J‖² in the lens's top output direction", "share →", NO_BAND)
    for ax in (a, c):
        ax.set_xlabel("Depth (% of blocks) →", fontsize=13.5, color=MUTED)
    gate = {}
    for label, col, n_total, adir, share_p, axis_p in MODELS:
        for lens, ls in (("shipped", "-"), ("logit", (0, (3, 2)))):
            rs = rows(adir / f"bilingual_depth_{lens}.json")
            a.plot([r["depth_0_100"] for r in rs], [r["eta2_script3"]["p50"] for r in rs], color=col, lw=2 if lens == "shipped" else 1.3,
                   ls=ls, label=f"{label} {'J-lens' if lens == 'shipped' else 'logit lens'}")
        gate[label] = {tag: rows(adir / f"bilingual_depth_logit{tag}.json")[-1]["han_minus_latin_mean_logit_sd"]["p50"] for tag in ("", "_wikizh")}
        ck = json.loads(share_p.read_text())
        ax_js = json.loads(axis_p.read_text())["layers"]
        aucs = [ax_js[k]["script_separation"] for k in ax_js]
        c.plot([100 * L / (n_total - 1) for L in ck["layers"]], [100 * s for s in ck["removed_share"]], color=col, lw=2, marker="o", ms=3.5,
               label=f"{label}: Chinese-vs-Latin AUC {min(aucs):.2f}–{max(aucs):.2f}")
    a.set_ylim(0, 0.58)
    a.legend(fontsize=10, frameon=False, loc="upper right", ncol=1)
    c.yaxis.set_major_formatter(lambda v, p: f"{v:.0f}%")
    c.set_ylim(0, 48)
    c.legend(fontsize=11.5, frameon=False, loc="upper right")

    style(b, "B  The output layer's own logits: Chinese minus Latin", "logit-vector sd →", NO_BAND)
    b.set_xlim(-0.6, 2.6)
    b.set_xticks([0, 1, 2])
    b.set_xticklabels([m[0] for m in MODELS], fontsize=13, color=MUTED)
    b.set_xlabel("")
    w = 0.34
    for i, (label, col, *_rest) in enumerate(MODELS):
        b.bar(i - w / 2, gate[label][""], w, color=col, label="English prose" if i == 0 else None)
        b.bar(i + w / 2, gate[label]["_wikizh"], w, color=col, alpha=0.45, hatch="//", edgecolor=col, lw=0, label="Chinese prose" if i == 0 else None)
        for x, v in ((i - w / 2, gate[label][""]), (i + w / 2, gate[label]["_wikizh"])):
            b.text(x, v + (0.05 if v >= 0 else -0.05), f"{v:+.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=12, color=INK)
    b.axhline(0, color=AXIS, lw=0.9)
    b.set_ylim(-1.9, 1.9)
    b.legend(fontsize=12.5, frameon=False, loc="lower right")

    # no figure title: the post's caption carries it
    for ax in (a, b, c):
        ax._left_title.set_fontsize(15); ax.yaxis.label.set_size(13.5); ax.tick_params(labelsize=13)
    fig.tight_layout(rect=[0, 0.01, 1, 0.99])
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"script_gate.{ext}", dpi=170, facecolor="white")
    print(f"wrote {OUT / 'script_gate.png'}")


if __name__ == "__main__":
    main()
