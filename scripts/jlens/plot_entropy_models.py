"""
plot_entropy_models.py — mean readout entropy across depth, J-lens against logit lens, for Claude Sonnet 4.5 (the
paper's Figure 55 data, not redistributed), DeepSeek-V3 and the two Qwen3.5 models. Held-out WikiText, 12 positions x
100 paragraphs for the open models. Inputs: audit_v2/lead/entropy_by_depth_<model>.json, the paper's appendix spec.

    python scripts/jlens/plot_entropy_models.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_fig28 import AXIS, MUTED, OUT, PAPER_BAND, style
from plot_hero_offset import BLUE, GRAY, DOT, BAND_LIGHT

LEAD = Path("/workspace/jlens_blogpost/audit_v2/lead")
SONNET = Path("/workspace/jlens_blogpost/paper_data/appendix.json")
MODELS = (("Claude Sonnet 4.5 (paper's data)", None), ("DeepSeek-V3", "v3"), ("Qwen3.5-397B", "qwen35_397b_fp8"), ("Qwen3.5-122B", "qwen35_122b"))


def main():
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2), sharey=True)
    for ax, (title, key) in zip(axes, MODELS):
        style(ax, title, "mean readout entropy, nats →" if ax is axes[0] else "", (0, 0))
        ax.axvspan(*PAPER_BAND, color=BAND_LIGHT, zorder=0, lw=0)
        ax.set_xlim(0, 100); ax.set_ylim(0, 12.6)
        ax.set_xlabel("Depth (% of layers) →", fontsize=9, color=MUTED)
        if key is None:
            spec = json.loads(SONNET.read_text())
            p = [q for q in spec["panels"] if "ntropy" in q["title"]][0]
            ser = {q["label"]: np.array(q["y"]) for q in p["series"]}
            x = np.arange(25) / 24 * 100
            ax.plot(x, ser["jacobian"], color=BLUE, lw=2.0, zorder=3)
            ax.plot(x, ser["logit"], color=GRAY, ls=DOT, lw=1.6, zorder=3)
        else:
            d = json.loads((LEAD / f"entropy_by_depth_{key}.json").read_text())
            x = d["depth"]
            ax.plot(x, [r["mean"] for r in d["arms"]["released"]], color=BLUE, lw=2.0, zorder=3)
            ax.plot(x, [r["mean"] for r in d["arms"]["logit"]], color=GRAY, ls=DOT, lw=1.6, zorder=3)
    axes[0].text(4, 5.4, "J-lens", fontsize=8, color=BLUE)
    axes[0].text(55, 7.8, "logit lens", fontsize=8, color=GRAY)
    fig.tight_layout(w_pad=1.2)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"entropy_models.{ext}", dpi=170, facecolor="white")
    print(f"wrote {OUT / 'entropy_models.png'}")


if __name__ == "__main__":
    main()
