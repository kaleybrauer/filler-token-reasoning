"""
plot_lens_cka.py — how much of each lens's layer-by-layer CKA structure survives two checks.

Rows (columns: DeepSeek-V3, Qwen3.5-397B, Qwen3.5-122B):
  1  the lens as fitted (V3 100 prompts, Qwen 500)
  2  a smaller fit of the same model: V3's 25-prompt subset, Qwen3.5-397B's 24-prompt praxagent lens
  3  each layer's top output singular direction projected out of its Jacobian before comparing layers
     (cka_layers.py --remove-top-output-dirs 1, a diagnostic, not a lens)
Right: D mean CKA against layer distance for each fitted and smaller lens; E the share of each layer's |J|_F^2 in
its top output direction.

Every 4th V3 layer and every 3rd Qwen layer. CKA between two layers depends only on those two Jacobians, so a
subsampled full matrix equals a run on the subset. Each title gives the 3-block score (mean within-block minus
between-block CKA, segmentation drawn dashed) and the same score for decay alone: a matrix that keeps only the
mean CKA at each layer distance. Smooth decay by itself earns a positive score.

    python scripts/jlens/plot_lens_cka.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cka_layers import decay_only, three_blocks
from plot_fig28 import AXIS, GRID, INK, MUTED, OUT

REPO = Path(__file__).resolve().parents[2]
J, Q = REPO / "outputs/jlens", REPO / "outputs/jlens/qwen35"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
CKA_MIN = 0.3
COLS = [  # label, colour, decoder blocks, stride, {row: [candidate files, first existing wins]}
    ("DeepSeek-V3", BLUE, 61, 4, {"fitted": [J / "cka_layers_v3_every4.json", J / "cka_layers.json"],
                                  "small": [J / "cka_layers_v3_every4_n25.json"],
                                  "no_top1": [J / "cka_layers_v3_every4_no_top1.json"]}),
    ("Qwen3.5-397B", ORANGE, 60, 3, {"fitted": [Q / "cka_qwen35_397b.json"],
                                     "small": [Q / "cka_qwen35_397b_n24.json"],
                                     "no_top1": [Q / "cka_qwen35_397b_n500_every3_no_top1.json"]}),
    ("Qwen3.5-122B", AQUA, 48, 3, {"fitted": [Q / "cka_qwen35_122b_every3.json", Q / "cka_qwen35_122b.json"],
                                   "small": [],
                                   "no_top1": [Q / "cka_qwen35_122b_every3_no_top1.json"]}),
]
ROWS = [("fitted", "as fitted (V3 100 prompts, Qwen 500)"),
        ("small", "smaller fit (V3 25 prompts, Qwen3.5-397B 24)"),
        ("no_top1", "top output direction removed")]


def load_matrix(paths, stride):
    for p in paths:
        if p.exists():
            ck = json.loads(p.read_text())
            C, layers = np.array(ck["cka"]), np.array(ck["layers"])
            keep = np.flatnonzero(layers % stride == 0)
            return C[np.ix_(keep, keep)], layers[keep], ck
    return None, None, None


def style_line(ax):
    ax.tick_params(labelsize=7.5, colors=MUTED)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    for s_ in ("left", "bottom"):
        ax.spines[s_].set_color(AXIS)
    ax.grid(axis="y", color=GRID, lw=0.8)


def main():
    fig = plt.figure(figsize=(15, 14.5))
    gs = fig.add_gridspec(3, 4, left=0.06, right=0.97, top=0.88, bottom=0.09, wspace=0.35, hspace=0.62,
                          width_ratios=[1, 1, 1, 1.15])
    im = None
    shares, lags = [], []
    for r, (key, row_label) in enumerate(ROWS):
        for c, (label, col, n_total, stride, files) in enumerate(COLS):
            ax = fig.add_subplot(gs[r, c])
            C, layers, ck = load_matrix(files[key], stride)
            if C is None:
                ax.set_title(label, fontsize=9.5, color=INK, loc="left")
                ax.text(0.5, 0.5, "no such lens" if not files[key] else "pending", ha="center", va="center",
                        color=MUTED, transform=ax.transAxes, fontsize=9)
                ax.axis("off")
            else:
                score, b1, b2 = three_blocks(C)
                null = three_blocks(decay_only(C))[0]
                depth = 100 * layers / (n_total - 1)
                h = (depth[1] - depth[0]) / 2
                im = ax.imshow(C, vmin=CKA_MIN, vmax=1, cmap="Blues", origin="lower",
                               extent=(depth[0] - h, depth[-1] + h, depth[0] - h, depth[-1] + h))
                for edge in (depth[b1] - h, depth[b2] - h):
                    ax.axvline(edge, color="white", lw=0.8, ls=(0, (3, 2)))
                    ax.axhline(edge, color="white", lw=0.8, ls=(0, (3, 2)))
                ax.set_title(f"{label}\nblock score {score:.2f} · decay alone {null:.2f}", fontsize=9.5, color=INK, loc="left")
                ax.tick_params(labelsize=7.5, colors=MUTED)
                ax.set_xlabel("depth (%) →", fontsize=8.5, color=MUTED)
                if c == 0:
                    ax.set_ylabel("depth (%) →", fontsize=8.5, color=MUTED)
                if key in ("fitted", "small"):
                    lags.append((label, col, key, 100 * stride / (n_total - 1),
                                 np.array([np.diagonal(C, k).mean() for k in range(len(C))])))
                if key == "no_top1" and ck.get("removed_share"):
                    shares.append((label, col, depth, np.array(ck["removed_share"])[np.flatnonzero(np.array(ck["layers"]) % stride == 0)]))
            if c == 0:
                ax.text(0, 1.24, f"{'ABC'[r]}  {row_label}", transform=ax.transAxes, fontsize=11, color=INK, va="bottom")
    if im is not None:
        holder = fig.add_subplot(gs[0, 3])
        holder.axis("off")
        cax = holder.inset_axes([0.0, 0.05, 0.07, 0.85])
        fig.colorbar(im, cax=cax, label="linear CKA")

    ax = fig.add_subplot(gs[1, 3])
    for label, col, key, step, prof in lags:
        ax.plot(step * np.arange(len(prof)), prof, color=col, lw=2 if key == "fitted" else 1.2,
                ls="-" if key == "fitted" else (0, (2, 2)), label=f"{label} {'as fitted' if key == 'fitted' else 'smaller fit'}")
    ax.set_xlim(0, 100)
    ax.set_title("D  mean CKA against layer distance", fontsize=9.5, color=INK, loc="left")
    ax.set_xlabel("layer distance (% of depth) →", fontsize=8.5, color=MUTED)
    style_line(ax)
    ax.legend(fontsize=7, frameon=False, loc="lower left")

    ax = fig.add_subplot(gs[2, 3])
    for label, col, depth, s in shares:
        ax.plot(depth, 100 * s, color=col, lw=2, marker="o", ms=3.5, label=label)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, None)
    ax.set_title("E  share of each layer's |J|² in its\ntop output direction", fontsize=9.5, color=INK, loc="left")
    ax.set_xlabel("depth (%) →", fontsize=8.5, color=MUTED)
    ax.yaxis.set_major_formatter(lambda v, p: f"{v:.0f}%")
    style_line(ax)
    ax.legend(fontsize=7.5, frameon=False)

    fig.suptitle("Layer-by-layer CKA of the J-lens: which structure survives a smaller fit and the removal of one direction",
                 fontsize=12.5, color=INK, x=0.06, ha="left")
    fig.text(0.06, 0.015, "Every 4th V3 layer, every 3rd Qwen layer; depth = layer / (blocks - 1). Dashed lines: the 3-block segmentation "
             "maximising mean within-block minus between-block CKA.\nDecay alone: that score for a matrix keeping only the mean CKA at "
             "each layer distance. Qwen3.5-397B lenses: dallinmj (500 passages, FP8 weights, bf16 compute) and praxagent\n(24 prompts, "
             "bf16 weights), which also differ in passage selection. V3's 25-prompt lens is a subset of the shipped lens's prompts.",
             fontsize=8, color=MUTED)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"lens_cka.{ext}", dpi=160)
    print(f"wrote {OUT / 'lens_cka.png'}")


if __name__ == "__main__":
    main()
