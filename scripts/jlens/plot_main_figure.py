"""
plot_main_figure.py — the post's first figure, drawn from saved results (no model or lens reads, runs in seconds).

  no title (the post's caption carries the takeaway); one shared legend; three panels:
  (a) at half depth, how many of a readout's top 20 tokens are Chinese-character tokens (released lens, penultimate
      lens, logit lens), with the vocabulary baseline and the majority line marked and the key shares annotated
  (b) script share of readout variance across depth (released, the paired 10-prompt final-target lens as a faint
      control, the penultimate lens, the logit lens with the model's own output)
  (c) the Jacobian's own anatomy: |cosine| between each layer's top direction and the top direction of the last
      layer's own Jacobian (top), and whether that direction separates Chinese from Latin tokens (bottom)

Inputs: results/jlens_workspace/hero_main_panel_a.json (plot_hero_main.py), outputs/jlens/bilingual_depth_{shipped,
target60_same,target59,logit}.json, /workspace/jlens_blogpost/audit_v2/lead/jacobian_anatomy.json.

    python scripts/jlens/plot_main_figure.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

REPO = Path(__file__).resolve().parents[2]
J = REPO / "outputs/jlens"
OUT = REPO / "results/jlens_workspace"
ANATOMY = Path("/workspace/jlens_blogpost/audit_v2/lead/jacobian_anatomy.json")

INK, MUTED, AXIS, GRID = "#1b2733", "#5f6f80", "#b9c4cf", "#edf1f5"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#7d8a97"
BAND = "#f4f6f9"
BAND_X = (37.5, 91.7)          # Sonnet 4.5's workspace band, % depth
SOLID_W, CTRL_W = 2.6, 1.3

plt.rcParams.update({"font.size": 17, "axes.titlesize": 16.5, "axes.labelsize": 17, "xtick.labelsize": 16,
                     "ytick.labelsize": 16, "font.family": "DejaVu Sans"})
NOTE = 16            # annotation text
NOTE_A = 13.5        # panel (a) notes


def style(ax, title, xlabel, ylabel, band=False):
    if band:
        ax.axvspan(*BAND_X, color=BAND, zorder=0, lw=0)
    ax.set_title(title, loc="left", color=INK, pad=16, fontweight="semibold")
    ax.set_xlabel(xlabel, color=MUTED); ax.set_ylabel(ylabel, color=MUTED)
    ax.tick_params(colors=MUTED, length=3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.grid(axis="y", color=GRID, lw=0.9); ax.set_axisbelow(True)


def pct_log_axis(ax, lo, hi):
    ax.set_yscale("log"); ax.set_ylim(lo, hi)
    ticks = [t for t in (1e-3, 1e-2, 1e-1, 1.0) if lo <= t <= hi]
    ax.yaxis.set_major_locator(FixedLocator(ticks)); ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100 * v:g}%"))


def depth_rows(name, key="eta2_script3", upto_output=False):
    rs = sorted(json.loads((J / f"bilingual_depth_{name}.json").read_text())["per_layer"], key=lambda r: r["layer"])
    rs = [r for r in rs if upto_output or r["layer"] < 59]
    return [r["depth_0_100"] for r in rs], [r[key]["p50"] for r in rs]


def check_overlaps(fig):
    """Print any text that leaves the figure or overlaps other text (titles, labels, notes, legend)."""
    fig.canvas.draw(); r = fig.canvas.get_renderer(); W, H = fig.bbox.width, fig.bbox.height
    texts = [t for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text().strip()]
    boxes = [(t.get_text().replace("\n", " ")[:30], t.get_window_extent(r)) for t in texts]
    for name, bb in boxes:
        if bb.x0 < 0 or bb.y0 < 0 or bb.x1 > W or bb.y1 > H:
            print(f"  OUTSIDE: {name!r}")
    for i, (n1, b1) in enumerate(boxes):
        for n2, b2 in boxes[i + 1:]:
            if b1.overlaps(b2):
                print(f"  OVERLAP: {n1!r} x {n2!r}")


def main():
    A = json.loads((OUT / "hero_main_panel_a.json").read_text())
    an = json.loads(ANATOMY.read_text())
    fig = plt.figure(figsize=(14, 7.3))
    gs = GridSpec(2, 3, figure=fig, left=0.07, right=0.975, top=0.744, bottom=0.108, wspace=0.55, hspace=0.80,
                  width_ratios=[1.15, 1.0, 1.0])
    a = fig.add_subplot(gs[:, 0]); b = fig.add_subplot(gs[:, 1])
    c1 = fig.add_subplot(gs[0, 2]); c2 = fig.add_subplot(gs[1, 2])

    # no title or subtitle: the caption carries them. One shared legend above the panels.
    handles = [Line2D([], [], color=BLUE, lw=SOLID_W, label="Final-layer target, 100 prompts"),
               Line2D([], [], color=BLUE, lw=CTRL_W, ls=(0, (4, 2)), alpha=0.55, label="Final-layer target, 10 prompt subset"),
               Line2D([], [], color=ORANGE, lw=SOLID_W, label="Penultimate target, 10 prompt subset"),
               Line2D([], [], color=GRAY, lw=1.8, ls=(0, (1.5, 1.5)), label="Logit lens")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.03, 0.995), ncol=3, frameon=False, fontsize=16.5,
               handlelength=2.2, columnspacing=1.5, handletextpad=0.6)

    # (a) at half depth, readouts become Chinese-heavy
    style(a, "(a) The target layer changes\nthe readouts' language", "Chinese characters in the top 20 tokens", "Share of readouts")
    xs = np.arange(A["k"] + 1)
    n = A["n_activations"]
    series_a = [("logit lens", GRAY, (0, (1.5, 1.5)), 1.8, 1.0),
                ("J-lens, final-layer target, same 10 prompts", BLUE, (0, (4, 2)), CTRL_W, 0.55),
                ("J-lens, penultimate target", ORANGE, "-", SOLID_W, 1.0),
                ("J-lens, final-layer target", BLUE, "-", SOLID_W, 1.0)]
    for name, col, ls, lw, alpha in series_a:
        if name not in A["hist"]:                     # older panel data has three series
            continue
        a.plot(xs, np.array(A["hist"][name]) / n, color=col, ls=ls, lw=lw, alpha=alpha, drawstyle="steps-mid",
               zorder=2 if alpha < 1 else 3)
    base = A["vocab_han_share"] * A["k"]
    a.axvline(base, color=AXIS, lw=1.1, ls=(0, (3, 2)), zorder=1)
    a.text(base - 0.3, 0.395, "vocabulary baseline", fontsize=NOTE_A, color=MUTED, va="top", ha="right", rotation=90)
    a.axvline(10.5, color=AXIS, lw=1.1, ls=(0, (3, 2)), zorder=1)
    a.text(10.75, 0.39, "majority\nChinese →", fontsize=NOTE_A, color=MUTED, va="top")
    s = A["summary"]
    fin, pen = s["J-lens, final-layer target"], s["J-lens, penultimate target"]
    all20 = A["hist"]["J-lens, final-layer target"][-1] / n
    a.text(18.8, all20 - 0.012, f"{all20:.0%} are all\nChinese", fontsize=NOTE_A, color=INK, ha="right", va="top")
    ctrl = s.get("J-lens, final-layer target, same 10 prompts")
    note = f"Mostly Chinese:\n{fin['frac_majority_han']:.0%} final target\n"
    if ctrl is not None:
        note += f"{ctrl['frac_majority_han']:.0%} final, 10 prompts\n"
    note += f"{pen['frac_majority_han']:.0%} penultimate"
    a.text(11.0, 0.315, note, fontsize=NOTE_A, color=INK, va="top", linespacing=1.4)
    a.set_xlim(-0.5, A["k"] + 0.5); a.set_ylim(0, 0.40)
    a.set_xticks([0, 5, 10, 15, 20])
    a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))

    # (b) the script shift at every depth
    style(b, "(b) The shift runs\nthrough every layer", "Depth (% of layers)",
          "Language share of readout variance", band=True)
    x, y = depth_rows("target60_same"); b.plot(x, y, color=BLUE, lw=CTRL_W, ls=(0, (4, 2)), alpha=0.55, zorder=2)
    x, y = depth_rows("shipped"); b.plot(x, y, color=BLUE, lw=SOLID_W, zorder=4)
    x, y = depth_rows("target59"); b.plot(x, y, color=ORANGE, lw=SOLID_W, zorder=4)
    x, y = depth_rows("logit", upto_output=True); b.plot(x, y, color=GRAY, lw=1.8, ls=(0, (1.5, 1.5)), zorder=3)
    b.plot([x[-1]], [y[-1]], marker="D", ms=6.5, color=GRAY, mec="white", mew=0.9, zorder=5, clip_on=False)
    b.annotate("model's own\noutput", xy=(x[-1], y[-1]), xytext=(64, 0.075), fontsize=NOTE, color=MUTED, ha="center", va="center",
               arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.9))
    pct_log_axis(b, 1e-3, 1.0); b.set_xlim(0, 100)

    # (c) the mechanism: one Jacobian direction, and it separates the scripts
    rows = an["lenses"]
    for ax, key, title, ylabel in ((c1, "cos_with_last_block_top", "(c) The direction comes\nfrom the last layer…",
                                    "Cosine with the last\nlayer's top direction"),
                                   (c2, "auc_script_top1", "…and it separates\nthe two languages", "AUC\n(0.5 = chance)")):
        style(ax, title, "", ylabel, band=True)
        for name, col, lw, ls, alpha, z in (("target60_same", BLUE, CTRL_W, (0, (4, 2)), 0.55, 2), ("released", BLUE, SOLID_W, "-", 1, 4),
                                            ("target59", ORANGE, SOLID_W, "-", 1, 4)):
            ax.plot([100 * r["layer"] / 60 for r in rows[name]], [r[key] for r in rows[name]], color=col, lw=lw, ls=ls, alpha=alpha, zorder=z)
        ax.set_xlim(0, 100)
    c2.set_xlabel("Depth (% of layers)", color=MUTED)
    c1.set_ylim(0, 1.04); c1.set_yticks([0, 0.5, 1.0])
    c2.set_ylim(0.45, 1.04); c2.set_yticks([0.5, 0.75, 1.0])
    c2.axhline(0.5, color=AXIS, lw=0.9, zorder=1)

    for ax in (a, b):                      # (a) and (b) shorter than the (c) stack and dropped to its bottom,
        pos = ax.get_position()            # with the lost height added to the title pad so the titles do not move
        h = 0.92 * pos.height
        ax.set_position([pos.x0, pos.y0, pos.width, h])
        ax.set_title(ax.get_title(loc="left"), loc="left", color=INK, fontweight="semibold",
                     pad=16 + (pos.height - h) * fig.get_figheight() * 72)
    check_overlaps(fig)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"main_figure.{ext}", dpi=210, facecolor="white")
    print(f"wrote {OUT / 'main_figure.png'}")


if __name__ == "__main__":
    main()
