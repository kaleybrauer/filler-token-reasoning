"""
plot_fig28.py — recreate the paper's Figure 28 ("Quantitative signatures of the
workspace's start and end") from our DeepSeek V3 lens.

The paper's four panels, and what we can supply:
  (a) next-token prediction accuracy: fraction of positions where any top-k J-lens
      token matches the model's top-1 prediction                      -- HAVE
  (b) excess kurtosis of the J-lens readout (logit distribution over the vocabulary
      at one (position, layer), across a set of activations)          -- HAVE
  (c) autocorrelation of the top-1 lens token across positions vs a position-shuffled
      null                                                            -- NOT MEASURED
      Needs many positions of generic text; our cached activations are a single
      readout position per item. Our own multi-position extractions are filler-token
      prompts, where repeated filler would inflate autocorrelation on its own, so
      they are not a substitute. Drawn as an empty panel rather than omitted.
  (d) fraction of residual-stream dimensions needed for a given share of variance
      across the J-lens vectors                                       -- HAVE

If the two half-corpus runs exist, each panel also gets a band spanned by the two
50-prompt half-lenses: the within-model fitting variability at half the corpus, which
is the floor any feature has to clear to be worth interpreting.

    python scripts/jlens/plot_fig28.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/jlens_workspace"
S1, S2, INK, MUTED, BAND = "#2a78d6", "#eb6834", "#1b2733", "#6E8091", "#dbe4ec"
PAPER_BAND = (38, 92)      # workspace region reported for Sonnet 4.5


def load(label):
    p = REPO / f"outputs/jlens/workspace_signatures_{label}.json"
    return json.load(open(p)) if p.exists() else None


def series(d, key):
    r = sorted(d["per_layer"], key=lambda x: x["layer"])
    return np.array([x["depth_0_100"] for x in r]), np.array([x[key] for x in r], float)


def band(ax, halves, key):
    """Shade between the two half-corpus curves, interpolated onto a common axis."""
    if len(halves) != 2:
        return None
    xs = [series(h, key) for h in halves]
    grid = np.union1d(xs[0][0], xs[1][0])
    ys = [np.interp(grid, x, y) for x, y in xs]
    ax.fill_between(grid, np.minimum(*ys), np.maximum(*ys), color=S1, alpha=0.16,
                    linewidth=0, zorder=1)
    return grid


def panel(ax, letter, title, ylabel, main, halves, key, ylim=None, pct=False, extra=None):
    ax.axvspan(*PAPER_BAND, color=BAND, zorder=0)
    if main is not None:
        band(ax, halves, key)
        x, y = series(main, key)
        ax.plot(x, y * (100 if pct else 1), color=S1, lw=2, zorder=3, solid_capstyle="round")
        if extra:
            x2, y2 = series(main, extra[0])
            ax.plot(x2, y2 * (100 if pct else 1), color=S2, lw=2, zorder=3,
                    solid_capstyle="round")
    ax.set_xlim(0, 100)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlabel("depth through the model (%)", fontsize=9, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.set_title(f"({letter})  {title}", fontsize=10.5, color=INK, loc="left", pad=8)
    ax.tick_params(labelsize=8.5, colors=MUTED, length=3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3ced9")
    ax.grid(axis="y", color="#eef2f6", lw=0.8, zorder=0)
    ax.set_axisbelow(True)


def main():
    m = load("shipped100")
    if m is None:
        sys.exit("no workspace_signatures_shipped100.json yet")
    halves = [h for h in (load("n50"), load("n50b")) if h is not None]
    topk = m.get("topk", 10)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    fig.patch.set_facecolor("white")
    for a in axes.ravel():
        a.set_facecolor("white")

    panel(axes[0, 0], "a", f"Next-token prediction accuracy (top-{topk})",
          "% of items", m, halves, f"nexttok_top{topk}", ylim=(-3, 100), pct=True)
    # The early sensory layers spike far off the scale that the rest of the curve needs;
    # clip to the informative range and disclose the peak rather than let it flatten
    # the structure this panel exists to show.
    _, ku = series(m, "kurtosis")
    kx, _ = series(m, "kurtosis")
    peak_i = int(np.argmax(ku))
    panel(axes[0, 1], "b", "Excess kurtosis of the J-lens readout",
          "excess kurtosis", m, halves, "kurtosis", ylim=(-1.3, 2.3))
    axes[0, 1].axhline(0, color="#c3ced9", lw=0.9, zorder=2)
    axes[0, 1].annotate(f"peak {ku[peak_i]:.1f} at depth {kx[peak_i]:.0f}% (off scale)",
                        xy=(kx[peak_i], 2.3), xytext=(kx[peak_i] + 5, 1.75),
                        fontsize=8, color=S2,
                        arrowprops=dict(arrowstyle="->", color=S2, lw=1.1))

    ax = axes[1, 0]
    ax.axvspan(*PAPER_BAND, color=BAND, zorder=0)
    ax.text(50, 0.5, "(c) not measured\n\nautocorrelation of the top-1 token across\n"
                     "positions needs many positions of generic text;\n"
                     "our cached activations are one readout position\n"
                     "per item, and our multi-position extractions are\n"
                     "filler prompts, where repetition would inflate it",
            ha="center", va="center", fontsize=8.5, color=MUTED, linespacing=1.6)
    ax.set_xlim(0, 100); ax.set_ylim(0, 1); ax.set_yticks([])
    ax.set_xlabel("depth through the model (%)", fontsize=9, color=MUTED)
    ax.set_title("(c)  Top-1 token autocorrelation across positions", fontsize=10.5,
                 color=INK, loc="left", pad=8)
    ax.tick_params(labelsize=8.5, colors=MUTED, length=3)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#c3ced9")

    panel(axes[1, 1], "d", "J-space dimensionality",
          "fraction of residual-stream dims", m, halves, "eff_dim_90",
          ylim=(-0.03, 0.72), extra=("eff_dim_50",))
    axes[1, 1].text(97, 0.645, "90% of variance", ha="right", fontsize=8.5, color=S1)
    axes[1, 1].text(97, 0.155, "50% of variance", ha="right", fontsize=8.5, color=S2)

    sub = (f"DeepSeek-V3-0324 (AWQ int4), J-lens fitted on {m['n_prompts']} prompts, "
           f"{m['n_layers_total']} layers, d={m['d_model']}; readout statistics over "
           f"{m['n_eval_states']} cached activations")
    if halves:
        sub += "; shaded curve band = two disjoint 50-prompt half-fits"
    fig.suptitle("Quantitative signatures of the workspace's start and end — DeepSeek V3",
                 fontsize=13, color=INK, x=0.011, ha="left", y=0.985)
    fig.text(0.011, 0.938, sub, fontsize=8.5, color=MUTED, ha="left")
    fig.text(0.011, 0.028, "Grey band: the workspace region reported for Claude Sonnet 4.5 "
                           "(depth 38–92), for comparison. Recreation of Figure 28 of "
                           "Anthropic, 'Verbalizable Representations Form a Global",
             fontsize=7.6, color=MUTED, ha="left")
    fig.text(0.011, 0.010, "Workspace in Language Models'. The panel measures are our "
                           "implementations of the published descriptions; the repository "
                           "ships no code for them.",
             fontsize=7.6, color=MUTED, ha="left")
    fig.tight_layout(rect=[0, 0.048, 1, 0.925])
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig28_v3.{ext}", dpi=200, facecolor="white")
    print("wrote", OUT / "fig28_v3.png", "and .pdf",
          f"(halves: {len(halves)})")


if __name__ == "__main__":
    main()
