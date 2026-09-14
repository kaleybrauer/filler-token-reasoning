"""
plot_fig28.py — recreate the paper's Figure 28 from our DeepSeek V3 lens, matching
its panel specification series for series.

The published figure is JavaScript-rendered from a small spec file; the panels are

  (a) J-lens next token prediction accuracy   series: top-k  1,2,4,8,16,32,64,128
  (b) J-lens unembedding kurtosis             series: percentile 1,10,25,50,75,90,99
  (c) J-lens top-1 autocorrelation            series: token offset 1,2,4,8,16,32
  (d) J-space dimensionality                  series: variance explained .9 .95 .98 .99 .995

with a viridis ordinal ramp per panel, a shaded workspace band, and one annotation
each. We reproduce (a), (b) and (d). Panel (c) needs the top-1 readout at many
positions of generic text; our cached activations are one readout position per item,
and our own multi-position extractions are filler-token prompts where repetition
would inflate autocorrelation by itself, so it is drawn empty rather than faked.

With --reference <spec.json> the published Sonnet 4.5 panels are rendered through the
same code into a second file, so the two can be set side by side. That file is the
paper's own and is not redistributed here; fetch it from the article's data directory.

    python scripts/jlens/plot_fig28.py --reference outputs/jlens/reference_fig28_sonnet45.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/jlens_workspace"
INK, MUTED, GRID, AXIS = "#1b2733", "#6E8091", "#eef2f6", "#c3ced9"
BAND_C = "#dfe6ee"
TOPKS = (1, 2, 4, 8, 16, 32, 64, 128)
KURT_PCTS = (1, 10, 25, 50, 75, 90, 99)
VAR_SHARES = (0.9, 0.95, 0.98, 0.99, 0.995)
PAPER_BAND = (9 / 24 * 100, 22 / 24 * 100)      # band [9,22] of 25 reindexed layers


def viridis(n):
    return [cm.viridis(i / (n - 1) if n > 1 else 0.5) for i in range(n)]


def style(ax, title, ylabel, band):
    ax.axvspan(*band, color=BAND_C, zorder=0, lw=0)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Layer (reindexed) →", fontsize=9, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=8)
    ax.tick_params(labelsize=8.5, colors=MUTED, length=3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def draw(ax, x, series, labels, legend_title, pct=False):
    cols = viridis(len(series))
    for y, c in zip(series, cols):
        ax.plot(x, np.array(y) * (100 if pct else 1), color=c, lw=1.9,
                solid_capstyle="round", zorder=3)
    # direct end-labels on the extremes of the ramp, legend title above them
    for idx, va in ((0, "center"), (len(series) - 1, "center")):
        y = np.array(series[idx]) * (100 if pct else 1)
        ax.annotate(labels[idx], xy=(x[-1], y[-1]), xytext=(4, 0),
                    textcoords="offset points", fontsize=8, color=cols[idx],
                    va=va, ha="left", annotation_clip=False)
    ax.text(1.005, 1.02, legend_title, transform=ax.transAxes, fontsize=7.8,
            color=MUTED, ha="left", va="bottom")
    if pct:
        ax.yaxis.set_major_formatter(lambda v, p: f"{v:.0f}%")


def band_between(ax, grids, pct=False):
    for g in grids:
        ax.fill_between(g[0], np.minimum(g[1], g[2]) * (100 if pct else 1),
                        np.maximum(g[1], g[2]) * (100 if pct else 1),
                        color="#9aa7b4", alpha=0.20, lw=0, zorder=1)


def halves_band(main_x, halves, key, pct):
    if len(halves) != 2:
        return []
    xs = [(np.array([r["depth_0_100"] for r in sorted(h["per_layer"], key=lambda z: z["layer"])]),
           np.array([r[key] for r in sorted(h["per_layer"], key=lambda z: z["layer"])], float))
          for h in halves]
    grid = np.union1d(xs[0][0], xs[1][0])
    return [(grid, np.interp(grid, *xs[0]), np.interp(grid, *xs[1]))]


def our_figure(m, halves, out):
    rows = sorted(m["per_layer"], key=lambda z: z["layer"])
    x = np.array([r["depth_0_100"] for r in rows])
    get = lambda k: np.array([r[k] for r in rows], float)

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.4))
    fig.patch.set_facecolor("white")
    for a in axes.ravel():
        a.set_facecolor("white")

    ax = axes[0, 0]
    style(ax, "(a)  J-lens next token prediction accuracy", "Top-k accuracy →", PAPER_BAND)
    band_between(ax, halves_band(x, halves, "nexttok_top1", True), True)
    draw(ax, x, [get(f"nexttok_top{k}") for k in TOPKS], [str(k) for k in TOPKS], "top-k", pct=True)
    ax.set_ylim(-3, 103)
    ax.annotate("transition to\nnext-token prediction", xy=(97, 55), xytext=(66, 72),
                fontsize=8, color=INK, linespacing=1.4,
                arrowprops=dict(arrowstyle="->", color=INK, lw=1))

    ax = axes[0, 1]
    style(ax, "(b)  J-lens unembedding kurtosis", "Excess kurtosis →", PAPER_BAND)
    ku = [get(f"kurtosis_p{q}") for q in KURT_PCTS]
    band_between(ax, halves_band(x, halves, "kurtosis_p50", False))
    draw(ax, x, ku, [str(q) for q in KURT_PCTS], "percentile")
    hi = max(float(np.percentile(s, 97)) for s in ku)
    ax.set_ylim(min(float(s.min()) for s in ku) - 0.3, hi * 1.35 + 0.3)
    pk = max((float(s.max()), float(x[int(np.argmax(s))]), q) for s, q in zip(ku, KURT_PCTS))
    if pk[0] > hi * 1.35:
        ax.annotate(f"p{pk[2]} peaks at {pk[0]:.0f}\n(off scale, depth {pk[1]:.0f}%)",
                    xy=(pk[1], ax.get_ylim()[1]), xytext=(pk[1] + 6, ax.get_ylim()[1] * 0.72),
                    fontsize=7.6, color=MUTED, linespacing=1.4,
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9))
    ax.axhline(0, color=AXIS, lw=0.9, zorder=2)

    ax = axes[1, 0]
    style(ax, "(c)  J-lens top-1 autocorrelation", "Δlog p (vs null) →", PAPER_BAND)
    ax.set_ylim(0, 1); ax.set_yticks([])
    ax.text(50, 0.52, "not measured", ha="center", fontsize=9.5, color=INK)
    ax.text(50, 0.36, "needs the top-1 readout at many positions of generic text.\n"
                      "Our cached activations are one readout position per item, and our\n"
                      "multi-position extractions are filler prompts, where repetition\n"
                      "would inflate autocorrelation on its own.",
            ha="center", va="top", fontsize=7.8, color=MUTED, linespacing=1.6)

    ax = axes[1, 1]
    style(ax, "(d)  J-space dimensionality", "Fraction of dimensions →", PAPER_BAND)
    band_between(ax, halves_band(x, halves, "eff_dim_900", True), True)
    draw(ax, x, [get(f"eff_dim_{int(v*1000)}") for v in VAR_SHARES],
         [f"{v:g}" for v in VAR_SHARES], "variance explained", pct=True)
    ax.set_ylim(-3, 103)
    ax.annotate("J-space effective rank\ncollapses pre-workspace", xy=(14, 4),
                xytext=(24, 42), fontsize=8, color=INK, linespacing=1.4,
                arrowprops=dict(arrowstyle="->", color=INK, lw=1))

    axes[0, 1].text(np.mean(PAPER_BAND), axes[0, 1].get_ylim()[1], "workspace layers",
                    ha="center", va="bottom", fontsize=8, color=MUTED)

    sub = (f"DeepSeek-V3-0324 (AWQ int4) · J-lens fitted on {m['n_prompts']} prompts · "
           f"{m['n_layers_total']} layers, d={m['d_model']} · readout statistics over "
           f"{m['n_eval_states']} cached activations")
    if len(halves) == 2:
        sub += " · grey ribbon: two disjoint 50-prompt half-fits"
    fig.suptitle("Quantitative signatures of the workspace's start and end — DeepSeek V3",
                 fontsize=13.5, color=INK, x=0.008, ha="left", y=0.986)
    fig.text(0.008, 0.940, sub, fontsize=8.4, color=MUTED, ha="left")
    fig.text(0.008, 0.026, "Shaded band: the workspace layers reported for Claude Sonnet 4.5 "
                           "(reindexed depth 37.5–91.7%), for comparison. Panels follow the series "
                           "specification of Figure 28 of Anthropic,", fontsize=7.5, color=MUTED)
    fig.text(0.008, 0.008, "'Verbalizable Representations Form a Global Workspace in Language "
                           "Models'; the measures are our implementations of the published "
                           "descriptions, since the reference repository ships no code for them.",
             fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=[0, 0.045, 0.985, 0.928])
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix("." + ext), dpi=200, facecolor="white")
    plt.close(fig)
    print("wrote", out.with_suffix(".png"))


def reference_figure(spec, out):
    """The published Sonnet 4.5 panels, drawn with the same code for side-by-side use."""
    n = spec["n_layers"]
    blo, bhi = [b / (n - 1) * 100 for b in spec["band"]]
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.4))
    fig.patch.set_facecolor("white")
    for ax, p in zip(axes.ravel(), spec["panels"]):
        ax.set_facecolor("white")
        style(ax, "  ".join(p["title"].split(" ", 1)), p["ylabel"], (blo, bhi))
        xs = np.array(p["series"][0]["x"], float) / (n - 1) * 100
        pct = "accuracy" in p["ylabel"].lower() or "fraction" in p["ylabel"].lower()
        draw(ax, xs, [s["y"] for s in p["series"]], [s["label"] for s in p["series"]],
             p.get("legend_title", ""), pct=pct)
    fig.suptitle("Quantitative signatures of the workspace's start and end — "
                 "Claude Sonnet 4.5 (published)", fontsize=13.5, color=INK, x=0.008,
                 ha="left", y=0.986)
    fig.text(0.008, 0.940, "Figure 28 of Anthropic, 'Verbalizable Representations Form a "
                           "Global Workspace in Language Models', redrawn from the article's "
                           "own figure data for comparison.", fontsize=8.4, color=MUTED)
    fig.tight_layout(rect=[0, 0.01, 0.985, 0.928])
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix("." + ext), dpi=200, facecolor="white")
    plt.close(fig)
    print("wrote", out.with_suffix(".png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="shipped100")
    ap.add_argument("--halves", default="n50,n50b")
    ap.add_argument("--reference", type=Path, default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    def load(lbl):
        p = REPO / f"outputs/jlens/workspace_signatures_{lbl}.json"
        return json.load(open(p)) if p.exists() else None

    m = load(args.label)
    if m is None:
        sys.exit(f"no workspace_signatures_{args.label}.json yet")
    if "nexttok_top128" not in m["per_layer"][0]:
        sys.exit("that run predates the full Figure-28 series — recompute first")
    halves = [h for h in (load(l) for l in args.halves.split(",")) if h is not None]
    our_figure(m, halves, OUT / "fig28_v3")
    if args.reference and args.reference.exists():
        reference_figure(json.load(open(args.reference)), OUT / "fig28_sonnet45_reference")


if __name__ == "__main__":
    main()
