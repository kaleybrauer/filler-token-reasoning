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
OFFSETS = (1, 2, 4, 8, 16, 32)
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
    # direct end-labels on the extremes of the ramp; nudge apart where the curves converge
    ends = [(i, float(np.array(series[i])[-1] * (100 if pct else 1))) for i in (0, len(series) - 1)]
    lo_, hi_ = ax.get_ylim() if ax.get_ylim() != (0.0, 1.0) else (min(e[1] for e in ends), max(e[1] for e in ends))
    span = max(hi_ - lo_, 1e-9)
    gap = abs(ends[0][1] - ends[1][1]) / span
    for n, (idx, yv) in enumerate(ends):
        dy = 0 if gap > 0.06 else (6 if n == 1 else -6)
        ax.annotate(labels[idx], xy=(x[-1], yv), xytext=(4, dy),
                    textcoords="offset points", fontsize=8, color=cols[idx],
                    va="center", ha="left", annotation_clip=False)
    ax.text(1.005, 1.075, legend_title, transform=ax.transAxes, fontsize=7.8,
            color=MUTED, ha="left", va="bottom")
    if pct:
        ax.yaxis.set_major_formatter(lambda v, p: f"{v:.0f}%")


def band_between(ax, grids, pct=False):
    for g in grids:
        ax.fill_between(g[0], np.minimum(g[1], g[2]) * (100 if pct else 1),
                        np.maximum(g[1], g[2]) * (100 if pct else 1),
                        color="#9aa7b4", alpha=0.20, lw=0, zorder=1)


def ribbon(ax, halves, key, pct=False):
    """Shade between the two half-corpus curves for one series, on a common depth grid."""
    if len(halves) != 2:
        return
    xs = []
    for h in halves:
        rows = sorted(h["per_layer"], key=lambda z: z["layer"])
        xs.append((np.array([r["depth_0_100"] for r in rows], float),
                   np.array([r[key] for r in rows], float)))
    g = np.union1d(xs[0][0], xs[1][0])
    y0, y1 = np.interp(g, *xs[0]), np.interp(g, *xs[1])
    sc = 100 if pct else 1
    ax.fill_between(g, np.minimum(y0, y1) * sc, np.maximum(y0, y1) * sc,
                    color="#7d8b99", alpha=0.28, lw=0, zorder=2)


def our_figure(sig, rd, sig_halves, rd_halves, cka, out, rd_logit=None):
    """sig: lens-only signatures (panel d). rd: WikiText readouts (panels a-c). rd_logit: the same readouts with
    J = I (workspace_readouts.py --logit), drawn dashed on panels (a)-(c): what the residual stream shows unaided."""
    LOGIT_LS = (0, (3, 2))

    def logit_arm(ax, key, pct=False):
        if rd_logit is None:
            return
        rows = sorted(rd_logit["per_layer"], key=lambda z: z["layer"])
        lx = [r["depth_0_100"] for r in rows if r["layer"] < rd_logit["n_layers_total"] - 1]
        ly = [r[key] * (100 if pct else 1) for r in rows if r["layer"] < rd_logit["n_layers_total"] - 1]
        ax.plot(lx, ly, color=INK, lw=1.3, ls=LOGIT_LS, zorder=4)

    def logit_note(ax, text, xy):
        # one note per panel, in a corner the curves leave empty (labels at the line ends collide with them)
        if rd_logit is not None:
            ax.text(*xy, text, transform=ax.transAxes, fontsize=7.8, color=INK, ha="left", va="top", linespacing=1.4)
    def series(src, keys):
        rows = sorted(src["per_layer"], key=lambda z: z["layer"])
        x = np.array([r["depth_0_100"] for r in rows], float)
        return x, [np.array([r[k] for r in rows], float) for k in keys]

    # V3's layer CKA is near-uniform (>=0.95 between all layers up to ~67% depth), so its block segmentation
    # defines no band (separation score ~0.12, boundaries only where the late decline starts). Use it only
    # when the blocks actually separate.
    cka_ok = bool(cka) and cka.get("separation_score", 0) >= 0.3
    if cka and not cka_ok:
        cka = None
    band = tuple(cka["band_depth"]) if cka_ok else PAPER_BAND
    band_note = ("V3 workspace band from its own layer-CKA block structure "
                 f"({band[0]:.0f}–{band[1]:.0f}%); dashed lines: Sonnet 4.5's band "
                 "(37.5–91.7%)") if cka_ok else \
                ("shaded band: the workspace layers reported for Sonnet 4.5 (37.5–91.7%), for comparison; "
                 "V3's own layer CKA is near-uniform (>=0.95 among layers 0–40) and defines no band")

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.6))
    fig.patch.set_facecolor("white")
    for ax in axes.ravel():
        ax.set_facecolor("white")

    def frame(ax, title, ylabel):
        style(ax, title, ylabel, band)
        if cka:
            for e in PAPER_BAND:
                ax.axvline(e, color="#9aa7b4", lw=1, ls=(0, (4, 3)), zorder=1)

    ax = axes[0, 0]
    frame(ax, "(a)  J-lens next token prediction accuracy", "Top-k accuracy →")
    x, ys = series(rd, [f"nexttok_top{k}" for k in TOPKS])
    ribbon(ax, rd_halves, "nexttok_top8", pct=True)
    draw(ax, x, ys, [str(k) for k in TOPKS], "top-k", pct=True)
    logit_arm(ax, "nexttok_top8", pct=True)
    logit_arm(ax, "nexttok_top1", pct=True)
    logit_note(ax, "dashed: logit lens\n(top-8 above, top-1 below)", (0.03, 0.62))
    ax.set_ylim(-3, 103)
    ax.annotate("transition to\nnext-token prediction", xy=(95, 60), xytext=(58, 78),
                fontsize=8, color=INK, linespacing=1.4,
                arrowprops=dict(arrowstyle="->", color=INK, lw=1))

    ax = axes[0, 1]
    frame(ax, "(b)  J-lens unembedding kurtosis", "Excess kurtosis →")
    x, ys = series(rd, [f"kurtosis_p{q}" for q in KURT_PCTS])
    # Clip to a range that keeps the workspace band legible -- close to the published panel's
    # own scale (to ~18) -- and label every point that falls off it, rather than letting the
    # early sensory-layer spikes and the final-layer output flatten the band region.
    KTOP = 16.0
    ax.set_ylim(-1.6, KTOP)
    ribbon(ax, rd_halves, "kurtosis_p50")
    draw(ax, x, [np.minimum(y, KTOP) for y in ys], [str(q) for q in KURT_PCTS], "percentile")
    logit_arm(ax, "kurtosis_p50")
    logit_note(ax, "dashed: logit lens, median", (0.40, 0.42))
    ax.axhline(0, color=AXIS, lw=0.9, zorder=2)
    top = np.array(ys[-1])
    for i in np.where(top > KTOP)[0]:
        ax.annotate(f"{top[i]:.0f}", xy=(x[i], KTOP), xytext=(5, -3), textcoords="offset points",
                    ha="left", va="top", fontsize=7.4, color=MUTED, annotation_clip=False)

    ax = axes[1, 0]
    frame(ax, "(c)  J-lens top-1 autocorrelation", "Δlog p (vs null) →")
    x, ys = series(rd, [f"autocorr_d{o}" for o in OFFSETS])
    ribbon(ax, rd_halves, "autocorr_d1")
    draw(ax, x, ys, [str(o) for o in OFFSETS], "token offset")
    logit_arm(ax, "autocorr_d1")
    logit_note(ax, "dashed: logit lens, offset 1", (0.03, 0.95))
    ax.axhline(0, color=AXIS, lw=0.9, zorder=2)

    ax = axes[1, 1]
    frame(ax, "(d)  J-space dimensionality", "Fraction of dimensions →")
    x, ys = series(sig, [f"eff_dim_{int(v*1000)}" for v in VAR_SHARES])
    ribbon(ax, sig_halves, "eff_dim_900", pct=True)
    draw(ax, x, ys, [f"{v:g}" for v in VAR_SHARES], "variance explained", pct=True)
    ax.set_ylim(-3, 103)

    for ax_ in (axes[0, 1],):
        y_top = ax_.get_ylim()[1]
        ax_.text(np.mean(band), y_top - 0.03 * (y_top - ax_.get_ylim()[0]), "workspace layers",
                 ha="center", va="top", fontsize=8, color=MUTED, zorder=4)

    sub = (f"DeepSeek-V3-0324 (AWQ int4) · J-lens fitted on {sig['n_prompts']} prompts · "
           f"{len(rd['layers'])} evenly spaced of {rd['n_layers_total']} layers · readouts on "
           f"{rd['prompt_span'][1]-rd['prompt_span'][0]} held-out WikiText prompts x "
           f"{rd['n_positions']} positions")
    sub2 = []
    if len(rd_halves) == 2 or len(sig_halves) == 2:
        sub2.append("grey ribbon: two disjoint 50-prompt half-fits")
    if rd_logit is not None:
        sub2.append("dashed: the logit lens (J = I) on the same activations")
    sub += "\n" + " · ".join(sub2)
    fig.suptitle("Quantitative signatures of the workspace's start and end — DeepSeek V3",
                 fontsize=13.5, color=INK, x=0.008, ha="left", y=0.988)
    fig.text(0.008, 0.958, sub, fontsize=8.1, color=MUTED, ha="left", va="top", linespacing=1.45)
    fig.text(0.008, 0.028, band_note + ".\nRibbons shown on one series per panel (top-8, p50, "
             "offset 1, 0.9).", fontsize=7.5, color=MUTED, va="bottom")
    fig.text(0.008, 0.010, "Panel specification follows Figure 28 of Anthropic, 'Verbalizable "
             "Representations Form a Global Workspace in Language Models'; the measures are our "
             "implementations of the published descriptions.", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=[0, 0.045, 0.985, 0.918])
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
    ap.add_argument("--reference", type=Path, default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    def load(name):
        p = REPO / f"outputs/jlens/{name}.json"
        return json.load(open(p)) if p.exists() else None

    sig = load("workspace_signatures_shipped100")
    rd = load("workspace_readouts_shipped100")
    if sig is None or rd is None:
        sys.exit("need workspace_signatures_shipped100.json and workspace_readouts_shipped100.json")
    sig_h = [h for h in (load("workspace_signatures_n50"), load("workspace_signatures_n50b")) if h]
    rd_h = [h for h in (load("workspace_readouts_n50"), load("workspace_readouts_n50b")) if h]
    cka = load("cka_layers")
    our_figure(sig, rd, sig_h, rd_h, cka, OUT / "fig28_v3", rd_logit=load("workspace_readouts_logit"))
    if args.reference and args.reference.exists():
        reference_figure(json.load(open(args.reference)), OUT / "fig28_sonnet45_reference")


if __name__ == "__main__":
    main()
