"""
Paper-ready 3×2 heatmap figure for 2-fact decode at every (layer, position)
on dots_10:
    columns = correct (left) | wrong (right)
    rows    = A1 / A2 / A1+A2 within ±5

No titles on individual panels. Larger font for paper print.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


VARIABLE_LABELS = [
    ("A1",   r"$A_1$"),
    ("A2",   r"$A_2$"),
    ("A1A2", r"$A_1{+}A_2$"),
]

METRIC_SUFFIX = {
    "exact":   ("frac_{var}_exact",   "% exact match"),
    "within5": ("frac_{var}_within5", "% within ±5"),
}

POS_TICK_STEP = 2  # label every Nth position


def pos_label(p: str) -> str:
    if p.startswith("pos_"):
        return p.split("_")[1].lstrip("0") or "0"
    return p


def matrix_for(d: dict, metric: str, min_layer: int = 0) -> tuple[np.ndarray, list[int], list[str]]:
    layers = [l for l in d["_layers"] if l >= min_layer]
    positions = d["_positions"]
    M = np.full((len(layers), len(positions)), np.nan)
    for j, pos in enumerate(positions):
        for i, layer in enumerate(layers):
            entry = d.get(pos, {}).get(str(layer))
            if entry is None:
                continue
            M[i, j] = entry[metric] * 100
    return M, layers, positions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--correct-json", type=Path,
                    default=Path("results/unsupervised_decode_2fact_allpos/decode_2fact_dots_10.json"))
    ap.add_argument("--incorrect-json", type=Path,
                    default=Path("results/unsupervised_decode_2fact_allpos/decode_2fact_dots_10_incorrect.json"))
    ap.add_argument("--outfile-stem", type=Path, default=None,
                    help="Default: plotting/plots/decode_2fact_dots_10_correct_vs_wrong_<metric>")
    ap.add_argument("--metric", choices=list(METRIC_SUFFIX), default="exact",
                    help="Color heatmap by exact-match or within-±5 (default: exact)")
    ap.add_argument("--min-layer", type=int, default=0,
                    help="Drop layers below this index (default: 0, no filter)")
    ap.add_argument("--font-size", type=int, default=None,
                    help="Base font size (default: 22, or 40 with --poster). On a "
                         "fixed canvas, larger = text bigger relative to the heatmap.")
    ap.add_argument("--figsize", type=float, nargs=2, default=(23.0, 9.0),
                    metavar=("W", "H"), help="Figure size in inches (default: 23 9)")
    ap.add_argument("--poster", action="store_true",
                    help="Poster preset: large default font (40) so the figure reads "
                         "from a distance. Canvas stays fixed, so the fonts grow "
                         "relative to the heatmap.")
    args = ap.parse_args()
    if args.font_size is None:
        args.font_size = 40 if args.poster else 22
    if args.outfile_stem is None:
        suffix = f"_{args.metric}"
        if args.min_layer > 0:
            suffix += f"_L{args.min_layer}+"
        if args.poster:
            suffix += "_poster"
        args.outfile_stem = Path(
            f"plotting/plots/decode_2fact_dots_10_correct_vs_wrong{suffix}"
        )

    metric_template, cbar_label = METRIC_SUFFIX[args.metric]
    VARIABLES = [(metric_template.format(var=v), label)
                 for v, label in VARIABLE_LABELS]

    plt.rcParams.update({
        "font.size": args.font_size,
        "axes.labelsize": args.font_size,
        "xtick.labelsize": args.font_size - 4,
        "ytick.labelsize": args.font_size - 4,
    })

    correct = json.load(open(args.correct_json))
    wrong   = json.load(open(args.incorrect_json))

    n_correct = correct["_n"]
    n_wrong   = wrong["_n"]
    boundaries = correct.get("_boundaries", {}) or {}
    filler_end = boundaries.get("filler_end_offset")

    # Horizontal layout: rows = Correct / Wrong; columns = A1 / A2 / A1+A2.
    # Canvas size is fixed (independent of font) so a larger --font-size grows the
    # text *relative to* the heatmap — the point of the poster preset.
    # Widen the inter-panel gap with the font so the rightmost x-tick of one panel
    # doesn't run into the leftmost ("0") of the next (e.g. "16" + "0" -> "160").
    f = args.font_size / 22.0
    fig, axes = plt.subplots(
        2, 3, figsize=args.figsize,
        sharex=True, sharey=True,
        gridspec_kw={"hspace": 0.10, "wspace": 0.06 * f},
    )

    row_specs = [(correct, "Correct", n_correct), (wrong, "Wrong", n_wrong)]

    last_im = None
    for row, (data, _label, _n) in enumerate(row_specs):
        for col, (metric, var_label) in enumerate(VARIABLES):
            ax = axes[row, col]
            M, layers, positions = matrix_for(data, metric, min_layer=args.min_layer)
            im = ax.imshow(M, aspect="auto", cmap="RdYlGn",
                           vmin=0, vmax=100, interpolation="nearest")
            last_im = im

            # x ticks: adaptive step so we get ~6-10 labeled positions per panel.
            # Without this, dots_50 / counting_25 / counting_50 panels have so many
            # ticks that the labels overlap into illegibility.
            n_pos = len(positions)
            if n_pos <= 20:
                pos_step = 2
            elif n_pos <= 35:
                pos_step = 5
            elif n_pos <= 70:
                pos_step = 10
            else:
                pos_step = 20
            # Bigger fonts take more room per label, so coarsen the step in step
            # with the font (font 22 -> x1, font 40 -> x2) to keep labels legible.
            pos_step *= max(1, round(args.font_size / 22.0))
            tick_idxs = [i for i, p in enumerate(positions)
                         if int(pos_label(p)) % pos_step == 0]
            ax.set_xticks(tick_idxs)
            ax.set_xticklabels([pos_label(positions[i]) for i in tick_idxs])

            # y ticks every 5 layers (every 10 if many layers)
            tick_step = 10 if len(layers) > 25 else 5
            ax.set_yticks([i for i, l in enumerate(layers) if l % tick_step == 0])
            ax.set_yticklabels([str(l) for l in layers if l % tick_step == 0])

            # filler boundary line
            if filler_end is not None:
                # positions are pos_000..pos_NNN, with pos_index == offset value;
                # find the index of pos with offset == filler_end
                try:
                    fe_idx = positions.index(f"pos_{filler_end:03d}")
                    ax.axvline(fe_idx + 0.5, color="white", linewidth=1.5,
                               linestyle="--", alpha=0.8)
                except ValueError:
                    pass

            # axes labels: only outer
            if col == 0:
                ax.set_ylabel("Layer")
            if row == len(row_specs) - 1:
                ax.set_xlabel("Token offset")

    # Row labels (left side) — Correct / Wrong with n=...
    # Sit further left than the "Layer" ylabel + tick labels so they don't collide.
    # The gap to clear (ylabel + wider tick labels) grows with the font, so push
    # the labels out proportionally. Two text() calls so the count line can differ.
    row_label_x = -0.35 * f
    for row, (_data, label, n) in enumerate(row_specs):
        axes[row, 0].text(row_label_x, 0.55, label,
                          transform=axes[row, 0].transAxes,
                          fontsize=args.font_size - 4, fontweight="normal",
                          ha="center", va="center", rotation=0)
        axes[row, 0].text(row_label_x, 0.40, f"n = {n}",
                          transform=axes[row, 0].transAxes,
                          fontsize=args.font_size - 2, fontweight="normal",
                          fontstyle="italic",
                          ha="center", va="center", rotation=0)

    # Column labels (top) — variable name (A1, A2, A1+A2)
    for col, (_metric, label) in enumerate(VARIABLES):
        axes[0, col].text(0.5, 1.04, label,
                          transform=axes[0, col].transAxes,
                          fontsize=args.font_size + 6, fontweight="bold",
                          ha="center", va="bottom")

    # Layout — leave more space on left for "Layer" + row labels, right for cbar.
    # Widen the left margin with the font so the pushed-out row labels stay on
    # canvas (bbox_inches="tight" trims any excess at save time).
    fig.subplots_adjust(left=min(0.16 * f, 0.32), right=0.92, top=0.92, bottom=0.10)
    cbar_ax = fig.add_axes([0.94, 0.12, 0.014, 0.78])
    cbar = fig.colorbar(last_im, cax=cbar_ax)
    cbar.set_label(cbar_label, fontsize=args.font_size)
    cbar.ax.tick_params(labelsize=args.font_size - 4)

    args.outfile_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(args.outfile_stem.with_suffix(f".{ext}"),
                    dpi=200, bbox_inches="tight")
        print(f"Saved {args.outfile_stem.with_suffix('.' + ext)}")
    plt.close()


if __name__ == "__main__":
    main()
