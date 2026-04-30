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
    ap.add_argument("--font-size", type=int, default=22)
    args = ap.parse_args()
    if args.outfile_stem is None:
        suffix = f"_{args.metric}"
        if args.min_layer > 0:
            suffix += f"_L{args.min_layer}+"
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

    fig, axes = plt.subplots(
        3, 2, figsize=(13, 13),
        sharex=True, sharey=True,
        gridspec_kw={"hspace": 0.10, "wspace": 0.06},
    )

    last_im = None
    for row, (metric, var_label) in enumerate(VARIABLES):
        for col, (data, _label) in enumerate([(correct, "Correct"), (wrong, "Wrong")]):
            ax = axes[row, col]
            M, layers, positions = matrix_for(data, metric, min_layer=args.min_layer)
            im = ax.imshow(M, aspect="auto", cmap="RdYlGn",
                           vmin=0, vmax=100, interpolation="nearest")
            last_im = im

            # x ticks at every even-offset position
            n_pos = len(positions)
            tick_idxs = [i for i, p in enumerate(positions)
                         if int(pos_label(p)) % 2 == 0]
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
            if row == len(VARIABLES) - 1:
                ax.set_xlabel("Token offset")

    # Row labels (left side) — variable name
    for row, (_metric, label) in enumerate(VARIABLES):
        # Place to the LEFT of the axis using axes coordinates of leftmost subplot
        axes[row, 0].text(-0.27, 0.5, label, transform=axes[row, 0].transAxes,
                          fontsize=args.font_size + 6, fontweight="bold",
                          ha="center", va="center", rotation=0)

    # Column labels — Correct / Wrong, with n=...
    for col, (label, n) in enumerate([("Correct", n_correct), ("Wrong", n_wrong)]):
        axes[0, col].text(0.5, 1.04, f"{label}  (n={n})",
                          transform=axes[0, col].transAxes,
                          fontsize=args.font_size + 4, fontweight="bold",
                          ha="center", va="bottom")

    # Layout — leave space on right for shared colorbar
    fig.subplots_adjust(left=0.13, right=0.90, top=0.94, bottom=0.08)
    cbar_ax = fig.add_axes([0.92, 0.10, 0.018, 0.82])
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
