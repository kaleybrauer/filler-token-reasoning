"""Overlaid 'dominant-value' heatmap for the varbind decode.

Instead of 5 separate panels, colour each (layer, position) cell by WHICH target
value is most strongly decoded there (argmax over exact-match fractions), with
whiteness = how weakly anything is encoded. The serial chain
    B -> c1*B -> V -> c*V -> answer
then reads as a vertical colour sweep down the layers.

    no encoding -> white
    B (base, visible)        -> blue
    c1*B (chain product)     -> green
    V (queried value)        -> gold
    c*V (question product)   -> orange
    answer                   -> red

Reads decode_varbind_<cond>.json (needs the 5 frac_*_exact fields).

Usage:
    python scripts/analysis/varbind_overlay_heatmap.py --condition dots_10
    python scripts/analysis/varbind_overlay_heatmap.py --condition dots_10 --incorrect
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

# (json key, legend label, RGB) — ordered along the computation, cool -> warm.
TARGETS = [
    ("frac_B_exact",   "B  (base, visible)",       (0.12, 0.47, 0.71)),  # blue
    ("frac_C1B_exact", "c1·B  (chain product)",    (0.17, 0.63, 0.17)),  # green
    ("frac_QV_exact",  "V  (queried value)",       (0.85, 0.69, 0.00)),  # gold
    ("frac_QC_exact",  "c·V  (question product)",  (1.00, 0.50, 0.00)),  # orange
    ("frac_ANS_exact", "answer  (output)",         (0.84, 0.15, 0.16)),  # red
]


def pos_sort_key(p):
    if p == "question_end": return -2
    if p == "pre_filler": return -1
    if p.startswith("filler_k"): return int(p.split("k")[1])
    if p.startswith("pos_"): return int(p.split("_")[1])
    if p == "answer_prompt": return 99999
    return 9999


def pos_label(p):
    if p == "question_end": return "q_end"
    if p == "pre_filler": return "filler_label"
    if p == "answer_prompt": return "ans_prompt"
    if p.startswith("filler_k"): return f"k{p.split('k')[1]}"
    if p.startswith("pos_"): return p.split("_")[1].lstrip("0") or "0"
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="dots_10")
    ap.add_argument("--incorrect", action="store_true")
    ap.add_argument("--decode-dir", type=Path, default=Path("results/varbind_decode_heatmap"))
    ap.add_argument("--output-dir", type=Path, default=Path("results/varbind_decode_heatmap"))
    ap.add_argument("--min-layer", type=int, default=30,
                    help="Drop layers below this — the first ~30 layers aren't useful for "
                         "logit lens.")
    ap.add_argument("--mode", choices=["proportional", "dominant", "brightness"],
                    default="proportional",
                    help="proportional: convex blend of value colours weighted by exact-match "
                         "fraction (leftover -> white). dominant: colour by the single "
                         "most-decoded value. brightness: normalize the blend to a pure hue "
                         "(which values), then modulate brightness by the TOTAL decoded "
                         "fraction (keeps the composition, boosts weak-but-clean cells).")
    ap.add_argument("--vmax", type=float, default=0.6,
                    help="(dominant/brightness modes) decoded fraction that maps to full "
                         "saturation; lower values fade toward white.")
    args = ap.parse_args()

    suffix = f"{args.condition}_incorrect" if args.incorrect else args.condition
    r = json.load(open(args.decode_dir / f"decode_varbind_{suffix}.json"))
    positions = sorted(r["_positions"], key=pos_sort_key)
    layers = [l for l in r["_layers"] if l >= args.min_layer]
    boundaries = r.get("_boundaries")
    keys = [t[0] for t in TARGETS]
    colors = np.array([t[2] for t in TARGETS])

    H, W = len(layers), len(positions)
    img = np.ones((H, W, 3))           # default white = nothing decodes
    for j, pos in enumerate(positions):
        for i, layer in enumerate(layers):
            cell = r.get(pos, {}).get(str(layer))
            if not cell:
                continue
            fr = np.array([cell.get(k) or 0.0 for k in keys])
            if args.mode == "proportional":
                # Literal overlay: convex blend of each value's colour by the
                # fraction of examples decoding EXACTLY to it; the leftover
                # (non-target / nothing) stays white. Fractions are disjoint
                # (each example's argmax matches at most one target) so they
                # sum to <= 1.
                total = float(fr.sum())
                col = max(0.0, 1.0 - total) * np.ones(3) + (fr[:, None] * colors).sum(0)
                img[i, j] = np.clip(col, 0.0, 1.0)
            elif args.mode == "brightness":
                # Normalize the blend to a pure HUE (the composition — which values
                # decode here, regardless of how often), then modulate brightness by
                # the TOTAL decoded fraction (vmax-scaled). Keeps the mix colour but
                # doesn't wash out a cleanly-but-weakly-decoded cell.
                total = float(fr.sum())
                if total > 1e-9:
                    hue = (fr[:, None] * colors).sum(0) / total
                    b = min(total / args.vmax, 1.0)
                    img[i, j] = np.clip((1 - b) * np.ones(3) + b * hue, 0.0, 1.0)
            else:  # dominant (winner-take-all)
                d = int(np.argmax(fr))
                m = min(fr[d] / args.vmax, 1.0)      # saturation by strength
                img[i, j] = (1 - m) * np.ones(3) + m * colors[d]

    fig_w = max(8.0, W * 0.34)
    fig, ax = plt.subplots(figsize=(fig_w, 9))
    ax.imshow(img, aspect="auto", interpolation="nearest")

    is_allpos = any(p.startswith("pos_") for p in positions)
    if is_allpos:
        step = max(1, W // 20)
        idxs = list(range(0, W, step))
        if (W - 1) not in idxs:
            idxs.append(W - 1)
        ax.set_xticks(idxs)
        ax.set_xticklabels([pos_label(positions[i]) for i in idxs], rotation=45, ha="right")
        ax.set_xlabel("Offset from question_end")
        if boundaries and boundaries.get("filler_end_offset") is not None:
            ax.axvline(boundaries["filler_end_offset"] + 0.5, color="black",
                       lw=1.2, ls="--", alpha=0.5)
    else:
        ax.set_xticks(range(W))
        ax.set_xticklabels([pos_label(p) for p in positions], rotation=45, ha="right")

    ax.set_yticks(range(H))
    ax.set_yticklabels([str(l) if l % 5 == 0 else "" for l in layers])
    ax.set_ylabel("Layer")

    ax.legend(handles=[Patch(facecolor=c, label=name) for _, name, c in TARGETS],
              loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=11,
              title=("dominant value" if args.mode == "dominant" else "decoded value"),
              frameon=False)
    tag = " — model INCORRECT" if args.incorrect else ""
    if args.mode == "proportional":
        sub = "white = non-target / none · colour = proportional blend of which values decode here"
    elif args.mode == "brightness":
        sub = (f"white = nothing · hue = composition (which values) · "
               f"brightness = total decoded fraction (sat@{args.vmax:g})")
    else:
        sub = (f"white = nothing decodes · hue = dominant value · intensity = strength "
               f"(sat@{args.vmax:g})")
    ax.set_title(f"varbind {args.condition}{tag}: decoded-value map per (layer, position)\n"
                 f"{sub}  [exact match]", fontsize=12)
    fig.tight_layout()
    out = args.output_dir / f"overlay_varbind_{suffix}_{args.mode}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"saved {out}")


if __name__ == "__main__":
    main()
