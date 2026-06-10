"""Paper figure: filler vs chain-of-thought decoded-value maps, side by side.

The CoT analog of varbind_paper_figure.py. Same proportional (fractional)
convex-blend overlay — each (layer, position) cell is a blend of the five value
colours weighted by exact-match decode fraction (white = nothing decodes), so
saturation = encoding strength (honest about magnitude, no per-cell boosting).

The story it tells: in the FILLER panel the chain is stacked across LAYERS at the
filler/answer positions (a vertical depth ladder, one forward pass); in the CoT
panel it is unrolled across POSITIONS (a diagonal staircase — each value written
at its own token, computed shallowly from the previous written value).

Handles both decode-JSON schemas automatically:
  * filler  : frac_{B,C1B,QV,QC,ANS}_exact, positions pos_NNN / filler_k / answer_prompt
  * CoT     : frac_{x,c1x,y,c2y,ans}_exact, positions cot_NNN / gen_NNN

Usage:
    python scripts/analysis/varbind_cot_paper_figure.py            # filler dots_10 | CoT tf
    python scripts/analysis/varbind_cot_paper_figure.py \
        --json results/varbind_cot_decode_heatmap/decode_varbind_cot_tf.json \
        --titles "chain-of-thought"                                 # single CoT panel
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.offsetbox import TextArea, HPacker, AnchoredOffsetbox
import numpy as np

# (legend label, RGB) in computation order, cool -> warm. Terse; define in caption.
TARGETS = [
    ("x",      (0.12, 0.47, 0.71)),   # blue   — base value (visible)
    ("c₁·x",   (0.17, 0.63, 0.17)),   # green  — chain product (hidden)
    ("y",      (0.85, 0.69, 0.00)),   # gold   — queried value (hidden)
    ("c₂·y",   (1.00, 0.50, 0.00)),   # orange — question product (hidden)
    ("answer", (0.84, 0.15, 0.16)),   # red    — output
]
COLORS = np.array([t[1] for t in TARGETS])
COT_KEYS = ["frac_x_exact", "frac_c1x_exact", "frac_y_exact", "frac_c2y_exact", "frac_ans_exact"]
FILLER_KEYS = ["frac_B_exact", "frac_C1B_exact", "frac_QV_exact", "frac_QC_exact", "frac_ANS_exact"]

# Worked example as coloured segments. Each variable AND its value take that
# variable's legend colour; a value keeps its colour when reused in the next step
# (so the data-flow x→c₁·x→y→c₂·y→answer is visible). Coefficients/constants stay
# black. Numbers are example idx 0 (lef=58 → cok=93 → answer=183).
_BLACK = (0.15, 0.15, 0.15)
_BLUE, _GREEN, _GOLD, _ORANGE, _RED = (t[1] for t in TARGETS)
EXAMPLE_SEGMENTS = [
    ("example:   ", _BLACK),
    ("x", _BLUE), (" = ", _BLACK), ("58", _BLUE), ("   →   ", _BLACK),
    ("c₁·x", _GREEN), (" = 2·", _BLACK), ("58", _BLUE), (" = ", _BLACK), ("116", _GREEN),
    ("   →   ", _BLACK),
    ("y", _GOLD), (" = ", _BLACK), ("116", _GREEN), (" − 23 = ", _BLACK), ("93", _GOLD),
    ("   →   ", _BLACK),
    ("c₂·y", _ORANGE), (" = 2·", _BLACK), ("93", _GOLD), (" = ", _BLACK), ("186", _ORANGE),
    ("   →   ", _BLACK),
    ("answer", _RED), (" = ", _BLACK), ("186", _ORANGE), (" − 3 = ", _BLACK), ("183", _RED),
]


def off(p):
    if p == "question_end": return -2
    if p == "pre_filler": return -1
    if p.startswith("pos_") or p.startswith("cot_") or p.startswith("gen_"):
        return int(p.split("_")[1])
    if p.startswith("filler_k"): return int(p.split("k")[1])
    if p == "answer_prompt": return 99999
    return 9999


def pos_label(p):
    if p == "question_end": return "q_end"
    if p == "pre_filler": return "f_label"
    if p == "answer_prompt": return "ans"
    if p.startswith(("pos_", "cot_", "gen_")): return p.split("_")[1].lstrip("0") or "0"
    if p.startswith("filler_k"): return "k" + p.split("k")[1]
    return p


def keys_for(r):
    """Detect the JSON schema from a populated cell."""
    for p in r["_positions"]:
        for l in r["_layers"]:
            c = r.get(p, {}).get(str(l))
            if c:
                return COT_KEYS if "frac_x_exact" in c else FILLER_KEYS
    return FILLER_KEYS


def proportional_rgb(r, layers, keys):
    positions = sorted(r["_positions"], key=off)
    img = np.ones((len(layers), len(positions), 3))
    for j, p in enumerate(positions):
        for i, l in enumerate(layers):
            cell = r.get(p, {}).get(str(l))
            if not cell:
                continue
            fr = np.array([cell.get(k) or 0.0 for k in keys])
            total = float(fr.sum())
            img[i, j] = np.clip(max(0.0, 1.0 - total) * np.ones(3)
                                + (fr[:, None] * COLORS).sum(0), 0.0, 1.0)
    return img, positions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", nargs="+", default=[
        "results/varbind_decode_heatmap/decode_varbind_dots_10_all.json",
        "results/varbind_cot_decode_heatmap/decode_varbind_cot_tf.json",
    ], help="Decode JSON paths, one per panel (filler and/or CoT).")
    ap.add_argument("--titles", nargs="+", default=["filler (k=10)", "chain-of-thought"])
    ap.add_argument("--min-layer", type=int, default=30)
    ap.add_argument("--target-width", type=float, default=13.0,
                    help="Target figure width in inches (auto-scales per-position width).")
    ap.add_argument("--min-panel", type=float, default=4.0,
                    help="Floor on a panel's width units so few-position panels "
                         "(e.g. baseline: q_end+ans) stay legible.")
    ap.add_argument("--max-panel", type=float, default=30.0,
                    help="Cap on a panel's width units so the wide CoT panel "
                         "doesn't dwarf the others.")
    ap.add_argument("--legend-fontsize", type=float, default=20.0,
                    help="Font size for the bottom colour legend.")
    ap.add_argument("--no-example", action="store_true",
                    help="Omit the colour-coded worked-example line under the legend.")
    ap.add_argument("--mark-writes", action="store_true",
                    help="On CoT panels, draw faint dashes at each value's write position.")
    ap.add_argument("--output", type=Path,
                    default=Path("results/varbind_cot_decode_heatmap/figure_filler_vs_cot"))
    args = ap.parse_args()

    data, layers = [], None
    for jp in args.json:
        r = json.load(open(jp))
        layers = [l for l in r["_layers"] if l >= args.min_layer]
        keys = keys_for(r)
        img, positions = proportional_rgb(r, layers, keys)
        data.append((r, img, positions))

    n_pos = [len(d[2]) for d in data]
    widths = [min(max(n, args.min_panel), args.max_panel) for n in n_pos]
    # auto-scale per-position width so the whole figure lands near target-width
    per = (args.target_width - 0.6) / max(sum(widths), 1)
    has_caption = not args.no_example
    fig = plt.figure(figsize=(sum(widths) * per + 0.6, 7.2 if has_caption else 6.4))
    gs = fig.add_gridspec(1, len(data), width_ratios=widths, wspace=0.06)

    for k, (r, img, positions) in enumerate(data):
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(img, aspect="auto", interpolation="nearest")
        step = max(1, len(positions) // 12) if len(positions) > 6 else 1
        idxs = list(range(0, len(positions), step))
        if (len(positions) - 1) not in idxs:
            idxs.append(len(positions) - 1)
        ax.set_xticks(idxs)
        ax.set_xticklabels([pos_label(positions[i]) for i in idxs], rotation=45,
                           ha="right", fontsize=9)
        ax.set_xlabel("token position", fontsize=12)

        # filler: mark the filler -> answer boundary; CoT: optional write markers
        b = r.get("_boundaries") or {}
        fe = b.get("filler_end_offset")
        if fe is not None:
            col = {off(p): j for j, p in enumerate(positions)}
            if fe in col:
                ax.axvline(col[fe] + 0.5, color="black", lw=1.0, ls="--", alpha=0.5)
        if args.mark_writes and "_markers" in r:
            col = {off(p): j for j, p in enumerate(positions)}
            for offs in r["_markers"].values():
                for o in offs:
                    if o in col:
                        ax.axvline(col[o], color="black", lw=0.6, ls=":", alpha=0.35)

        ax.set_title(args.titles[k] if k < len(args.titles) else "", fontsize=14)
        if k == 0:
            ax.set_yticks(range(len(layers)))
            ax.set_yticklabels([str(l) if l % 5 == 0 else "" for l in layers], fontsize=10)
            ax.set_ylabel("layer", fontsize=12)
        else:
            ax.set_yticks([])

    legend_y = 0.20 if has_caption else 0.12
    fig.legend(handles=[Patch(facecolor=c, label=name) for name, c in TARGETS],
               loc="upper center", bbox_to_anchor=(0.5, legend_y), ncol=len(TARGETS),
               fontsize=args.legend_fontsize, frameon=False, handlelength=1.3,
               columnspacing=1.7, handletextpad=0.5)
    fig.subplots_adjust(top=0.93, bottom=0.30 if has_caption else 0.22,
                        left=0.06, right=0.99)
    if has_caption:
        children = [TextArea(t, textprops=dict(color=c, fontsize=args.legend_fontsize - 5))
                    for t, c in EXAMPLE_SEGMENTS]
        box = HPacker(children=children, align="baseline", pad=0, sep=0)
        fig.add_artist(AnchoredOffsetbox(loc="center", child=box, pad=0, borderpad=0,
                                         frameon=False, bbox_to_anchor=(0.5, 0.085),
                                         bbox_transform=fig.transFigure))
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.output}.{ext}", dpi=200, bbox_inches="tight")
    print(f"saved {args.output}.png / .pdf")


if __name__ == "__main__":
    main()
