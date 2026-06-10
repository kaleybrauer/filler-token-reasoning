"""Decode heatmaps for the varbind CHAIN-OF-THOUGHT conditions.

Sibling of decode_varbind_heatmap.py, for the pkls written by
scripts/extract/extract_varbind_cot.py (--mode teacher_forced | free_gen). The
scientific question: when the model VERBALISES the chain across token positions,
does the internal depth ladder (filler: x@L33 -> c1*x@L38 -> y@L44 -> c2*y@L51
-> answer@L60, stacked across LAYERS at one position in a single forward pass)
restructure into a ladder across POSITIONS?

Differences from the filler decoder:
  * Positions are cot_NNN / gen_NNN (offset from the assistant turn start), not
    pos_NNN / filler_k. The position name IS the offset, so examples align by name.
  * Ground truth is stored per-example in pkl["targets"] = {x, c1x, y, c2y, ans}
    (ints) — no dataset join needed.
  * pkl["value_positions"] = {label: [absolute token idx]} gives where each value
    is WRITTEN. We mark those positions on each panel, so the map reads as
    "decodable at (layer, position)" vs "written at position w".

For each (layer, position) we logit-lens (RMSNorm -> lm_head, argmax over
single-token integers 0..max_num_token) and score exact match against each of the
five values. Two read-outs:
  1. five-panel (layer x position) exact-match heatmap, with each target's write
     position(s) drawn as dashed lines on its panel;
  2. a "CoT ladder" summary: for each target, the emergence layer (first layer
     reaching the --ladder-thresh exact rate) at the position JUST BEFORE the
     value is written (w-1) — i.e. where the model has COMPUTED it and is about to
     emit it. This is the next-token-prediction site and the closest analog to the
     filler ladder's per-layer emergence.  (Decoding AT the write position w is
     near-trivial at low layers — the token itself is in the residual — so the
     pre-write column is the informative one.)

For free_gen, truth is always the TRUE value, so low decode where the model
emitted a WRONG intermediate is meaningful (the true value isn't represented
because a wrong one is being computed). value_positions is empty for a target the
model never wrote, so its marker aggregates only over examples that did write it.

Usage:
    python scripts/analysis/decode_varbind_cot_heatmap.py \
        --extraction-dir data/extracted_states_varbind_cot_tf --tag tf
    python scripts/analysis/decode_varbind_cot_heatmap.py \
        --extraction-dir data/extracted_states_varbind_cot_free --tag free \
        --subset correct
    python scripts/analysis/decode_varbind_cot_heatmap.py --tag tf --replot
"""

import argparse
import json
import os
from collections import Counter
from pathlib import Path

_NCPU = str(os.cpu_count() or 1)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, _NCPU)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from decode_varbind_heatmap import rms_norm, load_tokenizer_lite, load_pkls  # noqa: E402

plt.rcParams.update({"font.size": 20})

# Display order/labels for the five targets (paper notation x/y).
TARGETS = [
    ("x", "x: base value (visible, exact)"),
    ("c1x", "c₁·x: chain product (hidden, exact)"),
    ("y", "y: queried value (hidden, exact)"),
    ("c2y", "c₂·y: question product (hidden, exact)"),
    ("ans", "answer (final, exact)"),
]
# Filler-condition emergence layers, for side-by-side reference in the summary.
FILLER_LADDER = {"x": 33, "c1x": 38, "y": 44, "c2y": 51, "ans": 60}


def cot_pos_key(p):
    if p == "question_end":
        return -1
    if p.startswith(("cot_", "gen_")):
        return int(p.split("_")[1])
    return 10 ** 9


def cot_pos_label(p):
    if p == "question_end":
        return "q_end"
    if p.startswith(("cot_", "gen_")):
        return p.split("_")[1].lstrip("0") or "0"
    return p


def collect_positions(all_data, min_coverage):
    """Position names present in >= min_coverage of examples, sorted by offset."""
    n = len(all_data)
    counts = Counter()
    for d in all_data:
        counts.update(p for p in d["states"]
                      if p == "question_end" or p.startswith(("cot_", "gen_")))
    keep = [p for p, c in counts.items() if c >= min_coverage * n]
    return sorted(keep, key=cot_pos_key)


def write_offsets(all_data):
    """For each target label, the offsets (from assistant_start) where the value is
    written, kept if they occur in >= 25% of examples. Used to draw markers."""
    n = len(all_data)
    per = {lab: Counter() for lab, _ in TARGETS}
    for d in all_data:
        a0 = d["assistant_start"]
        vp = d.get("value_positions", {})
        for lab, _ in TARGETS:
            for abs_idx in vp.get(lab, []):
                per[lab][abs_idx - a0] += 1
    return {lab: sorted(off for off, c in per[lab].items() if c >= 0.25 * n)
            for lab, _ in TARGETS}


def decode(all_data, positions, lm_head_num, norm_weight, num_vals):
    """results[pos][layer_str][frac_<label>_exact] = exact-match fraction."""
    hidden = lm_head_num.shape[1]
    layers = sorted(all_data[0]["states"][positions[0] if positions[0] in all_data[0]["states"]
                                          else next(iter(all_data[0]["states"]))].keys())
    layer_strs = [str(l) for l in layers]
    truth = {lab: np.array([d["targets"][lab] for d in all_data]) for lab, _ in TARGETS}

    results = {"_positions": positions, "_layers": layers, "_n": len(all_data)}
    for pos in tqdm(positions, desc="  positions"):
        valid = [i for i, d in enumerate(all_data) if pos in d["states"]]
        if not valid:
            continue
        H = np.empty((len(valid), len(layers), hidden), dtype=np.float32)
        for r, i in enumerate(valid):
            st = all_data[i]["states"][pos]
            for li, l in enumerate(layers):
                H[r, li] = st[l]
        Hn = rms_norm(H, norm_weight)
        nv, L, _ = Hn.shape
        preds = num_vals[np.argmax(Hn.reshape(nv * L, hidden) @ lm_head_num.T,
                                   axis=1)].reshape(nv, L)
        vi = np.array(valid)
        results[pos] = {}
        for li, lstr in enumerate(layer_strs):
            p = preds[:, li]
            cell = {}
            for lab, _ in TARGETS:
                cell[f"frac_{lab}_exact"] = float(np.mean(p == truth[lab][vi]))
            results[pos][lstr] = cell
    return results


def ladder_summary(results, markers, thresh):
    """Print the CoT ladder: emergence layer at the pre-write position (w-1) per
    target, next to the filler-condition reference layer."""
    positions = results["_positions"]
    layers = results["_layers"]
    off_to_pos = {cot_pos_key(p): p for p in positions}
    print(f"\n  CoT ladder — emergence layer (first layer >= {thresh:.0%} exact) "
          f"at the pre-write position w-1:")
    print(f"    {'target':5s}  {'write@off':>9s}  {'pre-write':>9s}  "
          f"{'emerge L':>8s}  {'peak%@(L,off)':>16s}   filler L")
    for lab, _ in TARGETS:
        offs = markers.get(lab, [])
        w = offs[0] if offs else None              # first write offset
        pre = off_to_pos.get(w - 1) if w is not None else None
        emerge = None
        if pre is not None and pre in results:
            for l in layers:
                v = results[pre][str(l)][f"frac_{lab}_exact"]
                if v >= thresh:
                    emerge = l
                    break
        # overall peak for context
        best = (-1.0, None, None)
        for p in positions:
            for l in layers:
                v = results.get(p, {}).get(str(l), {}).get(f"frac_{lab}_exact", -1)
                if v > best[0]:
                    best = (v, l, cot_pos_key(p))
        print(f"    {lab:5s}  {str(w):>9s}  {str(w-1) if w is not None else '—':>9s}  "
              f"{str(emerge):>8s}  {best[0]*100:6.1f}%@(L{best[1]},{best[2]:>3})   "
              f"L{FILLER_LADDER[lab]}")


def plot(results, markers, tag, subset, output_dir):
    positions = results["_positions"]
    layers = results["_layers"]
    n_pos = len(positions)
    off_to_col = {cot_pos_key(p): j for j, p in enumerate(positions)}

    fig_width = max(7 * len(TARGETS), n_pos * 0.10 * len(TARGETS))
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(fig_width, 11))

    im = None
    for ax, (lab, title) in zip(axes, TARGETS):
        matrix = np.full((len(layers), n_pos), np.nan)
        for j, pos in enumerate(positions):
            for i, layer in enumerate(layers):
                c = results.get(pos, {}).get(str(layer))
                if c is not None:
                    matrix[i, j] = c[f"frac_{lab}_exact"] * 100
        im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn",
                       vmin=0, vmax=100, interpolation="nearest")
        # mark this target's write position(s)
        for off in markers.get(lab, []):
            if off in off_to_col:
                ax.axvline(off_to_col[off], color="black", lw=1.2, ls="--", alpha=0.8)

        tick_step = max(1, n_pos // 20)
        ticks = list(range(0, n_pos, tick_step))
        if (n_pos - 1) not in ticks:
            ticks.append(n_pos - 1)
        ax.set_xticks(ticks)
        ax.set_xticklabels([cot_pos_label(positions[i]) for i in ticks],
                           rotation=45, ha="right")
        ax.set_xlabel("Offset from assistant start")
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels([str(l) if l % 5 == 0 else "" for l in layers])
        ax.set_title(title, fontweight="bold", fontsize=18)
        if ax is axes[0]:
            ax.set_ylabel("Layer")

    mode = {"tf": "teacher-forced CoT", "free": "free-generated CoT"}.get(tag, tag)
    sub = {"all": "", "correct": " — model CORRECT", "incorrect": " — model INCORRECT"}.get(subset, "")
    fig.suptitle(f"varbind {mode}{sub}: what is encoded at each (layer, position)? "
                 f"(dashed = value written)", fontsize=22)
    fig.subplots_adjust(right=0.88, wspace=0.15, top=0.93)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.65])
    fig.colorbar(im, cax=cbar_ax, label="% exact match")
    sfx = "" if subset == "all" else f"_{subset}"
    for ext in ["png", "pdf"]:
        fig.savefig(output_dir / f"heatmap_varbind_cot_{tag}{sfx}_exact.{ext}",
                    dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved heatmap_varbind_cot_{tag}{sfx}_exact.png/.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extraction-dir", type=Path,
                    default=Path("data/extracted_states_varbind_cot_tf"),
                    help="Dir of prob_*.pkl from extract_varbind_cot.py")
    ap.add_argument("--tag", default="tf", help="Short label for filenames/titles (tf/free).")
    ap.add_argument("--subset", choices=["all", "correct", "incorrect"], default="all",
                    help="all (default; required for teacher_forced — no model_correct), "
                         "or filter free_gen by correctness.")
    ap.add_argument("--lm-head", type=Path,
                    default=Path("data/model_weights/deepseek_v3/lm_head_weight.npy"))
    ap.add_argument("--rms-norm", type=Path,
                    default=Path("data/model_weights/deepseek_v3/rms_norm_weight.npy"))
    ap.add_argument("--model-path", default="/workspace/models/deepseek-v3-awq")
    ap.add_argument("--output-dir", type=Path, default=Path("results/varbind_cot_decode_heatmap"))
    ap.add_argument("--max-num-token", type=int, default=1000)
    ap.add_argument("--min-coverage", type=float, default=0.5,
                    help="Keep positions present in >= this fraction of examples.")
    ap.add_argument("--ladder-thresh", type=float, default=0.5,
                    help="Exact-match threshold defining the emergence layer.")
    ap.add_argument("--replot", action="store_true",
                    help="Redraw from the saved decode JSON (no model weights).")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sfx = "" if args.subset == "all" else f"_{args.subset}"

    if args.replot:
        jpath = args.output_dir / f"decode_varbind_cot_{args.tag}{sfx}.json"
        results = json.load(open(jpath))
        markers = results.get("_markers", {})
        print(f"=== replot {jpath.name} (n={results.get('_n')}) ===")
        ladder_summary(results, markers, args.ladder_thresh)
        plot(results, markers, args.tag, args.subset, args.output_dir)
        return

    tok = load_tokenizer_lite(args.model_path)
    number_tokens = {}
    for val in range(args.max_num_token):
        ids = tok.encode(str(val), add_special_tokens=False)
        if len(ids) == 1:
            number_tokens[ids[0]] = val
    num_ids = sorted(number_tokens.keys())
    num_vals = np.array([number_tokens[t] for t in num_ids])
    lm_head_num = np.load(args.lm_head).astype(np.float32)[num_ids]
    norm_weight = np.load(args.rms_norm).astype(np.float32)
    print(f"single-token ints: {len(num_ids)}, lm_head_num {lm_head_num.shape}")

    files = sorted(args.extraction_dir.glob("prob_*.pkl"))
    all_data, n_total = load_pkls(files, args.subset)
    print(f"{len(all_data)}/{n_total} {args.subset} examples")
    if not all_data:
        print("no examples, abort")
        return

    positions = collect_positions(all_data, args.min_coverage)
    markers = write_offsets(all_data)
    print(f"positions: {len(positions)} (offset {cot_pos_key(positions[0])}.."
          f"{cot_pos_key(positions[-1])})")
    print("write offsets: " + "  ".join(f"{lab}={markers[lab]}" for lab, _ in TARGETS))

    results = decode(all_data, positions, lm_head_num, norm_weight, num_vals)
    results["_markers"] = markers
    results["_tag"] = args.tag
    outfile = args.output_dir / f"decode_varbind_cot_{args.tag}{sfx}.json"
    json.dump(results, open(outfile, "w"), indent=2)
    print(f"Saved {outfile}")

    ladder_summary(results, markers, args.ladder_thresh)
    plot(results, markers, args.tag, args.subset, args.output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
