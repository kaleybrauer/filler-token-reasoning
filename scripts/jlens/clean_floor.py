"""
clean_floor.py — same-n reproducibility floor for the PRE-FILTERED J-lens.

The floor reported during the fit (≈0.11 pass@10) is not usable any more: it was
computed on half-lenses that still contained the rank-1 giant prompts, and with the
pass@k metric that credited DeepSeek's bare space token for every numeric
intermediate. Both are fixed now, so the floor has to be re-measured on lenses that
are built the way the shipped one is.

Construction. `fit` accumulates a running SUM over a deterministic prompt order, so
two checkpoints difference to the mean over the prompts BETWEEN them, and the
per-prompt Jacobians refitted for the Frobenius pre-filter let the giants inside that
range be subtracted exactly:

    J(a..b, clean) = (sum_b − sum_a − Σ_{giants in (a,b]} J_p) / (b − a − n_giants)

Half A = prompts 0..59, half B = prompts 60..99 — DISJOINT, both pre-filtered at the
same threshold. Their gap on the paper's metric is the reproducibility floor: how much
a J-lens-vs-logit-lens difference has to move before it means anything.

Nothing is materialised: each layer is assembled on the fly and handed to the same
scoring code the shipped lenses use, so this costs two scorings, not two 6 GB builds.

    python scripts/jlens/clean_floor.py --out outputs/jlens/clean_floor.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_filtered_lens import LayerReader, logged_norms  # noqa: E402
from score_eval_states import WEIGHTS, score_lens  # noqa: E402


class RangeLens:
    """Clean mean Jacobian over prompts (lo, hi], one layer at a time."""

    def __init__(self, ckpt_hi: Path, ckpt_lo: Path | None, giant_files: dict[int, Path]):
        self.hi = LayerReader(ckpt_hi, "jacobian_sum")
        self.lo = LayerReader(ckpt_lo, "jacobian_sum") if ckpt_lo else None
        self.giants = {i: LayerReader(p, "J") for i, p in sorted(giant_files.items())}
        n_hi = self.hi.n_done
        n_lo = self.lo.n_done if self.lo else 0
        self.span = (n_lo, n_hi)
        self.n = n_hi - n_lo - len(self.giants)
        self.layers = self.hi.layers

    def __getitem__(self, L):
        S = self.hi.read(L).astype(np.float64)
        if self.lo is not None:
            S -= self.lo.read(L)
        for r in self.giants.values():
            S -= r.read(L)
        for r in self.added.values():
            S += r.read(L)
        return (S / self.n).astype(np.float32)

    def __iter__(self):
        return iter(self.layers)

    def __len__(self):
        return len(self.layers)

    def items(self):
        for L in self.layers:
            yield L, self[L]


class BlockLens:
    """Clean mean Jacobian over a UNION of prompt blocks, one layer at a time.

    Each block is a (hi, lo) pair of fit checkpoints; the running sum means
    hi - lo is exactly the prompts between them, so non-adjacent blocks can be
    summed to build a half that the snapshot schedule never stored directly.
    """

    def __init__(self, blocks, giant_files: dict[int, Path], add_files=None, label=""):
        self.terms = []
        self.n_raw = 0
        self.spans = []
        for hi, lo in blocks:
            rh = LayerReader(hi, "jacobian_sum")
            rl = LayerReader(lo, "jacobian_sum") if lo else None
            n_lo = rl.n_done if rl else 0
            self.terms.append((rh, rl))
            self.n_raw += rh.n_done - n_lo
            self.spans.append((n_lo, rh.n_done))
        self.giants = {i: LayerReader(p, "J") for i, p in sorted(giant_files.items())}
        self.added = {i: LayerReader(p, "J") for i, p in sorted((add_files or {}).items())}
        self.n = self.n_raw - len(self.giants) + len(self.added)
        self.layers = self.terms[0][0].layers
        self.label = label

    def __getitem__(self, L):
        S = None
        for rh, rl in self.terms:
            T = rh.read(L).astype(np.float64)
            if rl is not None:
                T -= rl.read(L)
            S = T if S is None else S + T
        for r in self.giants.values():
            S -= r.read(L)
        for r in self.added.values():
            S += r.read(L)
        return (S / self.n).astype(np.float32)

    def __iter__(self):
        return iter(self.layers)

    def __len__(self):
        return len(self.layers)

    def items(self):
        for L in self.layers:
            yield L, self[L]


def in_spans(i, spans):
    return any(lo <= i < hi for lo, hi in spans)


def build_halves(d: Path, per_prompt_dir: Path, exclude_above: float, mode: str):
    """(half_A, half_B): two disjoint halves of the SHIPPED lens's own corpus.

    The shipped lens averages 100 prompts: the 88 of prompts 0..99 that clear the
    Frobenius pre-filter, plus 12 freshly fitted replacements (100..111). A useful
    floor splits exactly that set, so each half is built the way the shipped lens is
    and the two together reconstruct it.

    'shipped50'  50 / 50 -- A = 0..19 + 30..59 (42 survivors) + replacements 100..107
                            B = 20..29 + 60..99 (46 survivors) + replacements 108..111
                 The replacements are dealt in index order, and the split of the
                 originals is the one that lets the counts land on 50 each.
    'even50'     50 / 50 of the ORIGINAL prompts only -> 42 / 46 after filtering.
    'contig60'   the single-boundary 60 / 40 split -> 49 / 39.

    No snapshot at n=50 is needed for any of these: the checkpoints hold a running
    sum, so differencing two of them yields exactly the prompts between, and a half
    can be assembled out of non-adjacent blocks.
    """
    ck = {n: d / f"lens_v3.snap{n}.fitckpt" for n in (10, 20, 30, 40, 60)}
    ck[100] = d / "lens_v3.fitckpt"
    norms = logged_norms()
    giants = {i: per_prompt_dir / f"J_p{i:04d}.pt"
              for i, v in norms.items() if v > exclude_above}
    missing = [i for i, p in giants.items() if not p.exists()]
    if missing:
        raise SystemExit(f"no per-prompt Jacobian for {sorted(missing)}")

    addA = addB = {}
    if mode in ("even50", "shipped50"):
        blocks_A = [(ck[20], None), (ck[60], ck[30])]
        blocks_B = [(ck[30], ck[20]), (ck[100], ck[60])]
    elif mode == "contig60":
        blocks_A, blocks_B = [(ck[60], None)], [(ck[100], ck[60])]
    else:
        raise SystemExit(f"unknown split mode {mode!r}")
    if mode == "shipped50":
        rep = {i: per_prompt_dir / f"J_p{i:04d}.pt" for i in range(100, 112)}
        missing = [i for i, p in rep.items() if not p.exists()]
        if missing:
            raise SystemExit(f"no per-prompt Jacobian for replacements {missing}")
        addA = {i: p for i, p in rep.items() if i < 108}
        addB = {i: p for i, p in rep.items() if i >= 108}

    def spans_of(bl):
        return [((LayerReader(lo, "jacobian_sum").n_done if lo else 0),
                 LayerReader(hi, "jacobian_sum").n_done) for hi, lo in bl]

    sA, sB = spans_of(blocks_A), spans_of(blocks_B)
    A = BlockLens(blocks_A, {i: p for i, p in giants.items() if in_spans(i, sA)}, addA, "A")
    B = BlockLens(blocks_B, {i: p for i, p in giants.items() if in_spans(i, sB)}, addB, "B")
    overlap = [i for i in range(100) if in_spans(i, sA) and in_spans(i, sB)]
    if overlap or (set(addA) & set(addB)):
        raise SystemExit("halves overlap - not a valid split")
    if A.n_raw + B.n_raw != 100:
        raise SystemExit(f"halves cover {A.n_raw + B.n_raw} original prompts, not 100")
    if mode == "shipped50" and A.n != B.n:
        raise SystemExit(f"shipped50 should be equal halves, got {A.n} and {B.n}")
    return A, B


# Nested subsets of the shipped lens's 100 prompts, for an n-stability sweep.
# Block granularity is set by which snapshots were kept (10/20/30/40/60/100), so the
# reachable sizes are these; an exact 75 is not reachable as a NESTED subset because
# prompts 60..99 are one atomic block of 39.
SUBSETS = {
    # (blocks, replacement index range) -- nested where the name implies it.
    # n50 and n50b are the two DISJOINT halves of the shipped corpus: together they
    # are exactly the shipped lens, and their signature curves give the within-model
    # noise floor that a cross-model claim has to clear.
    "n25":  (["b0", "b4"],                         (100, 100)),
    "n50":  (["b0", "b1", "b3", "b4"],             (100, 108)),
    "n50b": (["b2", "b5"],                         (108, 112)),
    "n61":  (["b0", "b1", "b2", "b3", "b4"],       (100, 112)),
    "n93":  (["b0", "b1", "b3", "b4", "b5"],       (100, 112)),
    "n100": (["b0", "b1", "b2", "b3", "b4", "b5"], (100, 112)),
}


def build_subset(d: Path, per_prompt_dir: Path, exclude_above: float, name: str):
    """A BlockLens over a named nested subset, built exactly like the shipped lens."""
    ck = {n: d / f"lens_v3.snap{n}.fitckpt" for n in (10, 20, 30, 40, 60)}
    ck[100] = d / "lens_v3.fitckpt"
    B = {"b0": (ck[10], None), "b1": (ck[20], ck[10]), "b2": (ck[30], ck[20]),
         "b3": (ck[40], ck[30]), "b4": (ck[60], ck[40]), "b5": (ck[100], ck[60])}
    names, (rep_lo, rep_hi) = SUBSETS[name]
    blocks = [B[k] for k in names]
    norms = logged_norms()
    giants = {i: per_prompt_dir / f"J_p{i:04d}.pt"
              for i, v in norms.items() if v > exclude_above}

    def spans_of(bl):
        return [((LayerReader(lo, "jacobian_sum").n_done if lo else 0),
                 LayerReader(hi, "jacobian_sum").n_done) for hi, lo in bl]

    sp = spans_of(blocks)
    sub = {i: p for i, p in giants.items() if in_spans(i, sp)}
    add = {i: per_prompt_dir / f"J_p{i:04d}.pt" for i in range(rep_lo, rep_hi)}
    return BlockLens(blocks, sub, add, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=REPO_ROOT / "outputs/jlens")
    ap.add_argument("--per-prompt-dir", type=Path,
                    default=REPO_ROOT / "outputs/jlens/per_prompt")
    ap.add_argument("--states", type=Path, default=REPO_ROOT / "outputs/jlens/eval_states.pt")
    ap.add_argument("--split-mode", choices=["shipped50", "even50", "contig60"], default="shipped50",
                    help="shipped50 (default): 50/50 split of the SHIPPED lens's own 100 "
                         "prompts, replacements included. even50: 50/50 of the originals "
                         "only (42/46 after filtering). contig60: the 60/40 boundary split.")
    ap.add_argument("--exclude-above", type=float, default=40.0)
    ap.add_argument("--sets", default="multihop,order-ops")
    ap.add_argument("--ks", default="1,5,10,50")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = str(args.threads)
    if args.out is None:
        args.out = REPO_ROOT / f"outputs/jlens/clean_floor_{args.split_mode}.json"

    import torch
    from extract.extract_hidden_states import load_tokenizer

    ks = [int(k) for k in args.ks.split(",")]
    slugs = [s.strip() for s in args.sets.split(",") if s.strip()]
    norms = logged_norms()
    giants = {i: args.per_prompt_dir / f"J_p{i:04d}.pt"
              for i, v in sorted(norms.items()) if v > args.exclude_above}
    missing = {i: p for i, p in giants.items() if not p.exists()}
    if missing:
        raise SystemExit(f"no per-prompt Jacobian for {sorted(missing)} — cannot subtract them")

    A, B = build_halves(args.dir, args.per_prompt_dir, args.exclude_above, args.split_mode)
    for h in (A, B):
        spans = ", ".join(f"{lo}..{hi-1}" for lo, hi in h.spans)
        print(f"half {h.label}: prompts {spans}  minus {sorted(h.giants)}"
              + (f"  plus {sorted(h.added)}" if h.added else "") + f"  -> n={h.n}", flush=True)

    states = torch.load(args.states, map_location="cpu", weights_only=False)
    if not states.get("unembed_check", {}).get("passed", False):
        raise SystemExit("eval states failed G-UNEMBED")
    tok = load_tokenizer(states["model_path"])
    rms_w = np.load(WEIGHTS / "rms_norm_weight.npy").astype(np.float32)
    lm_w = np.load(WEIGHTS / "lm_head_weight.npy").astype(np.float32)

    rep = {"split_mode": args.split_mode, "exclude_above": args.exclude_above,
           "half_A": {"spans": A.spans, "n": A.n, "excluded": sorted(A.giants),
                      "added": sorted(A.added)},
           "half_B": {"spans": B.spans, "n": B.n, "excluded": sorted(B.giants),
                      "added": sorted(B.added)}, "ks": ks}
    t0 = time.perf_counter()
    for name, lens in (("half_A", A), ("half_B", B)):
        for L in lens:                      # streamed finiteness guard
            if not np.isfinite(lens[L]).all():
                raise SystemExit(f"{name}: non-finite at L{L}")
        print(f" {name}: finite, scoring", flush=True)
        rep[name]["results"] = score_lens(states, lens, tok, rms_w, lm_w, ks, slugs)

    print("\nsame-n floor |A − B| (a real difference must clear this):")
    rep["floor"] = {}
    for s in slugs:
        rep["floor"][s] = {}
        for arm in ("jlens", "logit"):
            for k in ks:
                d = abs(rep["half_A"]["results"][s][arm][f"pass@{k}"]
                        - rep["half_B"]["results"][s][arm][f"pass@{k}"])
                rep["floor"][s][f"{arm}_pass@{k}"] = round(d, 4)
                if k == 10:
                    print(f"  {s:10s} {arm:6s} pass@10 floor {d:.4f}"
                          f"{'   <- must be 0.000' if arm == 'logit' else ''}")
    rep["elapsed_s"] = round(time.perf_counter() - t0, 1)
    args.out.write_text(json.dumps(rep, indent=1))
    print(f"\nwrote {args.out} in {rep['elapsed_s']/60:.1f} min")


if __name__ == "__main__":
    main()
