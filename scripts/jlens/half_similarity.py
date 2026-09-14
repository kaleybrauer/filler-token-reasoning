"""
half_similarity.py — how alike are two J-lenses fitted on disjoint halves of the corpus?

`clean_floor.py` asks whether the two halves SCORE the same. This asks whether they
ARE the same, per layer, three ways — the trio the paper uses to compare lenses to
each other (its Figure 56):

  cosine      <A,B> / (|A||B|) over the flattened matrices: do they point the same way
  rel. dist   |A - B|_F / max(|A|_F,|B|_F): how much of the magnitude differs
  top-1       fraction of the 551 cached eval states where the two lenses' top-1
              readout token is identical, per layer -- the behavioural question

Halves are built exactly as in clean_floor.py: a difference of two fit checkpoints,
minus the pre-filtered giant prompts inside the range, so both halves are the object
the shipped lens is.

    python scripts/jlens/half_similarity.py --out outputs/jlens/half_similarity.json
"""

from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from clean_floor import build_halves                    # noqa: E402
from score_eval_states import WEIGHTS, rms_norm         # noqa: E402


def top1(H, J, rms_w, lm_w, eps, chunk=64):
    out = np.empty(H.shape[0], dtype=np.int64)
    for s in range(0, H.shape[0], chunk):
        h = H[s:s + chunk].astype(np.float32) @ J.T
        out[s:s + chunk] = (rms_norm(h, rms_w, eps) @ lm_w.T).argmax(1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=REPO_ROOT / "outputs/jlens")
    ap.add_argument("--per-prompt-dir", type=Path, default=REPO_ROOT / "outputs/jlens/per_prompt")
    ap.add_argument("--states", type=Path, default=REPO_ROOT / "outputs/jlens/eval_states.pt")
    ap.add_argument("--split-mode", choices=["shipped50", "even50", "contig60"], default="shipped50",
                    help="shipped50: split the SHIPPED lens's own 100 prompts in half, so "
                         "each half is built exactly the way the lens in use is built.")
    ap.add_argument("--exclude-above", type=float, default=40.0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = str(args.threads)
    if args.out is None:
        args.out = REPO_ROOT / f"outputs/jlens/half_similarity_{args.split_mode}.json"

    import torch
    A, B = build_halves(args.dir, args.per_prompt_dir, args.exclude_above, args.split_mode)
    for h in (A, B):
        spans = ", ".join(f"{lo}..{hi-1}" for lo, hi in h.spans)
        print(f"half {h.label}: prompts {spans}  minus {sorted(h.giants)}"
              + (f"  plus {sorted(h.added)}" if h.added else "") + f"  -> n={h.n}", flush=True)

    states = torch.load(args.states, map_location="cpu", weights_only=False)
    eps = states["rms_norm_eps"]
    H = np.concatenate([states["sets"][s]["H"].numpy() for s in sorted(states["sets"])])
    rms_w = np.load(WEIGHTS / "rms_norm_weight.npy").astype(np.float32)
    lm_w = np.load(WEIGHTS / "lm_head_weight.npy").astype(np.float32)
    n_items = H.shape[0]
    print(f"{n_items} eval states", flush=True)

    rows = []
    t0 = time.time()
    print(f"{'L':>3} {'cosine':>8} {'rel.dist':>9} {'top1 agree':>11} {'|A|':>9} {'|B|':>9}")
    for L in A.layers:
        a, b = A[L], B[L]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        cos = float((a * b).sum() / (na * nb))
        rel = float(np.linalg.norm(a - b) / max(na, nb))
        agree = float((top1(H[:, L], a, rms_w, lm_w, eps)
                       == top1(H[:, L], b, rms_w, lm_w, eps)).mean())
        rows.append({"layer": int(L), "cosine": round(cos, 4), "rel_dist": round(rel, 4),
                     "top1_agreement": round(agree, 4),
                     "fro_A": round(float(na), 2), "fro_B": round(float(nb), 2)})
        print(f"{L:3d} {cos:8.4f} {rel:9.4f} {agree:11.3f} {na:9.1f} {nb:9.1f}", flush=True)

    rep = {"split_mode": args.split_mode, "exclude_above": args.exclude_above, "n_items": n_items,
           "half_A": {"spans": A.spans, "n": A.n}, "half_B": {"spans": B.spans, "n": B.n},
           "per_layer": rows, "elapsed_s": round(time.time() - t0, 1)}
    args.out.write_text(json.dumps(rep, indent=1))
    print(f"\nwrote {args.out} in {rep['elapsed_s']/60:.1f} min")


if __name__ == "__main__":
    main()
