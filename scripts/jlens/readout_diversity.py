"""
readout_diversity.py — is a lens's top-1 readout degenerate?

For each lens layer, the top-1 vocab token of the lens readout for every cached
eval item (all six sets, 551 items). Reports per layer: the most common top-1
token and the fraction of items sharing it, the number of distinct top-1 tokens,
and the same for the logit-lens arm. A lens dominated by a rank-1 giant pushes
the same token everywhere; that inflates pass@1 wherever that token happens to
be an intermediate, so a high pass@1 is only meaningful if diversity is normal.
CPU, one lens layer in memory at a time.

    python scripts/jlens/readout_diversity.py --lens outputs/jlens/lens_v3_filtered40.pt --out X.json
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
from collections import Counter
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_eval_states import WEIGHTS, rms_norm  # noqa: E402
from score_lazy import LazyLens  # noqa: E402


def top1(H, rms_w, lm_w, eps, chunk=64):
    out = np.empty(H.shape[0], dtype=np.int64)
    for s in range(0, H.shape[0], chunk):
        out[s:s + chunk] = (rms_norm(H[s:s + chunk], rms_w, eps) @ lm_w.T).argmax(1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=Path, default=REPO_ROOT / "outputs/jlens/eval_states.pt")
    ap.add_argument("--lens", type=Path, required=True)
    ap.add_argument("--layers", default="0,5,10,15,20,25,30,35,40,45,50,55,59")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(args.threads)
    import torch
    from extract.extract_hidden_states import load_tokenizer

    states = torch.load(args.states, map_location="cpu", weights_only=False)
    tok = load_tokenizer(states["model_path"])
    eps = states["rms_norm_eps"]
    rms_w = np.load(WEIGHTS / "rms_norm_weight.npy").astype(np.float32)
    lm_w = np.load(WEIGHTS / "lm_head_weight.npy").astype(np.float32)
    sets = sorted(states["sets"])
    H_all = np.concatenate([states["sets"][s]["H"].numpy() for s in sets])  # [n, 61, d] fp16
    n = H_all.shape[0]
    lens = LazyLens(args.lens)
    layers = [int(x) for x in args.layers.split(",") if int(x) in lens.layers]
    res = {"lens": str(args.lens), "n_items": n, "sets": sets, "per_layer": {}}
    t0 = time.time()
    print(f"{n} items; lens {args.lens}")
    print(f"{'L':>3s} | {'jlens: top token':22s} share  distinct | {'logit: top token':22s} share  distinct")
    for L in layers:
        h = H_all[:, L].astype(np.float32)
        a = top1(h @ lens[L].T, rms_w, lm_w, eps)
        b = top1(h, rms_w, lm_w, eps)
        row = {}
        for arm, ids in (("jlens", a), ("logit", b)):
            c = Counter(ids.tolist()).most_common(3)
            row[arm] = {"top": [(tok.decode([t]), k / n) for t, k in c],
                        "top_share": c[0][1] / n, "n_distinct": len(set(ids.tolist()))}
        res["per_layer"][L] = row
        j, g = row["jlens"], row["logit"]
        print(f"{L:3d} | {j['top'][0][0]!r:22s} {j['top_share']:.2f}   {j['n_distinct']:4d}     | "
              f"{g['top'][0][0]!r:22s} {g['top_share']:.2f}   {g['n_distinct']:4d}", flush=True)
    res["elapsed_s"] = round(time.time() - t0, 1)
    args.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
