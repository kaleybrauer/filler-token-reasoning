"""
workspace_signatures.py — the paper's workspace-structure measures, on our lens.

Anthropic's global-workspace paper identifies three functional regions across depth
(sensory / workspace / motor) from several per-layer statistics. Their repo ships no
code for these, so these are OUR implementations of the measures as the paper
describes them; each docstring quotes the description being implemented.

Per layer L, given J_L and the model's readout matrix U = W_U * rms_weight (the
per-example normalisation is a positive scalar and cannot change a ranking, so the
readout DIRECTION for token t is row t of U J_L; the logit lens is U itself):

  eff_dim_{50,90}   "the effective linear dimensionality of the J-space: ... the
                    fraction of residual-stream dimensions needed to capture a given
                    share of the variance across the J-lens vectors W_U J_l".
                    Eigenvalues of (U J)^T (U J) = J^T (U^T U) J; report the fraction
                    of d_model needed for 50% / 90% of the spectrum.
  part_ratio        participation ratio (sum l)^2 / sum l^2, over d_model -- a
                    threshold-free companion to the above, cheap and standard.
  cos_to_logit      "the mean cosine similarity between the two methods' vectors for
                    the same vocabulary token, averaged over the vocabulary."
  nexttok_top{k}    "the fraction of positions at which any top-k J-lens token matches
                    the model's top-1 prediction."  The model's own top-1 is the
                    final-layer readout on the same cached state.
  kurtosis          "excess kurtosis of the J-lens readout distribution; high kurtosis
                    indicates a readout sharply peaked on a few tokens."
  top1_agree_logit  top-1 agreement between the J-lens and the logit lens.
  sym_kl_logit      symmetric KL between the two readout distributions.

Layers are also reported reindexed to [0,100] so the band is comparable with the
paper's (~38 to ~92 on that scale).

    python scripts/jlens/workspace_signatures.py --lens outputs/jlens/lens_v3_filtered40.pt
"""

from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_eval_states import WEIGHTS                    # noqa: E402
from score_lazy import LazyLens                          # noqa: E402


def eff_dims(M, d):
    """Fraction of dimensions holding 50% / 90% of the spectrum, plus participation ratio."""
    lam = np.linalg.eigvalsh(M.astype(np.float64))[::-1]
    lam = np.clip(lam, 0, None)
    tot = lam.sum()
    if tot <= 0:
        return dict(eff_dim_50=float("nan"), eff_dim_90=float("nan"), part_ratio=float("nan"))
    c = np.cumsum(lam) / tot
    return dict(eff_dim_50=float((np.searchsorted(c, 0.50) + 1) / d),
                eff_dim_90=float((np.searchsorted(c, 0.90) + 1) / d),
                part_ratio=float(tot ** 2 / (lam ** 2).sum() / d))



def centred_gram(U, chunk=16384):
    """Gram of the MEAN-CENTRED readout rows, accumulated in float64.

    The paper's J-space dimensionality is "the variance across the J-lens vectors",
    so the rows must be centred. Doing it as G - n*outer(mean,mean) AFTER forming an
    uncentred float32 Gram is catastrophic cancellation: an unembedding's rows share a
    huge common direction (for pythia-70m the mean row is 98% as long as a typical row
    and the uncentred spectrum is 96% rank-1), so the centred part is a few percent of
    the total and is lost. Subtract first, accumulate in float64.
    """
    import numpy as np
    d = U.shape[1]
    ubar = U.mean(0, dtype=np.float64).astype(np.float32)
    G = np.zeros((d, d), dtype=np.float64)
    for s in range(0, U.shape[0], chunk):
        C = U[s:s + chunk] - ubar
        G += (C.T @ C).astype(np.float64)
    return G.astype(np.float32), ubar


def cosines_to_logit(U, J, ubar=None, chunk=16384):
    """Mean over the vocabulary of cos(J^T U_t, U_t) -- the paper's lens-agreement
    measure ("mean cosine similarity between the two methods' vectors for the same
    vocabulary token, averaged over the vocabulary"). Also returns the centred variant,
    which removes the shared common direction that otherwise dominates both vectors.
    """
    import numpy as np
    raw = cen = 0.0
    n = U.shape[0]
    for s in range(0, n, chunk):
        C = U[s:s + chunk]
        P = C @ J
        raw += float((( P * C).sum(1) /
                      (np.linalg.norm(P, axis=1) * np.linalg.norm(C, axis=1) + 1e-9)).sum())
        if ubar is not None:
            Cc = C - ubar
            Pc = Cc @ J
            cen += float(((Pc * Cc).sum(1) /
                          (np.linalg.norm(Pc, axis=1) * np.linalg.norm(Cc, axis=1) + 1e-9)).sum())
    return raw / n, (cen / n if ubar is not None else float("nan"))


def readout_stats(h, J, U, k, chunk=64):
    """Per-layer readout statistics against the model's own final-layer prediction.

    h is [n_items, n_layers, d] fp16; the caller passes h[:, L] and h[:, -1].
    """
    n = h[0].shape[0]
    hits = agree = 0
    kurt = []
    kl = []
    for s in range(0, n, chunk):
        hl = h[0][s:s + chunk].astype(np.float32)
        hf = h[1][s:s + chunk].astype(np.float32)
        zj = (hl @ J.T) @ U.T                       # J-lens logits
        zg = hl @ U.T                               # logit-lens logits at the same layer
        zf = hf @ U.T                               # the model's own final-layer logits
        model_top1 = zf.argmax(1)
        topk = np.argpartition(-zj, k - 1, axis=1)[:, :k]
        hits += int((topk == model_top1[:, None]).any(1).sum())
        agree += int((zj.argmax(1) == zg.argmax(1)).sum())
        m = zj - zj.mean(1, keepdims=True)
        sd = m.std(1, keepdims=True) + 1e-9
        kurt.append((((m / sd) ** 4).mean(1) - 3.0))
        pj = np.exp(zj - zj.max(1, keepdims=True)); pj /= pj.sum(1, keepdims=True)
        pg = np.exp(zg - zg.max(1, keepdims=True)); pg /= pg.sum(1, keepdims=True)
        lj, lg = np.log(pj + 1e-12), np.log(pg + 1e-12)
        kl.append(((pj * (lj - lg)).sum(1) + (pg * (lg - lj)).sum(1)))
    return dict(**{f"nexttok_top{k}": hits / n},
                top1_agree_logit=agree / n,
                kurtosis=float(np.concatenate(kurt).mean()),
                sym_kl_logit=float(np.concatenate(kl).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", type=Path, default=REPO_ROOT / "outputs/jlens/lens_v3_filtered40.pt")
    ap.add_argument("--subset", default=None,
                    help="Name from clean_floor.SUBSETS (n25/n50/n61/n89/n100): build the "
                         "lens from a nested subset of the shipped corpus instead of a file, "
                         "so the n-stability sweep uses the shipped construction throughout.")
    ap.add_argument("--states", type=Path, default=REPO_ROOT / "outputs/jlens/eval_states.pt")
    ap.add_argument("--lm-head", type=Path, default=WEIGHTS / "lm_head_weight.npy")
    ap.add_argument("--rms-norm", type=Path, default=WEIGHTS / "rms_norm_weight.npy")
    ap.add_argument("--n-layers-total", type=int, default=61,
                    help="For reindexing depth to [0,100] as the paper does")
    ap.add_argument("--layers", default=None, help="comma list; default every fitted layer")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--vocab-chunk", type=int, default=16384)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = str(args.threads)
    label = args.label or (args.subset or args.lens.stem)
    if args.out is None:
        args.out = REPO_ROOT / f"outputs/jlens/workspace_signatures_{label}.json"

    import torch
    if args.subset:
        from clean_floor import build_subset
        lens = build_subset(REPO_ROOT / "outputs/jlens",
                            REPO_ROOT / "outputs/jlens/per_prompt", 40.0, args.subset)
        print(f"subset {args.subset}: blocks {lens.spans}  n={lens.n}", flush=True)
    else:
        lens = LazyLens(args.lens)
    layers = ([int(x) for x in args.layers.split(",")] if args.layers else lens.layers)
    d = lens.cur.d if hasattr(lens, "cur") else lens.terms[0][0].d

    # Readout matrix: W_U folded with the final RMSNorm's learned per-dimension weight.
    W = np.load(args.lm_head)
    U = (W.astype(np.float32) * np.load(args.rms_norm).astype(np.float32)[None, :])
    del W
    print(f"readout U {U.shape} ({U.nbytes/1e9:.2f} GB)", flush=True)

    # G = U^T U, accumulated in vocab chunks so the Gram matrix never needs a copy of U.
    t0 = time.time()
    G, ubar = centred_gram(U, args.vocab_chunk)
    print(f"  centred Gram in {time.time()-t0:.0f}s", flush=True)

    states = torch.load(args.states, map_location="cpu", weights_only=False)
    H = np.concatenate([states["sets"][s]["H"].numpy() for s in sorted(states["sets"])])
    H_final = H[:, states["n_layers"] - 1]
    print(f"{H.shape[0]} eval states, final layer = {states['n_layers']-1}", flush=True)

    rows = []
    print(f"{'L':>3} {'depth':>6} {'eff50':>7} {'eff90':>7} {'PR':>7} {'cos':>7} "
          f"{'nextTop':>8} {'agree':>7} {'kurt':>9} {'symKL':>7}")
    for L in layers:
        t = time.time()
        J = lens[L]
        M = J.T @ G @ J
        r = {"layer": int(L), "depth_0_100": round(100 * L / (args.n_layers_total - 1), 1)}
        r.update(eff_dims(M, d))
        # mean over vocabulary of cos((U J)_t, U_t), in chunks
        r["cos_to_logit"], r["cos_to_logit_centred"] = cosines_to_logit(
            U, J, ubar, args.vocab_chunk)
        r.update(readout_stats((H[:, L], H_final), J, U, args.topk))
        r["secs"] = round(time.time() - t, 1)
        rows.append(r)
        print(f"{L:3d} {r['depth_0_100']:6.1f} {r['eff_dim_50']:7.4f} {r['eff_dim_90']:7.4f} "
              f"{r['part_ratio']:7.4f} {r['cos_to_logit']:7.4f} "
              f"{r[f'nexttok_top{args.topk}']:8.3f} {r['top1_agree_logit']:7.3f} "
              f"{r['kurtosis']:9.1f} {r['sym_kl_logit']:7.2f}", flush=True)

    out = {"lens": str(args.lens), "label": label, "n_prompts": lens.n, "d_model": d,
           "n_layers_total": args.n_layers_total, "topk": args.topk,
           "n_eval_states": int(H.shape[0]), "per_layer": rows}
    args.out.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
