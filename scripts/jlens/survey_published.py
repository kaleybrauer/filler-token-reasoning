"""
survey_published.py — exploratory cross-model geometry from the published lenses.

Stage 1 of the reviewer's plan: ask whether an obvious scale/architecture trend
exists in the published lenses before spending GPU on split-half refits. Only the
two signatures that need no model execution are computed here, which is what makes
39 models affordable:

  eff_dim_{50,90}      fraction of residual-stream dimensions holding 50% / 90% of
                       the variance across the J-lens vectors U J_l  (the paper's
                       own J-space dimensionality measure, reported as a fraction of
                       d_model so it is comparable across widths)
  part_ratio           participation ratio of the same spectrum, over d_model
  cos_to_logit_global  cosine between the J-lens and logit-lens readout directions

U is the unembedding folded with the final norm's learned per-dimension weight.
Depth is reindexed to [0,100] so curves from models of different depth align.

⚠ The activation-dependent signatures (next-token agreement, kurtosis, symmetric KL)
are NOT here: they need forward passes through each model. And a published lens gives
ONE curve with no error bar -- a within-model floor needs two split-half refits, which
this script cannot produce. Treat every curve here as exploratory.

⚠ LayerNorm models (GPT-2, Pythia) centre before scaling, so folding the norm weight
is exact for the scale but drops the mean-subtraction; their cosine values carry a
small extra approximation. RMSNorm models (Qwen, Gemma, Llama, gpt-oss) are exact.

    python scripts/jlens/survey_published.py --model Qwen/Qwen3-4B --family qwen3-4b
"""

from __future__ import annotations

import argparse, gc, json, os, sys, time
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from workspace_signatures import eff_dims  # noqa: E402

LENS_REPO = "neuronpedia/jacobian-lens"
NORM_KEYS = ["model.norm.weight", "model.language_model.norm.weight",
             "transformer.ln_f.weight", "ln_f.weight",
             "gpt_neox.final_layer_norm.weight"]
HEAD_KEYS = ["lm_head.weight", "embed_out.weight", "wte.weight",
             "model.embed_tokens.weight", "transformer.wte.weight"]


def pick(index: dict, keys):
    """First (tensor name, shard file) in `keys` that the index actually has."""
    wm = index["weight_map"]
    for k in keys:
        if k in wm:
            return k, wm[k]
    return None, None


def load_readout(model_id: str, cache: Path, token=None):
    """U = W_unembed * final_norm_weight, pulling only the shards that hold them."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    try:
        idx = json.load(open(hf_hub_download(model_id, "model.safetensors.index.json",
                                             cache_dir=cache, token=token)))
    except Exception:
        idx = {"weight_map": {}}
    if not idx["weight_map"]:                       # single-shard model
        f = hf_hub_download(model_id, "model.safetensors", cache_dir=cache, token=token)
        with safe_open(f, framework="np") as h:
            names = list(h.keys())
            idx = {"weight_map": {n: "model.safetensors" for n in names}}
    hk, hf_ = pick(idx, HEAD_KEYS)
    nk, nf = pick(idx, NORM_KEYS)
    if hk is None:
        raise SystemExit(f"{model_id}: no unembedding among {HEAD_KEYS}; has "
                         f"{list(idx['weight_map'])[:6]}")
    out = {}
    for key, shard in [(hk, hf_), (nk, nf)]:
        if key is None:
            continue
        p = hf_hub_download(model_id, shard, cache_dir=cache, token=token)
        with safe_open(p, framework="np") as h:
            out[key] = np.asarray(h.get_tensor(key)).astype(np.float32)
    W = out[hk]
    g = out.get(nk)
    tied = hk in ("model.embed_tokens.weight", "wte.weight", "transformer.wte.weight")
    if g is not None:
        if g.shape[0] != W.shape[1]:
            raise SystemExit(f"{model_id}: norm {g.shape} vs unembed {W.shape}")
        W = W * g[None, :]
    return W, dict(unembed_key=hk, norm_key=nk, tied_embeddings=bool(tied),
                   has_norm=g is not None)



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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id, for the unembedding")
    ap.add_argument("--family", required=True, help="directory name in the lens repo")
    ap.add_argument("--lens-file", default=None, help="override the lens path in the repo")
    ap.add_argument("--revision", default="qwen-n1000")
    ap.add_argument("--cache", type=Path, default=Path("/workspace/.cache/huggingface"))
    ap.add_argument("--every", type=int, default=1, help="compute every Nth layer")
    ap.add_argument("--vocab-chunk", type=int, default=16384)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--outdir", type=Path, default=REPO_ROOT / "outputs/jlens_survey")
    args = ap.parse_args()
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = str(args.threads)
    token = os.environ.get("HF_TOKEN")

    import torch
    from huggingface_hub import hf_hub_download, list_repo_files

    lens_file = args.lens_file
    if lens_file is None:
        files = [f for f in list_repo_files(LENS_REPO, revision=args.revision)
                 if f.startswith(args.family + "/") and f.endswith(".pt")]
        if not files:
            raise SystemExit(f"no lens under {args.family}/ in {LENS_REPO}")
        lens_file = sorted(files, key=lambda f: ("n1000" not in f, f))[0]
    print(f"lens  {lens_file}", flush=True)
    lp = hf_hub_download(LENS_REPO, lens_file, revision=args.revision,
                         cache_dir=args.cache, token=token)
    ck = torch.load(lp, map_location="cpu", weights_only=True)
    Js = ck["J"]
    layers = sorted(int(x) for x in Js)
    d = int(ck.get("d_model") or Js[layers[0]].shape[0])
    n_prompts = int(ck.get("n_prompts", -1))
    print(f"  {len(layers)} layers [{layers[0]}..{layers[-1]}], d={d}, n_prompts={n_prompts}",
          flush=True)

    U, meta = load_readout(args.model, args.cache, token)
    print(f"  readout {U.shape} via {meta['unembed_key']}"
          f"{' (tied)' if meta['tied_embeddings'] else ''}"
          f"{'' if meta['has_norm'] else '  [NO FINAL NORM FOUND]'}", flush=True)

    t0 = time.time()
    G, ubar = centred_gram(U, args.vocab_chunk)
    print(f"  centred Gram in {time.time()-t0:.0f}s", flush=True)

    sel = [L for i, L in enumerate(layers) if i % args.every == 0]
    depth_den = max(layers) if max(layers) > 0 else 1
    rows = []
    for L in sel:
        t = time.time()
        J = Js[L].float().numpy() if hasattr(Js[L], "float") else np.asarray(Js[L], np.float32)
        r = {"layer": int(L), "depth_0_100": round(100 * L / depth_den, 1)}
        r.update(eff_dims(J.T @ G @ J, d))
        r["cos_to_logit"], r["cos_to_logit_centred"] = cosines_to_logit(
            U, J, ubar, args.vocab_chunk)
        r["secs"] = round(time.time() - t, 1)
        rows.append(r)
        print(f"  L{L:3d} depth {r['depth_0_100']:5.1f}  eff50 {r['eff_dim_50']:.4f}  "
              f"eff90 {r['eff_dim_90']:.4f}  PR {r['part_ratio']:.4f}  "
              f"cos {r['cos_to_logit']:.4f}", flush=True)
        del J; gc.collect()

    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / f"survey_{args.family}.json"
    out.write_text(json.dumps({"model": args.model, "family": args.family,
                               "lens_file": lens_file, "revision": args.revision,
                               "d_model": d, "n_layers": len(layers),
                               "n_prompts": n_prompts, "readout": meta,
                               "per_layer": rows}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
