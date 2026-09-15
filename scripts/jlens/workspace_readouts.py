"""
workspace_readouts.py — Figure 28 panels (a), (b), (c) on held-out pretraining-like text.

Reads the multi-position WikiText residuals (extract_wikitext_states.py: held-out corpus
prompts 200-299, 112 positions after the 16 attention-sink positions, all 61 layers) and
applies a J-lens at each selected layer:

  (a) next-token prediction accuracy, for every k in TOPKS: the fraction of positions at
      which any top-k J-lens token matches the MODEL'S OWN top-1 prediction at that
      position (the final-layer readout, which G-UNEMBED verified against the model).
  (b) excess kurtosis of each activation's readout logit vector over the vocabulary,
      reported as percentiles across activations -- "the excess kurtosis of the logit
      distribution for the readout of a single (position, layer) across a large data set
      of activations".
  (c) top-1 autocorrelation across positions: for the lens's top-1 token at position t,
      its log-probability at position t+delta, minus the same quantity at a position drawn
      from a within-prompt shuffle -- "Delta log p (vs null)". Log-probabilities depend on
      the logit scale, so the readout applies the full RMSNorm, not just its weight.

Default layers are 25 evenly spaced across the 61, the subsampling the paper's figure uses
(its own text notes subsampling sharpens transitions, so matching it matters for a direct
comparison). Layer 60 is the target, where the lens is the identity by construction.

One state is excluded: prompt 215 at stored position 69, whose layer-60 residual saturates
fp16 (four components at 65504, inside the model's own forward). It feeds the model's top-1
prediction used by panel (a), so the position is dropped at every layer.

Regime: these activations were extracted one prompt at a time while the lens was fitted in
batches of 64 copies. A GPU check on 32 held-out prompts found panel (a) accuracy identical
across the two regimes to within 0.3 points at every layer, so they are used as extracted.

    python scripts/jlens/workspace_readouts.py --lens outputs/jlens/lens_v3_filtered40.pt
    python scripts/jlens/workspace_readouts.py --subset n50
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_lazy import LazyLens                # noqa: E402
import models                                  # noqa: E402

TOPKS = (1, 2, 4, 8, 16, 32, 64, 128)
KURT_PCTS = (1, 10, 25, 50, 75, 90, 99)
OFFSETS = (1, 2, 4, 8, 16, 32)


def logits_of(x, U, eps):
    """Full readout: RMSNorm (scale included, weight folded into U) then unembed."""
    x = x / np.sqrt((x.astype(np.float64) ** 2).mean(-1, keepdims=True) + eps).astype(np.float32)
    return x @ U.T


def log_softmax(z):
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="v3", help="models.MODELS key: v3, qwen35_122b, qwen35_397b_fp8")
    ap.add_argument("--states", type=Path, default=None, help="default: the model's WikiText states")
    ap.add_argument("--lens", type=Path, default=None, help="default: the model's lens")
    ap.add_argument("--subset", default=None, help="build the lens from a named half/subset")
    ap.add_argument("--n-layers-shown", type=int, default=25,
                    help="evenly spaced layers, as the paper's figure shows")
    ap.add_argument("--max-prompts", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = str(args.threads)
    import torch

    m = models.get(args.model)
    args.states = args.states or m["states"]["wikitext"]
    args.lens = args.lens or m["lens"]
    label = args.label or (args.subset or ("shipped100" if args.model == "v3" else "shipped"))
    out_path = args.out or (REPO / f"outputs/jlens/workspace_readouts_{label}.json" if args.model == "v3"
                            else models.QS / args.model / "analysis" / f"workspace_readouts_{label}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.subset:
        if args.model != "v3":
            raise SystemExit("--subset builds V3 half/subset lenses only")
        from clean_floor import build_subset
        lens = build_subset(REPO / "outputs/jlens", REPO / "outputs/jlens/per_prompt", 40.0, args.subset)
    else:
        lens = LazyLens(args.lens)
    print(f"lens: {label}  n={lens.n}", flush=True)

    S = torch.load(args.states, map_location="cpu", weights_only=False, mmap=True)
    if not S["unembed_check"]["passed"]:
        raise SystemExit("WikiText states failed G-UNEMBED")
    H = S["H"]                                   # [P, T, L, d] fp16
    P, T, NL, d = H.shape
    if args.max_prompts:
        P = min(P, args.max_prompts)
    eps = float(S["rms_norm_eps"])
    lo = S["prompt_span"][0]
    n_total = S["n_layers"]
    target = n_total - 1
    layers = sorted({int(round(x)) for x in np.linspace(0, target, args.n_layers_shown)})
    print(f"states {tuple(H.shape)}; prompts {lo}..{lo+P-1}; layers {layers}", flush=True)

    # Exclude any position whose residual reaches the fp16 limit at a layer read here (V3: prompt 215,
    # stored position 69 at layer 60, clipped inside the model's own forward; Qwen3.5: values the
    # extraction clamped), at every layer.
    keep = np.ones((P, T), bool)
    for L in sorted(set(layers) | {target}):
        keep &= ~(H[:P, :, L].float().abs() >= 65504).any(-1).numpy()
    excluded = [(int(p) + lo, int(t)) for p, t in np.argwhere(~keep)]
    print(f"excluded {len(excluded)} saturated state(s): {excluded[:10]}", flush=True)
    if args.model == "v3" and args.states == m["states"]["wikitext"] and P == H.shape[0] and excluded != m["saturated"]:
        raise SystemExit(f"saturation scan found {excluded}, expected {m['saturated']}")

    U = (np.load(m["unembed_dir"] / "lm_head_weight.npy").astype(np.float32)
         * np.load(m["unembed_dir"] / "rms_norm_weight.npy").astype(np.float32)[None, :])

    # The model's own next-token prediction at every position: its final-layer readout.
    t0 = time.time()
    model_top1 = np.empty((P, T), np.int64)
    for p in range(P):
        model_top1[p] = logits_of(H[p, :, target].float().numpy(), U, eps).argmax(1)
    print(f"model top-1 computed in {time.time()-t0:.0f}s", flush=True)

    rng = np.random.default_rng(args.seed)
    perms = [rng.permutation(T) for _ in range(P)]
    kmax = max(TOPKS)
    rows = []
    print(f"{'L':>3} {'depth':>6} {'top1':>6} {'top8':>6} {'top128':>7} {'kurt p50':>9} "
          f"{'kurt p99':>9} {'ac d1':>7} {'ac d32':>7}", flush=True)
    for L in layers:
        tl = time.time()
        J = None if L >= target else lens[L]
        hits = {k: 0 for k in TOPKS}
        n_used = 0
        kurt_all = []
        ac_num = {dlt: 0.0 for dlt in OFFSETS}
        ac_null = {dlt: 0.0 for dlt in OFFSETS}
        ac_n = {dlt: 0 for dlt in OFFSETS}
        for p in range(P):
            h = H[p, :, L].float().numpy()
            if J is not None:
                h = h @ J.T
            z = logits_of(h, U, eps)                               # [T, V]
            m = keep[p]
            # (a)
            part = np.argpartition(-z, kmax - 1, axis=1)[:, :kmax]
            order = np.take_along_axis(part, np.argsort(-np.take_along_axis(z, part, 1), 1), 1)
            for k in TOPKS:
                hits[k] += int(((order[:, :k] == model_top1[p][:, None]).any(1) & m).sum())
            n_used += int(m.sum())
            # (b)
            c = z - z.mean(1, keepdims=True)
            ku = ((c / (c.std(1, keepdims=True) + 1e-9)) ** 4).mean(1) - 3.0
            kurt_all.append(ku[m])
            # (c)
            lp = log_softmax(z)
            top1 = order[:, 0]
            perm = perms[p]
            for dlt in OFFSETS:
                t = np.arange(T - dlt)
                ok = m[t] & m[t + dlt] & m[perm[t]]
                if not ok.any():
                    continue
                tt = t[ok]
                ac_num[dlt] += float(lp[tt + dlt, top1[tt]].sum())
                ac_null[dlt] += float(lp[perm[tt], top1[tt]].sum())
                ac_n[dlt] += int(ok.sum())
        ku = np.concatenate(kurt_all)
        r = {"layer": int(L), "depth_0_100": round(100 * L / target, 2),
             **{f"nexttok_top{k}": hits[k] / n_used for k in TOPKS},
             **{f"kurtosis_p{q}": float(np.percentile(ku, q)) for q in KURT_PCTS},
             **{f"autocorr_d{dlt}": (ac_num[dlt] - ac_null[dlt]) / max(ac_n[dlt], 1) for dlt in OFFSETS},
             "n_activations": n_used, "secs": round(time.time() - tl, 1)}
        rows.append(r)
        print(f"{L:3d} {r['depth_0_100']:6.1f} {r['nexttok_top1']:6.3f} {r['nexttok_top8']:6.3f} "
              f"{r['nexttok_top128']:7.3f} {r['kurtosis_p50']:9.3f} {r['kurtosis_p99']:9.2f} "
              f"{r['autocorr_d1']:7.3f} {r['autocorr_d32']:7.3f}   ({r['secs']}s)", flush=True)

    out = {"label": label, "lens": str(args.lens) if not args.subset else None, "subset": args.subset,
           "n_prompts_lens": lens.n, "states": str(args.states), "prompt_span": [lo, lo + P],
           "n_positions": T, "n_layers_total": n_total, "layers": layers, "excluded": excluded, "model": args.model,
           "topks": list(TOPKS), "kurt_pcts": list(KURT_PCTS), "offsets": list(OFFSETS),
           "per_layer": rows}
    out_path.write_text(json.dumps(out, indent=1))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
