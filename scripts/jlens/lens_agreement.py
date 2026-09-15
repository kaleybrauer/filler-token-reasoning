"""
lens_agreement.py — how much do two independent J-lens fits of the same model agree, layer by layer?

For Qwen3.5-397B: the 500-prompt lens (FP8 weights, bf16 compute) against praxagent's 24-prompt lens
(bf16 weights). For DeepSeek-V3: the two disjoint 50-prompt halves of the shipped lens, the reference for
what fitting noise alone does. Per layer:
  rel_frob         |J_a - J_b|_F / |J_a|_F
  cos_frob         <J_a, J_b>_F / (|J_a| |J_b|)
  readout_cos      mean over sampled tokens t of cos(J_a^T u_t, J_b^T u_t), u_t the centred unembedding row:
                   do the two lenses read token t out of the same residual direction?
  top_dir_cos      |cos| between the two lenses' top output singular directions
A lens argument is a file path or, for V3, a clean_floor.SUBSETS name (n50, n50b, ...).

    python scripts/jlens/lens_agreement.py --lens-a n50 --lens-b n50b --out outputs/jlens/lens_agreement_v3_halves.json
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(Path(__file__).resolve().parent))


def open_lens(spec):
    from clean_floor import SUBSETS, build_subset
    from score_lazy import LazyLens
    if spec in SUBSETS:
        return build_subset(REPO / "outputs/jlens", REPO / "outputs/jlens/per_prompt", 40.0, spec)
    return LazyLens(Path(spec))


def top_output_dir(J, iters=40, seed=0):
    v = np.random.default_rng(seed).standard_normal(J.shape[1])
    for _ in range(iters):
        v = J.T @ (J @ v); v /= np.linalg.norm(v)
    u = J @ v
    return u / np.linalg.norm(u)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens-a", required=True)
    ap.add_argument("--lens-b", required=True)
    ap.add_argument("--unembed-dir", type=Path, default=REPO / "data/model_weights/deepseek_v3")
    ap.add_argument("--n-layers-total", type=int, default=61)
    ap.add_argument("--n-tokens", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = str(args.threads)
    W = np.load(args.unembed_dir / "lm_head_weight.npy", mmap_mode="r")
    g = np.load(args.unembed_dir / "rms_norm_weight.npy").astype(np.float64)
    V = W.shape[0]
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(V, args.n_tokens, replace=False))
    mean_row = sum(np.asarray(W[s:s + 16384], np.float64).sum(0) for s in range(0, V, 16384)) / V
    Ut = (np.asarray(W[idx], np.float64) - mean_row) * g                     # [n, d] centred readout rows
    A, B = open_lens(args.lens_a), open_lens(args.lens_b)
    layers = sorted(set(A.layers) & set(B.layers))
    rows = []
    print(f"{len(layers)} shared layers; {args.n_tokens} sampled tokens", flush=True)
    for L in layers:
        t0 = time.time()
        Ja, Jb = np.asarray(A[L], np.float64), np.asarray(B[L], np.float64)
        na, nb = np.linalg.norm(Ja), np.linalg.norm(Jb)
        Pa, Pb = Ut @ Ja, Ut @ Jb
        rc = (Pa * Pb).sum(1) / (np.linalg.norm(Pa, axis=1) * np.linalg.norm(Pb, axis=1) + 1e-12)
        r = {"layer": int(L), "depth_0_100": round(100 * L / (args.n_layers_total - 1), 2),
             "rel_frob": float(np.linalg.norm(Ja - Jb) / na), "cos_frob": float((Ja * Jb).sum() / (na * nb)),
             "readout_cos_mean": float(rc.mean()), "readout_cos_p10": float(np.percentile(rc, 10)),
             "top_dir_cos": float(abs(top_output_dir(Ja) @ top_output_dir(Jb))), "secs": round(time.time() - t0, 1)}
        rows.append(r)
        print(f"  L{L:3d} depth {r['depth_0_100']:5.1f}  rel_frob {r['rel_frob']:.3f}  cos_frob {r['cos_frob']:.3f}  "
              f"readout_cos {r['readout_cos_mean']:.3f} (p10 {r['readout_cos_p10']:.3f})  top_dir {r['top_dir_cos']:.3f}", flush=True)
        del Ja, Jb, Pa, Pb
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"lens_a": args.lens_a, "lens_b": args.lens_b, "unembed_dir": str(args.unembed_dir),
                                    "n_tokens": args.n_tokens, "per_layer": rows}, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
