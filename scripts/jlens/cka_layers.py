"""
cka_layers.py — layer-by-layer similarity of the J-space, and V3's own workspace band.

This is the measurement behind the paper's Figure 27, the one that DEFINES the three
functional regions: "For each pair of layers, we compute the similarity of the J-space's
geometry. We do so using centered kernel alignment (CKA), which compares, for each pair
of layers, the matrices of pairwise similarities among J-lens vectors... The resulting
matrix has a clear block structure: an early block encompassing roughly the first third
of the model, a long middle block, and a small late block."

We need it because shading THEIR band on our figure would beg the question. Deriving V3's
band the way they derived theirs makes the comparison a result rather than an assumption.

It needs no activations. Linear CKA between the J-lens vector sets X = U_c J_a and
Y = U_c J_b (U_c = the mean-centred readout, i.e. unembedding folded with the final norm)
reduces to the d x d Gram:

    CKA(a,b) = ||J_a' G J_b||_F^2 / ( ||J_a' G J_a||_F * ||J_b' G J_b||_F ),   G = U_c' U_c

so the whole matrix costs two d-cubed products per layer pair and never forms a
vocabulary-sized matrix. Centring is what makes it *centered* kernel alignment, and it
matters here for the same reason as in the dimensionality measure: unembedding rows share
a large common direction that would otherwise dominate every entry.

Band: a 3-block segmentation chosen to maximise mean within-block CKA minus mean
between-block CKA, over all boundary pairs -- the block structure the paper reads off by
eye, picked by a stated criterion instead.

    python scripts/jlens/cka_layers.py --lens outputs/jlens/lens_v3_filtered40.pt
"""

from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_eval_states import WEIGHTS                    # noqa: E402
from workspace_signatures import centred_gram            # noqa: E402
from score_lazy import LazyLens                          # noqa: E402


def three_blocks(C):
    """Boundaries (b1, b2) maximising within-block minus between-block mean CKA."""
    n = C.shape[0]
    best = None
    for b1 in range(2, n - 3):
        for b2 in range(b1 + 2, n - 1):
            segs = [range(0, b1), range(b1, b2), range(b2, n)]
            win = bet = wn = bn = 0.0
            for i, s1 in enumerate(segs):
                for j, s2 in enumerate(segs):
                    blk = C[np.ix_(list(s1), list(s2))]
                    if i == j:
                        win += blk.sum(); wn += blk.size
                    else:
                        bet += blk.sum(); bn += blk.size
            score = win / wn - bet / bn
            if best is None or score > best[0]:
                best = (score, b1, b2)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", type=Path, default=REPO / "outputs/jlens/lens_v3_filtered40.pt")
    ap.add_argument("--lm-head", type=Path, default=WEIGHTS / "lm_head_weight.npy")
    ap.add_argument("--rms-norm", type=Path, default=WEIGHTS / "rms_norm_weight.npy")
    ap.add_argument("--n-layers-total", type=int, default=61)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", type=Path, default=REPO / "outputs/jlens/cka_layers.json")
    args = ap.parse_args()
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = str(args.threads)

    # Gram first, then release the unembedding before the lens is loaded: both at once
    # is 3.7 GB + 6.3 GB and the box has 16.
    U = (np.load(args.lm_head).astype(np.float32)
         * np.load(args.rms_norm).astype(np.float32)[None, :])
    t0 = time.time()
    G, _ = centred_gram(U)
    d = G.shape[0]
    del U
    print(f"centred Gram {G.shape} in {time.time()-t0:.0f}s", flush=True)

    lens = LazyLens(args.lens)
    layers = lens.layers
    n = len(layers)
    print(f"loading {n} layers fp16 ({n * d * d * 2 / 1e9:.1f} GB)", flush=True)
    J = np.empty((n, d, d), dtype=np.float16)
    for i, L in enumerate(layers):
        J[i] = lens[L].astype(np.float16)

    C = np.zeros((n, n), dtype=np.float64)
    diag = np.zeros(n)
    t0 = time.time()
    for a in range(n):
        A = J[a].astype(np.float32).T @ G          # J_a' G
        for b in range(a, n):
            M = A @ J[b].astype(np.float32)        # J_a' G J_b
            v = float(np.linalg.norm(M)) ** 2
            if a == b:
                diag[a] = np.sqrt(v)               # ||J_a' G J_a||_F
            C[a, b] = C[b, a] = v
        el = time.time() - t0
        print(f"  layer {a+1}/{n}  {el/(a+1):.1f}s/row  "
              f"eta {(n-a-1)*el/(a+1)/60:.1f} min", flush=True)
    C /= np.outer(diag, diag)

    score, b1, b2 = three_blocks(C)
    depth = lambda L: 100 * L / (args.n_layers_total - 1)
    band = (depth(layers[b1]), depth(layers[b2]))
    print(f"\n3-block segmentation: layers [0..{layers[b1]-1}] "
          f"[{layers[b1]}..{layers[b2]-1}] [{layers[b2]}..{layers[-1]}]")
    print(f"  separation score {score:.4f}")
    print(f"  middle block = depth {band[0]:.1f}-{band[1]:.1f}%   "
          f"(Sonnet 4.5 reported 37.5-91.7%)")

    args.out.write_text(json.dumps(
        {"lens": str(args.lens), "layers": layers, "n_layers_total": args.n_layers_total,
         "cka": C.round(5).tolist(), "boundaries_layer": [int(layers[b1]), int(layers[b2])],
         "band_depth": list(band), "separation_score": float(score),
         "paper_band_depth": [37.5, 91.7]}, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
