"""
kurtosis_diagnostic.py — is V3's early readout kurtosis a property of the transport?

Hypothesis. If J_L is dominated by one direction, then J_L h is roughly a multiple of the top
left-singular vector u1 whatever h is, and the readout logits are roughly U u1 -- the same
peaked vector for every activation. Readout kurtosis would then measure the unembedding
kurtosis of a single transport direction, not content.

Per plotted layer this reports:
  s1_share   top singular value / Frobenius norm of J_L  (1.0 = rank one)
  kurt_u1    excess kurtosis across the vocabulary of U u1, U = W_U * final-norm weight
  top_u1     the tokens U u1 pushes hardest
and compares against the measured readout kurtosis from the WikiText panels.

A DIAGNOSTIC, not a proposed lens. Low-memory: one lens layer at a time, and the
unembedding memory-mapped and read in chunks.
"""
import json, os, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("OMP_NUM_THREADS", "2")
from score_lazy import LazyLens
from extract.extract_hidden_states import load_tokenizer

W = np.load(REPO / "data/model_weights/deepseek_v3/lm_head_weight.npy", mmap_mode="r")
g = np.load(REPO / "data/model_weights/deepseek_v3/rms_norm_weight.npy").astype(np.float32)
tok = load_tokenizer("/workspace/models/deepseek-v3-awq")
lens = LazyLens(REPO / "outputs/jlens/lens_v3_filtered40.pt")
rd = json.load(open(REPO / "outputs/jlens/workspace_readouts_shipped100.json"))
meas = {r["layer"]: r for r in rd["per_layer"]}


def top_pair(J, iters=40):
    v = np.random.default_rng(0).standard_normal(J.shape[1]).astype(np.float32)
    for _ in range(iters):
        v = J.T @ (J @ v); v /= np.linalg.norm(v)
    u = J @ v; s = float(np.linalg.norm(u))
    return u / s, s


def unembed(u, chunk=16384):
    out = np.empty(W.shape[0], np.float32)
    for s in range(0, W.shape[0], chunk):
        out[s:s + chunk] = (np.asarray(W[s:s + chunk], np.float32) * g) @ u
    return out


rows = []
print(f"{'L':>3} {'depth':>6} {'s1_share':>9} {'kurt(U u1)':>11} {'meas p50':>9} {'meas p99':>9}   top tokens of U u1")
for L in [l for l in rd["layers"] if l < 60]:
    J = lens[L]
    u1, s1 = top_pair(J)
    z = unembed(u1)
    c = z - z.mean()
    k = float((c ** 4).mean() / (c ** 2).mean() ** 2 - 3)
    top = [tok.decode([int(i)]) for i in np.argsort(-z)[:5]]
    r = dict(layer=L, depth=meas[L]["depth_0_100"], s1_share=s1 / float(np.linalg.norm(J)),
             kurt_u1=k, meas_p50=meas[L]["kurtosis_p50"], meas_p99=meas[L]["kurtosis_p99"], top=top)
    rows.append(r)
    print(f"{L:3d} {r['depth']:6.1f} {r['s1_share']:9.3f} {k:11.1f} {r['meas_p50']:9.2f} {r['meas_p99']:9.2f}   {top}", flush=True)
x = np.array([r["s1_share"] for r in rows]); y = np.array([r["meas_p99"] for r in rows])
ku = np.array([r["kurt_u1"] for r in rows])
print(f"\ncorr(s1_share, measured p99) = {np.corrcoef(x, y)[0,1]:.3f}")
print(f"corr(kurt(U u1), measured p99) = {np.corrcoef(ku, y)[0,1]:.3f}")
(REPO / "outputs/jlens/kurtosis_diagnostic.json").write_text(json.dumps(rows, indent=1))
