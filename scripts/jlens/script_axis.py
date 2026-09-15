"""
script_axis.py — is V3's dominant J-lens direction really a script axis, and whose is it?

Two questions, cheap, no activations:
 1. Classify every vocabulary token by script (CJK / Latin letters / other) and measure how
    cleanly the projection onto a direction separates CJK from Latin tokens (AUC; 0.5 = no
    separation, 0 or 1 = perfect).
 2. Compare the lens's dominant output direction u1 with the top principal direction of the
    centred unembedding itself. If they coincide, the script split is a property of V3's output
    embedding geometry that any lens -- the logit lens included -- would inherit, not something
    the Jacobian discovered.
Low-memory: the unembedding is memory-mapped and read in chunks.
"""
import os, sys, json
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("OMP_NUM_THREADS", "2")
from score_lazy import LazyLens
from extract.extract_hidden_states import load_tokenizer

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--lens", type=Path, default=REPO / "outputs/jlens/lens_v3_filtered40.pt")
ap.add_argument("--unembed-dir", type=Path, default=REPO / "data/model_weights/deepseek_v3",
                help="lm_head_weight.npy + rms_norm_weight.npy (effective multiplier)")
ap.add_argument("--tokenizer", default="/workspace/models/deepseek-v3-awq",
                help="V3's tokenizer.json path (loaded with load_tokenizer) or a hub id (AutoTokenizer)")
ap.add_argument("--layers", nargs="+", type=int, default=[0, 15, 30, 45, 59])
ap.add_argument("--out", type=Path, default=REPO / "outputs/jlens/script_axis.json")
args = ap.parse_args()
W = np.load(args.unembed_dir / "lm_head_weight.npy", mmap_mode="r")
g = np.load(args.unembed_dir / "rms_norm_weight.npy").astype(np.float32)
V, d = W.shape
if Path(args.tokenizer).exists():
    tok = load_tokenizer(args.tokenizer)
else:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
n_tok = len(tok)

def script(s):
    if any("\u4e00" <= c <= "\u9fff" or "\u3400" <= c <= "\u4dbf" for c in s): return "cjk"
    if any(("a" <= c <= "z") or ("A" <= c <= "Z") for c in s): return "latin"
    return "other"
cls = np.array([script(tok.decode([i])) if i < n_tok else "other" for i in range(V)])
cnt = {k: int((cls == k).sum()) for k in ("cjk", "latin", "other")}
print("vocabulary by script:", cnt, {k: round(v / V, 3) for k, v in cnt.items()}, flush=True)

def project(u, chunk=16384):
    out = np.empty(V, np.float32)
    for s in range(0, V, chunk):
        out[s:s+chunk] = (np.asarray(W[s:s+chunk], np.float32) * g) @ u
    return out

def auc(pos, neg):
    x = np.concatenate([pos, neg]); r = np.empty(len(x)); r[np.argsort(x)] = np.arange(1, len(x) + 1)
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

def report(name, u):
    z = project(u)
    a = auc(z[cls == "cjk"], z[cls == "latin"])
    sep = max(a, 1 - a)
    print(f"  {name:28s} CJK-vs-Latin AUC {a:.3f} (separation {sep:.3f})  "
          f"mean proj  cjk {z[cls=='cjk'].mean():+.3f}  latin {z[cls=='latin'].mean():+.3f}  "
          f"other {z[cls=='other'].mean():+.3f}", flush=True)
    return z, sep

# top principal directions of the centred unembedding
ubar = np.zeros(d, np.float64)
for s in range(0, V, 16384): ubar += (np.asarray(W[s:s+16384], np.float32) * g).sum(0)
ubar = (ubar / V).astype(np.float32)
G = np.zeros((d, d), np.float64)
for s in range(0, V, 16384):
    C = np.asarray(W[s:s+16384], np.float32) * g - ubar
    G += (C.T @ C).astype(np.float64)
ev, evec = np.linalg.eigh(G)
ev, evec = ev[::-1], evec[:, ::-1]
print(f"\nunembedding PCA: top-5 variance shares {np.round(ev[:5] / ev.sum(), 4).tolist()}", flush=True)
print("\n[1] how script-aligned are directions?")
pcs = {}
for i in range(4):
    _, sep = report(f"unembedding PC{i+1}", evec[:, i].astype(np.float32)); pcs[i] = evec[:, i]
rng = np.random.default_rng(0)
report("random direction (control)", (lambda r: r / np.linalg.norm(r))(rng.standard_normal(d).astype(np.float32)))

print("\n[2] the lens's dominant output direction u1, by layer")
lens = LazyLens(args.lens)
out = {"vocab_by_script": cnt, "pc_variance_share": (ev[:5] / ev.sum()).tolist(), "layers": {}}
for L in args.layers:
    J = lens[L]
    v = rng.standard_normal(d).astype(np.float32)
    for _ in range(40): v = J.T @ (J @ v); v /= np.linalg.norm(v)
    u1 = J @ v; u1 /= np.linalg.norm(u1)
    _, sep = report(f"u1 at layer {L}", u1)
    cos_pc1 = abs(float(u1 @ pcs[0]))
    best = max(range(4), key=lambda i: abs(float(u1 @ pcs[i])))
    print(f"      |cos(u1, PC1)| {cos_pc1:.3f}   most aligned unembedding PC: PC{best+1} "
          f"(|cos| {abs(float(u1 @ pcs[best])):.3f})", flush=True)
    out["layers"][L] = {"script_separation": sep, "cos_pc1": cos_pc1, "best_pc": best + 1}
args.out.write_text(json.dumps(out, indent=1))
