"""
giant_structure.py — what does a giant-norm block Jacobian look like?

For a block mean J (from two consecutive snapshots) at a few layers: top
singular pair (u, v) by power iteration, how much of ||J||_F it carries, how
sparse v is (a massive-activation coordinate would show as one dominant entry),
and the unembedding of u (which tokens the giant direction pushes on).
"""
import argparse, json, os, sys, zipfile
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extract.extract_hidden_states import load_tokenizer

D = 7168
W = Path("data/model_weights/deepseek_v3")


def read_layer(path, L):
    zf = zipfile.ZipFile(path)
    pre = next(n for n in zf.namelist() if "/data/" in n).rsplit("/data/", 1)[0]
    with zf.open(f"{pre}/data/{L}") as f:
        buf = f.read()
    fd = os.open(path, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
    return np.frombuffer(buf, dtype=np.float32).reshape(D, D)


def top_pair(J, iters=30):
    v = np.random.default_rng(0).standard_normal(D).astype(np.float32)
    for _ in range(iters):
        v = J.T @ (J @ v); v /= np.linalg.norm(v)
    u = J @ v; s = np.linalg.norm(u); u /= s
    return u, v, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="10,30,50")
    ap.add_argument("--out", type=Path, default=Path("outputs/jlens/giant_structure.json"))
    args = ap.parse_args()
    tok = load_tokenizer("/workspace/models/deepseek-v3-awq")
    lm = np.load(W / "lm_head_weight.npy").astype(np.float32)
    rms_w = np.load(W / "rms_norm_weight.npy").astype(np.float32)
    D_ = Path("outputs/jlens")
    blocks = {"0..9 (clean-ish)": (None, D_ / "lens_v3.snap10.fitckpt", 10),
              "20..29 (giant 3947)": (D_ / "lens_v3.snap20.fitckpt", D_ / "lens_v3.snap30.fitckpt", 10),
              "30..39 (clean)": (D_ / "lens_v3.snap30.fitckpt", D_ / "lens_v3.snap40.fitckpt", 10),
              "60..99 (giant 5038)": (D_ / "lens_v3.snap60.fitckpt", D_ / "lens_v3.fitckpt", 40)}
    out = {}
    for L in [int(x) for x in args.layers.split(",")]:
        for name, (a, b, n) in blocks.items():
            J = read_layer(b, L)
            if a is not None:
                J = J - read_layer(a, L)
            J = J / n
            u, v, s = top_pair(J)
            fro = float(np.linalg.norm(J))
            vs = np.sort(np.abs(v))[::-1]
            us = np.sort(np.abs(u))[::-1]
            # unembed u: RMSNorm scale is irrelevant for the ranking, so use norm-weight * u
            logits = (u * rms_w) @ lm.T
            top = [tok.decode([int(i)]) for i in np.argsort(-logits)[:8]]
            bot = [tok.decode([int(i)]) for i in np.argsort(logits)[:8]]
            r = {"fro": fro, "top_sv": float(s), "sv_share": float(s / fro),
                 "v_top3_abs": vs[:3].tolist(), "v_top_coord": int(np.argmax(np.abs(v))),
                 "u_top3_abs": us[:3].tolist(), "u_top_coord": int(np.argmax(np.abs(u))),
                 "u_unembed_top": top, "u_unembed_bottom": bot}
            out[f"L{L} {name}"] = r
            print(f"L{L:02d} {name:22s} fro {fro:8.1f} top_sv {s:8.1f} ({s/fro:.2f})  "
                  f"|v| top3 {vs[:3].round(3).tolist()} @{r['v_top_coord']}  "
                  f"|u| top3 {us[:3].round(3).tolist()} @{r['u_top_coord']}\n"
                  f"      u-> {top}\n      u<- {bot}", flush=True)
    args.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
