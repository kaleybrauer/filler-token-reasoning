"""
penult_quick.py — early read on the penultimate-target test from per-prompt files, no lens build.
(PENULT_RUNBOOK.md §5)

For each per-prompt Jacobian file given (e.g. idx 100 fitted to target 60 and to target 59) it
transports the held-out WikiText states at a few layers and prints the median script-class eta^2
and Han-minus-Latin offset of the readout over the vocabulary, next to the logit lens. It uses
var_vocab(W g h) = h^T C h, with C the covariance of the norm-weighted unembedding rows, so every
activation is used and no vocabulary-sized logit matrix is formed. Script classes are
bilingual_diagnostics.py's (vocab_scripts.json) collapsed to Latin / Han / everything else, as in
its eta2_script3. Mean-lens files (build_mean_lens.py output) are accepted too.

    python scripts/jlens/penult_quick.py outputs/jlens/per_prompt/J_p0100.pt \
        outputs/jlens/per_prompt_target59/J_p0100.pt
Reference values (all 11,199 activations; 2026-09-17), eta^2 p50 and offset p50 at layers 20 / 30 / 45:
    idx 100 alone, target 60          0.093 / 0.157 / 0.211     -0.46 / -0.79 / -1.00 sd
    idx 100..104 mean lens, target 60 0.411 / 0.383 / 0.241     +1.33 / +0.91 / -0.96 sd
    logit lens                        0.003 / 0.003 / 0.003
One prompt's early layers vary a lot (single-prompt eta^2 at L18 ranges 0.07-0.47 across idx 100-107),
so read a one-prompt comparison at L30 and L45. The 5-prompt lens matches bilingual_diagnostics.py's
2,400-activation depth run to 0.01 (0.408 / 0.374 at L20 / L30).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path, help="per-prompt J_p*.pt files")
    ap.add_argument("--layers", nargs="+", type=int, default=[20, 30, 45])
    ap.add_argument("--states", type=Path, default=REPO / "outputs/jlens/wikitext_states.pt")
    ap.add_argument("--unembed-dir", type=Path, default=REPO / "data/model_weights/deepseek_v3")
    ap.add_argument("--vocab", type=Path, default=REPO / "outputs/jlens/vocab_scripts.json")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = str(args.threads)
    import torch
    torch.set_num_threads(args.threads)
    from build_filtered_lens import LayerReader

    t0 = time.time()
    voc = json.loads(args.vocab.read_text())
    cls_names, cls = voc["classes"], np.asarray(voc["cls"])
    c3 = np.where(cls == cls_names.index("Latin"), 0, np.where(cls == cls_names.index("Han"), 1, 2))
    W = np.load(args.unembed_dir / "lm_head_weight.npy", mmap_mode="r")
    g = np.load(args.unembed_dir / "rms_norm_weight.npy").astype(np.float32)
    V, d = W.shape
    assert len(c3) == V, (len(c3), V)
    sums, C = np.zeros((3, d)), np.zeros((d, d))
    for s in range(0, V, 8192):
        Wc = torch.from_numpy(np.asarray(W[s:s + 8192], np.float32) * g)
        C += (Wc.T @ Wc).double().numpy()
        for k in range(3):
            sums[k] += Wc.numpy()[c3[s:s + 8192] == k].sum(0, dtype=np.float64)
    n_k = np.bincount(c3, minlength=3).astype(np.float64)
    mean_all = sums.sum(0) / V
    Ct = torch.from_numpy(C / V - np.outer(mean_all, mean_all)).float()
    M = torch.from_numpy(np.vstack([sums / n_k[:, None], mean_all[None]])).float()   # Latin, Han, rest, all
    nk = torch.from_numpy(n_k).float()
    print(f"unembedding covariance and class means in {time.time() - t0:.0f}s "
          f"(Latin {int(n_k[0])}, Han {int(n_k[1])}, rest {int(n_k[2])})", flush=True)

    S = torch.load(args.states, map_location="cpu", weights_only=False, mmap=True)
    H = S["H"]
    keep = (H[:, :, S["n_layers"] - 1].float().abs().amax(-1) < 65504).reshape(-1)

    def stats(X):
        X = X / X.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-12)
        var = ((X @ Ct) * X).sum(1)
        mk = X @ M.T
        eta = (nk * (mk[:, :3] - mk[:, 3:4]) ** 2).sum(1) / V / var
        off = (mk[:, 1] - mk[:, 0]) / var.sqrt()
        return {"eta2_p50": float(eta.median()), "offset_sd_p50": float(off.median()),
                "offset_sd_p10": float(off.quantile(0.1)), "offset_sd_p90": float(off.quantile(0.9))}

    readers, labels = [], []
    for f in args.files:
        ck = torch.load(f, map_location="cpu", weights_only=False, mmap=True)
        if "settings" in ck:
            labels.append(f"{f.parent.name}/{f.name} (idx {ck['index']}, target {ck['settings']['target_layer']}, "
                          f"max||J||/sqrt(d) {ck['max_norm_over_sqrt_d']:.3f})")
        else:
            labels.append(f"{f.name} (mean lens, n={ck.get('n_prompts')}, source layers 0..{max(ck['J'])})")
        del ck
        readers.append(LayerReader(f, "J"))
    report = {"files": labels, "n_activations": int(keep.sum()), "layers": {}}
    print(f"{int(keep.sum())} activations; files:\n  " + "\n  ".join(f"[{i}] {s}" for i, s in enumerate(labels)), flush=True)
    print(f"{'L':>3} | {'lens':>9} | {'eta2 p50':>8} | Han-Latin offset sd p10 / p50 / p90", flush=True)
    for L in args.layers:
        h = H[:, :, L].reshape(-1, d)[keep].float()
        rows = {"logit": stats(h)}
        for i, r in enumerate(readers):
            if L in r.layers:
                J = torch.from_numpy(r.read(L).astype(np.float32))
                rows[f"[{i}]"] = stats(h @ J.T)
        report["layers"][L] = rows
        for name, x in rows.items():
            print(f"{L:3d} | {name:>9} | {x['eta2_p50']:8.4f} | {x['offset_sd_p10']:+.2f} / {x['offset_sd_p50']:+.2f} / "
                  f"{x['offset_sd_p90']:+.2f}", flush=True)
    if args.out:
        args.out.write_text(json.dumps(report, indent=1))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
