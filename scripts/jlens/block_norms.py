"""
block_norms.py — per-block Jacobian magnitude from the fit's snapshot chain.

The fit stores a running SUM, so consecutive snapshots (n=a < b) give the mean
Jacobian over the DISJOINT prompt block a..b-1:  (sum_b - sum_a) / (b - a).
This reports, per block and per layer, ||mean J||_F and ||mean J - I||_F, so a
block dominated by a giant-norm prompt is visible. Everything is read under
mmap one layer at a time (each checkpoint is 12.3 GiB fp32).

    python scripts/jlens/block_norms.py --out outputs/jlens/block_norms.json
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch

CHAIN = [("lens_v3.snap10.fitckpt", 10), ("lens_v3.snap20.fitckpt", 20),
         ("lens_v3.snap30.fitckpt", 30), ("lens_v3.snap40.fitckpt", 40),
         ("lens_v3.snap60.fitckpt", 60), ("lens_v3.fitckpt", 100)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("outputs/jlens"))
    ap.add_argument("--out", type=Path, default=Path("outputs/jlens/block_norms.json"))
    args = ap.parse_args()

    cks = []
    for name, n in CHAIN:
        ck = torch.load(args.dir / name, map_location="cpu", weights_only=True, mmap=True)
        assert int(ck["n_done"]) == n, (name, ck["n_done"])
        cks.append(ck["jacobian_sum"])
    layers = sorted(cks[0])
    d = cks[0][layers[0]].shape[0]
    I = torch.eye(d)

    blocks = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 60), (60, 100)]
    out = {"blocks": [f"{a}..{b-1}" for a, b in blocks], "layers": layers,
           "fro": {}, "fro_minus_I": {}, "mean_diag": {}, "top_sv": {}}
    t0 = time.time()
    for L in layers:
        sums = [ck[L].float() for ck in cks]   # one layer of every checkpoint
        rows = {k: [] for k in ("fro", "fro_minus_I", "mean_diag", "top_sv")}
        for i, (a, b) in enumerate(blocks):
            S = sums[i] if a == 0 else sums[i] - sums[i - 1]
            J = S / (b - a)
            rows["fro"].append(float(J.norm()))
            rows["fro_minus_I"].append(float((J - I).norm()))
            rows["mean_diag"].append(float(J.diagonal().mean()))
            # largest singular value via a few power iterations (cheap, d=7168)
            v = torch.randn(d); 
            for _ in range(8):
                v = J.T @ (J @ v); v = v / v.norm()
            rows["top_sv"].append(float((J @ v).norm()))
        for k in rows:
            out[k][L] = rows[k]
        print(f"L{L:02d} fro " + " ".join(f"{x:8.1f}" for x in rows["fro"])
              + "   top_sv " + " ".join(f"{x:8.1f}" for x in rows["top_sv"]),
              flush=True)
    print(f"done in {(time.time()-t0)/60:.1f} min")
    args.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
