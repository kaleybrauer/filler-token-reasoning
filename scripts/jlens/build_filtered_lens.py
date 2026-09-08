"""
build_filtered_lens.py — materialise a J-lens from the n=100 running sum, MINUS
refitted per-prompt Jacobians (the paper's cross-prompt Frobenius pre-filter),
PLUS newly fitted per-prompt Jacobians. CPU only, one layer in memory at a time.

    lens = ( sum_100  -  sum_{p in exclude} J_p  +  sum_{p in add} J_p ) / n

Writes a JacobianLens.save()-format file (keys J, n_prompts, source_layers,
d_model; fp16 like the reference) that apply_lens.load_jlens, jlens.JacobianLens
and both --jlens consumers read, plus a <out>.meta.json provenance sidecar.

Threshold: --exclude-above T removes every ORIGINAL prompt whose logged
max||J||/sqrt(d) exceeds T (their per-prompt files must exist), and refuses any
--add prompt whose refit norm exceeds T, so one rule governs both sets.

Usage:
    python scripts/jlens/build_filtered_lens.py \
        --base outputs/jlens/lens_v3.fitckpt \
        --per-prompt-dir outputs/jlens/per_prompt \
        --exclude-above 40 --add-range 100:115 \
        --out outputs/jlens/lens_v3_filtered40.pt
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
FIT_LOGS = [REPO_ROOT / "logs/jlens_validate_fit.log", REPO_ROOT / "logs/jlens_fit_POISONED_run.log"]


# ------------------------------------------------------------ layer-wise I/O

class LayerReader:
    """Read one layer of a torch .pt (checkpoint or per-prompt file) at a time.

    torch checkpoints are zip files with each storage stored UNCOMPRESSED as
    data/<k>, k in pickling order — for {layer: tensor} dicts written in layer
    order, k == layer. That is verified per file against a memory-mapped view
    (then released), because on a cgroup-limited box holding 60 layers, or even
    an mmap of them, gets the process killed.
    """

    def __init__(self, path: Path, key: str):
        self.path = Path(path)
        self.zf = zipfile.ZipFile(self.path)
        names = [n for n in self.zf.namelist() if "/data/" in n]
        self.prefix = names[0].rsplit("/data/", 1)[0]
        ck = torch.load(self.path, map_location="cpu", weights_only=True, mmap=True)
        if key not in ck:
            raise ValueError(f"{path}: no key {key!r} (keys {sorted(ck)})")
        d = ck[key]
        self.layers = sorted(int(l) for l in d)
        t0 = d[self.layers[0]]
        self.dtype = {torch.float32: np.float32, torch.float16: np.float16}[t0.dtype]
        self.d = int(t0.shape[0])
        self.meta = {k: v for k, v in ck.items() if k != key and not isinstance(v, dict)}
        self.n_done = int(ck["n_done"]) if "n_done" in ck else None
        # tensors of `key` are the first len(layers) storages iff `key` was the
        # first tensor-bearing entry pickled; verify on two layers, then drop mmap
        for L in (self.layers[0], self.layers[-1]):
            if not np.array_equal(self.read(L), d[L].numpy()):
                raise RuntimeError(f"{path}: zip entry order != layer order at L{L}")
        del ck, d

    def read(self, L: int) -> np.ndarray:
        with self.zf.open(f"{self.prefix}/data/{L}") as f:
            buf = f.read()
        fd = os.open(self.path, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
        return np.frombuffer(buf, dtype=self.dtype).reshape(self.d, self.d)


def logged_norms() -> dict[int, float]:
    """{0-based idx: max||J||/sqrt(d)} from the original fit logs."""
    out = {}
    for f in FIT_LOGS:
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            m = re.search(r"prompt (\d+)/\d+.*max\|\|J\|\|/sqrt\(d\)=([0-9.]+)", line)
            if m:
                out[int(m.group(1)) - 1] = float(m.group(2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=REPO_ROOT / "outputs/jlens/lens_v3.fitckpt")
    ap.add_argument("--per-prompt-dir", type=Path, default=REPO_ROOT / "outputs/jlens/per_prompt")
    ap.add_argument("--exclude-above", type=float, default=None,
                    help="Drop original prompts with logged max||J||/sqrt(d) > T; "
                         "refuse --add prompts above T too")
    ap.add_argument("--exclude", default="", help="Extra 0-based indices to drop, comma list")
    ap.add_argument("--keep", default="", help="0-based indices to KEEP even if above "
                    "--exclude-above (e.g. a prompt whose original settings cannot be "
                    "reproduced, so it cannot be subtracted exactly); recorded in the sidecar")
    ap.add_argument("--add", default="", help="0-based indices of per-prompt files to add")
    ap.add_argument("--add-range", default=None, help="a:b half-open, files that exist are added")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--fit-logs", default=None, help="Override the fit logs the per-prompt "
                    "norms are read from (comma list)")
    args = ap.parse_args()
    if args.fit_logs:
        FIT_LOGS[:] = [Path(x) for x in args.fit_logs.split(",")]

    base = LayerReader(args.base, "jacobian_sum")
    n_base = base.n_done
    norms = logged_norms()
    d, layers = base.d, base.layers

    exclude = {int(x) for x in args.exclude.split(",") if x.strip()}
    if args.exclude_above is not None:
        exclude |= {i for i, v in norms.items() if i < n_base and v > args.exclude_above}
    add = {int(x) for x in args.add.split(",") if x.strip()}
    if args.add_range:
        a, b = (int(x) for x in args.add_range.split(":"))
        add |= {i for i in range(a, b) if (args.per_prompt_dir / f"J_p{i:04d}.pt").exists()}
    keep = {int(x) for x in args.keep.split(",") if x.strip()}
    exclude -= keep
    if exclude & add:
        raise SystemExit(f"indices both excluded and added: {sorted(exclude & add)}")
    if any(i >= n_base for i in exclude):
        raise SystemExit(f"cannot exclude indices >= n_base={n_base}: {sorted(exclude)}")
    if any(i < n_base for i in add):
        raise SystemExit(f"--add indices must be >= n_base={n_base} (already in the sum): {sorted(add)}")

    readers, refit_norm, rejected_add = {}, {}, []
    for i in sorted(exclude | add):
        p = args.per_prompt_dir / f"J_p{i:04d}.pt"
        if not p.exists():
            raise SystemExit(f"missing per-prompt file for idx {i}: {p}")
        r = LayerReader(p, "J")
        if r.layers != layers or r.d != d:
            raise SystemExit(f"{p}: layers/d mismatch")
        info = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
        refit_norm[i] = float(info["max_norm_over_sqrt_d"])
        del info
        if i in exclude:
            if r.dtype != np.float32:
                print(f"  WARNING idx {i}: subtracting an fp16 per-prompt file (rounding "
                      f"residual ~5e-4 x {refit_norm[i]:.0f} = {5e-4 * refit_norm[i]:.1f} "
                      f"in max||J||/sqrt(d) units)")
            if i in norms and abs(refit_norm[i] - norms[i]) > 1e-3 * max(1.0, norms[i]):
                raise SystemExit(f"idx {i}: refit norm {refit_norm[i]:.3f} != logged "
                                 f"{norms[i]:.3f} — the refit did not reproduce the original "
                                 f"J_p, so subtracting it would leave a residual. Check settings.")
        if i in add and args.exclude_above is not None and refit_norm[i] > args.exclude_above:
            rejected_add.append(i)
            continue
        readers[i] = r
    add -= set(rejected_add)
    n = n_base - len(exclude) + len(add)
    print(f"base n={n_base}; exclude {len(exclude)} {sorted(exclude)}; add {len(add)} "
          f"{sorted(add)}; rejected adds {rejected_add}; -> n={n}", flush=True)

    out_dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    J_out, per_layer = {}, {}
    I_fro = np.sqrt(d)
    t0 = time.time()
    for L in layers:
        S = base.read(L).astype(np.float64)
        for i in exclude:
            S -= readers[i].read(L).astype(np.float64)
        for i in add:
            S += readers[i].read(L).astype(np.float64)
        J = S / n
        before = np.linalg.norm(base.read(L).astype(np.float64) / n_base)
        after = np.linalg.norm(J)
        per_layer[L] = {"fro_before": round(float(before), 2), "fro_after": round(float(after), 2),
                        "fro_minus_I_after": round(float(np.linalg.norm(J - np.eye(d))), 2),
                        "mean_diag_after": round(float(np.trace(J) / d), 4)}
        if not np.isfinite(J).all():
            raise SystemExit(f"non-finite lens at L{L} — refusing to write")
        J_out[L] = torch.from_numpy(J.astype(np.float32)).to(out_dtype)
        print(f"  L{L:02d} ||J||_F {before:9.1f} -> {after:8.1f}   (I = {I_fro:.1f})", flush=True)
    print(f"built in {(time.time() - t0) / 60:.1f} min", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"J": J_out, "n_prompts": n, "source_layers": layers, "d_model": d}, args.out)
    meta = {
        "built": time.strftime("%F %T"), "base": str(args.base), "n_base": n_base,
        "base_meta": {k: (v if isinstance(v, (int, float, str, list)) else str(v))
                      for k, v in base.meta.items()},
        "rule": {"exclude_above_max_norm_over_sqrt_d": args.exclude_above,
                 "reference": "paper appendix: cross-prompt Frobenius pre-filter (mean + N sigma); "
                              "T chosen on the logged per-prompt max||J||/sqrt(d)"},
        "excluded": {str(i): {"logged_norm": norms.get(i), "refit_norm": refit_norm[i]} for i in sorted(exclude)},
        "added": {str(i): {"refit_norm": refit_norm[i]} for i in sorted(add)},
        "rejected_adds": {str(i): refit_norm[i] for i in rejected_add},
        "kept_despite_threshold": {str(i): norms.get(i) for i in sorted(keep)},
        "n_prompts": n, "dtype": args.dtype, "per_layer": per_layer,
        "corpus": str(REPO_ROOT / "scripts/jlens/prompts_wikitext.json"),
        "settings_note": "estimator = jlens.fit reference (cotangent at all valid targets >= source, "
                         "mean over sources, skip_first=16, plain mean over prompts); "
                         "cotangent_scale 1/64 (exact by linearity); AWQ int4 weights",
    }
    args.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {args.out} (n={n}) + {args.out.with_suffix('.meta.json')}")


if __name__ == "__main__":
    main()
