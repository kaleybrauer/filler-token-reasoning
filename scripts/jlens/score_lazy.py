"""
score_lazy.py — score_eval_states.py's metric with the lens read under mmap.

The fit checkpoints are 12.3 GiB fp32 each; score_eval_states._load_ckpt
materialises them as float64 (24 GiB) which does not fit a 16 GiB box. This
driver builds the same lenses layer-by-layer from memory-mapped checkpoints and
hands a dict-like lazy object to the unchanged scoring functions.

    # all 100 prompts (fit checkpoint) — or any finished lens .pt
    python scripts/jlens/score_lazy.py --lens outputs/jlens/lens_v3.fitckpt --out X.json
    # prompts 10..99 only (subtract the n=10 snapshot)
    python scripts/jlens/score_lazy.py --lens outputs/jlens/lens_v3.fitckpt \
        --subtract outputs/jlens/lens_v3.snap10.fitckpt --out Y.json
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_eval_states import WEIGHTS, score_lens, warn_if_degenerate  # noqa: E402


class LazyLens:
    """{layer -> J float32} computed on access, one layer at a time.

    Accepts a fit checkpoint (running sum, key `jacobian_sum`, divided by n_done,
    optionally minus a baseline checkpoint) or a finished lens file (key `J`, as
    written by JacobianLens.save / build_filtered_lens.py). Layers are read as raw
    zip entries via build_filtered_lens.LayerReader — torch.load(mmap=True) plus a
    60-layer sweep got this process OOM-killed on the 16 GB cgroup box.
    """

    def __init__(self, cur, base=None):
        from build_filtered_lens import LayerReader
        import zipfile
        names = zipfile.ZipFile(cur).namelist()
        self.is_ckpt = any(n.endswith("/data.pkl") for n in names) and \
            "jacobian_sum" in torch_keys(cur)
        key = "jacobian_sum" if self.is_ckpt else "J"
        self.cur = LayerReader(cur, key)
        self.base = LayerReader(base, "jacobian_sum") if base else None
        if self.is_ckpt:
            self.n_cur = self.cur.n_done
            self.n_base = self.base.n_done if self.base else 0
            self.n = self.n_cur - self.n_base
        else:
            if base:
                raise SystemExit("--subtract only applies to fit checkpoints")
            self.n = int(self.cur.meta.get("n_prompts", 0))
            self.n_cur, self.n_base = self.n, 0
        self.layers = self.cur.layers

    def __getitem__(self, L):
        S = self.cur.read(L).astype(np.float32)
        if not self.is_ckpt:
            return S
        if self.base is not None:
            S -= self.base.read(L)
        S /= self.n
        return S

    def __iter__(self):
        return iter(self.layers)

    def __len__(self):
        return len(self.layers)

    def items(self):
        for L in self.layers:
            yield L, self[L]


def torch_keys(path):
    import torch
    return set(torch.load(path, map_location="cpu", weights_only=True, mmap=True).keys())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=Path, default=REPO_ROOT / "outputs/jlens/eval_states.pt")
    ap.add_argument("--lens", type=Path, required=True)
    ap.add_argument("--subtract", type=Path, default=None)
    ap.add_argument("--sets", default="multihop,order-ops")
    ap.add_argument("--ks", default="1,5,10,50,100")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(args.threads)

    import torch
    from extract.extract_hidden_states import load_tokenizer

    ks = [int(k) for k in args.ks.split(",")]
    slugs = [s.strip() for s in args.sets.split(",") if s.strip()]
    states = torch.load(args.states, map_location="cpu", weights_only=False)
    chk = states.get("unembed_check", {})
    if not chk.get("passed", False):
        raise SystemExit(f"eval states failed G-UNEMBED: {chk}")
    tokenizer = load_tokenizer(states["model_path"])
    rms_w = np.load(WEIGHTS / "rms_norm_weight.npy").astype(np.float32)
    lm_w = np.load(WEIGHTS / "lm_head_weight.npy").astype(np.float32)

    lens = LazyLens(args.lens, args.subtract)
    span = (f"{lens.n_base}..{lens.n_cur - 1}" if lens.is_ckpt else f"lens file, n={lens.n}")
    print(f"lens: {args.lens}" + (f" MINUS {args.subtract}" if args.subtract else "")
          + f" -> {'prompts ' if lens.is_ckpt else ''}{span} (n={lens.n})", flush=True)
    t0 = time.perf_counter()
    # score_eval_states.assert_finite_lens does sorted(lens.items()), which
    # materialises every layer at once (60 x 205 MB) — stream it instead.
    for L in lens:
        if not np.isfinite(lens[L]).all():
            raise SystemExit(f"REFUSING TO SCORE {span}: non-finite entries at L{L}")
    print("  finiteness check passed (streamed)", flush=True)
    report = {"states": str(args.states), "sets": slugs, "ks": ks, "lens": str(args.lens),
              "subtract": str(args.subtract) if args.subtract else None,
              "prompt_span": span, "n_prompts": lens.n,
              "layers": [lens.layers[0], lens.layers[-1]]}
    report["results"] = score_lens(states, lens, tokenizer, rms_w, lm_w, ks, slugs)
    warn_if_degenerate(report["results"])
    report["elapsed_s"] = round(time.perf_counter() - t0, 1)
    print(f"scored in {report['elapsed_s'] / 60:.1f} min", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
