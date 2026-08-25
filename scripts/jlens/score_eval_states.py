"""
score_eval_states.py — the paper's lens-quality metric, offline, on cached states.

Scores a fitted lens (or a live fit checkpoint) on the evaluation sets shipped
with the jacobian-lens repo, using the states cached by extract_eval_states.py.
No model, no GPU — so this runs on CPU alongside a fit that is occupying all
three cards, and the same fit can be scored again at every checkpoint to give
lens quality as a function of n.

Metric (data/evaluations/README.md): readout at the token preceding `target`;
pass@k = mean over items of the fraction of `intermediates` whose min-over-layers
lens rank is <= k. Two arms on identical states:

    J-lens      lens_L(h) = unembed( J_L h )
    logit lens  lens_L(h) = unembed( h )      (the J_L = I special case)

Both arms are restricted to the lens's fitted layers, so a band-only fit reports
"min over the fitted band" for both.

Why this exists (and why it is not a homemade stopping rule): the paper fixes
n=1000 and states no convergence criterion, and its §9.3 reports varying "the
number of contexts over which we average" only as a qualitative robustness claim,
with no curve. n=1000 is ~27 days on this hardware, so the n we can reach has to
be justified by measurement rather than by copying theirs. The metric measured
here is the paper's own, on the paper's own sets.

Usage:
    # one lens / the live fit checkpoint
    python scripts/jlens/score_eval_states.py --lens outputs/jlens/lens_v3.fitckpt

    # the same, appended to a curve file for plotting quality vs n
    python scripts/jlens/score_eval_states.py --lens outputs/jlens/lens_v3.fitckpt \
        --append outputs/jlens/quality_vs_n.jsonl

    # same-n noise floor: two DISJOINT half-lenses derived from two checkpoints
    python scripts/jlens/score_eval_states.py --split-half \
        outputs/jlens/lens_v3.n20.fitckpt outputs/jlens/lens_v3.n40.fitckpt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

WEIGHTS = REPO_ROOT / "data/model_weights/deepseek_v3"


# --------------------------------------------------------------- lens loading

def _load_ckpt(path: Path):
    """(jacobian_sum {layer: [d,d] float64}, n_done) from a fit checkpoint."""
    import torch

    ck = torch.load(path, map_location="cpu", weights_only=True)
    if "jacobian_sum" not in ck:
        raise ValueError(f"{path}: not a fit checkpoint (keys {sorted(ck)})")
    return {int(l): J.double().numpy() for l, J in ck["jacobian_sum"].items()}, int(ck["n_done"])


def load_lens(path: Path) -> tuple[dict[int, np.ndarray], int]:
    """{layer -> J float32}, n_prompts. Accepts a lens file or a fit checkpoint."""
    from apply_lens import load_jlens

    lens = load_jlens(path)
    n = 0
    if path.suffix == ".fitckpt" or path.name.endswith(".fitckpt"):
        _, n = _load_ckpt(path)
    return lens, n


def split_half(ckpt_a: Path, ckpt_b: Path) -> tuple[dict, dict, int, int]:
    """Two lenses over DISJOINT prompt sets, from two checkpoints of one run.

    `fit` accumulates a running SUM over a deterministic prompt order and resumes
    from `next_idx`, so a checkpoint at n=a covers prompts 1..a and one at n=b>a
    covers 1..b. Then

        mean(1..a)   = sum_a / a
        mean(a+1..b) = (sum_b - sum_a) / (b - a)

    are fitted on disjoint prompts. Their difference on a metric is that metric's
    same-n reproducibility floor — the error bar an n -> n' step has to clear
    before it counts as a real change rather than which prompts happened to land.
    Costs no GPU: it is arithmetic on checkpoints the fit already writes.
    """
    sum_a, n_a = _load_ckpt(ckpt_a)
    sum_b, n_b = _load_ckpt(ckpt_b)
    if n_b <= n_a:
        raise SystemExit(f"need n_b > n_a, got n_a={n_a} n_b={n_b}")
    layers = sorted(set(sum_a) & set(sum_b))
    first = {l: (sum_a[l] / n_a).astype(np.float32) for l in layers}
    second = {l: ((sum_b[l] - sum_a[l]) / (n_b - n_a)).astype(np.float32) for l in layers}
    return first, second, n_a, n_b - n_a


# --------------------------------------------------------------- the metric

def variant_token_ids(tokenizer, word: str) -> list[int]:
    """First token id of each surface variant of `word`.

    `intermediates` are words, not ids, and the paper does not pin the variant
    convention; an intermediate is credited at its best rank over
    {w, " "+w, lower(w), " "+lower(w)}. Keep in sync with the GPU-path twin in
    eval_lens_quality.py.
    """
    ids = []
    for surface in {word, " " + word, word.lower(), " " + word.lower()}:
        enc = tokenizer.encode(surface, add_special_tokens=False)
        if enc:
            ids.append(enc[0])
    return sorted(set(ids))


def rms_norm(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * w


def best_ranks(H: np.ndarray, lens: dict[int, np.ndarray] | None, layers: list[int],
               item_ids: list[list[list[int]]], rms_w: np.ndarray, lm_w: np.ndarray,
               eps: float, chunk: int = 64) -> list[list[int]]:
    """Best (lowest) full-vocab rank of every intermediate, over layers+variants.

    H is [n_items, n_layers, d]; `item_ids[i][j]` are the surface-variant token
    ids of item i's j-th intermediate; `lens=None` is the logit-lens arm.

    Layers are the outer loop and every intermediate is ranked from the same
    readout, so the [n_items, vocab] logits are built once per layer rather than
    once per intermediate. They are discarded immediately, so peak memory is one
    chunk's readout (~32 MB at chunk=64).
    """
    n = H.shape[0]
    best = [[np.iinfo(np.int64).max] * len(ids) for ids in item_ids]
    for L in layers:
        h = H[:, L].astype(np.float32)
        if lens is not None:
            h = h @ lens[L].T
        for s in range(0, n, chunk):
            logits = rms_norm(h[s:s + chunk], rms_w, eps) @ lm_w.T   # [c, vocab]
            for r in range(logits.shape[0]):
                i, row = s + r, logits[r]
                for j, ids in enumerate(item_ids[i]):
                    if not ids:
                        continue
                    rank = int((row > row[ids].max()).sum()) + 1
                    if rank < best[i][j]:
                        best[i][j] = rank
    return best


def score_set(H: np.ndarray, meta: dict, lens: dict | None, layers: list[int],
              tokenizer, rms_w, lm_w, eps, ks) -> tuple[dict, dict]:
    """pass@k / median rank over one eval set, for one arm.

    Also returns the per-item pass fractions, so the caller can difference the
    two arms item-by-item — see `paired_delta`.
    """
    item_ids = [[variant_token_ids(tokenizer, w) for w in words]
                for words in meta["intermediates"]]
    ranks = best_ranks(H, lens, layers, item_ids, rms_w, lm_w, eps)

    flat = np.array([r for row in ranks for r in row], dtype=np.float64)
    out = {"n_items": int(H.shape[0]), "n_intermediates": int(flat.size),
           "median_rank": float(np.median(flat)),
           "mean_log10_rank": round(float(np.log10(flat).mean()), 3)}
    per_item = {}
    for k in ks:
        # Published metric: mean over ITEMS of the fraction of that item's
        # intermediates at rank <= k. The spread over items is the sampling
        # error bar — the thing the removed convergence monitor reported without.
        v = np.array([np.mean([r <= k for r in row]) for row in ranks])
        per_item[k] = v
        out[f"pass@{k}"] = round(float(v.mean()), 4)
        out[f"pass@{k}_se"] = round(float(v.std(ddof=1) / np.sqrt(len(v))), 4)
    return out, per_item


def paired_delta(a: dict, b: dict, ks) -> dict:
    """J-lens minus logit lens, differenced PER ITEM.

    Both arms are read from the same cached states on the same items, so the two
    pass@k estimates are paired, not independent. Their per-arm SEs are dominated
    by which items are easy — variance shared by both arms, which cancels in the
    difference. Differencing first gives the SE of the quantity actually being
    judged (does the transport help?), and it is typically several times tighter
    than |se_a| + |se_b| would suggest.

    `n_better`/`n_worse` are the item counts behind the mean, so a delta driven
    by a couple of items is visible as such rather than hidden in an average.
    """
    out = {}
    for k in ks:
        d = a[k] - b[k]
        out[f"dpass@{k}"] = round(float(d.mean()), 4)
        out[f"dpass@{k}_se"] = round(float(d.std(ddof=1) / np.sqrt(len(d))), 4)
        out[f"dpass@{k}_n_better"] = int((d > 0).sum())
        out[f"dpass@{k}_n_worse"] = int((d < 0).sum())
    return out


# --------------------------------------------------------------- driver

def score_lens(states, lens, tokenizer, rms_w, lm_w, ks, slugs) -> dict:
    layers = sorted(lens) if lens is not None else None
    eps = states["rms_norm_eps"]
    res = {}
    for slug in slugs:
        s = states["sets"][slug]
        H = s["H"].numpy()
        use_layers = layers if layers is not None else list(range(states["n_layers"]))
        j, j_item = score_set(H, s, lens, use_layers, tokenizer, rms_w, lm_w, eps, ks)
        g, g_item = score_set(H, s, None, use_layers, tokenizer, rms_w, lm_w, eps, ks)
        res[slug] = {"jlens": j, "logit": g, "paired": paired_delta(j_item, g_item, ks)}
        d = res[slug]["paired"]
        print(f"  {slug:14s} jlens median {j['median_rank']:>7.0f} pass@10 "
              f"{j['pass@10']:.3f}  |  logit median {g['median_rank']:>7.0f} pass@10 "
              f"{g['pass@10']:.3f}  |  paired d(pass@10) {d['dpass@10']:+.3f}"
              f"+-{d['dpass@10_se']:.3f} ({d['dpass@10_n_better']}+/"
              f"{d['dpass@10_n_worse']}-)", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=Path,
                    default=REPO_ROOT / "outputs/jlens/eval_states.pt")
    ap.add_argument("--lens", type=Path, default=None,
                    help="Lens file or fit checkpoint (a checkpoint's running sum "
                         "is divided by n_done, so a run in progress can be scored)")
    ap.add_argument("--split-half", nargs=2, type=Path, default=None,
                    metavar=("CKPT_A", "CKPT_B"),
                    help="Same-n noise floor from two checkpoints of one run")
    ap.add_argument("--sets", default="multihop,order-ops")
    ap.add_argument("--ks", default="1,5,10,50,100")
    ap.add_argument("--threads", type=int, default=32,
                    help="BLAS threads. Capped by default so scoring stays polite "
                         "to a fit running on the same box.")
    ap.add_argument("--append", type=Path, default=None,
                    help="Append the result as one JSON line (KB, not the 11.5 GiB "
                         "the lens itself would cost) for a quality-vs-n curve")
    ap.add_argument("--out", type=Path, default=None)
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
    print(f"states: {args.states}  sets {sorted(states['sets'])}  "
          f"G-UNEMBED top1 {chk['top1_agreement']:.3f} corr {chk['logit_corr']}")

    tokenizer = load_tokenizer(states["model_path"])
    rms_w = np.load(WEIGHTS / "rms_norm_weight.npy").astype(np.float32)
    lm_w = np.load(WEIGHTS / "lm_head_weight.npy").astype(np.float32)

    t0 = time.perf_counter()
    report = {"states": str(args.states), "sets": slugs, "ks": ks}

    if args.split_half:
        a, b = args.split_half
        first, second, n_first, n_second = split_half(a, b)
        print(f"split-half: prompts 1..{n_first} (n={n_first}) vs "
              f"{n_first + 1}..{n_first + n_second} (n={n_second}) — disjoint")
        print(" half A:")
        report["half_A"] = score_lens(states, first, tokenizer, rms_w, lm_w, ks, slugs)
        print(" half B:")
        report["half_B"] = score_lens(states, second, tokenizer, rms_w, lm_w, ks, slugs)
        report["n_half_A"], report["n_half_B"] = n_first, n_second
        print("\nsame-n floor |A - B| (a real n->n' step must clear this):")
        for slug in slugs:
            for arm in ("jlens", "logit"):
                for k in ks:
                    d = abs(report["half_A"][slug][arm][f"pass@{k}"]
                            - report["half_B"][slug][arm][f"pass@{k}"])
                    if k == 10:
                        print(f"  {slug:14s} {arm:6s} pass@10 floor {d:.4f}")
    else:
        if args.lens is None:
            raise SystemExit("need --lens or --split-half")
        lens, n_done = load_lens(args.lens)
        layers = sorted(lens)
        print(f"lens: {args.lens}  n_prompts={n_done}  "
              f"{len(layers)} layers [{layers[0]}..{layers[-1]}]")
        report["lens"] = str(args.lens)
        report["n_prompts"] = n_done
        report["layers"] = [layers[0], layers[-1]]
        report["results"] = score_lens(states, lens, tokenizer, rms_w, lm_w, ks, slugs)

    report["elapsed_s"] = round(time.perf_counter() - t0, 1)
    print(f"\nscored in {report['elapsed_s'] / 60:.1f} min")

    if args.append:
        args.append.parent.mkdir(parents=True, exist_ok=True)
        with args.append.open("a") as f:
            f.write(json.dumps(report) + "\n")
        print(f"appended to {args.append}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
