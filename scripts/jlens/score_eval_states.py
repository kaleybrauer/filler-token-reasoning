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


def subtract_lens(baseline: Path, current: Path) -> tuple[dict, int, int, int]:
    """The lens over only the prompts fitted AFTER `baseline`.

    Same identity the split-half uses, applied to the live checkpoint:

        mean(b+1..n) = (sum_n - sum_b) / (n - b)

    Prompts 0-7 of this run were preserved verbatim from the superseded
    prefix-sampled corpus and are eight paragraphs of ONE WikiText article
    (`--preserve-prefix 8`, needed because jlens.fit resumes by prompt index).

    Nothing about those eight Jacobians is wrong — each is a valid measurement.
    The problem is that they are correlated draws, not independent ones, so an
    unweighted mean over them is pseudo-replication: it estimates the Jacobian of
    a narrower text distribution than the "generic text" the lens is meant to
    average over. In effective-sample terms the lens at raw n covers roughly
    (n - 8) + 1 distinct articles, so raw n=10 is ~3 articles, not 10.

    Subtracting the n=10 checkpoint drops all eight (and two distinct-article
    prompts with them, which is cheap) and leaves a lens fitted on one paragraph
    per article — the corpus actually intended. It also makes a quality-vs-n curve
    interpretable: uncorrected, early movement is the 8/n weight of that one
    article falling away, which reads as convergence but is not.

    Returns (J, n_effective, n_current, n_baseline).
    """
    sum_b, n_b = _load_ckpt(baseline)
    sum_n, n_n = _load_ckpt(current)
    if n_n <= n_b:
        raise SystemExit(f"current n={n_n} is not past baseline n={n_b} yet")
    layers = sorted(set(sum_b) & set(sum_n))
    J = {l: ((sum_n[l] - sum_b[l]) / (n_n - n_b)).astype(np.float32) for l in layers}
    return J, n_n - n_b, n_n, n_b


def split_half(*ckpts: Path) -> tuple[dict, dict, int, int, str]:
    """Two lenses over DISJOINT prompt sets, from checkpoints of one run.

    `fit` accumulates a running SUM over a deterministic prompt order and resumes
    from `next_idx`, so a checkpoint at n=a covers prompts 1..a. Differencing two
    checkpoints therefore recovers the mean over the prompts BETWEEN them:

        mean(a+1..b) = (sum_b - sum_a) / (b - a)

    The gap between two such disjoint means is the metric's same-n reproducibility
    floor — the error bar an n -> n' step has to clear before it counts as a real
    change rather than as which prompts happened to land. Costs no GPU: it is
    arithmetic on checkpoints the fit already writes.

    TWO checkpoints (a, b) give halves 1..a and a+1..b. THREE (a, b, c) give
    a+1..b and b+1..c, skipping the prefix entirely — which is what this fit needs,
    because prompts 0-7 were preserved verbatim from the superseded prefix-sampled
    corpus and are eight paragraphs of ONE WikiText article, i.e. correlated draws
    rather than eight independent ones. A half holding them is not exchangeable
    with a half of distinct articles, so a two-checkpoint floor would fold that
    difference in sampling breadth into what is meant to be prompt-sampling noise
    alone, and come out too wide — making the lens look converged earlier than it is.
    """
    sums, ns = [], []
    for c in ckpts:
        s, n = _load_ckpt(c)
        sums.append(s)
        ns.append(n)
    if any(ns[i] >= ns[i + 1] for i in range(len(ns) - 1)):
        raise SystemExit(f"checkpoints must be strictly increasing in n, got {ns}")
    layers = sorted(set.intersection(*(set(s) for s in sums)))

    if len(ckpts) == 2:
        (sa, sb), (na, nb) = sums, ns
        first = {l: (sa[l] / na).astype(np.float32) for l in layers}
        second = {l: ((sb[l] - sa[l]) / (nb - na)).astype(np.float32) for l in layers}
        return first, second, na, nb - na, f"1..{na} vs {na + 1}..{nb}"
    if len(ckpts) == 3:
        (sa, sb, sc), (na, nb, nc) = sums, ns
        first = {l: ((sb[l] - sa[l]) / (nb - na)).astype(np.float32) for l in layers}
        second = {l: ((sc[l] - sb[l]) / (nc - nb)).astype(np.float32) for l in layers}
        return first, second, nb - na, nc - nb, f"{na + 1}..{nb} vs {nb + 1}..{nc}"
    raise SystemExit(f"--split-half takes 2 or 3 checkpoints, got {len(ckpts)}")


# --------------------------------------------------------------- the metric

def assert_finite_lens(lens: dict, label: str) -> None:
    """Refuse to score a non-finite lens.

    The 2026-08-25 run reported pass@10 = 1.000 and median_rank = 1 at every one of
    14 curve points across 25 hours. That was not a good lens, it was NaN: with NaN
    logits `(row > row[ids].max()).sum()` is all-False, so every intermediate scores
    rank 1 and every pass@k is a perfect 1.0. A perfect score is the SIGNATURE of
    this failure, not evidence against it. One isfinite check here would have caught
    it 20 minutes after the first bad prompt instead of a day later.
    """
    bad = [l for l, J in sorted(lens.items()) if not np.isfinite(J).all()]
    if bad:
        raise SystemExit(
            f"REFUSING TO SCORE {label}: non-finite entries in {len(bad)} of "
            f"{len(lens)} layers (e.g. L{bad[0]}). A NaN lens scores a perfect "
            f"1.000 rather than erroring, so this is a hard stop."
        )


def warn_if_degenerate(res: dict) -> None:
    """A pass@1 of exactly 1.000 means something is wrong, not that the lens is perfect."""
    for slug, arms in res.items():
        for arm, v in arms.items():
            if arm == "paired":
                continue
            if v.get("pass@1") == 1.0 and v.get("median_rank") == 1.0:
                print(f"  !! {slug}/{arm}: pass@1=1.000 and median_rank=1 exactly — "
                      f"that is the NaN signature, not a perfect lens. Check the input.",
                      flush=True)


def variant_token_ids(tokenizer, word: str) -> list[int]:
    """Token id of each SINGLE-TOKEN surface variant of `word`.

    `intermediates` are words, not ids; the paper picks them to be single-token
    words and credits an intermediate at its best rank over surface variants
    {w, " "+w, lower(w), " "+lower(w)}. Only variants that tokenize to ONE token
    are credited. Taking the first token of a multi-token variant is wrong on
    DeepSeek's tokenizer: " 5" -> [" ", "5"], so the bare space would be credited
    for every numeric intermediate, and "squared" -> ["s", "quared"] would credit
    "s". That inflated pass@1 for any lens whose readout is whitespace (found
    2026-09-08: a giant-dominated lens scored 54 numeric rank-1 "hits" at layer 3
    through the space token). Words with no single-token variant are skipped and
    counted by the caller. Keep in sync with the twin in the other script.
    """
    ids = []
    for surface in {word, " " + word, word.lower(), " " + word.lower()}:
        enc = tokenizer.encode(surface, add_special_tokens=False)
        if len(enc) == 1:
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
    ap.add_argument("--subtract", type=Path, default=None,
                    help="Baseline checkpoint to subtract from --lens, giving the "
                         "lens over only the prompts fitted after it. Use the n=10 "
                         "snapshot to drop the one-article prompt-0-7 prefix.")
    ap.add_argument("--split-half", nargs="+", type=Path, default=None,
                    metavar="CKPT",
                    help="Same-n noise floor from checkpoints of one run. Two -> "
                         "halves 1..a and a+1..b. THREE -> a+1..b and b+1..c, which "
                         "skips the single-article prompt-0-7 prefix (see split_half)")
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
        first, second, n_first, n_second, span = split_half(*args.split_half)
        assert_finite_lens(first, 'half A'); assert_finite_lens(second, 'half B')
        print(f"split-half: prompts {span} — disjoint, n={n_first} vs n={n_second}")
        report["span"] = span
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
        if args.subtract:
            lens, n_done, n_cur, n_base = subtract_lens(args.subtract, args.lens)
            assert_finite_lens(lens, f'{args.lens} minus {args.subtract}')
            report["baseline"] = str(args.subtract)
            report["baseline_n"] = n_base
            report["n_raw"] = n_cur
            print(f"lens: {args.lens} MINUS {args.subtract}  -> prompts "
                  f"{n_base + 1}..{n_cur} (n={n_done}, prefix-free)")
        else:
            lens, n_done = load_lens(args.lens)
            assert_finite_lens(lens, str(args.lens))
            print(f"lens: {args.lens}  n_prompts={n_done}")
        layers = sorted(lens)
        print(f"  {len(layers)} layers [{layers[0]}..{layers[-1]}]")
        report["lens"] = str(args.lens)
        report["n_prompts"] = n_done
        report["layers"] = [layers[0], layers[-1]]
        report["results"] = score_lens(states, lens, tokenizer, rms_w, lm_w, ks, slugs)
        warn_if_degenerate(report["results"])

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
