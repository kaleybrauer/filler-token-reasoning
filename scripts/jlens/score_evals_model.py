"""
score_evals_model.py — the paper's lens-eval metric for any model in models.py, J-lens against logit lens, with the
wrong-answer control, on the layers the lens holds.

Reads <states> (eval_states.pt: extract_eval_states.py for V3, extract_qwen35_states.py --steps evals for Qwen3.5),
the model's cached unembedding (models.py unembed_dir) and tokenizer, and scores each arm with score_eval_states's
rule: an intermediate is credited at the best rank, over layers, of any single-token surface variant. Differences
from score_eval_states.py: (1) intermediates with no single-token variant are SKIPPED rather than counted as misses
(Qwen's tokenizer splits every multi-digit number into digits, so 22 of order-ops' 110 intermediates have no
single-token form there); (2) both arms use the layers the lens holds (the 122B lens holds 0-46 of 48);
(3) 20 wrong answers per item (other items' intermediates from the same set) are scored the same way, so block-level
lifts show up as wrong-answer credit.

    python scripts/jlens/score_evals_model.py --model qwen35_122b
    python scripts/jlens/score_evals_model.py --model v3 --layers 0:59      # reproduces the V3 numbers
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(Path(__file__).resolve().parent))
import models                                                          # noqa: E402
from score_eval_states import variant_token_ids, rms_norm, paired_delta  # noqa: E402
from score_lazy import LazyLens                                        # noqa: E402

SETS = ["multihop", "order-ops", "association", "multilingual", "typo"]
KS = [5, 10, 50]
N_WRONG = 20


LOCAL_TOK = {"Qwen/Qwen3.5-122B-A10B": Path("/workspace/models/qwen35-122b-tokenizer")}   # hub id -> local copy


def load_tok(name, m):
    if m.get("tokenizer_kind") == "auto" and LOCAL_TOK.get(m["tokenizer"], Path("/nonexistent")).exists():
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(str(LOCAL_TOK[m["tokenizer"]]))
    return models.load_tokenizer_for(name)


def best_ranks(H, J_of, layers, targets, rms_w, lm_w, eps, chunk=32):
    n, V = H.shape[0], lm_w.shape[0]
    best = [np.full(len(t), np.iinfo(np.int64).max) for t in targets]
    for L in layers:
        h = H[:, L].astype(np.float32)
        J = J_of(L)
        if J is not None:
            h = h @ J.T
        for s in range(0, n, chunk):
            z = rms_norm(h[s:s + chunk], rms_w, eps) @ lm_w.T
            zs = np.sort(z, axis=1)
            for r in range(z.shape[0]):
                i = s + r
                if not targets[i]:
                    continue
                vals = np.array([z[r, ids].max() for ids in targets[i]])
                np.minimum(best[i], V - np.searchsorted(zs[r], vals, side="right") + 1, out=best[i])
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--states", type=Path, default=None)
    ap.add_argument("--lens", type=Path, default=None, help="default: the model's lens in models.py")
    ap.add_argument("--layers", default=None, help="a:b (half-open) or comma list; default: the lens's layers")
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = str(args.threads)
    import torch
    torch.set_num_threads(args.threads)
    m = models.get(args.model)
    states_path = args.states or (REPO / "outputs/jlens/eval_states.pt" if args.model == "v3" else models.QS / args.model / "eval_states.pt")
    states = torch.load(states_path, map_location="cpu", weights_only=False)
    if not states["unembed_check"]["passed"]:
        raise SystemExit(f"{states_path}: unembed check failed {states['unembed_check']}")
    tok = load_tok(args.model, m)
    rms_w = np.load(m["unembed_dir"] / "rms_norm_weight.npy").astype(np.float32)
    lm_w = np.load(m["unembed_dir"] / "lm_head_weight.npy").astype(np.float32)
    eps = float(states["rms_norm_eps"])
    lens = LazyLens(args.lens or m["lens"])
    if args.layers:
        layers = list(range(*map(int, args.layers.split(":")))) if ":" in args.layers else [int(x) for x in args.layers.split(",")]
    else:
        layers = list(lens.layers)
    missing = [L for L in layers if L not in lens.layers]
    if missing:
        raise SystemExit(f"lens lacks layers {missing[:5]}...")
    print(f"{args.model}: states {states_path.name}, lens n={lens.n} layers {layers[0]}..{layers[-1]} ({len(layers)}), vocab {lm_w.shape[0]}", flush=True)
    rng = np.random.default_rng(0)
    out = {"model": args.model, "states": str(states_path), "lens": str(args.lens or m["lens"]), "layers": layers, "ks": KS, "sets": {}}
    for slug in args.sets.split(","):
        s = states["sets"][slug]; H = s["H"].numpy(); n = H.shape[0]
        true = [[ids for ids in (variant_token_ids(tok, w) for w in words) if ids] for words in s["intermediates"]]
        n_words = sum(len(w) for w in s["intermediates"]); n_scored = sum(len(t) for t in true)
        pool = [(j, ids) for j, item in enumerate(true) for ids in item]
        wrong = []
        for i, item in enumerate(true):
            own = {t for ids in item for t in ids}
            cand = [ids for j, ids in pool if j != i and own.isdisjoint(ids)]
            pick = rng.choice(len(cand), size=min(N_WRONG, len(cand)), replace=False)
            wrong.append([cand[k] for k in pick])
        targets = [t + w for t, w in zip(true, wrong)]
        per, res = {}, {}
        for arm in ("logit", "jlens"):
            t0 = time.time()
            br = best_ranks(H, (lambda L: None) if arm == "logit" else (lambda L: lens[L]), layers, targets, rms_w, lm_w, eps)
            per[arm] = {}
            for kind in ("true", "wrong"):
                for k in KS:
                    v = np.array([np.mean(br[i][:len(true[i])] <= k) if kind == "true" and len(true[i]) else
                                  np.mean(br[i][len(true[i]):] <= k) if kind == "wrong" and len(wrong[i]) else np.nan for i in range(n)])
                    per[arm][(kind, k)] = v
            res[arm] = {f"{kind}_pass@{k}": round(float(np.nanmean(per[arm][(kind, k)])), 4) for kind in ("true", "wrong") for k in KS}
            print(f"  {slug:13s} {arm:6s} " + "  ".join(f"@{k}: true {res[arm][f'true_pass@{k}']:.3f} wrong {res[arm][f'wrong_pass@{k}']:.3f}" for k in KS) + f"  ({time.time()-t0:.0f}s)", flush=True)
        paired = {}
        for k in KS:
            for kind in ("true", "wrong"):
                d = per["jlens"][(kind, k)] - per["logit"][(kind, k)]; d = d[~np.isnan(d)]
                paired[f"{kind}@{k}"] = [round(float(d.mean()), 4), round(float(d.std(ddof=1) / np.sqrt(len(d))), 4)]
            e = (per["jlens"][("true", k)] - per["jlens"][("wrong", k)]) - (per["logit"][("true", k)] - per["logit"][("wrong", k)]); e = e[~np.isnan(e)]
            paired[f"excess@{k}"] = [round(float(e.mean()), 4), round(float(e.std(ddof=1) / np.sqrt(len(e))), 4)]
        print(f"  {slug:13s} J-logit @10: true {paired['true@10'][0]:+.3f}±{paired['true@10'][1]:.3f}  wrong {paired['wrong@10'][0]:+.3f}  "
              f"excess {paired['excess@10'][0]:+.3f}±{paired['excess@10'][1]:.3f}   (scored {n_scored}/{n_words} intermediates)", flush=True)
        out["sets"][slug] = {"n_items": n, "n_intermediates": n_words, "n_scored": n_scored, "arms": res, "paired": paired}
        out_path = args.out or (REPO / f"outputs/jlens/evals_{args.model}.json" if args.model == "v3" else models.QS / args.model / "analysis" / "evals_jlens_vs_logit.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=1))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
