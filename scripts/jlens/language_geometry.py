"""
language_geometry.py — for a fixed concept, is V3's representation language-neutral, language-
conditioned, or hybrid (a shared meaning component plus a language component)? Residual-space tests
on the matched concept states (extract_concept_states.py): each of N concepts in English, Chinese
and Japanese, under two content-neutral templates (replicates) and bare.

Readout co-occurrence of translations cannot separate these hypotheses: V3's unembedding already
places translations near each other (nearest-Han-row retrieval from W_U alone is a true translation
52% top-1), so every test here is in residual space, on unit-normalised states, per layer, for the
raw residual h and the J-transported J_L h (shipped lens, and halves A and B as a noise floor).

  anova       x[lang, concept, template] = mu + a_lang + b_concept + (ab)_lang,concept + e.
              Shares of total sum of squares; interaction vs replicate noise (F, per-dimension
              degrees of freedom); how many principal directions carry the interaction. Separately
              for concepts whose Japanese and Chinese strings are identical and those that differ.
                neutral:      a ~ 0, b large
                hybrid:       a clearly > 0, b large, (ab) ~ replicate noise (F ~ 1)
                conditioned:  (ab) well above noise (F >> 1)
  retrieval   nearest cross-language neighbour of each concept (top-1, chance 1/N): raw; per-language
              centred; after projecting out the k leading interaction directions (fit on half the
              concepts, scored on the other half) and after projecting out k random directions.
              Ceiling: same language across the two templates.
  direction   the zh-en and ja-zh mean differences: cross-validated AUC; cosine between templates;
              cosine with the Wikipedia zh-en mean difference; |cosine| with the lens's top input
              singular direction (V3's script axis) and, after transport, with its output direction.

    python scripts/jlens/language_geometry.py --states outputs/jlens/concept_states.pt
    python scripts/jlens/language_geometry.py --synthetic      # checks the tests on planted structure
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(Path(__file__).resolve().parent))
LANGS = ("en", "zh", "ja")
KS = (0, 1, 2, 4, 8, 16, 32)


def unit(X):
    return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-12)


def anova(X):
    """X [n_lang, n_concept, n_template, d] (unit rows). Sums of squares over vectors."""
    nl, nc, nt, d = X.shape
    mu = X.mean((0, 1, 2))
    m_lc = X.mean(2)
    a = X.mean((1, 2)) - mu
    b = X.mean((0, 2)) - mu
    ab = m_lc - mu - a[:, None] - b[None, :]
    e = X - m_lc[:, :, None]
    ss = {"lang": nc * nt * (a ** 2).sum(), "concept": nl * nt * (b ** 2).sum(),
          "interaction": nt * (ab ** 2).sum(), "replicate": (e ** 2).sum()}
    tot = ((X - mu) ** 2).sum()
    df_inter, df_err = (nl - 1) * (nc - 1), nl * nc * (nt - 1)
    F = (ss["interaction"] / df_inter) / (ss["replicate"] / df_err) if nt > 1 else float("nan")
    s = np.linalg.svd(ab.reshape(-1, d), compute_uv=False) ** 2
    return {"share": {k: float(v / tot) for k, v in ss.items()}, "F_interaction_vs_replicate": float(F),
            "interaction_top_k_share": {k: float(s[:k].sum() / s.sum()) for k in (1, 4, 16, 64)}}, ab


def ranks(A, B):
    S = unit(A) @ unit(B).T
    return (S > np.diag(S)[:, None]).sum(1) + 1


def ret(A, B):
    r = ranks(A, B)
    return {"top1": float((r == 1).mean()), "top5": float((r <= 5).mean())}


def project_out(X, Q):
    return X - (X @ Q) @ Q.T if Q is not None and Q.shape[1] else X


def auc(pos, neg):
    x = np.concatenate([pos, neg]); r = np.empty(len(x)); r[np.argsort(x)] = np.arange(1, len(x) + 1)
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def analyse_layer(X, same, rng, extra_dirs=None):
    """X [3, N, 2, d] contextual states (unit). same: bool [N] ja/zh identical strings."""
    out = {}
    out["anova"], _ = anova(X)
    for name, m in (("same_ja_zh_string", same), ("different_ja_zh_string", ~same)):
        if m.sum() >= 20:
            out[f"anova_{name}"], _ = anova(X[:, m])
    N = X.shape[1]
    perm = rng.permutation(N); tr, te = perm[: N // 2], perm[N // 2:]
    _, ab_tr = anova(X[:, tr])
    Vt = np.linalg.svd(ab_tr.reshape(-1, X.shape[-1]), full_matrices=False)[2]
    R = np.linalg.qr(rng.standard_normal((X.shape[-1], max(KS))))[0]
    r = {}
    for i, j in ((0, 1), (0, 2), (1, 2)):
        pair = f"{LANGS[i]}-{LANGS[j]}"
        A, B = X[i, :, 0], X[j, :, 0]
        r[pair] = {"raw": ret(A, B),
                   "centred": ret(A - A.mean(0), B - B.mean(0)),
                   "remove_interaction_k": {k: ret(project_out(A[te] - A[te].mean(0), Vt[:k].T),
                                                   project_out(B[te] - B[te].mean(0), Vt[:k].T)) for k in KS},
                   "remove_random_k": {k: ret(project_out(A[te] - A[te].mean(0), R[:, :k]),
                                              project_out(B[te] - B[te].mean(0), R[:, :k])) for k in KS}}
        if pair == "zh-ja":
            for name, m in (("same_string", same), ("different_string", ~same)):
                r[pair][f"centred_{name}"] = ret(A[m] - A[m].mean(0), B[m] - B[m].mean(0))
    for li, lang in enumerate(LANGS):
        r[f"{lang} template1-template2 (ceiling)"] = ret(X[li, :, 0], X[li, :, 1])
    out["retrieval"] = r
    dirs = {}
    for i, j in ((0, 1), (1, 2), (0, 2)):
        pair = f"{LANGS[j]}-{LANGS[i]}"
        dt = [X[j, :, t].mean(0) - X[i, :, t].mean(0) for t in (0, 1)]
        d_tr = X[j, tr].mean((0, 1)) - X[i, tr].mean((0, 1))
        d = X[j].mean((0, 1)) - X[i].mean((0, 1))
        dirs[pair] = {"norm": float(np.linalg.norm(d)),
                      "auc_cross_validated": auc(X[j, te].reshape(-1, X.shape[-1]) @ d_tr,
                                                 X[i, te].reshape(-1, X.shape[-1]) @ d_tr),
                      "cos_template1_template2": float(unit(dt[0]) @ unit(dt[1]))}
        for name, v in (extra_dirs or {}).items():
            if v is not None and pair in ("zh-en",) or (v is not None and name.startswith("lens")):
                dirs[pair][f"abs_cos_{name}"] = float(abs(unit(d) @ unit(v)))
    out["language_directions"] = dirs
    return out


def synthetic(regime, N=300, d=128, noise=0.9, seed=0):
    rng = np.random.default_rng(seed)
    meaning = rng.standard_normal((N, d))
    lang = rng.standard_normal((3, d)) * 2.5
    X = np.empty((3, N, 2, d))
    for l in range(3):
        for t in range(2):
            eps = noise * rng.standard_normal((N, d))
            if regime == "neutral":
                X[l, :, t] = meaning + eps
            elif regime == "hybrid":
                X[l, :, t] = meaning + lang[l] + eps
            else:   # conditioned: each language maps meaning through its own rotation
                Q = np.linalg.qr(np.random.default_rng(100 + l).standard_normal((d, d)))[0]
                X[l, :, t] = meaning @ (Q if l else np.eye(d)) * 0.9 + 0.3 * meaning + lang[l] + eps
    same = np.zeros(N, bool); same[: N // 3] = True
    return unit(X), same


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=Path, default=REPO / "outputs/jlens/concept_states.pt")
    ap.add_argument("--lenses", nargs="+", default=["raw", "shipped", "n50", "n50b"])
    ap.add_argument("--layers", nargs="+", type=int, default=None)
    ap.add_argument("--wiki", nargs=2, type=Path, default=[REPO / "outputs/jlens/wiki_zh_states.pt",
                                                           REPO / "outputs/jlens/wiki_en_states.pt"])
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", type=Path, default=REPO / "outputs/jlens/language_geometry.json")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    if args.synthetic:
        for regime in ("neutral", "hybrid", "conditioned"):
            X, same = synthetic(regime)
            o = analyse_layer(X, same, np.random.default_rng(0))
            a = o["anova"]; rz = o["retrieval"]["en-zh"]
            print(f"{regime:12s} shares lang {a['share']['lang']:.3f} concept {a['share']['concept']:.3f} "
                  f"inter {a['share']['interaction']:.3f} repl {a['share']['replicate']:.3f} | F {a['F_interaction_vs_replicate']:.2f} | "
                  f"en-zh top1 raw {rz['raw']['top1']:.2f} centred {rz['centred']['top1']:.2f} "
                  f"remove-inter k=16 {rz['remove_interaction_k'][16]['top1']:.2f} | ceiling {o['retrieval']['en template1-template2 (ceiling)']['top1']:.2f} | "
                  f"zh-en AUC {o['language_directions']['zh-en']['auc_cross_validated']:.2f}")
        return

    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = str(args.threads)
    import torch
    S = torch.load(args.states, map_location="cpu", mmap=True, weights_only=False)
    spec, H = S["spec"], S["H"]
    concepts, prompts = spec["concepts"], spec["prompts"]
    N = len(concepts)
    idx = {(p["lang"], p["template"], p["concept"]): i for i, p in enumerate(prompts)}
    order = np.array([[[idx[(lang, t, c)] for t in (1, 2)] for c in range(N)] for lang in LANGS])  # [3, N, 2]
    same = np.array([c["same_ja_zh"] for c in concepts])
    target = S["n_layers"] - 1
    layers = args.layers or sorted({int(round(x)) for x in np.linspace(0, target, 25)})

    wiki = {}
    if all(p.exists() for p in args.wiki):
        W = [torch.load(p, map_location="cpu", mmap=True, weights_only=False)["H"] for p in args.wiki]
        pos = np.arange(0, W[0].shape[1], 8)
        for L in layers:
            wiki[L] = unit(W[0][:, pos, L].float().numpy().reshape(-1, W[0].shape[-1])).mean(0) \
                      - unit(W[1][:, pos, L].float().numpy().reshape(-1, W[1].shape[-1])).mean(0)

    lenses = {}
    for name in args.lenses:
        if name == "raw":
            lenses[name] = None
        elif name == "shipped":
            from score_lazy import LazyLens
            lenses[name] = LazyLens(REPO / "outputs/jlens/lens_v3_filtered40.pt")
        else:
            from clean_floor import build_subset
            lenses[name] = build_subset(REPO / "outputs/jlens", REPO / "outputs/jlens/per_prompt", 40.0, name)

    report = {"states": str(args.states), "n_concepts": N, "n_same_ja_zh": int(same.sum()), "layers": layers, "per_lens": {}}
    for name, lens in lenses.items():
        rows = {}
        print(f"\n--- {name} ---\n{'L':>3} {'lang':>6} {'concept':>8} {'inter':>6} {'repl':>6} {'F':>6} | en-zh top1 raw/centred "
              f"| zh-ja centred same/diff | zh-en AUC cos(t1,t2) cos(wiki) | ceiling en", flush=True)
        for L in layers:
            t0 = time.time()
            X = H[torch.from_numpy(order.reshape(-1))][:, L].float().numpy()
            J = None
            if lens is not None and L < target:
                J = np.asarray(lens[L], np.float32)
                X = X @ J.T
            X = unit(X.reshape(3, N, 2, -1).astype(np.float64))
            extra = {"wiki_zh_minus_en": wiki.get(L)}
            if J is not None:
                Jd = J.astype(np.float64)
                vecs = np.linalg.qr(np.random.default_rng(L).standard_normal((Jd.shape[1], 1)))[0]
                for _ in range(40):
                    vecs = Jd.T @ (Jd @ vecs); vecs /= np.linalg.norm(vecs)
                extra["lens_top_output_direction"] = (Jd @ vecs)[:, 0]
            o = analyse_layer(X, same, np.random.default_rng(L), extra)
            o["secs"] = round(time.time() - t0, 1)
            rows[L] = o
            a, r, dz = o["anova"], o["retrieval"], o["language_directions"]["zh-en"]
            zj = r["zh-ja"]
            print(f"{L:3d} {a['share']['lang']:6.3f} {a['share']['concept']:8.3f} {a['share']['interaction']:6.3f} "
                  f"{a['share']['replicate']:6.3f} {a['F_interaction_vs_replicate']:6.2f} | "
                  f"{r['en-zh']['raw']['top1']:.2f}/{r['en-zh']['centred']['top1']:.2f} | "
                  f"{zj.get('centred_same_string', {}).get('top1', float('nan')):.2f}/{zj.get('centred_different_string', {}).get('top1', float('nan')):.2f} | "
                  f"{dz['auc_cross_validated']:.2f} {dz['cos_template1_template2']:.2f} {dz.get('abs_cos_wiki_zh_minus_en', float('nan')):.2f} | "
                  f"{r['en template1-template2 (ceiling)']['top1']:.2f}   ({o['secs']}s)", flush=True)
        report["per_lens"][name] = rows
        args.out.write_text(json.dumps(report, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
