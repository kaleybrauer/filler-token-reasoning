"""
bilingual_diagnostics.py — before reading V3's Figure 28 panel (b) as a different workspace,
rule out its bilingual vocabulary. CPU tests on the held-out WikiText states
(extract_wikitext_states.py), against the shipped lens.

Why. V3's panel (b) differs from Sonnet's in two ways: broad early kurtosis spikes (at 8% and
25% depth) and a flat workspace band whose MEDIAN readout kurtosis is negative. A logit vector
over 129k tokens only goes negative when it is two-level, and 27% of V3's vocabulary is Han:
an English context that pushes every Han token down together makes exactly that shape. The
lens's dominant output direction separates Han from Latin tokens almost perfectly at every
layer (script_axis.py), so script structure has to be separated from the workspace before the
band can be compared with Sonnet's.

  spikes  At each spike layer, every held-out activation's readout kurtosis. For the top 1%:
          input context, top-50 readout tokens and their scripts, which prompts they come
          from, and whether the same activations are spikes under each half-fit lens and
          under the logit lens (estimation noise would not repeat across halves).
  depth   At the 25 panel layers, on an evenly spaced subsample of positions:
            - script make-up and letter-script entropy of the top-50/100 readout tokens;
            - excess kurtosis over matched vocabulary subsets: Latin tokens only, Han tokens
              only, and uniformly random subsets of those two sizes (kurtosis depends on the
              population measured, so a subset is only compared with its size-matched
              random control);
            - kurtosis after removing each script class's mean logit, and eta^2, the share of
              a readout's variance across the vocabulary that the class means explain.
          A script OFFSET shows as a large eta^2 and kurtosis restored once the class means
          are removed. DILUTION across translation equivalents shows as Han tokens standing
          out individually within their class alongside the Latin top tokens; the
          within-class top-k at the stored layers are saved for that check.

Kurtosis, eta^2 and top-k do not depend on the readout's scale; the RMSNorm scale is applied
anyway, as in workspace_readouts.py, to keep logits O(10). The unembedding is streamed in
chunks from its fp16 file and moments are accumulated per chunk, so no [activations x
vocabulary] matrix is ever materialised.

    python scripts/jlens/bilingual_diagnostics.py spikes --layers 5 15 --lenses shipped n50 n50b logit
    python scripts/jlens/bilingual_diagnostics.py depth --lenses shipped logit
"""
from __future__ import annotations

import argparse, json, os, sys, time, unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

LETTER_SCRIPTS = ("Latin", "Han", "Kana", "Hangul", "Cyrillic", "Greek", "Arabic", "Hebrew",
                  "Indic", "Thai", "OtherLetter")
NONLETTER = ("digit", "punct", "space", "byte", "special")
CLASSES = LETTER_SCRIPTS + NONLETTER
LATIN, HAN = CLASSES.index("Latin"), CLASSES.index("Han")
N_LETTER = len(LETTER_SCRIPTS)
_PREFIX = (("CJK", "Han"), ("HIRAGANA", "Kana"), ("KATAKANA", "Kana"),
           ("HALFWIDTH KATAKANA", "Kana"), ("HANGUL", "Hangul"), ("LATIN", "Latin"),
           ("FULLWIDTH LATIN", "Latin"), ("CYRILLIC", "Cyrillic"), ("GREEK", "Greek"),
           ("ARABIC", "Arabic"), ("HEBREW", "Hebrew"), ("DEVANAGARI", "Indic"),
           ("BENGALI", "Indic"), ("GURMUKHI", "Indic"), ("GUJARATI", "Indic"), ("ORIYA", "Indic"),
           ("TAMIL", "Indic"), ("TELUGU", "Indic"), ("KANNADA", "Indic"), ("MALAYALAM", "Indic"),
           ("SINHALA", "Indic"), ("THAI", "Thai"))
KURT_PCTS = (1, 10, 25, 50, 75, 90, 99)
KURT_VARIANTS = ("full", "offset_removed", "latin", "rand_latin_size", "han", "rand_han_size")
import models                    # noqa: E402  per-model paths (V3 by default, Qwen3.5 for the comparison)


# ---------------------------------------------------------------- vocabulary scripts

def char_script(c, _cache={}):
    s = _cache.get(c)
    if s is None:
        name = unicodedata.name(c, "")
        s = _cache[c] = next((sc for pre, sc in _PREFIX if name.startswith(pre)), "OtherLetter")
    return s


def token_class(piece):
    letters = [c for c in piece if unicodedata.category(c).startswith("L")]
    if letters:
        return Counter(char_script(c) for c in letters).most_common(1)[0][0]
    if "�" in piece:
        return "byte"            # a partial UTF-8 sequence: a fragment of a longer character
    if any(unicodedata.category(c) == "Nd" for c in piece):
        return "digit"
    return "space" if not piece.strip() else "punct"


def load_vocab(V, model="v3"):
    cache = models.get(model)["vocab_cache"]
    if cache.exists():
        v = json.loads(cache.read_text())
        if v["classes"] == list(CLASSES) and len(v["cls"]) == V:
            return np.array(v["cls"], np.int64), v["pieces"]
    tok = models.load_tokenizer_for(model)
    special = set(tok.all_special_ids) | set(getattr(tok, "added_tokens_decoder", None) or {})
    n_tok = len(tok)
    pieces, cls = [], []
    for i in range(V):
        piece = tok.decode([i]) if i < n_tok else ""
        pieces.append(piece)
        cls.append(CLASSES.index("special" if (i >= n_tok or i in special) else token_class(piece)))
    cache.write_text(json.dumps({"classes": list(CLASSES), "cls": cls, "pieces": pieces}))
    return np.array(cls, np.int64), pieces


def vocab_summary(cls):
    counts = np.bincount(cls, minlength=len(CLASSES))
    p = counts[:N_LETTER] / counts[:N_LETTER].sum()
    ent = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
    return {"counts": dict(zip(CLASSES, counts.tolist())),
            "shares": {k: round(v / counts.sum(), 4) for k, v in zip(CLASSES, counts.tolist())},
            "letter_script_entropy_bits": ent}


# ---------------------------------------------------------------- streamed readouts

class Unembed:
    """lm_head in float32 chunks from its fp16 file. The RMSNorm weight is applied on the
    residual side, (x*g) @ W.T == x @ (W*g).T, so the matrix is never copied whole."""

    def __init__(self, torch, cls, unembed_dir, chunk=16384):
        self.torch = torch
        self.W = np.load(unembed_dir / "lm_head_weight.npy", mmap_mode="r")
        self.g = torch.from_numpy(np.load(unembed_dir / "rms_norm_weight.npy").astype(np.float32))
        self.V, self.d = self.W.shape
        self.chunk = chunk
        self.cls = cls
        self.counts = np.bincount(cls, minlength=len(CLASSES))
        sums = np.zeros((len(CLASSES), self.d))
        for s, Wc in self.chunks():
            c = cls[s:s + Wc.shape[0]]
            for k in np.unique(c):
                sums[k] += Wc.numpy()[c == k].sum(0, dtype=np.float64)
        self.class_mean_rows = torch.from_numpy(
            (sums / np.maximum(self.counts, 1)[:, None]).astype(np.float32))
        self.mean_row = torch.from_numpy((sums.sum(0) / self.V).astype(np.float32))

    def chunks(self):
        for s in range(0, self.V, self.chunk):
            yield s, self.torch.from_numpy(np.asarray(self.W[s:s + self.chunk], dtype=np.float32))


class TopK:
    def __init__(self, torch, N, k):
        self.t, self.k = torch, k
        self.v = torch.full((N, k), -float("inf"))
        self.i = torch.full((N, k), -1, dtype=torch.long)

    def update(self, g0, g1, z, gids):
        v, j = self.t.topk(z, min(self.k, z.shape[1]), dim=1)
        cv = self.t.cat([self.v[g0:g1], v], 1)
        ci = self.t.cat([self.i[g0:g1], gids[j]], 1)
        v2, j2 = self.t.topk(cv, self.k, dim=1)
        self.v[g0:g1], self.i[g0:g1] = v2, self.t.gather(ci, 1, j2)


def readout_stats(torch, X, emb, rand_sets, k_full=100, class_k=0, group=2048):
    """Scale-free statistics of the readouts of X [N, d] (already lens-transported), streamed
    over the vocabulary. Moments are raw power sums of (z - vocabulary mean), per script class
    and per random subset; everything else is derived from them."""
    n_cls, N = len(CLASSES), X.shape[0]
    X = X.float()
    X = X / (torch.linalg.vector_norm(X, dim=1, keepdim=True) / emb.d ** 0.5 + 1e-6)
    X = X * emb.g
    M = X @ emb.class_mean_rows.T                       # class-mean logit, [N, n_cls]
    mu = X @ emb.mean_row                               # vocabulary-mean logit, [N]
    cls_t = torch.from_numpy(emb.cls)
    mom = np.zeros((N, n_cls, 4))
    rmom = np.zeros((N, len(rand_sets), 4))
    top, off = TopK(torch, N, k_full), TopK(torch, N, k_full)
    ctop = {LATIN: TopK(torch, N, class_k), HAN: TopK(torch, N, class_k)} if class_k else {}
    for s, Wc in emb.chunks():
        e = s + Wc.shape[0]
        c = cls_t[s:e]
        gids = torch.arange(s, e)
        perm = torch.argsort(c, stable=True)
        present, sizes = torch.unique(c[perm], return_counts=True)
        present, sizes = present.numpy(), sizes.tolist()
        rcols = [torch.from_numpy(np.flatnonzero(r[s:e])) for r in rand_sets]
        kcols = {k: torch.nonzero(c == k).flatten() for k in ctop}
        for g0 in range(0, N, group):
            g1 = min(g0 + group, N)
            z = X[g0:g1] @ Wc.T                                           # [G, C]
            top.update(g0, g1, z, gids)
            off.update(g0, g1, z - M[g0:g1].index_select(1, c), gids)
            for k, cols in kcols.items():
                if len(cols):
                    ctop[k].update(g0, g1, z.index_select(1, cols), cols + s)
            zc = z - mu[g0:g1, None]
            p1 = zc.index_select(1, perm)
            p2 = p1 * p1
            for j, pw in enumerate((p1, p2, p2 * p1, p2 * p2)):
                mom[g0:g1, present, j] += torch.stack(
                    [b.sum(1) for b in torch.split(pw, sizes, 1)], 1).double().numpy()
            for r, cols in enumerate(rcols):
                q1 = zc.index_select(1, cols)
                q2 = q1 * q1
                for j, pw in enumerate((q1, q2, q2 * q1, q2 * q2)):
                    rmom[g0:g1, r, j] += pw.sum(1).double().numpy()
    out = moments_summary(mom, rmom, emb.counts, [int(r.sum()) for r in rand_sets])
    out["top_ids"], out["offset_removed_top_ids"] = top.i.numpy(), off.i.numpy()
    for k, t in ctop.items():
        out[f"{CLASSES[k].lower()}_top_ids"] = t.i.numpy()
    return out


def _central(n, S):
    a, e2, e3, e4 = (S[..., j] / n for j in range(4))
    return e2 - a ** 2, e4 - 4 * a * e3 + 6 * a ** 2 * e2 - 3 * a ** 4


def moments_summary(mom, rmom, counts, rand_sizes):
    V = counts.sum()
    pres = counts > 0
    m2f, m4f = _central(V, mom.sum(1))
    m2k, m4k = _central(np.maximum(counts, 1)[None, :], mom)
    within = (counts[pres][None, :] * m2k[:, pres]).sum(1)
    rest = [k for k in range(len(CLASSES)) if k not in (LATIN, HAN) and pres[k]]
    m2r, _ = _central(counts[rest].sum(), mom[:, rest].sum(1))
    within3 = counts[LATIN] * m2k[:, LATIN] + counts[HAN] * m2k[:, HAN] + counts[rest].sum() * m2r
    kurt = {"full": m4f / m2f ** 2 - 3,
            "offset_removed": ((counts[pres][None, :] * m4k[:, pres]).sum(1) / V)
                              / (within / V) ** 2 - 3,
            "latin": m4k[:, LATIN] / m2k[:, LATIN] ** 2 - 3,
            "han": m4k[:, HAN] / m2k[:, HAN] ** 2 - 3}
    for name, r, n in zip(("rand_latin_size", "rand_han_size"), range(rmom.shape[1]), rand_sizes):
        m2, m4 = _central(n, rmom[:, r])
        kurt[name] = m4 / m2 ** 2 - 3
    return {"kurt": kurt, "eta2_script16": 1 - within / (V * m2f),
            "eta2_script3": 1 - within3 / (V * m2f),
            "class_mean_offset_han_minus_latin": (mom[:, HAN, 0] / max(counts[HAN], 1)
                                                  - mom[:, LATIN, 0] / max(counts[LATIN], 1))
                                                 / np.sqrt(m2f)}


def script_stats(ids, cls, K):
    c = cls[ids[:, :K]]
    cnt = np.stack([(c == k).sum(1) for k in range(len(CLASSES))], 1)
    nl = cnt[:, :N_LETTER].sum(1)
    p = cnt[:, :N_LETTER] / np.maximum(nl, 1)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.where(p > 0, p * np.log2(p), 0.0).sum(1)
    return {"frac_latin": cnt[:, LATIN] / K, "frac_han": cnt[:, HAN] / K,
            "frac_other_letter": (nl - cnt[:, LATIN] - cnt[:, HAN]) / K,
            "frac_nonletter": 1 - nl / K, "entropy": ent,
            "mixed": (cnt[:, LATIN] > 0) & (cnt[:, HAN] > 0), "top1": c[:, 0]}


def summarize_script(st, m=slice(None)):
    return {"frac_latin": float(st["frac_latin"][m].mean()),
            "frac_han": float(st["frac_han"][m].mean()),
            "frac_other_letter": float(st["frac_other_letter"][m].mean()),
            "frac_nonletter": float(st["frac_nonletter"][m].mean()),
            "entropy_mean": float(st["entropy"][m].mean()),
            "entropy_p50": float(np.percentile(st["entropy"][m], 50)),
            "entropy_p90": float(np.percentile(st["entropy"][m], 90)),
            "mixed_latin_han": float(st["mixed"][m].mean()),
            "top1_latin": float((st["top1"][m] == LATIN).mean()),
            "top1_han": float((st["top1"][m] == HAN).mean())}


def pcts(x):
    x = x[np.isfinite(x)]
    return {f"p{q}": float(np.percentile(x, q)) for q in KURT_PCTS}


# ---------------------------------------------------------------- shared setup

def setup(args):
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = str(args.threads)
    import torch
    torch.set_num_threads(args.threads)
    m = models.get(args.model)
    if args.states is None:
        args.states = m["states"]["wikitext"]
    if args.out_dir is None:
        args.out_dir = REPO / "outputs/jlens" if args.model == "v3" else models.QS / args.model / "analysis"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    S = torch.load(args.states, map_location="cpu", weights_only=False, mmap=True)
    if not S["unembed_check"]["passed"]:
        raise SystemExit(f"{args.states} failed G-UNEMBED")
    H = S["H"]
    P = min(H.shape[0], args.max_prompts or H.shape[0])
    lo = S["prompt_span"][0]
    # Exclude any position whose final-layer residual hits the fp16 limit, at every layer (for
    # the WikiText states this finds exactly SATURATED).
    sat = (H[:P, :, S["n_layers"] - 1].float().abs().amax(-1) >= 65504).numpy()
    keep = ~sat
    found = [(int(p) + lo, int(t)) for p, t in np.argwhere(sat)]
    print(f"excluded saturated (prompt, position): {found}", flush=True)
    if args.model == "v3" and args.states == m["states"]["wikitext"] and P == H.shape[0] and found != m["saturated"]:
        raise SystemExit(f"saturation scan found {found}, workspace_readouts.py excludes {m['saturated']}")
    V = np.load(m["unembed_dir"] / "lm_head_weight.npy", mmap_mode="r").shape[0]
    cls, pieces = load_vocab(V, args.model)
    t0 = time.time()
    emb = Unembed(torch, cls, m["unembed_dir"])
    rng = np.random.default_rng(args.seed)
    rand_sets = []
    for n in (emb.counts[LATIN], emb.counts[HAN]):
        r = np.zeros(V, bool)
        r[rng.choice(V, size=int(n), replace=False)] = True
        rand_sets.append(r)
    print(f"states {tuple(H.shape)} prompts {lo}..{lo+P-1}; vocabulary {vocab_summary(cls)['shares']}"
          f"  (class means {time.time()-t0:.0f}s)", flush=True)
    return torch, S, H, P, lo, keep, cls, pieces, emb, rand_sets


def transport(torch, Xraw, lens, L, target):
    if lens is None or L >= target:
        return Xraw
    J = lens[L]
    J = J if isinstance(J, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(J, np.float32))
    return Xraw @ J.float().T


def show(piece):
    return repr(piece)[1:-1]


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


# ---------------------------------------------------------------- spikes

def cmd_spikes(args):
    torch, S, H, P, lo, keep, cls, pieces, emb, rand_sets = setup(args)
    target = S["n_layers"] - 1
    ids_in = S["input_ids"][:P].numpy()
    T, d = H.shape[1], H.shape[3]
    valid = keep.reshape(-1)
    ref_path = REPO / "outputs/jlens/workspace_readouts_shipped100.json"
    ref = ({r["layer"]: r for r in json.loads(ref_path.read_text())["per_layer"]}
           if ref_path.exists() and args.model == "v3" and args.states == models.get("v3")["states"]["wikitext"] else {})
    lenses = {name: models.open_lens(args.model, name) for name in args.lenses}
    tok = models.load_tokenizer_for(args.model)
    report = {"vocab": vocab_summary(cls), "layers": {}}
    for L in args.layers:
        Xraw = H[:P, :, L].reshape(-1, d).float()
        res = {}
        for name, lens in lenses.items():
            t0 = time.time()
            res[name] = readout_stats(torch, transport(torch, Xraw, lens, L, target), emb,
                                      rand_sets, k_full=50)
            print(f"L{L} {name}: {time.time()-t0:.0f}s", flush=True)
        main = res[args.lenses[0]]
        ku = main["kurt"]["full"]
        thr = np.percentile(ku[valid], 99)
        spike = valid & (ku >= thr)
        rest = valid & ~spike
        sp_idx = np.flatnonzero(spike)
        sp_idx = sp_idx[np.argsort(-ku[sp_idx])]
        p_of, t_of = np.divmod(np.arange(len(valid)), T)

        lay = {"depth_0_100": round(100 * L / target, 2), "n_activations": int(valid.sum()),
               "kurtosis_pcts": {n: pcts(r["kurt"]["full"][valid]) for n, r in res.items()},
               "spike_flat_indices": sp_idx.tolist(),      # into [prompt x position], main lens
               "n_positions": int(T)}
        if L in ref and args.lenses[0] == "shipped" and P == H.shape[0]:
            lay["check_vs_workspace_readouts"] = {f"p{q}": [lay["kurtosis_pcts"]["shipped"][f"p{q}"],
                                                             ref[L][f"kurtosis_p{q}"]] for q in KURT_PCTS}
        print(f"\n=== layer {L} ({lay['depth_0_100']}% depth)  top-1% threshold kurtosis {thr:.2f}, "
              f"{len(sp_idx)} activations ===")
        for n in res:
            k = lay["kurtosis_pcts"][n]
            print(f"  kurtosis {n:8s} " + "  ".join(f"p{q} {k[f'p{q}']:6.2f}" for q in KURT_PCTS))
        if "check_vs_workspace_readouts" in lay:
            dev = max(abs(a - b) for a, b in lay["check_vs_workspace_readouts"].values())
            print(f"  reproduces panel (b) percentiles to within {dev:.4f}")

        # where the spikes are
        pc = Counter((p_of[sp_idx] + lo).tolist())
        lay["prompts"] = {"n_distinct": len(pc), "most_common": pc.most_common(8)}
        lay["position_quartiles"] = np.percentile(t_of[sp_idx], [25, 50, 75]).tolist()
        print(f"  from {len(pc)} distinct prompts; most spikes per prompt {pc.most_common(6)}; "
              f"stored-position quartiles {lay['position_quartiles']}")
        inp = Counter(pieces[ids_in[p, t]] for p, t in zip(p_of[sp_idx], t_of[sp_idx]))
        base = Counter(pieces[i] for i in ids_in.reshape(-1)[valid])
        lay["input_tokens"] = [(tk, n, round(n / len(sp_idx) / (base[tk] / valid.sum()), 1))
                               for tk, n in inp.most_common(15)]
        print("  input token at the spike (count, enrichment over all positions): "
              + ", ".join(f"'{show(tk)}' {n} x{e}" for tk, n, e in lay["input_tokens"]))
        nxt = Counter(pieces[ids_in[p, t + 1]] for p, t in zip(p_of[sp_idx], t_of[sp_idx]) if t + 1 < T)
        lay["next_tokens"] = nxt.most_common(10)
        print("  actual next token: " + ", ".join(f"'{show(tk)}' {n}" for tk, n in nxt.most_common(10)))

        # what the spike readouts say
        top = main["top_ids"]
        t1 = Counter(pieces[i] for i in top[sp_idx, 0])
        t10 = Counter(pieces[i] for i in top[sp_idx, :10].reshape(-1))
        lay["readout_top1"] = t1.most_common(15)
        lay["readout_top10_frequent"] = t10.most_common(30)
        print("  readout top-1: " + ", ".join(f"'{show(tk)}' {n}" for tk, n in t1.most_common(15)))
        print("  most frequent in readout top-10: "
              + ", ".join(f"'{show(tk)}' {n}" for tk, n in t10.most_common(30)))
        for src in ("top_ids", "offset_removed_top_ids"):
            st = script_stats(main[src], cls, 50)
            lay[f"scripts_{src}"] = {"spikes": summarize_script(st, spike), "rest": summarize_script(st, rest)}
            a, b = lay[f"scripts_{src}"]["spikes"], lay[f"scripts_{src}"]["rest"]
            print(f"  top-50 scripts [{src}]  spikes: Latin {a['frac_latin']:.2f} Han {a['frac_han']:.2f} "
                  f"other-letter {a['frac_other_letter']:.2f} non-letter {a['frac_nonletter']:.2f}"
                  f"  |  rest: Latin {b['frac_latin']:.2f} Han {b['frac_han']:.2f} "
                  f"other-letter {b['frac_other_letter']:.2f} non-letter {b['frac_nonletter']:.2f}")
        e3 = main["eta2_script3"]
        lay["eta2_script3_median"] = {"spikes": float(np.median(e3[spike])), "rest": float(np.median(e3[rest]))}
        print(f"  script-class share of readout variance (eta^2, median): spikes "
              f"{lay['eta2_script3_median']['spikes']:.3f}  rest {lay['eta2_script3_median']['rest']:.3f}")

        # do the same activations spike under the other lenses?
        lay["replication"] = {}
        for n, r in res.items():
            if n == args.lenses[0]:
                continue
            k2 = r["kurt"]["full"]
            r1, r5 = np.percentile(k2[valid], 99), np.percentile(k2[valid], 95)
            rep = {"spearman_vs_main": spearman(ku[valid], k2[valid]),
                   "main_spikes_in_its_top1pct": float((k2[sp_idx] >= r1).mean()),
                   "main_spikes_in_its_top5pct": float((k2[sp_idx] >= r5).mean())}
            lay["replication"][n] = rep
            print(f"  vs {n:8s}: Spearman {rep['spearman_vs_main']:.3f};  of the {args.lenses[0]} spikes, "
                  f"{rep['main_spikes_in_its_top1pct']:.0%} are in its top 1%, "
                  f"{rep['main_spikes_in_its_top5pct']:.0%} in its top 5%")

        ex = []
        pick = list(sp_idx[:4]) + list(np.random.default_rng(L).choice(sp_idx[4:], size=min(8, len(sp_idx) - 4),
                                                                         replace=False))
        for i in pick:
            p, t = int(p_of[i]), int(t_of[i])
            ctx = tok.decode(ids_in[p, max(0, t - 12):t + 1].tolist())
            e = {"prompt": p + lo, "position": t, "kurtosis": float(ku[i]),
                 "context": ctx, "next": pieces[ids_in[p, t + 1]] if t + 1 < T else None,
                 "top50": [pieces[j] for j in top[i]],
                 "offset_removed_top20": [pieces[j] for j in main["offset_removed_top_ids"][i, :20]],
                 "kurtosis_other_lenses": {n: float(r["kurt"]["full"][i]) for n, r in res.items()}}
            ex.append(e)
            print(f"\n  [prompt {e['prompt']} pos {t}] kurtosis {e['kurtosis']:.1f}  "
                  + " ".join(f"{n}={v:.1f}" for n, v in e["kurtosis_other_lenses"].items()))
            print(f"    context: ...{show(ctx[-90:])}  ||next: '{show(e['next'] or '')}'")
            print("    top-50: " + " | ".join(show(x) for x in e["top50"]))
        lay["examples"] = ex
        report["layers"][L] = lay
        print(flush=True)
    out = args.out_dir / f"bilingual_spikes{args.tag}.json"
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"wrote {out}")


# ---------------------------------------------------------------- depth

def cmd_depth(args):
    torch, S, H, P, lo, keep, cls, pieces, emb, rand_sets = setup(args)
    target = S["n_layers"] - 1
    T, d = H.shape[1], H.shape[3]
    layers = args.layers or sorted({int(round(x)) for x in np.linspace(0, target, 25)})
    # top-k lists are kept at the same fractions of depth in every model: V3's 0, 15, 25, 35, 45, 55, 60
    store_layers = args.store_layers or [int(round(target * k / 24)) for k in (0, 6, 10, 14, 18, 22, 24)]
    pos = np.unique(np.round(np.linspace(0, T - 1, args.positions_per_prompt)).astype(int))
    sel = keep[:, pos].reshape(-1)
    print(f"{len(pos)} positions per prompt -> {int(sel.sum())} activations per layer; layers {layers}",
          flush=True)
    for name in args.lenses:
        lens = models.open_lens(args.model, name)
        # stamp the lens's identity into the output: a few-prompt lens can be rebuilt from other prompts, and a
        # consumer that loads readouts by lens NAME would otherwise pair a new lens with a stale run (see
        # penult_compare.py).
        lens_file = getattr(getattr(lens, "cur", None), "path", None)
        lens_n = getattr(lens, "n", None)
        rows, store = [], {}
        heads = {"full": "full", "offset_removed": "offset-rm", "latin": "latin",
                 "rand_latin_size": "rand-L", "han": "han", "rand_han_size": "rand-H"}
        print(f"\n--- lens {name} ---\n{'L':>3} {'depth':>5} | kurt p50 / p99: "
              + " ".join(f"{heads[v]:>11}" for v in KURT_VARIANTS)
              + " | eta2 p50 | top100 H(bits) Han% mixed | offset-rm Han% H", flush=True)
        for L in layers:
            t0 = time.time()
            X = H[:P][:, pos, L].reshape(-1, d)[torch.from_numpy(sel)].float()
            r = readout_stats(torch, transport(torch, X, lens, L, target), emb, rand_sets,
                              k_full=100, class_k=100 if L in store_layers else 0)
            row = {"layer": int(L), "depth_0_100": round(100 * L / target, 2),
                   "n_activations": int(sel.sum()),
                   "kurtosis": {v: pcts(r["kurt"][v]) for v in KURT_VARIANTS},
                   "eta2_script3": pcts(r["eta2_script3"]), "eta2_script16": pcts(r["eta2_script16"]),
                   "han_minus_latin_mean_logit_sd": pcts(r["class_mean_offset_han_minus_latin"]),
                   "topk_scripts": {}}
            for src in ("top_ids", "offset_removed_top_ids"):
                for K in (50, 100):
                    row["topk_scripts"][f"{src}@{K}"] = summarize_script(script_stats(r[src], cls, K))
            row["secs"] = round(time.time() - t0, 1)
            rows.append(row)
            if L in store_layers:
                for key in ("top_ids", "offset_removed_top_ids", "latin_top_ids", "han_top_ids"):
                    store[f"L{L}_{key}"] = r[key].astype(np.int32)
            kk = row["kurtosis"]
            f100, o100 = row["topk_scripts"]["top_ids@100"], row["topk_scripts"]["offset_removed_top_ids@100"]
            print(f"{L:3d} {row['depth_0_100']:5.1f} | " + " ".join(
                f"{kk[v]['p50']:5.2f}/{kk[v]['p99']:5.1f}" for v in KURT_VARIANTS)
                + f" | {row['eta2_script3']['p50']:.3f} | {f100['entropy_mean']:.2f} {f100['frac_han']:5.1%} "
                  f"{f100['mixed_latin_han']:5.1%} | {o100['frac_han']:5.1%} {o100['entropy_mean']:.2f}"
                  f"   ({row['secs']}s)", flush=True)
        out = args.out_dir / f"bilingual_depth_{name}{args.tag}.json"
        out.write_text(json.dumps({"lens": name, "lens_file": str(lens_file) if lens_file else None,
                                   "lens_n_prompts": lens_n, "states": str(args.states), "corpus": S.get("corpus"),
                                   "positions": pos.tolist(), "vocab": vocab_summary(cls),
                                   "kurt_variants": KURT_VARIANTS, "per_layer": rows}, indent=1))
        np.savez_compressed(args.out_dir / f"bilingual_depth_{name}{args.tag}_topk.npz",
                            positions=pos, keep=sel, **store)
        print(f"wrote {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("spikes", "depth"):
        p = sub.add_parser(name)
        p.add_argument("--lenses", nargs="+", default=["shipped", "logit"] if name == "depth"
                       else ["shipped", "n50", "n50b", "logit"])
        p.add_argument("--layers", nargs="+", type=int, default=[5, 15] if name == "spikes" else None)
        p.add_argument("--max-prompts", type=int, default=None)
        p.add_argument("--threads", type=int, default=4)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--tag", default="")
        p.add_argument("--model", default="v3", help="models.MODELS key: v3, qwen35_122b, qwen35_397b_fp8")
        p.add_argument("--states", type=Path, default=None, help="default: the model's WikiText states")
        p.add_argument("--out-dir", type=Path, default=None, help="default: outputs/jlens (V3), outputs/jlens_qwen35/<model>/analysis")
    sub.choices["depth"].add_argument("--positions-per-prompt", type=int, default=24)
    sub.choices["depth"].add_argument("--store-layers", nargs="+", type=int, default=None,
                                      help="layers whose top-k lists are saved; default 0, 25, 42, 58, 75, 92, 100%% of depth")
    args = ap.parse_args()
    {"spikes": cmd_spikes, "depth": cmd_depth}[args.cmd](args)


if __name__ == "__main__":
    main()
