"""
translation_pairs.py — are the Han tokens in V3's J-lens readouts translations of the Latin ones?

bilingual_diagnostics.py found Han tokens standing out individually (script offset removed) in
English-text readouts from ~40% depth, peaking near 75%, and examples show one concept in both
scripts (' felt' | '覺得' | '他觉得'). Examples can be cherry-picked, so this counts it. At each
stored layer and for each activation: do any of its top-10 Latin readout tokens and any of its
top-10 Han readout tokens form a dictionary translation pair? The null pairs each activation's
Latin list with the Han list of an activation from a DIFFERENT prompt at the same layer, so
generic vocabulary overlap and dictionary over-generation count equally in both.

Dictionary: CC-CEDICT (CC BY-SA 4.0; Traditional and Simplified headwords with English glosses),
downloaded once to data/dictionaries/. Every content word of a gloss maps to the headword; English
inflections are reduced by suffix stripping. A Han token matches a headword when equal, or when one
contains the other and the contained string has at least two characters ('他认为' matches '认为').

    python scripts/jlens/translation_pairs.py outputs/jlens/bilingual_depth_shipped_topk.npz
"""
from __future__ import annotations

import argparse, json, re, sys, urllib.request, zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
CEDICT_URL = "https://www.mdbg.net/chinese/export/cedict/cedict_1_0_ts_utf-8_mdbg.zip"
STOP = set("""the and for with that this from are was were been has have had not but its his her
their they them you your our all can may will would could should one two out off into onto over
under also more most less than then there here what which who whom whose when where why how very
such some each other another used person thing something someone kind sth sb""".split())
SKIP_GLOSS = ("CL:", "variant of", "old variant", "see ", "surname", "Taiwan pr.", "abbr. for",
              "used in", "erhua variant", "Japanese variant")


def load_cedict():
    path = REPO / "data/dictionaries/cedict_1_0_ts_utf-8_mdbg.zip"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(CEDICT_URL, path)
    with zipfile.ZipFile(path) as z:
        text = z.read(z.namelist()[0]).decode("utf-8")
    line_re = re.compile(r"^(\S+) (\S+) \[[^\]]*\] /(.*)/\s*$")
    en2zh = {}
    for line in text.splitlines():
        m = line_re.match(line)
        if not m:
            continue
        trad, simp, glosses = m.groups()
        for g in glosses.split("/"):
            if g.startswith(SKIP_GLOSS):
                continue
            for w in re.findall(r"[a-z]+", re.sub(r"\([^)]*\)", " ", g.lower())):
                if len(w) >= 3 and w not in STOP:
                    en2zh.setdefault(w, set()).update({trad, simp})
    return en2zh


def lemmas(w):
    out = {w}
    for suf, rep in (("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", ""), ("d", ""), ("ly", "")):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            out.add(w[: len(w) - len(suf)] + rep)
    return out


def english_words(pieces_):
    words = set()
    for p in pieces_:
        w = p.strip().lower()
        if re.fullmatch(r"[a-z]{3,}", w) and w not in STOP:
            words |= lemmas(w)
    return words


def translates(heads, tok):
    return tok in heads or any((len(h) >= 2 and h in tok) or (len(tok) >= 2 and tok in h) for h in heads)


def score(latin_rows, han_rows, en2zh, pieces, k):
    """per activation: any pair matches; share of Han top-k tokens translating a Latin top-k token"""
    hit, prec = np.zeros(len(latin_rows), bool), np.zeros(len(latin_rows))
    for i, (lat, han) in enumerate(zip(latin_rows, han_rows)):
        heads = set().union(*(en2zh.get(w, set()) for w in english_words(pieces[j] for j in lat[:k])))
        if not heads:
            continue
        m = [translates(heads, pieces[j].strip()) for j in han[:k]]
        hit[i], prec[i] = any(m), np.mean(m)
    return hit, prec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+", type=Path)
    ap.add_argument("--model", default="v3", help="models.MODELS key, for the vocabulary pieces and output path")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n-perm", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    en2zh = load_cedict()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import models
    v = json.loads(models.get(args.model)["vocab_cache"].read_text())
    pieces = v["pieces"]
    print(f"CC-CEDICT: {len(en2zh)} English words -> Chinese headwords")
    report = {}
    for path in args.npz:
        z = np.load(path)
        n_pos = len(z["positions"])
        prompt_of = np.flatnonzero(z["keep"]) // n_pos
        layers = sorted(int(k[1:].split("_")[0]) for k in z.files if k.endswith("_latin_top_ids"))
        rng = np.random.default_rng(args.seed)
        print(f"\n{path.name}  (top-{args.k} Latin x top-{args.k} Han, null = another prompt's Han list)")
        print(f"{'layer':>5} {'pairs found':>12} {'null':>7} {'ratio':>6} | {'Han precision':>13} {'null':>6}")
        rows = {}
        for L in layers:
            lat, han = z[f"L{L}_latin_top_ids"], z[f"L{L}_han_top_ids"]
            hit, prec = score(lat, han, en2zh, pieces, args.k)
            nh, np_ = [], []
            for _ in range(args.n_perm):
                perm = rng.permutation(len(lat))
                ok = prompt_of[perm] != prompt_of
                h0, p0 = score(lat[ok], han[perm][ok], en2zh, pieces, args.k)
                nh.append(h0.mean())
                np_.append(p0.mean())
            rows[L] = {"hit_rate": float(hit.mean()), "null_hit_rate": float(np.mean(nh)),
                       "han_precision": float(prec.mean()), "null_han_precision": float(np.mean(np_)),
                       "n": int(len(lat))}
            r = rows[L]
            print(f"{L:5d} {r['hit_rate']:12.1%} {r['null_hit_rate']:7.1%} "
                  f"{r['hit_rate'] / max(r['null_hit_rate'], 1e-9):6.1f} | "
                  f"{r['han_precision']:13.1%} {r['null_han_precision']:6.1%}", flush=True)
        report[path.name] = rows
    out = (REPO / "outputs/jlens/translation_pairs.json" if args.model == "v3"
           else models.QS / args.model / "analysis" / "translation_pairs.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev.update(report)
    out.write_text(json.dumps(prev, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
