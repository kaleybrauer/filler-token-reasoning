"""
build_wiki_corpus.py — Chinese and English held-out prose from ONE Wikipedia dump, for the
English-vs-Chinese readout comparison (bilingual_diagnostics.py).

The Figure 28 panels were measured on WikiText-103, which is English and carries its own
preprocessing (" @-@ " hyphens, spaces before punctuation). Contrasting it with Chinese
Wikipedia would change two things at once, so both languages come from the same dump
(wikimedia/wikipedia 20231101) through the same loader as prompts_wikitext.json: one middle
paragraph per article, sampled across the whole article stream.

Filters. A paragraph must tokenise to at least --min-tokens (so a 128-token window fits after
BOS) and be written in the target script (Han share of letters for zh, Latin for en). Chinese
Wikipedia mixes Traditional and Simplified articles; V3's Chinese is mostly Simplified, so zh
paragraphs must use Simplified forms of common characters several times more often than
Traditional ones.

Rows come from the Hugging Face datasets-server JSON API (no pyarrow needed). Output has the
same keys as prompts_wikitext.json, plus titles and dataset offsets for provenance.

    python scripts/jlens/build_wiki_corpus.py --lang zh
    python scripts/jlens/build_wiki_corpus.py --lang en
"""
from __future__ import annotations

import argparse, json, os, sys, time, unicodedata, urllib.error, urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

API = "https://datasets-server.huggingface.co/rows?dataset=wikimedia/wikipedia&config={config}&split=train&offset={offset}&length={length}"
CONFIG = {"zh": "20231101.zh", "en": "20231101.en"}
TRAD = set("們這國說時會來對發學經過與電開關見長門問間書車東無為從後當點實現體種還應")
SIMP = set("们这国说时会来对发学经过与电开关见长门问间书车东无为从后当点实现体种还应")


def fetch(config, offset, length, tries=8):
    """Authenticated requests (HF_TOKEN from the environment) get far higher rate limits;
    a 429 waits for Retry-After, or backs off exponentially."""
    headers = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"} if os.environ.get("HF_TOKEN") else {}
    req = urllib.request.Request(API.format(config=config, offset=offset, length=length), headers=headers)
    for k in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())
        except Exception as e:                      # rate limits and transient errors
            if k == tries - 1:
                raise
            wait = min(300, 15 * 2 ** k)
            if isinstance(e, urllib.error.HTTPError) and e.headers.get("Retry-After", "").isdigit():
                wait = int(e.headers["Retry-After"]) + 1
            print(f"  {type(e).__name__} {getattr(e, 'code', '')}: retry in {wait}s", flush=True)
            time.sleep(wait)


def script_share(text, prefix):
    letters = [c for c in text if unicodedata.category(c).startswith("L")]
    if not letters:
        return 0.0
    return sum(unicodedata.name(c, "").startswith(prefix) for c in letters) / len(letters)


def qualifies(para, lang, tok, min_tokens):
    if lang == "zh":
        simp, trad = sum(c in SIMP for c in para), sum(c in TRAD for c in para)
        if script_share(para, "CJK") < 0.8 or simp < 3 or simp < 4 * trad:
            return False
    elif script_share(para, "LATIN") < 0.95:
        return False
    return len(tok(para, add_special_tokens=False)["input_ids"]) >= min_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=sorted(CONFIG), required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--min-tokens", type=int, default=140)
    ap.add_argument("--windows", type=int, default=600, help="evenly spaced sampling windows")
    ap.add_argument("--rows-per-window", type=int, default=20)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    from extract.extract_hidden_states import load_tokenizer
    tok = load_tokenizer("/workspace/models/deepseek-v3-awq")
    config = CONFIG[args.lang]
    total = fetch(config, 0, 1)["num_rows_total"]
    out = args.out or REPO / f"scripts/jlens/prompts_wiki_{args.lang}.json"

    prompts, titles, offsets, scanned = [], [], [], 0
    for w in range(args.windows):
        if len(prompts) >= args.n:
            break
        off = int((w + 0.5) * total / args.windows)
        for i, r in enumerate(fetch(config, off, args.rows_per_window)["rows"]):
            scanned += 1
            paras = [p.strip() for p in r["row"]["text"].split("\n")]
            good = [p for p in paras if qualifies(p, args.lang, tok, args.min_tokens)]
            if good:                                   # the qualifying paragraph nearest the middle
                prompts.append(good[len(good) // 2])
                titles.append(r["row"]["title"])
                offsets.append(off + i)
                break
        if (w + 1) % 25 == 0:
            print(f"  window {w+1}/{args.windows}: {len(prompts)} prompts from {scanned} articles", flush=True)
        time.sleep(1.0)
    if len(prompts) < args.n:
        raise SystemExit(f"only {len(prompts)} qualifying paragraphs; raise --windows")
    tmp = out.with_suffix(".tmp")               # atomic: the GPU extraction polls for this file
    tmp.write_text(json.dumps({
        "source": f"wikimedia/wikipedia:{config}:train",
        "loader": "one middle qualifying paragraph per article, one article per evenly spaced window",
        "filters": {"min_tokens": args.min_tokens,
                    "script": "Han >= 80% of letters, Simplified markers >= 4x Traditional" if args.lang == "zh"
                              else "Latin >= 95% of letters"},
        "n": len(prompts), "n_articles_scanned": scanned, "n_rows_total": total,
        "titles": titles, "offsets": offsets, "prompts": prompts}, indent=1, ensure_ascii=False))
    tmp.replace(out)
    print(f"wrote {out}: {len(prompts)} prompts from {scanned} articles")


if __name__ == "__main__":
    main()
