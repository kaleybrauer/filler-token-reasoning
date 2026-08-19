"""
prepare_prompts.py — build the WikiText-103 prompt corpus for the J-lens fit.

The lens is an average Jacobian over generic text, not over our task prompts:
E_{prompt,t}[d h_final / d h_L]. Anthropic's reference fit uses ~100-1000
web-text passages truncated to 128 tokens; the two published frontier-scale
fits used 24 (praxagent, Qwen-397B) and measured the curve flat from 60
(xiangchensong, DeepSeek-V4-Flash).

SAMPLING (`--mode spread`, the default). The reference loader
`jlens.examples.load_wikitext_prompts` takes the first n records of >= min_chars
off the stream. On WikiText-103 that is badly degenerate: consecutive records are
consecutive paragraphs of the SAME article, so the first 8 prompts came from one
article and the first 60 from eight. An average over near-duplicate contexts
converges to the wrong thing -- it estimates the Jacobian of that article, not of
generic text. This script instead walks the stream article by article and takes
ONE middle paragraph from each, so n prompts come from ~n distinct articles.
`--mode prefix` reproduces the reference behaviour for comparison.

`--preserve-prefix K` copies the first K prompts verbatim out of an existing
corpus file. That is not cosmetic: `jlens.fit` resumes by prompt INDEX, so a fit
already n prompts deep can only be continued against a corpus whose first n
entries are unchanged. The shipped corpus was built with `--preserve-prefix 8`
against the superseded prefix-sampled file, which is why prompts 0-7 are
prefix-sampled and 8+ are spread-sampled.

The committed `prompts_wikitext.json` is the authoritative artifact -- it is what
the lens was actually fitted on, and it self-documents its provenance in its own
header fields. This script regenerates it; it does not define it.

Cached to JSON so the fit itself never depends on the network.

Usage:
    python scripts/jlens/prepare_prompts.py --n 1000 \
        --preserve-prefix 8 --preserve-from scripts/jlens/prompts_wikitext_prefix.json.bak \
        --out scripts/jlens/prompts_wikitext.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

DATASET = ("Salesforce/wikitext", "wikitext-103-raw-v1", "train")

# ' = Title = ' opens an article; ' = = Section = = ' does not.
ARTICLE_RE = re.compile(r"^\s=\s([^=].*?)\s=\s*$")


def load_spread(n_pool: int, min_chars: int):
    """One middle paragraph per article, walking the stream.

    Buffers only the paragraphs of an article that already clear `min_chars`,
    then emits the MIDDLE one -- taking the first would collect nothing but
    lead/definitional prose. Returns (paragraphs, distinct contributing titles).
    """
    from datasets import load_dataset

    path, name, split = DATASET
    ds = load_dataset(path, name, split=split, streaming=True)

    title, buf, out, titles = None, [], [], []

    def flush():
        if title and buf:
            out.append(buf[len(buf) // 2])
            titles.append(title)

    for rec in ds:
        t = rec["text"]
        m = ARTICLE_RE.match(t.rstrip("\n"))
        if m:
            flush()
            buf = []
            title = m.group(1)
            if len(out) >= n_pool:
                break
            continue
        if len(t.strip()) >= min_chars:
            buf.append(t)
    flush()
    return out, len(set(titles))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--min-chars", type=int, default=600,
                    help="jlens.examples default; 128 tokens is ~600 chars")
    ap.add_argument("--mode", choices=["spread", "prefix"], default="spread",
                    help="spread: one middle paragraph per article (what we fit on). "
                         "prefix: the reference loader's first-n-records (degenerate).")
    ap.add_argument("--pool", type=int, default=1200,
                    help="Candidate paragraphs to collect before de-duplicating "
                         "against the preserved prefix and trimming to --n")
    ap.add_argument("--preserve-prefix", type=int, default=0,
                    help="Copy the first K prompts verbatim from --preserve-from, so a "
                         "fit already K prompts deep stays resumable (fit resumes by index)")
    ap.add_argument("--preserve-from", type=Path, default=None)
    ap.add_argument("--out", type=Path,
                    default=Path("scripts/jlens/prompts_wikitext.json"))
    args = ap.parse_args()

    if args.mode == "prefix":
        from jlens.examples import load_wikitext_prompts
        prompts = load_wikitext_prompts(args.n, min_chars=args.min_chars)
        n_articles, loader = None, "jlens.examples.load_wikitext_prompts (reference, prefix)"
    else:
        pool, n_articles = load_spread(args.pool, args.min_chars)
        loader = "one middle paragraph per article, sampled across the stream"

    note = None
    if args.preserve_prefix:
        if args.preserve_from is None:
            raise SystemExit("--preserve-prefix needs --preserve-from")
        old = json.loads(args.preserve_from.read_text())["prompts"]
        k = args.preserve_prefix
        if len(old) < k:
            raise SystemExit(f"{args.preserve_from} has only {len(old)} prompts, need {k}")
        kept = old[:k]
        # Drop pool entries the prefix already contributes, THEN trim -- otherwise
        # the same paragraph is averaged twice.
        keptset = set(kept)
        prompts = kept + [x for x in pool if x not in keptset][: args.n - k]
        note = (f"first {k} preserved verbatim from the prefix-sampled file so "
                f"jlens.fit's index-based resume stays aligned")
    elif args.mode == "spread":
        prompts = pool[: args.n]

    payload = {
        "source": f"{DATASET[0]}:{DATASET[1]}:{DATASET[2]}",
        "loader": loader,
        "min_chars": args.min_chars,
        "n": len(prompts),
        "prompts": prompts,
    }
    if note:
        payload["note"] = note
    if n_articles is not None:
        payload["n_articles_sampled"] = n_articles
    # Key order matches the shipped artifact.
    order = ["source", "loader", "note", "min_chars", "n", "n_articles_sampled", "prompts"]
    payload = {k: payload[k] for k in order if k in payload}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=1)
    lens = [len(p) for p in prompts]
    print(f"wrote {len(prompts)} prompts -> {args.out}")
    if n_articles is not None:
        print(f"  from {n_articles} articles")
    print(f"  chars: min {min(lens)}  median {sorted(lens)[len(lens) // 2]}  max {max(lens)}")


if __name__ == "__main__":
    main()
