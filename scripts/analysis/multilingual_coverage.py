"""
For letterpos / capitalpos tasks, compute the fraction of examples whose top-K
aggregated tokens contain a substring of the correct answer in English or
Chinese (plus the chemical symbol for elements).

Reads:
  - data/element_aliases.json (for letterpos)
  - data/capital_aliases.json (for capitalpos)
  - <aggregated-dir>/aggregated_<cond>.json

Per (task, condition, K), reports:
  fraction of examples with at least one matching token in top-K.

Usage:
    python scripts/analysis/multilingual_coverage.py \
        --task letterpos \
        --aliases data/element_aliases.json \
        --aggregated-dir outputs/deepseek_letterpos_aggregated

    python scripts/analysis/multilingual_coverage.py \
        --task capitalpos \
        --aliases data/capital_aliases.json \
        --aggregated-dir outputs/deepseek_capitalpos_aggregated
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

CONDS = ["dots_10", "dots_25", "dots_50", "counting_10", "counting_25", "counting_50"]


def is_cjk(s: str) -> bool:
    """True if any character is in CJK / fullwidth / Cyrillic / Arabic ranges."""
    return any(ord(c) > 0x2E80 for c in s)


def token_matches(token_str: str, ref: dict) -> tuple[bool, str | None]:
    """Return (matched, which_alias). Alias is the matched reference string or
    "symbol:<X>" if matched on chemical symbol."""
    t = token_str.strip()
    if not t:
        return False, None

    # (a) symbol match — case-sensitive, exact token equality
    sym = ref.get("symbol")
    if sym:
        t_punc = t.rstrip(".,;:!?)]}>")
        if t_punc == sym or t_punc == sym.lower() and len(sym) >= 2:
            # only accept exact case for symbols (Ag, Hg, U) — case-folded form
            # would over-match common English words
            if t_punc == sym:
                return True, f"symbol:{sym}"

    # (b) name / alias substring match
    t_low = t.lower()
    for alias in ref["aliases"]:
        a = alias.lower().strip()
        if not a:
            continue
        # min length: 2 for CJK / non-Latin; 4 for Latin (3 too permissive)
        min_len = 2 if is_cjk(a) else 4
        if len(t_low) < min_len:
            continue
        # bidirectional substring
        if t_low in a or a in t_low:
            return True, alias
    return False, None


def analyze(aggregated_path: Path, aliases: dict, truth_field: str) -> dict:
    data = json.load(open(aggregated_path))
    results = {K: 0 for K in (1, 2, 3, 5, 10, 20, 50)}
    by_alias_type = {"english_name": 0, "english_alias": 0, "chinese": 0, "symbol": 0}
    n = 0
    not_in_aliases = 0
    for ex in data:
        truth = ex.get(truth_field)
        if not truth:
            continue
        if truth not in aliases:
            not_in_aliases += 1
            continue
        n += 1
        ref = aliases[truth]
        # Find min rank where any token matches; record alias type at first match
        min_rank = None
        first_alias = None
        for rank, tok in enumerate(ex["top_tokens"], 1):
            ok, alias = token_matches(tok["str"], ref)
            if ok:
                min_rank = rank
                first_alias = alias
                break
        if min_rank is not None:
            for K in results:
                if min_rank <= K:
                    results[K] += 1
            # Categorize alias type
            if first_alias.startswith("symbol:"):
                by_alias_type["symbol"] += 1
            elif is_cjk(first_alias):
                by_alias_type["chinese"] += 1
            elif first_alias.lower() == truth.lower() or truth.lower() in first_alias.lower():
                by_alias_type["english_name"] += 1
            else:
                by_alias_type["english_alias"] += 1
    return {"n": n, "not_in_aliases": not_in_aliases,
            "topk_hits": results, "by_alias_type": by_alias_type}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["letterpos", "capitalpos"], required=True)
    ap.add_argument("--aliases", type=Path, required=True)
    ap.add_argument("--aggregated-dir", type=Path, required=True)
    ap.add_argument("--conditions", nargs="+", default=CONDS)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    aliases = json.load(open(args.aliases))
    truth_field = "element" if args.task == "letterpos" else "intermediate"

    print(f"Aliases: {len(aliases)} entries from {args.aliases}")
    print(f"Truth field: {truth_field}")
    print()
    print(f"{'Condition':<14} {'N':>4} {'top-1':>10} {'top-2':>10} {'top-5':>10} "
          f"{'top-10':>10} {'top-20':>10}  | first-match: en-name | en-alias | zh | symbol")

    summary = {}
    for cond in args.conditions:
        path = args.aggregated_dir / f"aggregated_{cond}.json"
        if not path.exists():
            print(f"MISSING: {path}")
            continue
        r = analyze(path, aliases, truth_field)
        n = r["n"]
        h = r["topk_hits"]; bt = r["by_alias_type"]
        def pct(K): return f"{h[K]}/{n} ({h[K]/n:.0%})"
        print(f"  {cond:<12} {n:>4} {pct(1):>10} {pct(2):>10} {pct(5):>10} "
              f"{pct(10):>10} {pct(20):>10}  | "
              f"{bt['english_name']:>10} | {bt['english_alias']:>8} | "
              f"{bt['chinese']:>4} | {bt['symbol']:>6}")
        summary[cond] = r

    if args.output:
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved summary → {args.output}")


if __name__ == "__main__":
    main()
