"""
Build per-element multilingual alias table for the letterpos task.

For each chemical element with atomic number 1-100:
  - English name
  - Chemical symbol
  - Labels in: de, fr, es, it, pt, ru, zh, zh-hans, zh-hant, ja, ko, ar, la

Source: Wikidata (SPARQL + wbgetentities). Polite User-Agent included.

Output: data/element_aliases.json — keyed by English element name.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import requests

HEADERS = {"User-Agent": "filler-probing-research/1.0 (kaleybrauer@gmail.com)"}
LANGS = ["en", "zh", "zh-hans", "zh-hant"]


def fetch_element_qids() -> list[dict]:
    sparql = """
    SELECT ?element ?atomicNumber ?symbol ?nameEn WHERE {
      ?element wdt:P31 wd:Q11344 ;
               wdt:P1086 ?atomicNumber ;
               wdt:P246 ?symbol ;
               rdfs:label ?nameEn .
      FILTER(LANG(?nameEn) = 'en')
      FILTER(?atomicNumber >= 1 && ?atomicNumber <= 100)
    }
    ORDER BY ?atomicNumber
    """
    r = requests.get("https://query.wikidata.org/sparql",
                     params={"query": sparql, "format": "json"},
                     headers=HEADERS, timeout=30)
    r.raise_for_status()
    rows = r.json()["results"]["bindings"]
    out = []
    for row in rows:
        out.append({
            "qid": row["element"]["value"].split("/")[-1],
            "atomic_number": int(row["atomicNumber"]["value"]),
            "symbol": row["symbol"]["value"],
            "name": row["nameEn"]["value"].title(),  # "Silver", not "silver"
        })
    return out


def fetch_labels(qids: list[str]) -> dict[str, dict[str, str]]:
    """wbgetentities batch (up to 50 IDs per call)."""
    out = {}
    for i in range(0, len(qids), 50):
        batch = qids[i:i + 50]
        r = requests.get(
            "https://www.wikidata.org/w/api.php",
            params={
                "action": "wbgetentities",
                "ids": "|".join(batch),
                "props": "labels|aliases",
                "languages": "|".join(LANGS),
                "format": "json",
            },
            headers=HEADERS, timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        for qid, ent in data["entities"].items():
            out[qid] = {
                "labels": {lang: ent["labels"][lang]["value"]
                            for lang in LANGS if lang in ent.get("labels", {})},
                "aliases": {
                    lang: [a["value"] for a in ent.get("aliases", {}).get(lang, [])]
                    for lang in LANGS if lang in ent.get("aliases", {})
                },
            }
        time.sleep(0.5)  # polite
    return out


def main():
    print("Querying Wikidata SPARQL for chemical elements...")
    elements = fetch_element_qids()
    print(f"  {len(elements)} elements (Z=1..100)")

    qids = [e["qid"] for e in elements]
    print("Fetching labels via wbgetentities...")
    labels = fetch_labels(qids)

    out = {}
    for e in elements:
        ent = labels.get(e["qid"], {"labels": {}, "aliases": {}})
        # Collect all alias strings — use a set to dedupe
        alias_set = set()
        # Always include the English name and the symbol-as-text
        alias_set.add(e["name"])
        # Wikidata labels per language
        for lang_strs in ent["labels"].values():
            alias_set.add(lang_strs)
        # Wikidata aliases per language (alternative names)
        for lang_alist in ent["aliases"].values():
            for s in lang_alist:
                alias_set.add(s)

        # Filter: drop "Element N" / "elemento 47" / "80Hg" / "العنصر 47" /
        # "primer elemento" — these are metadata-style aliases that would
        # generate false positives via the word "element"/etc.
        import re
        bad_patterns = [
            re.compile(r"^\s*[Ee]lement[oó]?\s*\d+\s*$"),
            re.compile(r"^\s*[ée]l[ée]ment\s*\d+\s*$"),
            re.compile(r"^\s*\d+[A-Z][a-z]?\s*$"),
            re.compile(r"^\s*العنصر\b"),
            re.compile(r"^\s*عنصر\s+"),
            re.compile(r"^\s*элемент\s+\d", re.IGNORECASE),
            re.compile(r"^\s*primer elemento\s*$", re.IGNORECASE),
            re.compile(r"^\s*elemento\s+(uno|due|tre|cuatro|cinco)\s*$", re.IGNORECASE),
        ]
        def keep(s: str) -> bool:
            t = s.strip()
            if not t:
                return False
            return not any(p.match(t) for p in bad_patterns)
        aliases = sorted({s.strip() for s in alias_set if keep(s)})

        out[e["name"]] = {
            "atomic_number": e["atomic_number"],
            "symbol": e["symbol"],
            "qid": e["qid"],
            "aliases": aliases,
        }

    Path("data").mkdir(exist_ok=True)
    with open("data/element_aliases.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(out)} elements → data/element_aliases.json")
    # Preview a few
    for name in ["Hydrogen", "Silver", "Mercury", "Uranium"]:
        if name in out:
            e = out[name]
            print(f"\n{name} (Z={e['atomic_number']}, symbol={e['symbol']}):")
            for s in e["aliases"][:8]:
                print(f"    {s}")
            if len(e["aliases"]) > 8:
                print(f"    ... +{len(e['aliases']) - 8} more")


if __name__ == "__main__":
    main()
