"""
Build per-capital multilingual alias table for the capitalpos task — V2.

Uses (capital, region) pairs extracted from the question text in
data/capital_letter_position.json. For each pair, queries Wikidata SPARQL to
find the entity that is officially "capital of <region>" via the P36 property,
which disambiguates Pierre-the-name from Pierre-South-Dakota and similar.

Strategy per pair:
  1. Clean the region phrase ("the German state of Brandenburg" → "Brandenburg").
  2. SPARQL: find ?region with English label = cleaned name, and with P36 = a
     ?capital whose English label (case-insensitive) equals the dataset capital.
  3. If found, use that capital Q-ID. Else fall back to wbsearchentities by
     capital name (the v1 path).
  4. Apply manual Q-ID overrides for entries where the fallback returns a
     wrong-but-similar entity (Star Trek city entries, municipality vs. city).
  5. Fetch en + zh + zh-hans + zh-hant labels and aliases via wbgetentities.
  6. Clean aliases: drop "City of X" / nickname / municipality patterns and
     bare ISO codes; strip parentheticals; split on commas. Leaves only
     canonical name forms in EN and ZH.

Output: data/capital_aliases.json
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import requests

HEADERS = {"User-Agent": "filler-probing-research/1.0 (email-here)"}
LANGS = ["en", "zh", "zh-hans", "zh-hant"]

CITY_KEYWORDS = re.compile(
    r"\b(city|town|capital|municipality|metropolis|prefecture|county seat|"
    r"administrative seat|state capital|provincial capital|locality|village)\b",
    re.IGNORECASE,
)

REGION_CLEANUPS = [
    (re.compile(r"^the (?:German|Australian|Indian|Mexican|US|Canadian|"
                r"Brazilian|Italian|Spanish|French|Russian|Argentine|Argentinian) "
                r"(?:state|province|region|department|state) of ", re.I), ""),
    (re.compile(r"^the (?:state|province|region|department) of ", re.I), ""),
    (re.compile(r"^the ", re.I), ""),
]

# Manual Q-ID overrides for entries where wbsearchentities fallback returned
# a wrong-but-similarly-named entity. Each was verified by inspecting the
# Wikidata entity's English description and instance-of (P31) claims.
MANUAL_QID_OVERRIDES = {
    "Berlin":       "Q64",          # was Star Trek city entry
    "Paris":        "Q90",          # was Star Trek city entry
    "Sarajevo":     "Q11194",       # city, not Republika Srpska
    "Copenhagen":   "Q1748",        # was Star Trek city entry
    "Stockholm":    "Q1754",        # was Star Trek city entry
    "Reykjavik":    "Q1764",        # was Star Trek city entry
    "Moscow":       "Q649",         # was Star Trek city entry
    "Boston":       "Q100",         # was Star Trek city entry
    "Chilpancingo": "Q139492122",   # capital of Guerrero (was the municipality entry)
}

# Alias cleanup. DROP_PATTERNS match aliases that aren't canonical names —
# e.g. "City of Light" (a nickname for Paris) would otherwise survive prefix-
# stripping as "Light" and false-match any word "light" in the top-K tokens.
DROP_PATTERNS = [
    re.compile(r"^Capital of\b", re.I),
    re.compile(r"\bMunicipality\b", re.I),
    re.compile(r"\bCounty Seat\b", re.I),
    re.compile(r"^[A-Z][A-Z]$"),                    # bare 2-letter state abbrev
    re.compile(r"^[A-Z]{2,4}-[A-Z0-9]{2,4}$"),      # ISO codes like DE-BE
    re.compile(r"^(City|Town|Village) of\b", re.I), # nicknames like "City of Light"
]
NICKNAME_PATTERNS = [
    re.compile(r"^The\b", re.I),       # "The Hub", "The Big Apple"
    re.compile(r"\bCradle\b", re.I),
    re.compile(r"\bAthens of\b", re.I),
]


def clean_alias(s: str) -> str | None:
    """Return cleaned alias, or None if it should be dropped entirely."""
    s = s.strip()
    if not s:
        return None
    for p in DROP_PATTERNS + NICKNAME_PATTERNS:
        if p.search(s):
            return None
    s = re.sub(r"\s*\([^)]*\)\s*$", "", s).strip()  # drop trailing parenthetical
    if "," in s:
        s = s.split(",", 1)[0].strip()              # "Hartford, CT" → "Hartford"
    if not s:
        return None
    if re.fullmatch(r"[A-Z][A-Z]", s) or s.lower() in {
        "city", "town", "capital", "metropolis", "village"
    }:
        return None
    return s


def clean_region(name: str) -> list[str]:
    """Return candidate cleaned region names to try, in order of likelihood."""
    candidates = [name]
    for pat, repl in REGION_CLEANUPS:
        cleaned = pat.sub(repl, name).strip()
        if cleaned and cleaned not in candidates:
            candidates.append(cleaned)
    # Also strip trailing " Province"
    if name.endswith(" Province"):
        candidates.append(name[: -len(" Province")].strip())
    return candidates


def sparql_lookup(capital: str, region: str) -> str | None:
    """Find Q-ID of the city named `capital` that's the P36 capital of `region`."""
    cap_lower = capital.replace('"', '').lower()
    region_clean = region.replace('"', '')
    sparql = f'''
    SELECT ?capital WHERE {{
      ?region rdfs:label "{region_clean}"@en ;
              wdt:P36 ?capital .
      ?capital rdfs:label ?capLabel .
      FILTER(LANG(?capLabel) = "en")
      FILTER(LCASE(STR(?capLabel)) = "{cap_lower}")
    }} LIMIT 1
    '''
    try:
        r = requests.get("https://query.wikidata.org/sparql",
                         params={"query": sparql, "format": "json"},
                         headers=HEADERS, timeout=20)
        r.raise_for_status()
        rows = r.json().get("results", {}).get("bindings", [])
        if rows:
            return rows[0]["capital"]["value"].split("/")[-1]
    except Exception:
        return None
    return None


def search_capital(name: str) -> str | None:
    """Fallback: wbsearchentities by capital name."""
    r = requests.get(
        "https://www.wikidata.org/w/api.php",
        params={"action": "wbsearchentities", "search": name,
                "language": "en", "type": "item", "format": "json", "limit": 10},
        headers=HEADERS, timeout=20,
    )
    r.raise_for_status()
    results = r.json().get("search", [])
    for res in results:
        if CITY_KEYWORDS.search(res.get("description", "")):
            return res["id"]
    if results:
        return results[0]["id"]
    return None


def fetch_labels(qids: list[str]) -> dict[str, dict]:
    out = {}
    for i in range(0, len(qids), 50):
        batch = qids[i:i + 50]
        r = requests.get(
            "https://www.wikidata.org/w/api.php",
            params={"action": "wbgetentities", "ids": "|".join(batch),
                    "props": "labels|aliases",
                    "languages": "|".join(LANGS), "format": "json"},
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
        time.sleep(0.4)
    return out


def main():
    src = json.load(open("data/capital_letter_position.json"))
    pairs_dict: dict[str, str] = {}  # capital → region (first seen)
    for ex in src["examples"]:
        cap = ex["intermediate"]
        m = re.search(r"capital of (.+?)\?", ex["question"])
        if m and cap not in pairs_dict:
            pairs_dict[cap] = m.group(1).strip()
    print(f"{len(pairs_dict)} unique (capital, region) pairs")

    # Step 1: SPARQL lookup with region disambiguation
    qid_for: dict[str, str | None] = {}
    sparql_hits = 0
    fallback_hits = 0
    no_match = 0
    for i, (cap, region) in enumerate(pairs_dict.items(), 1):
        qid = None
        for region_candidate in clean_region(region):
            qid = sparql_lookup(cap, region_candidate)
            if qid:
                sparql_hits += 1
                break
            time.sleep(0.2)
        if qid is None:
            qid = search_capital(cap)
            if qid:
                fallback_hits += 1
            else:
                no_match += 1
        qid_for[cap] = qid
        if i % 25 == 0:
            print(f"  {i}/{len(pairs_dict)}  (sparql={sparql_hits} "
                  f"fallback={fallback_hits} miss={no_match})")
        time.sleep(0.2)
    print(f"\n  sparql_hits={sparql_hits}  fallback_hits={fallback_hits}  "
          f"no_match={no_match}")

    # Step 2: apply manual Q-ID overrides
    overridden = 0
    for cap, override_qid in MANUAL_QID_OVERRIDES.items():
        if cap in qid_for and qid_for[cap] != override_qid:
            qid_for[cap] = override_qid
            overridden += 1
    print(f"Applied {overridden} manual Q-ID overrides")

    # Step 3: fetch labels
    qids = sorted({q for q in qid_for.values() if q})
    print(f"\nFetching labels for {len(qids)} capitals...")
    labels = fetch_labels(qids)

    # Step 4: assemble — clean aliases and dedupe case-insensitively
    out = {}
    for name, qid in qid_for.items():
        ent = labels.get(qid, {"labels": {}, "aliases": {}}) if qid else {"labels": {}, "aliases": {}}
        raw = {name}
        for s in ent["labels"].values():
            raw.add(s)
        for alist in ent["aliases"].values():
            for s in alist:
                raw.add(s)
        cleaned = []
        seen: set[str] = set()
        for a in raw:
            c = clean_alias(a)
            if c and c.lower() not in seen:
                cleaned.append(c)
                seen.add(c.lower())
        if name not in seen:
            cleaned.insert(0, name)
        out[name] = {"qid": qid, "aliases": sorted(cleaned),
                     "region_used": pairs_dict[name]}

    Path("data").mkdir(exist_ok=True)
    with open("data/capital_aliases.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(out)} capitals → data/capital_aliases.json")
    for name in ["Berlin", "Pierre", "Bismarck", "Tokyo", "Mumbai", "Shimla",
                  "Chilpancingo", "Culiacán"]:
        if name in out:
            e = out[name]
            print(f"\n{name} (qid={e['qid']}, region={e['region_used']!r}):")
            for s in e["aliases"][:8]:
                print(f"    {s}")


if __name__ == "__main__":
    main()
