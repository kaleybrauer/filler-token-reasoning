"""
build_concept_prompts.py — matched concept prompts in English, Chinese and Japanese, for the
residual-space language tests (language_geometry.py).

A concept is one dictionary meaning with a SINGLE-TOKEN word in each language under the model's
tokenizer: English ' word' (CC-CEDICT and JMdict glosses), Chinese Simplified headword (CC-CEDICT),
Japanese kanji headword (JMdict, noun sense). Translations must be primary meanings: the Chinese
headword's whole first gloss is the English word; the English word is the first gloss of the Japanese
entry's first sense (frequency-tagged); the Japanese word is itself a CC-CEDICT word with that gloss. Chinese and Japanese share the Han script class, so
the zh/ja contrast separates language from script; concepts whose Japanese and Chinese forms are
the same string are kept and flagged (same script AND same string, different language context).

Each concept appears in every language under the same content-neutral templates, which end right
before the concept, and once bare (concept token after BOS, no language context). The concept must
be the final token and tokenise identically in and out of context; the template alone must
tokenise to a prefix of the full prompt. Concepts failing this in any prompt are dropped.

Dictionaries (downloaded once to data/dictionaries/): CC-CEDICT (CC BY-SA 4.0), JMdict_e (EDRDG,
CC BY-SA 4.0).

    python scripts/jlens/build_concept_prompts.py
"""
from __future__ import annotations

import argparse, gzip, json, re, sys, unicodedata
from pathlib import Path
from xml.etree import ElementTree as ET

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(Path(__file__).resolve().parent))
from translation_pairs import load_cedict, STOP   # noqa: E402

TEMPLATES = {
    "en": ["Here is one word:", "She wrote down a single word:"],   # concept appended as ' word'
    "zh": ["这里有一个词：", "她写下了一个词："],
    "ja": ["ここに一つの言葉があります：", "彼女は一つの言葉を書き留めた："],
}
JMDICT = REPO / "data/dictionaries/JMdict_e.gz"


def is_han(s):
    return bool(s) and all(unicodedata.name(c, "").startswith("CJK") for c in s)


def load_jmdict():
    """English gloss word -> {kanji headword: priority} for noun senses with kanji-only headwords."""
    en2ja = {}
    with gzip.open(JMDICT, "rb") as f:
        for _, entry in ET.iterparse(f):
            if entry.tag != "entry":
                continue
            kebs = [(k.findtext("keb"), any(p.text.startswith(("news1", "ichi1", "spec1", "nf0")) for p in k.findall("ke_pri")))
                    for k in entry.findall("k_ele")]
            kebs = [(k, pri) for k, pri in kebs if is_han(k) and len(k) <= 3]
            if not kebs:
                entry.clear(); continue
            for si, sense in enumerate(entry.findall("sense")):
                pos = " ".join(p.text or "" for p in sense.findall("pos"))
                if "noun" not in pos:
                    continue
                for gi, g in enumerate(sense.findall("gloss")):
                    txt = re.sub(r"\([^)]*\)", " ", (g.text or "").lower()).strip()
                    if not re.fullmatch(r"[a-z]+", txt) or len(txt) < 3 or txt in STOP:
                        continue
                    for k, pri in kebs:
                        cur = en2ja.setdefault(txt, {})
                        # 7 = frequency-tagged headword, first sense, first gloss: the word's primary meaning
                        cur[k] = max(cur.get(k, 0), 4 * pri + 2 * (si == 0) + (gi == 0))
            entry.clear()
    return en2ja


def cedict_first_gloss():
    """Chinese Simplified headword -> first English gloss word (for preferring exact-meaning headwords)."""
    import zipfile
    path = REPO / "data/dictionaries/cedict_1_0_ts_utf-8_mdbg.zip"
    with zipfile.ZipFile(path) as z:
        text = z.read(z.namelist()[0]).decode("utf-8")
    first = {}
    for line in text.splitlines():
        m = re.match(r"^(\S+) (\S+) \[[^\]]*\] /(.*)/\s*$", line)
        if m:
            g = " ".join(re.sub(r"\([^)]*\)", " ", m.group(3).split("/")[0].lower()).split())
            if re.fullmatch(r"[a-z]+", g):                 # a single-word first gloss only
                first.setdefault(m.group(2), g)
    return first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/workspace/models/deepseek-v3-awq")
    ap.add_argument("--max-concepts", type=int, default=600)
    ap.add_argument("--out", type=Path, default=REPO / "scripts/jlens/concept_prompts.json")
    args = ap.parse_args()
    from extract.extract_hidden_states import load_tokenizer
    tok = load_tokenizer(args.model_path)

    def single(s):
        ids = tok(s, add_special_tokens=False)["input_ids"]
        return ids[0] if len(ids) == 1 else None

    en2zh, en2ja, zh_first = load_cedict(), load_jmdict(), cedict_first_gloss()
    print(f"CC-CEDICT {len(en2zh)} English words; JMdict {len(en2ja)} English noun glosses", flush=True)
    cands = []
    for w in sorted(set(en2zh) & set(en2ja)):
        en_id = single(" " + w)
        if en_id is None:
            continue
        # strict: English is the FIRST gloss of the Chinese headword, and the primary meaning of the Japanese entry
        zh = [h for h in en2zh[w] if is_han(h) and len(h) <= 3 and zh_first.get(h) == w and single(h) is not None]
        # ...and the Japanese kanji word is also a Chinese (traditional or simplified) word with this gloss,
        # so a polysemous English word cannot pair two different senses across the languages
        ja = [h for h, p in en2ja[w].items() if p == 7 and h in en2zh[w] and single(h) is not None]
        if not zh or not ja:
            continue
        zh_h, ja_h = min(zh, key=single), min(ja, key=single)      # lowest BPE id = most frequent merge
        ids = (en_id, single(zh_h), single(ja_h))
        cands.append({"en": w, "zh": zh_h, "ja": ja_h, "en_id": ids[0], "zh_id": ids[1], "ja_id": ids[2],
                      "freq_rank_proxy": max(ids), "same_ja_zh": zh_h == ja_h})
    # common concepts first: rank by the rarest of the three tokens (BPE id is a frequency proxy)
    cands.sort(key=lambda c: (c["freq_rank_proxy"], c["en"]))
    print(f"{len(cands)} concepts pass the strict primary-meaning filter and are single-token in all three", flush=True)

    # in-context tokenisation check, then cap
    concepts, prompts = [], []
    for c in cands:
        ok = True
        cp = []
        for lang in ("en", "zh", "ja"):
            word = (" " if lang == "en" else "") + c[lang]
            for ti, tpl in enumerate([""] + TEMPLATES[lang]):
                text = tpl + word
                ids = tok(text, add_special_tokens=True)["input_ids"]
                pre = tok(tpl, add_special_tokens=True)["input_ids"] if tpl else [ids[0]]
                if ids[-1] != c[f"{lang}_id"] or ids[:len(pre)] != pre or len(ids) != len(pre) + 1:
                    ok = False; break
                cp.append({"concept": len(concepts), "lang": lang, "template": ti, "text": text,
                           "expect_last_id": c[f"{lang}_id"], "n_tokens": len(ids)})
            if not ok:
                break
        if ok:
            concepts.append(c); prompts.extend(cp)
        if len(concepts) >= args.max_concepts:
            break
    same = sum(c["same_ja_zh"] for c in concepts)
    args.out.write_text(json.dumps({"model_path": args.model_path, "templates": TEMPLATES,
                                    "template_0_is_bare": True, "concepts": concepts, "prompts": prompts},
                                   indent=1, ensure_ascii=False))
    print(f"wrote {args.out}: {len(concepts)} concepts ({same} with identical ja/zh strings), {len(prompts)} prompts")
    import random
    for c in random.Random(0).sample(concepts, 40):
        print(f"   {c['en']:14s} {c['zh']:4s} {c['ja']:4s}{'  (same string)' if c['same_ja_zh'] else ''}")


if __name__ == "__main__":
    main()
