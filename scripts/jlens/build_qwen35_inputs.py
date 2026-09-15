"""
build_qwen35_inputs.py — inputs for the Qwen3.5 extraction (extract_qwen35_states.py), derived from
the V3 inputs so the two models are compared on identical text and identical concepts.

  concept_prompts_qwen35.json
      the V3 concept set (concept_prompts.json, v1) minus the 19 concepts that share a word with
      another concept (as language_geometry.py drops them): 581 concepts, same templates and texts,
      with Qwen3.5's expected final token id and length for every prompt. Each concept keeps its V3
      index. Every prompt must end on the concept's single Qwen token with the template as a clean
      token prefix, or the build fails.
  qwen35_inputs_check.json
      tokenizer identity across the two Qwen3.5 checkpoints (sha256 of the tokenizer files on the
      Hub), and the Qwen token counts of the WikiText and Wikipedia paragraphs (all must reach 128).

Qwen3.5's tokenizer has no BOS token and jlens's force_bos cannot add one: the lens fit, the
extraction and these expectations all use the plain tokenizer output.

    python scripts/jlens/build_qwen35_inputs.py
"""
from __future__ import annotations

import hashlib, json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODELS = ("Qwen/Qwen3.5-122B-A10B", "Qwen/Qwen3.5-397B-A17B-FP8")
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
LANGS = ("en", "zh", "ja")
CORPORA = {"wikitext": ("scripts/jlens/prompts_wikitext.json", 200), "wiki_zh": ("scripts/jlens/prompts_wiki_zh.json", 0),
           "wiki_en": ("scripts/jlens/prompts_wiki_en.json", 0)}


def main():
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer
    api = HfApi()
    shas = {}
    for m in MODELS:
        files = set(api.list_repo_files(m))
        info = api.get_paths_info(m, [f for f in TOKENIZER_FILES if f in files])
        shas[m] = {i.path: (i.lfs.sha256 if i.lfs else i.blob_id) for i in info}
    same = shas[MODELS[0]] == shas[MODELS[1]]
    print(f"tokenizer files identical across {MODELS}: {same}  {shas[MODELS[0]]}")
    if not same:
        raise SystemExit("the two checkpoints ship different tokenizer files; build inputs per model")
    tok = AutoTokenizer.from_pretrained(MODELS[0])

    src = REPO / "scripts/jlens/concept_prompts.json"
    spec = json.loads(src.read_text())
    C, P = spec["concepts"], spec["prompts"]
    uses = {}
    for ci, c in enumerate(C):
        for lang in LANGS:
            uses.setdefault((lang, c[lang]), []).append(ci)
    shared = {ci for v in uses.values() if len(v) > 1 for ci in v}
    keep = [ci for ci in range(len(C)) if ci not in shared]
    new_index = {ci: k for k, ci in enumerate(keep)}

    def ids(text):
        return tok(text)["input_ids"]

    concepts, prompts, bad = [], [], []
    for ci in keep:
        c = dict(C[ci]); c["v3_index"] = ci
        for lang in LANGS:
            w = ids((" " if lang == "en" else "") + c[lang])
            if len(w) != 1:
                bad.append(("not single", lang, c[lang]))
            c[f"{lang}_id"] = w[0]
        concepts.append(c)
    for p in P:
        if p["concept"] not in new_index:
            continue
        c = concepts[new_index[p["concept"]]]
        word = (" " if p["lang"] == "en" else "") + c[p["lang"]]
        tpl = p["text"][: len(p["text"]) - len(word)]
        full = ids(p["text"])
        pre = ids(tpl) if tpl else []
        if full[-1] != c[f"{p['lang']}_id"] or full[: len(pre)] != pre or len(full) != len(pre) + 1:
            bad.append(("prefix", p["text"]))
        prompts.append({"concept": new_index[p["concept"]], "lang": p["lang"], "template": p["template"],
                        "text": p["text"], "expect_last_id": full[-1], "n_tokens": len(full)})
    if bad:
        raise SystemExit(f"{len(bad)} tokenisation failures, e.g. {bad[:5]}")
    texts = [q["text"] for q in prompts]
    token_seqs = {tuple(ids(t)) for t in texts}
    same_str = sum(c["same_ja_zh"] for c in concepts)
    # bare prompts are the word alone, so one string used by different concepts in different languages
    # (天: day in zh, sky in ja) is also a duplicate; only bare prompts, which the geometry tests don't use
    bare_owner = {}
    for q in prompts:
        if q["template"] == 0:
            bare_owner.setdefault(q["text"], set()).add(q["concept"])
    cross = sum(len(v) - 1 for v in bare_owner.values() if len(v) > 1)
    n_dup = len(prompts) - len(token_seqs)
    print(f"{len(concepts)} concepts, {len(prompts)} prompts, {n_dup} duplicate token sequences = {same_str} "
          f"same-string bare zh/ja pairs + {cross} bare strings shared across concepts")
    if n_dup != same_str + cross:
        raise SystemExit("duplicate count does not match the design")
    out = {"derived_from": str(src.relative_to(REPO)),
           "derived_from_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
           "tokenizer": MODELS[0], "tokenizer_sha256": shas[MODELS[0]], "bos": None,
           "templates": spec["templates"], "template_0_is_bare": True,
           "dropped_shared_word": [C[ci]["en"] for ci in sorted(shared)],
           "concepts": concepts, "prompts": prompts}
    (REPO / "scripts/jlens/concept_prompts_qwen35.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))

    counts = {}
    for name, (path, lo) in CORPORA.items():
        n = [len(ids(t)) for t in json.loads((REPO / path).read_text())["prompts"][lo:lo + 100]]
        counts[name] = {"start": lo, "min_tokens": min(n), "n_under_128": sum(x < 128 for x in n)}
        if counts[name]["n_under_128"]:
            raise SystemExit(f"{name}: {counts[name]['n_under_128']} paragraphs under 128 Qwen tokens")
    (REPO / "scripts/jlens/qwen35_inputs_check.json").write_text(json.dumps(
        {"tokenizer_files_identical": same, "tokenizer_sha256": shas, "paragraph_token_counts": counts,
         "concept_duplicates": n_dup, "bare_strings_shared_across_concepts": cross}, indent=1))
    print(f"paragraphs: {counts}")
    print("wrote scripts/jlens/concept_prompts_qwen35.json and scripts/jlens/qwen35_inputs_check.json")


if __name__ == "__main__":
    main()
