"""
llm_decode_batch.py

Batch-decode aggregated-residual top-50 JSONs (from aggregate_residuals_all_settings.py)
across multiple conditions × LLMs with threaded Anthropic API calls.

Supports both the 2-fact task (A1 + A2, decode two operands) and the 1-fact
task (fact_value + x where x is known, decode the one retrieved value). Pick
with --task {2fact, 1fact}. Within each task, --prompt toggles between a
"neutral" framing and an "addition"-specific framing. Output filenames include
both so runs don't clobber each other:
    llm_decode_<cond>_<model>_<prompt>_<task>.json

Usage:
    python scripts/decode/llm_decode_batch.py \
        --aggregated-dir outputs/deepseek_aggregated \
        --conditions dots_10 dots_25 dots_50 counting_10 counting_25 \
        --task 2fact --prompt neutral \
        --models haiku sonnet
"""

from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

# ----------------------------------------------------------------------------
# Prompts — 2-fact (decode two operands)
# ----------------------------------------------------------------------------

PROMPT_NEUTRAL_2FACT = (
    "These are potentially important tokens that were taken from a model's "
    "internal state during a task. The tokens in this list are hints about "
    "what specific numbers the model was 'thinking about' in its internal "
    "state while doing the calculation, when it has not yet gotten to "
    "outputting the final answer. The original prompt contained formatting "
    "words like 'Filler' and 'Answer' that are not relevant to the task. "
    "The model is Chinese, so it may use Chinese tokens as well as English.\n\n"
    "What specific two numbers was the model thinking about?\n\n"
    "Think briefly, then end your response with a JSON object on its own line: "
    "{\"n1\": <int>, \"confidence_n1\": <float 0-1>, "
    "\"n2\": <int>, \"confidence_n2\": <float 0-1>, "
    "\"backups\": [<int>, <int>, ... up to 10 ranked alternatives]}. "
    "If you cannot determine n1 or n2, output null for that field."
)

PROMPT_ADDITION_2FACT = (
    "These are potentially important tokens that were taken from a model doing "
    "an addition task. The model was computing the sum of two operands, and "
    "the tokens in this list are hints about what the two operands were — the "
    "numbers the model was 'thinking about' in its internal state while doing "
    "the calculation, when it has not yet gotten to outputting the final sum. "
    "The original prompt contained formatting words like 'Filler' and 'Answer' "
    "that are not relevant to the task. The model is Chinese, so it may use "
    "Chinese tokens as well as English.\n\n"
    "What two numbers are being added together in the task?\n\n"
    "Think briefly, then end your response with a JSON object on its own line: "
    "{\"n1\": <int>, \"confidence_n1\": <float 0-1>, "
    "\"n2\": <int>, \"confidence_n2\": <float 0-1>, "
    "\"backups\": [<int>, <int>, ... up to 10 ranked alternatives]}. "
    "If you cannot determine n1 or n2, output null for that field."
)

# ----------------------------------------------------------------------------
# Prompts — 1-fact (decode one retrieved value)
# ----------------------------------------------------------------------------

PROMPT_NEUTRAL_1FACT = (
    "These are potentially important tokens that were taken from a model's "
    "internal state during a task. The tokens in this list are hints about "
    "what specific number the model was 'thinking about' in its internal "
    "state while doing the calculation, when it has not yet gotten to "
    "outputting the final answer. The original prompt contained formatting "
    "words like 'Filler' and 'Answer' that are not relevant to the task. "
    "The model is Chinese, so it may use Chinese tokens as well as English.\n\n"
    "What specific number was the model thinking about?\n\n"
    "Think briefly, then end your response with a JSON object on its own line: "
    "{\"n\": <int>, \"confidence\": <float 0-1>, "
    "\"backups\": [<int>, <int>, ... up to 10 ranked alternatives]}. "
    "If you cannot determine n, output null for that field."
)

PROMPT_ADDITION_1FACT = (
    "These are potentially important tokens that were taken from a model doing "
    "an addition task. The model was computing the sum of a value it had to "
    "retrieve from memory and another value that was given directly in the "
    "prompt. The tokens in this list are hints about what the retrieved value "
    "was — the number the model was 'thinking about' in its internal state "
    "while doing the calculation, when it has not yet gotten to outputting "
    "the final sum. The original prompt contained formatting words like "
    "'Filler' and 'Answer' that are not relevant to the task. The model is "
    "Chinese, so it may use Chinese tokens as well as English.\n\n"
    "What number was the model retrieving from memory?\n\n"
    "Think briefly, then end your response with a JSON object on its own line: "
    "{\"n\": <int>, \"confidence\": <float 0-1>, "
    "\"backups\": [<int>, <int>, ... up to 10 ranked alternatives]}. "
    "If you cannot determine n, output null for that field."
)

PROMPT_NEUTRAL_LETTERPOS = (
    "These are potentially important tokens that were taken from a model's "
    "internal state during a task. The tokens in this list are hints about "
    "what specific concept, entity, or name the model was 'thinking about' "
    "in its internal state while doing the calculation, when it has not yet "
    "gotten to outputting the final answer. The original prompt contained "
    "formatting words like 'Filler' and 'Answer' that are not relevant to "
    "the task. The model is Chinese, so it may use Chinese tokens as well "
    "as English.\n\n"
    "What specific thing (person, place, object, concept, etc.) was the "
    "model thinking about?\n\n"
    "Think briefly, then end your response with a JSON object on its own line: "
    "{\"answer\": <string>, \"confidence\": <float 0-1>, "
    "\"backups\": [<string>, <string>, ... up to 10 ranked alternatives]}. "
    "If you cannot determine the answer, output null for that field."
)

PROMPT_CHEMISTRY_LETTERPOS = (
    "These are potentially important tokens that were taken from a model "
    "doing a chemistry task. The model was computing a property of a specific "
    "chemical element, and the tokens in this list are hints about which "
    "element the model was 'thinking about' in its internal state while doing "
    "the calculation, when it has not yet gotten to outputting the final "
    "answer. The original prompt contained formatting words like 'Filler' "
    "and 'Answer' that are not relevant to the task. The model is Chinese, "
    "so it may use Chinese tokens as well as English.\n\n"
    "Which chemical element was the model thinking about?\n\n"
    "Think briefly, then end your response with a JSON object on its own line: "
    "{\"answer\": <string — full element name in English, e.g. \"Chlorine\">, "
    "\"confidence\": <float 0-1>, "
    "\"backups\": [<string>, <string>, ... up to 10 ranked alternative element names]}. "
    "If you cannot determine the answer, output null for that field."
)

PROMPT_GEOGRAPHY_CAPITALPOS = (
    "These are potentially important tokens that were taken from a model "
    "doing a geography task. The model was computing a property of a specific "
    "city, and the tokens in this list are hints about which "
    "city the model was 'thinking about' in its internal state while doing "
    "the calculation, when it has not yet gotten to outputting the final "
    "answer. The original prompt contained formatting words like 'Filler' "
    "and 'Answer' that are not relevant to the task. The model is Chinese, "
    "so it may use Chinese tokens as well as English.\n\n"
    "Which city was the model thinking about?\n\n"
    "Think briefly, then end your response with a JSON object on its own line: "
    "{\"answer\": <string — full city name in English, e.g. \"Paris\">, "
    "\"confidence\": <float 0-1>, "
    "\"backups\": [<string>, <string>, ... up to 10 ranked alternative city names]}. "
    "If you cannot determine the answer, output null for that field."
)

# ----------------------------------------------------------------------------
# Prompts — varbind (decode the hidden serial intermediates; rank ALL numbers)
# ----------------------------------------------------------------------------
# The pooled filler-region tokens contain four numbers: the given base operand,
# the queried variable's value V (hidden), the scaled product c·V (hidden), and
# the final answer. We score all four downstream, so the judge is asked for a
# flat RANKED LIST of every number it can read — never told to exclude anything
# (over-exclusion risk) and never told the task structure (would invite ranking
# by arithmetic plausibility instead of token evidence). The judge has no access
# to the prompt's coefficients/operations, so it cannot back-solve V from the
# base and answer — any V/c·V recovery must come from the tokens.

PROMPT_NEUTRAL_VARBIND = (
    "These are potentially important tokens that were taken from a model's "
    "internal state during a task. The tokens in this list are hints about what "
    "specific numbers the model was 'thinking about' in its internal state while "
    "working through a calculation, before it output the final answer. The "
    "original prompt contained formatting words like 'Filler' and 'Answer' that "
    "are not relevant to the task. The model is Chinese, so it may use Chinese "
    "tokens as well as English.\n\n"
    "List every distinct number the model appears to have been thinking about, "
    "ranked from most to least strongly supported by these tokens.\n\n"
    "Think briefly, then end your response with a JSON object on its own line: "
    "{\"numbers\": [<int>, <int>, ... up to 10 distinct integers, most likely "
    "first]}. If you cannot determine any number, output an empty list."
)

# NOTE: there is deliberately no task-framed varbind prompt. To the judge the
# tokens are an unlabeled list of numbers, and "a value was hidden / computed" is
# not a property it can read off any individual candidate — so such a prompt
# cannot point at V or c·V specifically; it can only rank by token support, same
# as neutral. The per-target pointing is done in the SCORING (run_one_varbind
# scores V / c·V / B / answer separately against the neutral ranked list).

PROMPTS = {
    ("2fact",      "neutral"):   PROMPT_NEUTRAL_2FACT,
    ("2fact",      "addition"):  PROMPT_ADDITION_2FACT,
    ("1fact",      "neutral"):   PROMPT_NEUTRAL_1FACT,
    ("1fact",      "addition"):  PROMPT_ADDITION_1FACT,
    ("letterpos",  "neutral"):   PROMPT_NEUTRAL_LETTERPOS,
    ("letterpos",  "chemistry"): PROMPT_CHEMISTRY_LETTERPOS,
    ("capitalpos", "neutral"):   PROMPT_NEUTRAL_LETTERPOS,   # same neutral framing
    ("capitalpos", "geography"): PROMPT_GEOGRAPHY_CAPITALPOS,
    ("varbind",    "neutral"):   PROMPT_NEUTRAL_VARBIND,
}

MODEL_IDS = {
    "haiku":  "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
}


def build_prompt(task: str, mode: str) -> str:
    return PROMPTS[(task, mode)]


# ----------------------------------------------------------------------------
# JSON parsing + formatting
# ----------------------------------------------------------------------------

JSON_RE_2FACT     = re.compile(r'\{[^{}]*"n1"[^{}]*"backups"[^{}]*\[[^\]]*\][^{}]*\}', re.DOTALL)
JSON_ANY_2FACT    = re.compile(r'\{[^{}]*"n1"[^{}]*\}', re.DOTALL)
JSON_RE_1FACT     = re.compile(r'\{[^{}]*"n"[^{}]*"backups"[^{}]*\[[^\]]*\][^{}]*\}', re.DOTALL)
JSON_ANY_1FACT    = re.compile(r'\{[^{}]*"n"\s*:[^{}]*\}', re.DOTALL)
JSON_RE_LETTERPOS = re.compile(r'\{[^{}]*"answer"[^{}]*"backups"[^{}]*\[[^\]]*\][^{}]*\}', re.DOTALL)
JSON_ANY_LETTERPOS = re.compile(r'\{[^{}]*"answer"\s*:[^{}]*\}', re.DOTALL)
JSON_RE_VARBIND   = re.compile(r'\{[^{}]*"numbers"\s*:\s*\[[^\]]*\][^{}]*\}', re.DOTALL)


def parse(text: str, task: str):
    if task == "2fact":
        m = JSON_RE_2FACT.findall(text) or JSON_ANY_2FACT.findall(text)
    elif task == "1fact":
        m = JSON_RE_1FACT.findall(text) or JSON_ANY_1FACT.findall(text)
    elif task == "letterpos":
        m = JSON_RE_LETTERPOS.findall(text) or JSON_ANY_LETTERPOS.findall(text)
    elif task == "varbind":
        m = JSON_RE_VARBIND.findall(text)
    else:
        raise ValueError(f"unknown task {task!r}")
    if not m:
        return None
    try:
        return json.loads(m[-1])
    except Exception:
        return None


def format_tokens(top):
    return "\n".join(
        f"{i:2d}. {t['prob']:.3f}  {t['str']!r}" for i, t in enumerate(top, 1)
    )


def derangement(n: int, seed: int) -> np.ndarray:
    """Return a permutation of [0..n) with no fixed points.

    Copied verbatim from scripts/release/run_shuffled_control.py so the shuffled
    control here uses the IDENTICAL procedure (and same seed convention) as the
    paper's shuffled-tokens control.
    """
    rng = np.random.default_rng(seed)
    while True:
        perm = rng.permutation(n)
        if not np.any(perm == np.arange(n)):
            return perm
        # Resolve fixed points by swapping each with the next index
        fixed = np.where(perm == np.arange(n))[0]
        for i in fixed:
            j = (i + 1) % n
            perm[i], perm[j] = perm[j], perm[i]
        if not np.any(perm == np.arange(n)):
            return perm


def apply_shuffle(samples, seed):
    """Shuffled-tokens control: give each example a DIFFERENT example's top_tokens
    (derangement partner) while keeping its own truth fields, so the judge is
    scored against the original truth using a donor's tokens. Same procedure as
    the paper's run_shuffled_control.py."""
    n = len(samples)
    perm = derangement(n, seed)
    return [{**samples[i],
             "top_tokens": samples[int(perm[i])]["top_tokens"],
             "partner_idx": samples[int(perm[i])].get("idx")}
            for i in range(n)]


# ----------------------------------------------------------------------------
# Per-run driver
# ----------------------------------------------------------------------------

def run_one_2fact(client, label, model_id, prompt, samples, max_workers):
    n = len(samples)
    print(f"\n=== {label} (n={n}) ===", flush=True)
    results = [None] * n

    def worker(i):
        user = f"Top tokens (rank. score  'string'):\n{format_tokens(samples[i]['top_tokens'])}"
        try:
            resp = client.messages.create(
                model=model_id, max_tokens=1000,
                system=prompt,
                messages=[{"role": "user", "content": user}],
            )
            return i, resp.content[0].text, None
        except Exception as e:
            return i, None, str(e)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(worker, i) for i in range(n)]
        for f in tqdm(as_completed(futures), total=n, desc=label):
            i, text, err = f.result()
            results[i] = {"text": text, "error": err}

    scored = []
    for i, s in enumerate(samples):
        parsed = parse(results[i]["text"] or "", "2fact") or {}
        n1, n2 = parsed.get("n1"), parsed.get("n2")
        backups = parsed.get("backups", []) or []
        A1 = s["fact_value_1"]
        A2 = s["fact_value_2"]
        primary = {n1, n2}
        top_k = {}
        for K in [2, 3, 5, 10]:
            cand = [n1, n2] + backups[:K - 2]
            top_k[K] = {"a1": A1 in cand, "a2": A2 in cand,
                        "both": (A1 in cand) and (A2 in cand)}
        scored.append({
            "idx": s.get("idx"), "A1": A1, "A2": A2,
            "pred": [n1, n2], "backups": backups,
            "got_a1": A1 in primary, "got_a2": A2 in primary,
            "both": (A1 in primary) and (A2 in primary),
            "top_k": top_k, "raw": results[i]["text"],
        })

    summary = {}
    for K in [2, 3, 5, 10]:
        a1 = sum(r["top_k"][K]["a1"] for r in scored)
        a2 = sum(r["top_k"][K]["a2"] for r in scored)
        b  = sum(r["top_k"][K]["both"] for r in scored)
        summary[K] = (a1, a2, b)
        print(f"  top-{K:>2}: A1={a1}/{n}, A2={a2}/{n}, Both={b}/{n}", flush=True)
    return scored, summary


def run_one_1fact(client, label, model_id, prompt, samples, max_workers):
    n = len(samples)
    print(f"\n=== {label} (n={n}) ===", flush=True)
    results = [None] * n

    def worker(i):
        user = f"Top tokens (rank. score  'string'):\n{format_tokens(samples[i]['top_tokens'])}"
        try:
            resp = client.messages.create(
                model=model_id, max_tokens=1000,
                system=prompt,
                messages=[{"role": "user", "content": user}],
            )
            return i, resp.content[0].text, None
        except Exception as e:
            return i, None, str(e)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(worker, i) for i in range(n)]
        for f in tqdm(as_completed(futures), total=n, desc=label):
            i, text, err = f.result()
            results[i] = {"text": text, "error": err}

    scored = []
    for i, s in enumerate(samples):
        parsed = parse(results[i]["text"] or "", "1fact") or {}
        pred = parsed.get("n")
        backups = parsed.get("backups", []) or []
        A = s["fact_value"]
        top_k = {}
        for K in [1, 2, 3, 5, 10]:
            cand = [pred] + backups[:K - 1]
            top_k[K] = A in cand
        scored.append({
            "idx": s.get("idx"), "A": A,
            "pred": pred, "backups": backups,
            "got": A == pred,
            "top_k": top_k, "raw": results[i]["text"],
        })

    summary = {}
    for K in [1, 2, 3, 5, 10]:
        hits = sum(r["top_k"][K] for r in scored)
        summary[K] = hits
        print(f"  top-{K:>2}: A={hits}/{n}", flush=True)
    return scored, summary


def _matches_str(pred, truth):
    """Case-insensitive substring match (truth inside prediction string)."""
    return (isinstance(pred, str) and isinstance(truth, str)
            and truth.lower() in pred.lower())


def run_one_letterpos(client, label, model_id, prompt, samples, max_workers):
    n = len(samples)
    print(f"\n=== {label} (n={n}) ===", flush=True)
    results = [None] * n

    def worker(i):
        user = f"Top tokens (rank. score  'string'):\n{format_tokens(samples[i]['top_tokens'])}"
        try:
            resp = client.messages.create(
                model=model_id, max_tokens=1000,
                system=prompt,
                messages=[{"role": "user", "content": user}],
            )
            return i, resp.content[0].text, None
        except Exception as e:
            return i, None, str(e)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(worker, i) for i in range(n)]
        for f in tqdm(as_completed(futures), total=n, desc=label):
            i, text, err = f.result()
            results[i] = {"text": text, "error": err}

    scored = []
    for i, s in enumerate(samples):
        parsed = parse(results[i]["text"] or "", "letterpos") or {}
        pred = parsed.get("answer")
        backups = parsed.get("backups", []) or []
        truth = s["element"]
        top_k = {}
        for K in [1, 2, 3, 5, 10]:
            cand = [pred] + backups[: K - 1]
            top_k[K] = any(_matches_str(c, truth) for c in cand)
        scored.append({
            "idx": s.get("idx"), "element": truth,
            "atomic_number": s.get("atomic_number"),
            "pred": pred, "backups": backups,
            "got": _matches_str(pred, truth),
            "top_k": top_k, "raw": results[i]["text"],
        })

    summary = {}
    for K in [1, 2, 3, 5, 10]:
        hits = sum(r["top_k"][K] for r in scored)
        summary[K] = hits
        print(f"  top-{K:>2}: element={hits}/{n}", flush=True)
    return scored, summary


def run_one_capitalpos(client, label, model_id, prompt, samples, max_workers):
    n = len(samples)
    print(f"\n=== {label} (n={n}) ===", flush=True)
    results = [None] * n

    def worker(i):
        user = f"Top tokens (rank. score  'string'):\n{format_tokens(samples[i]['top_tokens'])}"
        try:
            resp = client.messages.create(
                model=model_id, max_tokens=1000,
                system=prompt,
                messages=[{"role": "user", "content": user}],
            )
            return i, resp.content[0].text, None
        except Exception as e:
            return i, None, str(e)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(worker, i) for i in range(n)]
        for f in tqdm(as_completed(futures), total=n, desc=label):
            i, text, err = f.result()
            results[i] = {"text": text, "error": err}

    scored = []
    for i, s in enumerate(samples):
        parsed = parse(results[i]["text"] or "", "letterpos") or {}
        pred = parsed.get("answer")
        backups = parsed.get("backups", []) or []
        truth = s.get("intermediate")
        top_k = {}
        for K in [1, 2, 3, 5, 10]:
            cand = [pred] + backups[: K - 1]
            top_k[K] = any(_matches_str(c, truth) for c in cand)
        scored.append({
            "idx": s.get("idx"), "intermediate": truth,
            "answer": s.get("answer"),
            "pred": pred, "backups": backups,
            "got": _matches_str(pred, truth),
            "top_k": top_k, "raw": results[i]["text"],
        })

    summary = {}
    for K in [1, 2, 3, 5, 10]:
        hits = sum(r["top_k"][K] for r in scored)
        summary[K] = hits
        print(f"  top-{K:>2}: city={hits}/{n}", flush=True)
    return scored, summary


def _to_int(x):
    try:
        return int(round(float(x)))
    except (ValueError, TypeError):
        return None


def build_base_map(dataset_path):
    """idx -> base literal value (the visible operand the queried chain reads),
    parsed from the source dataset. chain_len=1 only (else None)."""
    with open(dataset_path) as f:
        ds = json.load(f)
    out = {}
    for e in ds["examples"]:
        defs = {n: v for n, v in e["definitions"]}
        expr = defs.get(e["queried_term"])
        b = None
        if isinstance(expr, str):
            mm = re.search(r"the number for (\w+)", expr)
            if mm:
                rv = defs.get(mm.group(1))
                b = rv if isinstance(rv, int) else None
        out[e["idx"]] = b
    return out


def run_one_varbind(client, label, model_id, prompt, samples, max_workers, base_by_idx):
    """Decode the varbind hidden intermediates. The judge returns a ranked list
    of numbers; we score top-k presence of FOUR targets per example:
        V   = queried_value          (hidden — primary target)
        cV  = coefficient * V        (hidden — second target)
        B   = base literal           (visible input — expected distractor)
        ans = answer                 (visible output — expected distractor)
    V and cV are the result; B and ans are the salience baseline (we expect them
    to appear). The judge never sees the prompt's coefficients/operations, so it
    cannot back-solve V from B and ans — recovery must come from the tokens."""
    n = len(samples)
    print(f"\n=== {label} (n={n}) ===", flush=True)
    results = [None] * n

    def worker(i):
        user = f"Top tokens (rank. score  'string'):\n{format_tokens(samples[i]['top_tokens'])}"
        try:
            resp = client.messages.create(
                model=model_id, max_tokens=1000,
                system=prompt,
                messages=[{"role": "user", "content": user}],
            )
            return i, resp.content[0].text, None
        except Exception as e:
            return i, None, str(e)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(worker, i) for i in range(n)]
        for f in tqdm(as_completed(futures), total=n, desc=label):
            i, text, err = f.result()
            results[i] = {"text": text, "error": err}

    TARGETS = ["V", "cV", "B", "ans"]
    KS = [1, 2, 3, 5, 10]
    scored = []
    for i, s in enumerate(samples):
        parsed = parse(results[i]["text"] or "", "varbind") or {}
        nums = [x for x in (_to_int(v) for v in (parsed.get("numbers") or [])) if x is not None]

        V = _to_int(s.get("intermediate"))
        coef = _to_int(s.get("coefficient"))
        cV = coef * V if (coef is not None and coef > 0 and V is not None) else None
        ans = _to_int(s.get("answer"))
        B = _to_int(base_by_idx.get(s.get("idx")))
        truth = {"V": V, "cV": cV, "B": B, "ans": ans}

        top_k = {K: {t: (truth[t] is not None and truth[t] in nums[:K]) for t in TARGETS}
                 for K in KS}
        scored.append({
            "idx": s.get("idx"), "V": V, "cV": cV, "B": B, "answer": ans,
            "coefficient": coef, "numbers": nums,
            "top_k": top_k, "raw": results[i]["text"],
        })

    summary = {}
    for K in KS:
        summary[K] = {t: sum(r["top_k"][K][t] for r in scored) for t in TARGETS}
        print(f"  top-{K:>2}: V={summary[K]['V']}/{n}  cV={summary[K]['cV']}/{n}  "
              f"| B={summary[K]['B']}/{n}  ans={summary[K]['ans']}/{n}", flush=True)
    return scored, summary


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aggregated-dir", type=Path, required=True,
                    help="Directory containing aggregated_<cond>.json files")
    ap.add_argument("--outdir", type=Path, default=None,
                    help="Output dir for llm_decode_* files (default: same as --aggregated-dir)")
    ap.add_argument("--conditions", nargs="+", required=True,
                    help="Condition names")
    ap.add_argument("--task", choices=["2fact", "1fact", "letterpos", "capitalpos", "varbind"], default="2fact",
                    help="2fact: decode two operands. 1fact: decode one retrieved value. "
                         "letterpos: decode one element name (from an atomic-number → "
                         "element → Nth letter task). capitalpos: decode one capital city "
                         "name (from a country → capital → last letter task). varbind: decode "
                         "the hidden serial intermediates V and c·V (chained variable binding).")
    ap.add_argument("--prompt", choices=["neutral", "addition", "chemistry", "geography"], default="neutral",
                    help="Prompt framing. Valid pairs: (2fact, 1fact) × (neutral, addition), "
                         "(letterpos) × (neutral, chemistry), (capitalpos) × (neutral, geography), "
                         "(varbind) × (neutral).")
    ap.add_argument("--varbind-dataset", type=Path,
                    default=Path("data/chained_var_binding_dataset.json"),
                    help="Source dataset for varbind — joined by idx to recover the base "
                         "value B for scoring.")
    ap.add_argument("--shuffle", action="store_true",
                    help="Shuffled-tokens control: replace each example's tokens with a "
                         "derangement partner's and score against the ORIGINAL truth. Same "
                         "procedure as the paper's run_shuffled_control.py. Adds _shuffled "
                         "to output filenames.")
    ap.add_argument("--shuffle-seed", type=int, default=0,
                    help="Seed for the derangement (default 0, matching the paper control).")
    ap.add_argument("--models", nargs="+", default=["haiku", "sonnet"],
                    choices=list(MODEL_IDS))
    ap.add_argument("--threads", type=int, default=8,
                    help="ThreadPoolExecutor max_workers per run")
    ap.add_argument("--key-path", type=Path,
                    default=Path("/workspace/keys/anthropic_api_key"),
                    help="File with Anthropic API key (ignored if $ANTHROPIC_API_KEY already set)")
    ap.add_argument("--print-prompts", action="store_true",
                    help="Print the resolved prompt and exit")
    args = ap.parse_args()

    outdir = args.outdir or args.aggregated_dir
    outdir.mkdir(parents=True, exist_ok=True)
    prompt = build_prompt(args.task, args.prompt)

    if args.print_prompts:
        print(f"{'=' * 72}\n{args.task} / {args.prompt}\n{'=' * 72}")
        print(prompt)
        return

    if "ANTHROPIC_API_KEY" not in os.environ:
        os.environ["ANTHROPIC_API_KEY"] = args.key_path.read_text().strip()
    import anthropic
    client = anthropic.Anthropic()

    runner = {"2fact": run_one_2fact,
              "1fact": run_one_1fact,
              "letterpos": run_one_letterpos,
              "capitalpos": run_one_capitalpos}.get(args.task)
    base_by_idx = build_base_map(args.varbind_dataset) if args.task == "varbind" else None

    all_summaries = {}
    for cond in args.conditions:
        agg_path = args.aggregated_dir / f"aggregated_{cond}.json"
        if not agg_path.exists():
            print(f"MISSING: {agg_path}; skipping {cond}")
            continue
        with open(agg_path) as f:
            samples = json.load(f)
        if args.shuffle:
            samples = apply_shuffle(samples, args.shuffle_seed)
        print(f"\n{'#' * 72}\n# {cond}  (n={len(samples)}, task={args.task}, "
              f"prompt={args.prompt}{', SHUFFLED' if args.shuffle else ''})\n{'#' * 72}", flush=True)
        for mlabel in args.models:
            model_id = MODEL_IDS[mlabel]
            if args.task == "varbind":
                scored, summary = run_one_varbind(client, f"{cond}/{mlabel}", model_id,
                                                  prompt, samples, args.threads, base_by_idx)
            else:
                scored, summary = runner(client, f"{cond}/{mlabel}", model_id,
                                         prompt, samples, args.threads)
            shuf = "_shuffled" if args.shuffle else ""
            out_path = outdir / f"llm_decode_{cond}_{mlabel}_{args.prompt}_{args.task}{shuf}.json"
            with open(out_path, "w") as f:
                json.dump(scored, f, indent=2, ensure_ascii=False)
            print(f"  Saved → {out_path}")
            all_summaries[(cond, mlabel)] = (len(samples), summary)

    print(f"\n\n=== SUMMARY ({args.task} / {args.prompt}) ===")
    if args.task == "2fact":
        print(f"{'Condition':<14} {'Model':<7} {'N':>4}  {'prim-Both':>10} {'top-3-Both':>10} "
              f"{'top-5-Both':>10} {'top-10-Both':>12}")
        for (cond, mlabel), (n, summary) in all_summaries.items():
            prim, t3, t5, t10 = summary[2][2], summary[3][2], summary[5][2], summary[10][2]
            print(f"  {cond:<12} {mlabel:<7} {n:>4}   {prim:>4}/{n}     "
                  f"{t3:>4}/{n}     {t5:>4}/{n}     {t10:>4}/{n}")
    elif args.task == "varbind":
        print("targets V, c·V = hidden (result); B, ans = visible (baseline). values are % top-k.")
        print(f"{'Condition':<12} {'Model':<7} {'N':>4}  "
              f"{'V@1':>5} {'V@5':>5} {'V@10':>5}   {'cV@1':>5} {'cV@5':>5} {'cV@10':>5}   "
              f"{'B@10':>5} {'ans@10':>6}")
        for (cond, mlabel), (n, summary) in all_summaries.items():
            def pc(K, t):
                return 100.0 * summary[K][t] / n if n else 0.0
            print(f"  {cond:<10} {mlabel:<7} {n:>4}   "
                  f"{pc(1,'V'):5.1f} {pc(5,'V'):5.1f} {pc(10,'V'):5.1f}   "
                  f"{pc(1,'cV'):5.1f} {pc(5,'cV'):5.1f} {pc(10,'cV'):5.1f}   "
                  f"{pc(10,'B'):5.1f} {pc(10,'ans'):6.1f}")
    else:
        target = {"1fact": "A", "letterpos": "element", "capitalpos": "city"}.get(args.task, "element")
        print(f"{'Condition':<14} {'Model':<7} {'N':>4}  "
              f"{'top-1':>8} {'top-2':>8} {'top-3':>8} {'top-5':>8} {'top-10':>8}  ({target})")
        for (cond, mlabel), (n, summary) in all_summaries.items():
            print(f"  {cond:<12} {mlabel:<7} {n:>4}   "
                  f"{summary[1]:>4}/{n}  {summary[2]:>4}/{n}  {summary[3]:>4}/{n}  "
                  f"{summary[5]:>4}/{n}  {summary[10]:>4}/{n}")

    summary_path = outdir / f"llm_decode_summary_{args.prompt}_{args.task}{'_shuffled' if args.shuffle else ''}.json"
    with open(summary_path, "w") as f:
        json.dump({f"{c}_{m}": {"n": n, "summary": s}
                   for (c, m), (n, s) in all_summaries.items()}, f, indent=2)
    print(f"\nSaved summary → {summary_path}")


if __name__ == "__main__":
    main()
