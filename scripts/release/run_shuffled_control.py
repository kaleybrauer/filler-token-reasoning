"""
Shuffled-tokens control: each example's top-50 tokens are replaced with a
DIFFERENT example's top-50 from the same (model, task, condition); the judge
is then scored against the ORIGINAL example's ground truth. If accuracy
collapses, the judge is reading example-specific signal; if it stays high,
the judge is mostly using a task-level prior (general number / element /
city distribution).

Reads:  release/top_tokens/<model>_<task>_dots_10.json
Writes: release/judge_outputs_shuffled/<model>_<task>_dots_10_<judge>_<prompt>.json

Run:    python scripts/release/run_shuffled_control.py
"""
from __future__ import annotations

import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "decode"))

from llm_decode_batch import (   # noqa: E402
    PROMPTS, MODEL_IDS, format_tokens, parse, _matches_str
)

import anthropic   # noqa: E402

API_KEY_PATH = Path("/workspace/keys/anthropic_api_key")
client = anthropic.Anthropic(api_key=API_KEY_PATH.read_text().strip())

OUT_DIR = REPO / "release" / "judge_outputs_shuffled"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 0
N_WORKERS = 8

# (model, task, prompts_to_run)
COMBOS = [
    ("deepseek_v3", "1fact",      ["neutral"]),
    ("deepseek_v3", "2fact",      ["neutral"]),
    ("deepseek_v3", "letterpos",  ["neutral", "chemistry"]),
    ("deepseek_v3", "capitalpos", ["neutral", "geography"]),
    ("kimi_k2",     "2fact",      ["neutral"]),
    ("kimi_k2",     "letterpos",  ["neutral", "chemistry"]),
    ("kimi_k2",     "capitalpos", ["neutral", "geography"]),
]
JUDGES = ["haiku", "sonnet"]
COND = "dots_10"


def derangement(n: int, seed: int) -> np.ndarray:
    """Return a permutation of [0..n) with no fixed points."""
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


def truth_fields(task: str, rec: dict) -> dict:
    """Return the subset of fields needed for scoring + display."""
    if task == "1fact":
        return {"fact_value": rec.get("fact_value"),
                "answer":     rec.get("answer")}
    if task == "2fact":
        return {"fact_value_1": rec.get("fact_value_1"),
                "fact_value_2": rec.get("fact_value_2"),
                "answer":       rec.get("answer")}
    if task == "letterpos":
        return {"element":       rec.get("element"),
                "atomic_number": rec.get("atomic_number")}
    if task == "capitalpos":
        return {"intermediate":  rec.get("intermediate")}
    raise ValueError(task)


def score(task: str, parsed: dict, truth: dict) -> tuple:
    """Return (pred, backups, top_k_dict, got)."""
    if task in ("1fact", "2fact"):
        return _score_numeric(task, parsed, truth)
    return _score_string(parsed, truth)


def _score_numeric(task: str, parsed: dict, truth: dict):
    if task == "1fact":
        pred = parsed.get("n")
        backups = parsed.get("backups", []) or []
        gt = truth["fact_value"]
        top_k = {}
        for K in (1, 2, 3, 5, 10):
            cand = [pred] + backups[: K - 1]
            top_k[str(K)] = any(_n_eq(c, gt) for c in cand)
        return pred, backups, top_k, _n_eq(pred, gt)
    # 2fact
    n1 = parsed.get("n1"); n2 = parsed.get("n2")
    backups = parsed.get("backups", []) or []
    a1, a2 = truth["fact_value_1"], truth["fact_value_2"]
    pred = [n1, n2]
    top_k = {}
    for K in (1, 2, 3, 5, 10):
        # All candidates the judge offered up to rank K (n1, n2, then backups)
        cand = [n1, n2] + backups[: max(0, K - 2)]
        cand_set = {c for c in cand if c is not None}
        a1_hit = _n_eq_set(a1, cand_set)
        a2_hit = _n_eq_set(a2, cand_set)
        top_k[str(K)] = {"a1": a1_hit, "a2": a2_hit, "both": a1_hit and a2_hit}
    got_a1 = a1 is not None and (n1 == a1 or n2 == a1)
    got_a2 = a2 is not None and (n1 == a2 or n2 == a2)
    return pred, backups, top_k, {"got_a1": got_a1, "got_a2": got_a2,
                                  "both": got_a1 and got_a2}


def _n_eq(c, gt):
    if gt is None or c is None:
        return False
    try:
        return int(c) == int(gt)
    except (TypeError, ValueError):
        return False


def _n_eq_set(gt, cand_set):
    if gt is None:
        return False
    for c in cand_set:
        if _n_eq(c, gt):
            return True
    return False


def _score_string(parsed: dict, truth: dict):
    pred = parsed.get("answer")
    backups = parsed.get("backups", []) or []
    gt = truth.get("element") or truth.get("intermediate")
    top_k = {}
    for K in (1, 2, 3, 5, 10):
        cand = [pred] + backups[: K - 1]
        top_k[str(K)] = any(_matches_str(c, gt) for c in cand)
    return pred, backups, top_k, _matches_str(pred, gt)


def task_to_prompt_task(task: str) -> str:
    """Map our short task names to the (task) key in PROMPTS."""
    return {"1fact": "1fact", "2fact": "2fact",
            "letterpos": "letterpos", "capitalpos": "capitalpos"}[task]


def task_to_parse_task(task: str) -> str:
    """Map task to the key llm_decode_batch.parse() understands.
    capitalpos uses the same string-answer parser as letterpos."""
    if task == "capitalpos":
        return "letterpos"
    return task


def call_judge(top_tokens, system_prompt, model_id):
    user = f"Top tokens (rank. score  'string'):\n{format_tokens(top_tokens)}"
    resp = client.messages.create(
        model=model_id, max_tokens=1000,
        system=system_prompt,
        messages=[{"role": "user", "content": user}],
    )
    if not resp.content:
        return None  # refusal
    return resp.content[0].text


def run_one(model: str, task: str, prompt_kind: str, judge: str):
    out_path = OUT_DIR / f"{model}_{task}_{COND}_{judge}_{prompt_kind}.json"
    if out_path.exists():
        print(f"  [skip] exists: {out_path.name}")
        return

    src = json.load(open(REPO / "release" / "top_tokens" / f"{model}_{task}_{COND}.json"))
    n = len(src)
    perm = derangement(n, SEED)

    system_prompt = PROMPTS[(task_to_prompt_task(task), prompt_kind)]
    model_id = MODEL_IDS[judge]

    label = f"{model} {task} {prompt_kind} {judge}"
    print(f"\n=== {label}: n={n} ===", flush=True)

    out_records = [None] * n

    def worker(i):
        partner = int(perm[i])
        donor_tokens = src[partner]["top_tokens"]
        try:
            text = call_judge(donor_tokens, system_prompt, model_id)
        except Exception as e:
            return i, None, str(e)
        return i, text, None

    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = [ex.submit(worker, i) for i in range(n)]
        for f in tqdm(as_completed(futures), total=n, desc=label, leave=False):
            i, text, err = f.result()
            partner = int(perm[i])
            truth = truth_fields(task, src[i])
            parsed = parse(text or "", task_to_parse_task(task)) or {}
            pred, backups, top_k, got = score(task, parsed, truth)

            rec = {"idx": src[i].get("idx"),
                   "partner_idx": src[partner].get("idx"),
                   "raw": text,
                   "pred": pred, "backups": backups,
                   "top_k": top_k}
            if isinstance(got, dict):  # 2fact
                rec.update(got)
            else:
                rec["got"] = got
            rec.update(truth)
            if err:
                rec["error"] = err
            out_records[i] = rec

    json.dump(out_records, open(out_path, "w"), indent=2)
    print(f"  saved {out_path.name}")


def main():
    for model, task, prompts in COMBOS:
        for prompt_kind in prompts:
            for judge in JUDGES:
                run_one(model, task, prompt_kind, judge)
    print("\nAll runs complete.")


if __name__ == "__main__":
    main()
