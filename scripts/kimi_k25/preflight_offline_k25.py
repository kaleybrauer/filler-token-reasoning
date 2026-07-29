#!/usr/bin/env python3
"""Offline preflight for the Kimi K2.5 varbind run — gates 1, 3 and the token checks.

Everything here runs on CPU with the tokenizer alone, so it must pass before the
model is loaded. The live gates (2, 4, capture convention, 5) need the GPU and run
inside extract_hidden_states_vllm.py --preflight.

Usage:
    /root/.venvs/k25/bin/python scripts/kimi_k25/preflight_offline_k25.py \
        --model-path /workspace/models/kimi-k2.5 \
        --dataset data/chained_var_binding_easy_dataset.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from data.generate_1hop_dataset import make_filler_tokens  # noqa: E402
from extract.extract_hidden_states import CONDITIONS, find_filler_boundaries  # noqa: E402
from kimi_k25.prompt_k25 import (  # noqa: E402
    build_prompt_k25,
    check_prompt_tail,
    check_thinking_suppressed,
    check_v3_scaffold,
)

RUN_CONDITIONS = ["baseline", "dots_5", "dots_10", "dots_25", "dots_50",
                  "counting_5", "counting_10", "counting_25"]


def expected_filler_len(cond: str, k: int) -> int:
    """In-context filler span on the Kimi K2.5 tokenizer.

    dots_k   -> k     tokens: the space after "Filler:" merges into the first dot
                      ('Ġ.'), and the last dot merges with the following newlines
                      ('Ġ.ĊĊ'), so there are exactly k tokens.
    counting_k -> 2k  tokens: 'Ġ' + digit, repeated. NOT the 2k-1 recorded in
                      project_filler_token_counts / RUNBOOK gate 3 — that figure is
                      the *standalone* tokenization of "1 2 3 … k" (['1','Ġ','2',…]
                      = 2k-1). In context the leading space after "Filler:" cannot
                      merge into the digit, because space-prefixed numbers are not
                      tokens in Kimi's vocab (RUNBOOK §0), so it costs one extra
                      token. Verified token-by-token, see logs/preflight_offline.log.

    Side effect, in our favour: counting_25 is 50 tokens here, exactly matching
    dots_50, so the length-matched pair is cleaner than on V3 (49 vs 50).
    """
    return k if cond.startswith("dots") else 2 * k


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=Path, default=Path("/workspace/models/kimi-k2.5"))
    ap.add_argument("--dataset", type=Path,
                    default=Path("data/chained_var_binding_easy_dataset.json"))
    ap.add_argument("--n-examples", type=int, default=3,
                    help="Examples per condition to check")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    print(f"Loading tokenizer from {args.model_path} …")
    tok = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
    print(f"  {type(tok).__name__}, vocab {len(tok)}")

    data = json.loads(args.dataset.read_text())
    problems = data["examples"]
    few_shot = (data.get("few_shot_examples") or data.get("few_shot_facts"))[:5]
    print(f"  dataset: {len(problems)} examples, {len(few_shot)} few-shot in use\n")

    failures: list[str] = []
    lengths: dict[str, list[int]] = {}

    # ---- Gate 1 + gate 3, per condition -------------------------------------
    for cond in RUN_CONDITIONS:
        k, filler_type = CONDITIONS[cond]
        lengths[cond] = []
        for i in range(args.n_examples):
            rng = random.Random(i)
            text, ids = build_prompt_k25(tok, few_shot, problems[i], filler_type, k, rng=rng)
            lengths[cond].append(len(ids))

            for name, (ok, msg) in {
                "gate1/think": check_thinking_suppressed(ids),
                "gate1/tail": check_prompt_tail(text),
                "gate1/scaffold": check_v3_scaffold(text, k),
            }.items():
                if not ok:
                    failures.append(f"{cond}[{i}] {name}: {msg}")

            # Gate 3: the located filler region must decode back to exactly the
            # filler string the prompt builder emitted. This is stronger than a
            # token count and does not depend on the tokenizer's merge behaviour.
            if k > 0:
                qe, fs, fe = find_filler_boundaries(tok, torch.tensor([ids]), k)
                span = fe - fs + 1
                want = expected_filler_len(cond, k)
                if span != want:
                    failures.append(
                        f"{cond}[{i}] gate3: filler span {span} != expected {want} "
                        f"(question_end={qe}, filler={fs}..{fe})")
                expected_filler = make_filler_tokens(filler_type, k, rng=random.Random(i))
                got_filler = tok.decode(ids[fs:fe + 1]).strip()
                if got_filler != expected_filler.strip():
                    failures.append(
                        f"{cond}[{i}] gate3: filler region decodes to {got_filler[:60]!r}, "
                        f"expected {expected_filler[:60]!r}")
                n_items = len(got_filler.split())
                if n_items != k:
                    failures.append(
                        f"{cond}[{i}] gate3: filler region holds {n_items} items, expected {k}")
                if not (0 <= qe < fs <= fe < len(ids)):
                    failures.append(
                        f"{cond}[{i}] gate3: implausible ordering "
                        f"question_end={qe} filler={fs}..{fe} seq_len={len(ids)}")
            if i == 0:
                if k > 0:
                    qe, fs, fe = find_filler_boundaries(tok, torch.tensor([ids]), k)
                    print(f"{cond:12s} len={len(ids):4d}  question_end={qe:4d} "
                          f"filler={fs}..{fe} (span {fe - fs + 1}, want "
                          f"{expected_filler_len(cond, k)})")
                else:
                    print(f"{cond:12s} len={len(ids):4d}  (no filler)")

    # ---- BOS / double-special check -----------------------------------------
    print()
    _, ids0 = build_prompt_k25(tok, few_shot, problems[0], "dots", 10, rng=random.Random(0))
    bos = getattr(tok, "bos_token_id", None)
    if bos is not None and ids0[0] == bos:
        failures.append(f"tokenizer prepended BOS ({bos}) — every position would shift by 1")
    print(f"first 6 token ids: {ids0[:6]}  (bos_token_id={bos})")
    print(f"first tokens decoded: {tok.convert_ids_to_tokens(ids0[:4])}")

    # ---- Decode targets are single tokens -----------------------------------
    targets: set[int] = set()
    for p in problems:
        targets.add(p["answer"])
        targets.add(p["queried_value"])
        targets.add(p["coefficient"] * p["queried_value"])
    multi = [t for t in sorted(targets) if len(tok.encode(str(t), add_special_tokens=False)) != 1]
    print(f"\ndecode targets: {len(targets)} distinct, {len(multi)} multi-token")
    if multi:
        print(f"  multi-token: {multi[:20]}")

    # ints 0-999 single token (RUNBOOK §0 assertion, kept live)
    bad_ints = [n for n in range(1000)
                if len(tok.encode(str(n), add_special_tokens=False)) != 1]
    if bad_ints:
        failures.append(f"ints not single-token: {bad_ints[:10]} ({len(bad_ints)} total)")
    print(f"ints 0-999 single-token: {len(bad_ints) == 0}")

    # ---- Prompt lengths ------------------------------------------------------
    print("\nprompt lengths (min/max over checked examples):")
    overall_max = 0
    for cond in RUN_CONDITIONS:
        lo, hi = min(lengths[cond]), max(lengths[cond])
        overall_max = max(overall_max, hi)
        print(f"  {cond:12s} {lo:4d} .. {hi:4d}")
    print(f"  => max_model_len=2048 headroom: {2048 - overall_max} tokens")
    if overall_max > 2048:
        failures.append(f"prompt length {overall_max} exceeds max_model_len=2048")

    # ---- Verdict -------------------------------------------------------------
    print("\n" + "=" * 70)
    if failures:
        print(f"OFFLINE PREFLIGHT FAILED — {len(failures)} problem(s):")
        for f in failures:
            print(f"  FAIL {f}")
        return 1
    print("OFFLINE PREFLIGHT PASSED (gates 1, 3, token checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
