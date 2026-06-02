#!/usr/bin/env python3
"""Generate a chained variable-binding dataset for probing experiments.

Unlike the addition tasks (which test *retrieval* of a stored fact), this task
tests *transient computation*: each problem assigns fake CVC nonsense terms
either a literal value or an expression derived from an earlier term, and the
queried term sits at the end of a reference chain. The model must compute a
hidden intermediate (`queried_value`) that is never written in the prompt.

Example (chain_len=1 — one hidden indirection):

    zab = 45
    piy = 43
    xoc = twice the number for zab plus 14      # hidden intermediate: 104
    lej = three times the number for zab plus 6
    kox = 45
    Question: What is twice the number for xoc minus 35?   -> 2*104 - 35 = 173

The task / algorithm is ported from the `multi_hop` repo's
generate_chained_var_binding_dataset.py, but the OUTPUT matches this repo's
dataset format: a single JSON with `few_shot_examples` + `examples`, plus the
build_*_varbind prompt helpers that extract_hidden_states.py and the eval
scripts import (mirroring generate_1hop_dataset.py / generate_2fact_dataset.py /
generate_letterpos_dataset.py). It plugs into the local-GPU extraction pipeline
via `--dataset-type varbind`.

Usage:
    python scripts/data/generate_varbind_dataset.py --preview
    python scripts/data/generate_varbind_dataset.py --num-examples 800
    # depth-2 negative control:
    python scripts/data/generate_varbind_dataset.py --chain-len 2 \
        --output data/chained_var_binding_c2_dataset.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List

# Script lives at scripts/data/; reuse the shared filler helpers from the 1-hop
# generator (same import style as generate_2fact_dataset.py / _letterpos).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.generate_1hop_dataset import make_filler_tokens, filler_description

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
OUTPUT_PATH = Path("data/chained_var_binding_dataset.json")

SEED = 42
NUM_FEW_SHOT = 8          # reserved few-shot pool (extraction/eval use the first 5)
NUM_EXAMPLES = 500        # scored examples (matches the ~500-example probing scale)

# Problem-shape defaults (chain_len=1 is the unsaturated "sweet spot"; chain_len=2
# is the floor and serves as a negative control — no recoverable computation).
NUM_TERMS = 5
CHAIN_LEN = 1
VAL_MIN, VAL_MAX = 10, 99
CONST_MIN, CONST_MAX = 1, 50
MAX_COEF = 3

CONSONANTS = "bcdfghjklmnpqrstvwxyz"
VOWELS = "aeiou"
COEF_WORDS = {2: "twice", 3: "three times", 4: "four times", 5: "five times", 6: "six times"}


# ---------------------------------------------------------------------------
# Fake-term + chain construction (ported from multi_hop)
# ---------------------------------------------------------------------------

def load_real_words() -> set:
    """Load a dictionary word list so generated CVC terms are guaranteed non-words."""
    words: set = set()
    for path in ("/usr/share/dict/words", "/usr/share/dict/web2"):
        if os.path.exists(path):
            with open(path, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    words.add(line.strip().lower())
            break
    return words


def make_fake_term(real_words: set, rng: random.Random, existing: set) -> str:
    """Return a fresh consonant-vowel-consonant nonsense term (not a real word)."""
    for _ in range(2000):
        term = rng.choice(CONSONANTS) + rng.choice(VOWELS) + rng.choice(CONSONANTS)
        if term in real_words or term in existing:
            continue
        return term
    raise RuntimeError("Could not generate a unique fake term")


def topo_shuffle(terms: List[Dict], rng: random.Random) -> List[Dict]:
    """Random topological order: a referenced term always precedes its dependent."""
    name_to_term = {t["name"]: t for t in terms}
    deps = {t["name"]: ({t["ref"]} if t["kind"] == "derived" else set()) for t in terms}
    emitted, emitted_set = [], set()
    while len(emitted) < len(terms):
        ready = [n for n in deps if n not in emitted_set and deps[n] <= emitted_set]
        choice = rng.choice(ready)
        emitted.append(choice)
        emitted_set.add(choice)
    return [name_to_term[n] for n in emitted]


def derive_value(coef: int, op: str, const: int, ref_val: int) -> int:
    return coef * ref_val + const if op == "plus" else coef * ref_val - const


def generate_problem(rng, real_words, num_terms, chain_len,
                     val_min, val_max, const_min, const_max, max_coef) -> Dict:
    names: set = set()
    terms: List[Dict] = []

    def add_literal():
        name = make_fake_term(real_words, rng, names); names.add(name)
        val = rng.randint(val_min, val_max)
        t = {"name": name, "kind": "literal", "value": val, "depth": 0}
        terms.append(t); return t

    def add_derived(ref_term):
        name = make_fake_term(real_words, rng, names); names.add(name)
        # choose coef/op/const guaranteeing a non-negative derived value
        for _ in range(50):
            coef = rng.randint(2, max_coef)
            op = rng.choice(["plus", "minus"])
            const = rng.randint(const_min, const_max)
            val = derive_value(coef, op, const, ref_term["value"])
            if val >= 0:
                break
        else:
            op, const = "plus", rng.randint(const_min, const_max)
            coef = rng.randint(2, max_coef)
            val = derive_value(coef, op, const, ref_term["value"])
        t = {"name": name, "kind": "derived", "value": val, "coef": coef,
             "op": op, "const": const, "ref": ref_term["name"],
             "depth": ref_term["depth"] + 1}
        terms.append(t); return t

    # Relevant chain: 1 base literal + chain_len derived links (deepest = queried)
    base = add_literal()
    prev = base
    for _ in range(chain_len):
        prev = add_derived(prev)
    queried = prev

    # Distractors (literals + derived-from-anything) to reach num_terms total
    while len(terms) < num_terms:
        if rng.random() < 0.4 or len(terms) == 0:
            add_literal()
        else:
            add_derived(rng.choice(terms))

    # Question always multiplies the queried term (coef >= 2 -> intermediate exists)
    for _ in range(50):
        q_coef = rng.randint(2, max_coef)
        q_op = rng.choice(["plus", "minus"])
        q_const = rng.randint(const_min, const_max)
        answer = derive_value(q_coef, q_op, q_const, queried["value"])
        if answer >= 0:
            break
    else:
        q_op, q_const = "plus", rng.randint(const_min, const_max)
        q_coef = rng.randint(2, max_coef)
        answer = derive_value(q_coef, q_op, q_const, queried["value"])

    # Render display order + definitions
    display = topo_shuffle(terms, rng)
    definitions = []
    for t in display:
        if t["kind"] == "literal":
            definitions.append([t["name"], t["value"]])
        else:
            expr = f"{COEF_WORDS[t['coef']]} the number for {t['ref']} {t['op']} {t['const']}"
            definitions.append([t["name"], expr])

    question = f"What is {COEF_WORDS[q_coef]} the number for {queried['name']} {q_op} {q_const}?"

    return {
        "type": "chained_var_binding",
        "definitions": definitions,
        "queried_term": queried["name"],
        "queried_value": queried["value"],   # hidden intermediate (never stated in prompt)
        "coefficient": q_coef,
        "operation": q_op,
        "constant": q_const,
        "num_terms": num_terms,
        "chain_len": chain_len,
        "question": question,
        "answer": answer,
    }


def generate_unique_problems(n_total, rng, real_words, num_terms, chain_len,
                             val_min, val_max, const_min, const_max, max_coef) -> List[Dict]:
    """Generate n_total problems, de-duped on (definitions, question)."""
    problems, seen = [], set()
    attempts, max_attempts = 0, n_total * 300
    while len(problems) < n_total and attempts < max_attempts:
        attempts += 1
        p = generate_problem(rng, real_words, num_terms, chain_len,
                             val_min, val_max, const_min, const_max, max_coef)
        key = (tuple(tuple(d) for d in p["definitions"]), p["question"])
        if key in seen:
            continue
        seen.add(key)
        problems.append(p)
    if len(problems) < n_total:
        print(f"Warning: only generated {len(problems)}/{n_total} unique problems")
    return problems


# ---------------------------------------------------------------------------
# Prompt construction (mirrors build_*_letterpos in generate_letterpos_dataset.py)
# ---------------------------------------------------------------------------

def render_definitions(definitions: list) -> str:
    """Render the variable block: one `name = value-or-expr` per line."""
    return "\n".join(f"{name} = {value}" for name, value in definitions)


def build_system_message_varbind(filler_type: str, k: int) -> str:
    """System message for the chained variable-binding task."""
    base = (
        "You will be given a list of variable definitions followed by a question. "
        "Each variable equals either a number or an expression that refers to an "
        "earlier variable (for example 'twice the number for X plus 3'). Resolve "
        "the references to work out the value the question asks for, then answer "
        "immediately with just the number, nothing else. No explanation, no words, "
        "no reasoning, just the number."
    )
    if k > 0:
        desc = filler_description(filler_type)
        base += (
            f" After the question, there will be {k} filler tokens "
            f"(a sequence of {desc}) to give you extra space to process "
            f"the problem before answering."
        )
    return base


def build_user_turn_varbind(definitions: list, question: str, filler_type: str, k: int,
                            rng: random.Random | None = None) -> str:
    """User turn: the variable block, the question, then the Filler/Answer scaffold.

    The trailing `\\n\\nFiller: {filler}\\n\\nAnswer:` structure is identical to the
    other tasks, so find_filler_boundaries() locates the filler region unchanged.
    """
    var_block = render_definitions(definitions)
    question_line = f"{var_block}\nQuestion: {question}"
    if k > 0:
        filler = make_filler_tokens(filler_type, k, rng=rng)
        return f"{question_line}\n\nFiller: {filler}\n\nAnswer:"
    return f"{question_line}\n\nAnswer:"


def build_prompt_messages_varbind(
    few_shot_items: list[dict],
    target_item: dict,
    filler_type: str,
    k: int,
    rng: random.Random | None = None,
) -> list[dict]:
    """Full chat prompt for a chained variable-binding problem (bare assistant)."""
    system_msg = build_system_message_varbind(filler_type, k)
    messages = [{"role": "system", "content": system_msg}]

    for fs in few_shot_items:
        user_content = build_user_turn_varbind(
            fs["definitions"], fs["question"], filler_type, k, rng=rng
        )
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": str(fs["answer"])})

    user_content = build_user_turn_varbind(
        target_item["definitions"], target_item["question"], filler_type, k, rng=rng
    )
    messages.append({"role": "user", "content": user_content})

    return messages


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate chained variable-binding dataset for probing experiments."
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--num-examples", type=int, default=NUM_EXAMPLES,
                        help="Number of scored examples (default: 500)")
    parser.add_argument("--num-few-shot", type=int, default=NUM_FEW_SHOT,
                        help="Reserved few-shot problems (extraction uses the first 5)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--num-terms", type=int, default=NUM_TERMS,
                        help="Total terms per problem (>= chain_len + 1; >= 5 recommended)")
    parser.add_argument("--chain-len", type=int, default=CHAIN_LEN,
                        help="Reference-chain depth. 1 = unsaturated sweet spot; "
                             "2 = floor / negative control.")
    parser.add_argument("--val-min", type=int, default=VAL_MIN)
    parser.add_argument("--val-max", type=int, default=VAL_MAX)
    parser.add_argument("--const-min", type=int, default=CONST_MIN)
    parser.add_argument("--const-max", type=int, default=CONST_MAX)
    parser.add_argument("--max-coef", type=int, default=MAX_COEF,
                        help=f"Max coefficient (must be a key in {sorted(COEF_WORDS)})")
    parser.add_argument("--preview", action="store_true",
                        help="Print 3 example prompts (baseline, dots_5, counting_5) "
                             "to verify format, then exit.")
    args = parser.parse_args()

    assert args.num_terms >= args.chain_len + 1, "num_terms must be >= chain_len + 1"
    assert args.max_coef in COEF_WORDS, f"max_coef must be one of {sorted(COEF_WORDS)}"

    real_words = load_real_words()
    print(f"Loaded {len(real_words)} real words for filtering.")
    print(f"num_terms={args.num_terms}, chain_len={args.chain_len}, max_coef={args.max_coef}")

    # Generate few-shot pool + examples as one de-duped stream, then split. Each
    # problem is self-contained (random nonsense terms), so there is no shared
    # fact pool to isolate — de-duping on (definitions, question) is sufficient.
    n_total = args.num_few_shot + args.num_examples
    rng = random.Random(args.seed)
    all_problems = generate_unique_problems(
        n_total, rng, real_words, args.num_terms, args.chain_len,
        args.val_min, args.val_max, args.const_min, args.const_max, args.max_coef,
    )

    few_shot_items = all_problems[:args.num_few_shot]
    examples = []
    for i, p in enumerate(all_problems[args.num_few_shot:]):
        examples.append({"idx": i, **p})

    print(f"Reserved {len(few_shot_items)} few-shot, generated {len(examples)} examples.")

    dataset = {
        "task": "chained_var_binding",
        "chain_len": args.chain_len,
        "num_terms": args.num_terms,
        "few_shot_examples": few_shot_items,
        "examples": examples,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"Saved dataset to {args.output}")

    if examples:
        answers = [e["answer"] for e in examples]
        inters = [e["queried_value"] for e in examples]
        print(f"Answer range: {min(answers)}-{max(answers)} "
              f"({len(set(answers))} unique); "
              f"intermediate range: {min(inters)}-{max(inters)} "
              f"({len(set(inters))} unique).")

    # ------------------------------------------------------------------
    # Preview (optional)
    # ------------------------------------------------------------------
    if args.preview:
        print("\n" + "=" * 72)
        print("PREVIEW: example prompts")
        print("=" * 72)
        target = examples[0]
        for filler_type, k in [("none", 0), ("dots", 5), ("counting", 5)]:
            preview_rng = random.Random(args.seed)
            label = f"{filler_type} (k={k})" if k > 0 else "baseline (k=0)"
            print(f"\n{'─' * 72}")
            print(f"  Condition: {label}")
            print(f"  queried '{target['queried_term']}' = {target['queried_value']} "
                  f"(hidden), Answer={target['answer']}")
            print(f"{'─' * 72}")
            msgs = build_prompt_messages_varbind(
                few_shot_items[:5], target,
                "none" if k == 0 else filler_type, k, rng=preview_rng,
            )
            for msg in msgs:
                print(f"\n[{msg['role'].upper()}]")
                print(msg["content"])
        print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
