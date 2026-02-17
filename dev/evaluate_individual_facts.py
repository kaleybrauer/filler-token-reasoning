#!/usr/bin/env python3
"""
Evaluate which individual facts the model actually knows.

For each fact, asks the model the question and checks whether it returns
the correct integer. Outputs:
  - known_facts.json:   facts the model answered correctly
  - unknown_facts.json: facts the model got wrong
  - fact_eval_full.jsonl: detailed per-fact results

Use known_facts.json to filter your train/val/test sets so you only
train on questions the model can actually answer individually.
"""
import argparse
import json
import os
import pathlib
import re
import time
from typing import Any, Dict, List, Optional, Tuple

# Set up cache before importing transformers
def setup_cache():
    if os.path.exists("/workspace"):
        cache_path = "/workspace/.cache/huggingface"
        os.environ["HF_HOME"] = cache_path
        os.environ["HF_DATASETS_CACHE"] = f"{cache_path}/datasets"
        pathlib.Path(cache_path).mkdir(parents=True, exist_ok=True)

setup_cache()

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ─────────────────────────────────────────────────────────────────────────────
# Fact loading (shared with generate_prompt_format_test.py)
# ─────────────────────────────────────────────────────────────────────────────

INT_KEYS = ["answer", "value", "number", "n", "age", "atomic_number"]


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _extract_int_from_obj(obj: Any) -> Optional[int]:
    if _is_int(obj):
        return obj
    if isinstance(obj, dict):
        for k in INT_KEYS:
            if k in obj and _is_int(obj[k]):
                return obj[k]
    return None


def load_facts(path: pathlib.Path, kind: str) -> List[Tuple[str, int, str]]:
    """Load facts from JSON file. Returns list of (question_text, answer_int, kind)."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: List[Tuple[str, int, str]] = []

    def make_q(key_or_q: str) -> str:
        if "?" in key_or_q:
            return key_or_q.strip()
        if kind == "age":
            return f"At what age did {key_or_q} die?"
        if kind == "atomic":
            return f"What is the atomic number of {key_or_q}?"
        return key_or_q.strip()

    if isinstance(raw, dict):
        for k, v in raw.items():
            ans = _extract_int_from_obj(v)
            if ans is None:
                continue
            q = make_q(k)
            out.append((q, int(ans), kind))

    elif isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            ans = _extract_int_from_obj(item)
            if ans is None:
                continue

            q = None
            for qk in ["question", "q", "prompt", "text"]:
                if qk in item and isinstance(item[qk], str):
                    q = item[qk].strip()
                    break

            if q is None:
                for nk in ["name", "entity", "person", "element"]:
                    if nk in item and isinstance(item[nk], str):
                        q = make_q(item[nk])
                        break

            if q is None:
                continue

            out.append((q, int(ans), kind))

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────────────────────────────────────

USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
DTYPE = torch.bfloat16 if USE_BF16 else torch.float16


def load_model_and_tokenizer(
    model_name: str,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
) -> Tuple[Any, Any]:
    """Load model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if load_in_4bit:
        print("Using 4-bit quantization")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=DTYPE,
        )
    elif load_in_8bit:
        print("Using 8-bit quantization")
        quant_config = BitsAndBytesConfig(load_in_8bit=True)

    model_kwargs = dict(
        torch_dtype=DTYPE,
        quantization_config=quant_config,
        trust_remote_code=True,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    try:
        import flash_attn
        model_kwargs["attn_implementation"] = "flash_attention_2"
        print("Using Flash Attention 2")
    except ImportError:
        model_kwargs["attn_implementation"] = "sdpa"
        print("Using SDPA attention")

    print(f"Loading model: {model_name}")
    start = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    print(f"Loaded in {time.time() - start:.1f}s")

    model.eval()

    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated() / 1e9
        print(f"GPU memory: {mem:.2f} GB")

    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Mode Detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_model_mode(model_name: str) -> str:
    """Auto-detect whether a model is base or instruct from its name."""
    name_lower = model_name.lower()
    if "instruct" in name_lower or "chat" in name_lower:
        return "instruct"
    return "base"


# ─────────────────────────────────────────────────────────────────────────────
# Prompt Builders
# ─────────────────────────────────────────────────────────────────────────────

# Few-shot examples for base model — these should be easy, universally known
# facts so they never contaminate the evaluation.
BASE_FEWSHOT_EXAMPLES = [
    ("How many legs does a cat have?", 4),
    ("How many days are in a week?", 7),
    ("How many sides does a triangle have?", 3),
    ("How many hours are in a day?", 24),
    ("How many minutes are in an hour?", 60),
]


def build_prompt_instruct(
    question: str,
    tokenizer: Any,
) -> List[int]:
    """Build input_ids for an instruct model using chat template."""
    messages = [
        {
            "role": "system",
            "content": (
                "Answer the following question with just a single integer. "
                "No words, no explanation, just the number."
            ),
        },
        {"role": "user", "content": question},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return tokenizer.encode(prompt_text, add_special_tokens=False), prompt_text


def build_prompt_base(
    question: str,
    tokenizer: Any,
    num_shots: int = 5,
) -> List[int]:
    """Build input_ids for a base model using few-shot completion format."""
    lines = []
    for q, a in BASE_FEWSHOT_EXAMPLES[:num_shots]:
        lines.append(f"Q: {q}")
        lines.append(f"A: {a}")
        lines.append("")

    lines.append(f"Q: {question}")
    lines.append("A:")

    prompt_text = "\n".join(lines)
    bos_ids = [tokenizer.bos_token_id] if tokenizer.bos_token_id else []
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    return bos_ids + prompt_ids, prompt_text


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def parse_integer_answer(text: str) -> Optional[int]:
    """Extract integer from generated text."""
    text = text.strip()
    try:
        return int(text)
    except ValueError:
        pass
    # Try first integer in the text
    match = re.search(r'-?\d+', text)
    if match:
        return int(match.group())
    return None


@torch.no_grad()
def evaluate_fact(
    model: Any,
    tokenizer: Any,
    question: str,
    expected: int,
    mode: str = "instruct",
    num_shots: int = 5,
    max_new_tokens: int = 20,
) -> Dict[str, Any]:
    """Ask the model a single factual question and check the answer."""
    device = next(model.parameters()).device

    if mode == "instruct":
        input_ids, prompt_text = build_prompt_instruct(question, tokenizer)
    else:
        input_ids, prompt_text = build_prompt_base(question, tokenizer, num_shots=num_shots)

    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    # For base model, stop on newline as well as EOS — the answer is on one line
    eos_token_id = tokenizer.eos_token_id
    if mode == "base":
        newline_ids = tokenizer.encode("\n", add_special_tokens=False)
        if len(newline_ids) == 1:
            eos_token_id = [tokenizer.eos_token_id, newline_ids[0]]

    outputs = model.generate(
        input_ids=input_tensor,
        attention_mask=torch.ones_like(input_tensor),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=eos_token_id,
        use_cache=True,
    )

    generated_ids = outputs[0, len(input_ids):].tolist()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    predicted = parse_integer_answer(generated_text)

    return {
        "question": question,
        "expected": expected,
        "predicted": predicted,
        "correct": predicted == expected,
        "generated_text": generated_text,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate which individual facts the model knows"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Model name or path")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--mode", type=str, default="auto",
                        choices=["auto", "base", "instruct"],
                        help="Prompt mode: 'base' uses few-shot completion, "
                             "'instruct' uses chat template, "
                             "'auto' detects from model name (default: auto)")
    parser.add_argument("--num-shots", type=int, default=5,
                        help="Number of few-shot examples for base mode (default: 5)")
    parser.add_argument("--sources", type=str, required=True,
                        help="Directory containing fact JSON files")
    parser.add_argument("--outdir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--max-answer", type=int, default=1000,
                        help="Maximum answer value")
    parser.add_argument("--report-every", type=int, default=50,
                        help="Print running accuracy every N facts")
    parser.add_argument("--tolerance", type=int, default=0,
                        help="Accept answers within +/- tolerance of expected "
                             "(0 = exact match only)")

    args = parser.parse_args()

    # Load facts
    srcdir = pathlib.Path(args.sources)
    age = load_facts(srcdir / "age_facts.json", "age") if (srcdir / "age_facts.json").exists() else []
    atomic = load_facts(srcdir / "atomic_facts.json", "atomic") if (srcdir / "atomic_facts.json").exists() else []
    static = load_facts(srcdir / "static_facts.json", "static") if (srcdir / "static_facts.json").exists() else []

    all_facts = [(q, a, k) for (q, a, k) in (age + atomic + static) if 0 <= a < args.max_answer]
    print(f"Loaded {len(all_facts)} facts "
          f"(age={len(age)}, atomic={len(atomic)}, static={len(static)})")

    # Load model
    print()
    model, tokenizer = load_model_and_tokenizer(
        args.model,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
    )

    # Detect or set mode
    if args.mode == "auto":
        mode = detect_model_mode(args.model)
        print(f"\nAuto-detected mode: {mode}")
    else:
        mode = args.mode
        print(f"\nUsing mode: {mode}")

    # Show example prompt so user can verify it looks right
    example_q = all_facts[0][0] if all_facts else "What is the atomic number of Helium?"
    if mode == "instruct":
        _, example_prompt = build_prompt_instruct(example_q, tokenizer)
    else:
        _, example_prompt = build_prompt_base(example_q, tokenizer, num_shots=args.num_shots)
    print(f"\nExample prompt:")
    print("-" * 40)
    print(example_prompt)
    print("-" * 40)

    # Evaluate each fact
    print(f"\nEvaluating {len(all_facts)} individual facts...")
    if args.tolerance > 0:
        print(f"  Tolerance: +/- {args.tolerance}")
    print()

    results = []
    correct = 0
    total = 0
    start = time.time()

    # Track per-category stats
    category_stats = {}

    for question, expected, kind in tqdm(all_facts, desc="Facts"):
        result = evaluate_fact(model, tokenizer, question, expected,
                               mode=mode, num_shots=args.num_shots)
        result["kind"] = kind

        # Apply tolerance if specified
        if args.tolerance > 0 and result["predicted"] is not None:
            result["within_tolerance"] = abs(result["predicted"] - expected) <= args.tolerance
        else:
            result["within_tolerance"] = result["correct"]

        results.append(result)
        if result["correct"]:
            correct += 1
        total += 1

        # Per-category tracking
        if kind not in category_stats:
            category_stats[kind] = {"correct": 0, "total": 0}
        category_stats[kind]["total"] += 1
        if result["correct"]:
            category_stats[kind]["correct"] += 1

        # Periodic reporting
        if args.report_every > 0 and total % args.report_every == 0:
            acc = correct / total * 100
            elapsed = time.time() - start
            tqdm.write(f"  [{total}/{len(all_facts)}] {correct}/{total} ({acc:.1f}%) "
                       f"[{elapsed:.1f}s]")

    elapsed = time.time() - start

    # Split into known/unknown
    known = [r for r in results if r["correct"]]
    unknown = [r for r in results if not r["correct"]]

    if args.tolerance > 0:
        within_tol = [r for r in results if r["within_tolerance"]]

    # Print summary
    print(f"\n{'='*60}")
    print(f"RESULTS: {correct}/{total} ({correct/total*100:.1f}%) exact match")
    if args.tolerance > 0:
        n_tol = sum(1 for r in results if r["within_tolerance"])
        print(f"         {n_tol}/{total} ({n_tol/total*100:.1f}%) within tolerance +/- {args.tolerance}")
    print(f"Time: {elapsed:.1f}s ({elapsed/total:.2f}s per fact)")
    print(f"{'='*60}")

    # Per-category breakdown
    print(f"\nBy category:")
    for kind in sorted(category_stats.keys()):
        s = category_stats[kind]
        acc = s["correct"] / s["total"] * 100 if s["total"] > 0 else 0
        print(f"  {kind:<10}: {s['correct']:>4}/{s['total']:<4} ({acc:.1f}%)")

    # Show some wrong answers for inspection
    print(f"\nSample wrong answers (first 20):")
    print(f"{'Question':<55} {'Expected':>8} {'Got':>8} {'Raw output'}")
    print("-" * 100)
    for r in unknown[:20]:
        q_short = r["question"][:52] + "..." if len(r["question"]) > 55 else r["question"]
        raw_short = r["generated_text"][:30].replace("\n", "\\n")
        print(f"{q_short:<55} {r['expected']:>8} {str(r['predicted']):>8} {raw_short}")

    # Save outputs
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # known_facts.json — list of {question, answer, kind} the model got right
    known_facts = [
        {"question": r["question"], "answer": r["expected"], "kind": r["kind"]}
        for r in known
    ]
    (outdir / "known_facts.json").write_text(
        json.dumps(known_facts, indent=2, ensure_ascii=False)
    )

    # unknown_facts.json — same format for wrong answers
    unknown_facts = [
        {
            "question": r["question"],
            "answer": r["expected"],
            "kind": r["kind"],
            "model_predicted": r["predicted"],
            "model_raw": r["generated_text"],
        }
        for r in unknown
    ]
    (outdir / "unknown_facts.json").write_text(
        json.dumps(unknown_facts, indent=2, ensure_ascii=False)
    )

    # Full detailed results
    with (outdir / "fact_eval_full.jsonl").open("w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Summary JSON
    summary = {
        "model": args.model,
        "mode": mode,
        "num_shots": args.num_shots if mode == "base" else None,
        "total_facts": total,
        "known_count": len(known),
        "unknown_count": len(unknown),
        "accuracy": correct / total if total > 0 else 0,
        "category_stats": {
            kind: {
                "correct": s["correct"],
                "total": s["total"],
                "accuracy": s["correct"] / s["total"] if s["total"] > 0 else 0,
            }
            for kind, s in category_stats.items()
        },
        "tolerance": args.tolerance,
        "elapsed_seconds": elapsed,
    }
    if args.tolerance > 0:
        summary["within_tolerance_count"] = sum(1 for r in results if r["within_tolerance"])
        summary["within_tolerance_accuracy"] = summary["within_tolerance_count"] / total

    (outdir / "fact_eval_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nSaved to {outdir}/:")
    print(f"  known_facts.json     ({len(known)} facts)")
    print(f"  unknown_facts.json   ({len(unknown)} facts)")
    print(f"  fact_eval_full.jsonl ({total} detailed results)")
    print(f"  fact_eval_summary.json")


if __name__ == "__main__":
    main()