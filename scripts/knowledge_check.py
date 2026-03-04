#!/usr/bin/env python3
"""
Evaluate which individual facts the model actually knows.

For each fact, asks the model the question multiple times and checks
whether it returns the correct integer reliably. A fact is considered
"known" if the model answers correctly on at least --pass-threshold
fraction of trials.

Prompt format matches generate_2hop_dataset.py:
  instruction + "Question: ..." in user message, "Answer:" prefill.

Outputs:
  - known_facts.json:   facts the model answered reliably
  - unknown_facts.json: facts the model got wrong or was unreliable on
  - fact_eval_full.jsonl: detailed per-fact results
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

import sys
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Add scripts directory to path for imports
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "scripts"))
from generate_addition_dataset import build_user_message

# ─────────────────────────────────────────────────────────────────────────────
# Fact loading
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

# Few-shot examples for individual fact evaluation — simple, universally known
# facts that won't overlap with the evaluation pool.
INDIVIDUAL_FEWSHOT = [
    ("How many legs does a cat have?", 4),
    ("How many days are in a week?", 7),
    ("How many sides does a triangle have?", 3),
    ("How many hours are in a day?", 24),
    ("How many minutes are in an hour?", 60),
]


def build_prompt_instruct(
    question: str,
    tokenizer: Any,
    num_shots: int = 5,
) -> Tuple[List[int], str]:
    """Build input_ids for an instruct model using chat template.

    Uses the same format as eval_addition.py / generate_2hop_dataset.py:
    instruction in user message, "Question: ...", "Answer:" prefill.
    """
    messages: list = []

    # Few-shot examples
    for q, a in INDIVIDUAL_FEWSHOT[:num_shots]:
        messages.append({"role": "user", "content": build_user_message(q)})
        messages.append({"role": "assistant", "content": f"Answer: {a}"})

    # Test question
    messages.append({"role": "user", "content": build_user_message(question)})

    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_text += "Answer:"
    input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    return input_ids, prompt_text


def build_prompt_base(
    question: str,
    tokenizer: Any,
    num_shots: int = 5,
) -> Tuple[List[int], str]:
    """Build input_ids for a base model using few-shot completion format.

    Uses the same Question:/Answer: structure as the instruct variant,
    but as plain text without chat template.
    """
    lines = []
    for q, a in INDIVIDUAL_FEWSHOT[:num_shots]:
        lines.append(build_user_message(q))
        lines.append(f"Answer: {a}")
        lines.append("")

    lines.append(build_user_message(question))
    lines.append("Answer:")

    prompt_text = "\n".join(lines)
    bos_ids = [tokenizer.bos_token_id] if tokenizer.bos_token_id else []
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    return bos_ids + prompt_ids, prompt_text


# ─────────────────────────────────────────────────────────────────────────────
# Multihop Prompt Builders
# ─────────────────────────────────────────────────────────────────────────────

MULTIHOP_DIR = pathlib.Path(__file__).resolve().parent / "multihop"

MULTIHOP_INSTRUCTION = (
    "You will be given a question. Answer immediately using the format "
    "'Answer: [ANSWER]' where [ANSWER] is just the answer, nothing else. "
    "No explanation, no words, no reasoning, just the answer."
)

# Few-shot examples for multihop integer (hop1) facts.
MULTIHOP_INTEGER_FEWSHOT = [
    ("How many legs does an octopus have?", 8),
    ("How many sides does a hexagon have?", 6),
    ("How many hours are in a day?", 24),
    ("How many days are in a week?", 7),
    ("How many teeth do adult humans typically have?", 32),
]

# Few-shot examples for multihop string (hop2) facts.
MULTIHOP_STRING_FEWSHOT = [
    ("What is the capital of France?", "Paris"),
    ("What is the largest planet in the solar system?", "Jupiter"),
    ("What is the chemical symbol for gold?", "Au"),
    ("What country has the largest population?", "China"),
    ("What is the official language of Brazil?", "Portuguese"),
]


def normalize_answer(s: str) -> str:
    """Normalize string for case-insensitive, punctuation-insensitive comparison."""
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _fact_to_question(fact_str: str) -> str:
    """Convert a chain fact description to a question.

    If the fact string already contains '?', use it as-is.
    Otherwise, wrap it: 'What is the {fact}?'
    """
    if "?" in fact_str:
        return fact_str.strip()
    lowered = fact_str[0].lower() + fact_str[1:]
    return f"What is the {lowered}?"


def build_prompt_instruct_multihop(
    question: str,
    tokenizer: Any,
    fewshot: List[Tuple],
    num_shots: int = 5,
) -> Tuple[List[int], str]:
    """Build instruct prompt using MULTIHOP_INSTRUCTION and given few-shot examples."""
    messages: list = [{"role": "system", "content": MULTIHOP_INSTRUCTION}]
    for q, a in fewshot[:num_shots]:
        messages.append({"role": "user", "content": f"Question: {q}"})
        messages.append({"role": "assistant", "content": f"Answer: {a}"})
    messages.append({"role": "user", "content": f"Question: {question}"})
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_text += "Answer:"
    input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    return input_ids, prompt_text


def build_prompt_base_multihop(
    question: str,
    tokenizer: Any,
    fewshot: List[Tuple],
    num_shots: int = 5,
) -> Tuple[List[int], str]:
    """Build base model prompt using MULTIHOP_INSTRUCTION and given few-shot examples."""
    lines = [MULTIHOP_INSTRUCTION, ""]
    for q, a in fewshot[:num_shots]:
        lines.append(f"Question: {q}")
        lines.append(f"Answer: {a}")
        lines.append("")
    lines.append(f"Question: {question}")
    lines.append("Answer:")
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


def _get_model_device(model: Any) -> torch.device:
    """Get the input device for a model (handles device_map='auto')."""
    if hasattr(model, 'hf_device_map'):
        first_device = next(iter(model.hf_device_map.values()))
        if isinstance(first_device, str):
            return torch.device(first_device)
        return torch.device(f"cuda:{first_device}")
    if hasattr(model, 'device'):
        return model.device
    return next(model.parameters()).device


@torch.no_grad()
def evaluate_fact(
    model: Any,
    tokenizer: Any,
    question: str,
    expected: int,
    mode: str = "instruct",
    num_shots: int = 5,
    max_new_tokens: int = 20,
    num_trials: int = 1,
    temperature: float = 0.0,
    build_prompt: Optional[Any] = None,
) -> Dict[str, Any]:
    """Ask the model a factual question one or more times and check the answer.

    Args:
        num_trials: Number of times to ask (>=2 enables sampling-based eval).
        temperature: Sampling temperature for trials. If 0.0 and num_trials > 1,
            a default of 0.5 is used so that trials are not all identical.
        build_prompt: Optional callable(question, tokenizer, num_shots) -> (ids, text).
            If None, uses build_prompt_instruct/base based on mode.

    Returns:
        Dict with per-trial details and aggregate correct_count / trial_count.
    """
    device = _get_model_device(model)

    if build_prompt is not None:
        input_ids, prompt_text = build_prompt(question, tokenizer, num_shots)
    elif mode == "instruct":
        input_ids, prompt_text = build_prompt_instruct(question, tokenizer, num_shots=num_shots)
    else:
        input_ids, prompt_text = build_prompt_base(question, tokenizer, num_shots=num_shots)

    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    # For base model, stop on newline as well as EOS
    eos_token_id = tokenizer.eos_token_id
    if mode == "base":
        newline_ids = tokenizer.encode("\n", add_special_tokens=False)
        if len(newline_ids) == 1:
            eos_token_id = [tokenizer.eos_token_id, newline_ids[0]]

    # If multiple trials requested but temperature is 0, use a default
    use_temp = temperature
    if num_trials > 1 and use_temp == 0.0:
        use_temp = 0.5

    trial_results = []
    correct_count = 0

    for trial_idx in range(num_trials):
        # First trial is always greedy for reproducibility
        if trial_idx == 0:
            do_sample = False
            gen_temp = None
        else:
            do_sample = True
            gen_temp = use_temp

        gen_kwargs = dict(
            input_ids=input_tensor,
            attention_mask=torch.ones_like(input_tensor),
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_token_id,
            use_cache=True,
        )
        if gen_temp is not None:
            gen_kwargs["temperature"] = gen_temp

        outputs = model.generate(**gen_kwargs)

        generated_ids = outputs[0, len(input_ids):].tolist()
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        predicted = parse_integer_answer(generated_text)
        is_correct = (predicted == expected)

        if is_correct:
            correct_count += 1

        trial_results.append({
            "trial": trial_idx,
            "predicted": predicted,
            "correct": is_correct,
            "generated_text": generated_text,
        })

    return {
        "question": question,
        "expected": expected,
        "correct_count": correct_count,
        "trial_count": num_trials,
        "correct_frac": correct_count / num_trials,
        "trials": trial_results,
        # Back-compat: "predicted" and "generated_text" from the greedy (first) trial
        "predicted": trial_results[0]["predicted"],
        "generated_text": trial_results[0]["generated_text"],
    }


def parse_string_answer(text: str) -> str:
    """Extract first-line answer from generated text."""
    return text.strip().split("\n")[0].strip()


@torch.no_grad()
def evaluate_fact_string(
    model: Any,
    tokenizer: Any,
    question: str,
    expected: str,
    build_prompt: Any,
    mode: str = "instruct",
    num_shots: int = 5,
    max_new_tokens: int = 30,
    num_trials: int = 1,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """Ask the model a string-answer factual question one or more times."""
    device = _get_model_device(model)
    input_ids, prompt_text = build_prompt(question, tokenizer, num_shots)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    eos_token_id = tokenizer.eos_token_id
    if mode == "base":
        newline_ids = tokenizer.encode("\n", add_special_tokens=False)
        if len(newline_ids) == 1:
            eos_token_id = [tokenizer.eos_token_id, newline_ids[0]]

    use_temp = temperature
    if num_trials > 1 and use_temp == 0.0:
        use_temp = 0.5

    expected_norm = normalize_answer(expected)
    trial_results = []
    correct_count = 0

    for trial_idx in range(num_trials):
        do_sample = trial_idx > 0
        gen_kwargs = dict(
            input_ids=input_tensor,
            attention_mask=torch.ones_like(input_tensor),
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_token_id,
            use_cache=True,
        )
        if do_sample:
            gen_kwargs["temperature"] = use_temp

        outputs = model.generate(**gen_kwargs)
        generated_ids = outputs[0, len(input_ids):].tolist()
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        predicted = parse_string_answer(generated_text)
        is_correct = normalize_answer(predicted) == expected_norm

        if is_correct:
            correct_count += 1
        trial_results.append({
            "trial": trial_idx,
            "predicted": predicted,
            "correct": is_correct,
            "generated_text": generated_text,
        })

    return {
        "question": question,
        "expected": expected,
        "correct_count": correct_count,
        "trial_count": num_trials,
        "correct_frac": correct_count / num_trials,
        "trials": trial_results,
        "predicted": trial_results[0]["predicted"],
        "generated_text": trial_results[0]["generated_text"],
    }


def load_multihop_facts() -> Tuple[Dict, Dict, List]:
    """Load unique hop1 (integer) and hop2 facts from 2-hop problems.

    Returns:
        hop1_facts: {fact_key: (question, value)} for hop1 (integer values)
        hop2_facts: {fact_key: (question, value)} for hop2 (string or integer values)
        problems_2hop: full list of 2-hop problem dicts
    """
    sys.path.insert(0, str(MULTIHOP_DIR))
    from generate_dataset import generate_all_problems  # type: ignore

    _, problems_2hop, _, _ = generate_all_problems()

    hop1_facts: Dict[str, Tuple[str, Any]] = {}
    hop2_facts: Dict[str, Tuple[str, Any]] = {}

    for p in problems_2hop:
        chain = p.get("chain", [])
        if len(chain) >= 1:
            step = chain[0]
            key = step["fact"]
            if key not in hop1_facts:
                hop1_facts[key] = (_fact_to_question(key), step["value"])
        if len(chain) >= 2:
            step = chain[1]
            key = step["fact"]
            if key not in hop2_facts:
                hop2_facts[key] = (_fact_to_question(key), step["value"])

    return hop1_facts, hop2_facts, problems_2hop


def run_multihop(args: Any, model: Any, tokenizer: Any, mode: str) -> None:
    """Evaluate individual hop1 and hop2 facts for the multihop 2-hop dataset."""
    print("\nLoading multihop 2-hop problems...")
    hop1_facts, hop2_facts, problems_2hop = load_multihop_facts()
    print(f"  {len(problems_2hop)} total 2-hop problems")
    print(f"  {len(hop1_facts)} unique hop1 facts")
    print(f"  {len(hop2_facts)} unique hop2 facts")

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Build prompt builder functions matching (question, tokenizer, num_shots) signature
    if mode == "instruct":
        def int_prompt(q, tok, ns):
            return build_prompt_instruct_multihop(q, tok, MULTIHOP_INTEGER_FEWSHOT, ns)
        def str_prompt(q, tok, ns):
            return build_prompt_instruct_multihop(q, tok, MULTIHOP_STRING_FEWSHOT, ns)
    else:
        def int_prompt(q, tok, ns):
            return build_prompt_base_multihop(q, tok, MULTIHOP_INTEGER_FEWSHOT, ns)
        def str_prompt(q, tok, ns):
            return build_prompt_base_multihop(q, tok, MULTIHOP_STRING_FEWSHOT, ns)

    # Show example prompts
    first_hop1_q = next(iter(hop1_facts.values()))[0]
    _, ex_prompt = int_prompt(first_hop1_q, tokenizer, args.num_shots)
    print(f"\nExample hop1 prompt:")
    print("-" * 40)
    print(ex_prompt)
    print("-" * 40)

    # ── Evaluate hop1 facts (integer or string, but typically integer) ────────
    print(f"\nEvaluating {len(hop1_facts)} hop1 facts...")
    hop1_results: Dict[str, Dict] = {}
    start = time.time()
    for i, (key, (question, expected)) in enumerate(tqdm(hop1_facts.items(), desc="Hop1")):
        if isinstance(expected, int):
            result = evaluate_fact(
                model, tokenizer, question, expected,
                mode=mode, num_shots=args.num_shots,
                num_trials=args.num_trials, temperature=args.temperature,
                build_prompt=int_prompt,
            )
        else:
            result = evaluate_fact_string(
                model, tokenizer, question, str(expected),
                build_prompt=int_prompt,
                mode=mode, num_shots=args.num_shots,
                num_trials=args.num_trials, temperature=args.temperature,
            )
        result["kind"] = "multihop_hop1"
        result["fact_key"] = key
        result["known"] = result["correct_frac"] >= args.pass_threshold
        hop1_results[key] = result

        if args.report_every > 0 and (i + 1) % args.report_every == 0:
            n_known = sum(1 for r in hop1_results.values() if r["known"])
            tqdm.write(f"  [{i+1}/{len(hop1_facts)}] {n_known} known so far")

    hop1_known_count = sum(1 for r in hop1_results.values() if r["known"])
    print(f"Hop1: {hop1_known_count}/{len(hop1_results)} known "
          f"({hop1_known_count/len(hop1_results)*100:.1f}%)")

    # ── Evaluate hop2 facts (string or integer) ───────────────────────────────
    print(f"\nEvaluating {len(hop2_facts)} hop2 facts...")
    hop2_results: Dict[str, Dict] = {}
    for i, (key, (question, expected)) in enumerate(tqdm(hop2_facts.items(), desc="Hop2")):
        if isinstance(expected, int):
            result = evaluate_fact(
                model, tokenizer, question, expected,
                mode=mode, num_shots=args.num_shots,
                num_trials=args.num_trials, temperature=args.temperature,
                build_prompt=str_prompt,
            )
        else:
            result = evaluate_fact_string(
                model, tokenizer, question, str(expected),
                build_prompt=str_prompt,
                mode=mode, num_shots=args.num_shots,
                num_trials=args.num_trials, temperature=args.temperature,
            )
        result["kind"] = "multihop_hop2"
        result["fact_key"] = key
        result["known"] = result["correct_frac"] >= args.pass_threshold
        hop2_results[key] = result

        if args.report_every > 0 and (i + 1) % args.report_every == 0:
            n_known = sum(1 for r in hop2_results.values() if r["known"])
            tqdm.write(f"  [{i+1}/{len(hop2_facts)}] {n_known} known so far")

    hop2_known_count = sum(1 for r in hop2_results.values() if r["known"])
    print(f"Hop2: {hop2_known_count}/{len(hop2_results)} known "
          f"({hop2_known_count/len(hop2_results)*100:.1f}%)")

    elapsed = time.time() - start

    # ── Determine known problems ──────────────────────────────────────────────
    known_problems = []
    unknown_problems = []
    for p in problems_2hop:
        chain = p.get("chain", [])
        h1_key = chain[0]["fact"] if len(chain) >= 1 else None
        h2_key = chain[1]["fact"] if len(chain) >= 2 else None
        h1_known = hop1_results.get(h1_key, {}).get("known", False) if h1_key else False
        h2_known = hop2_results.get(h2_key, {}).get("known", False) if h2_key else False
        if h1_known and h2_known:
            known_problems.append(p)
        else:
            unknown_problems.append(p)

    print(f"\nProblems where both hops known: {len(known_problems)}/{len(problems_2hop)} "
          f"({len(known_problems)/len(problems_2hop)*100:.1f}%)")

    # ── Save outputs ──────────────────────────────────────────────────────────
    (outdir / "known_problems.json").write_text(
        json.dumps(known_problems, indent=2, ensure_ascii=False)
    )
    (outdir / "known_hop1_facts.json").write_text(
        json.dumps(
            [{"question": r["question"], "answer": r["expected"],
              "fact_key": r["fact_key"]}
             for r in hop1_results.values() if r["known"]],
            indent=2, ensure_ascii=False,
        )
    )
    (outdir / "known_hop2_facts.json").write_text(
        json.dumps(
            [{"question": r["question"], "answer": r["expected"],
              "fact_key": r["fact_key"]}
             for r in hop2_results.values() if r["known"]],
            indent=2, ensure_ascii=False,
        )
    )

    total_facts = len(hop1_results) + len(hop2_results)
    with (outdir / "fact_eval_full.jsonl").open("w") as f:
        for r in hop1_results.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        for r in hop2_results.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "model": args.model,
        "mode": mode,
        "task": "multihop",
        "num_shots": args.num_shots,
        "num_trials": args.num_trials,
        "pass_threshold": args.pass_threshold,
        "total_problems_2hop": len(problems_2hop),
        "known_problems": len(known_problems),
        "known_problems_frac": len(known_problems) / len(problems_2hop) if problems_2hop else 0,
        "hop1_facts": len(hop1_results),
        "hop1_known": hop1_known_count,
        "hop1_accuracy": hop1_known_count / len(hop1_results) if hop1_results else 0,
        "hop2_facts": len(hop2_results),
        "hop2_known": hop2_known_count,
        "hop2_accuracy": hop2_known_count / len(hop2_results) if hop2_results else 0,
        "elapsed_seconds": elapsed,
    }
    (outdir / "fact_eval_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nSaved to {outdir}/:")
    print(f"  known_problems.json     ({len(known_problems)} problems)")
    print(f"  known_hop1_facts.json   ({hop1_known_count} facts)")
    print(f"  known_hop2_facts.json   ({hop2_known_count} facts)")
    print(f"  fact_eval_full.jsonl    ({total_facts} results)")
    print(f"  fact_eval_summary.json")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate which individual facts the model knows"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Model name or path")
    parser.add_argument("--task", type=str, default="addition",
                        choices=["addition", "multihop"],
                        help="Which fact set to evaluate: 'addition' uses --sources JSON files; "
                             "'multihop' loads 2-hop problems from scripts/multihop/ (default: addition)")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--mode", type=str, default="auto",
                        choices=["auto", "base", "instruct"],
                        help="Prompt mode: 'base' uses few-shot completion, "
                             "'instruct' uses chat template, "
                             "'auto' detects from model name (default: auto)")
    parser.add_argument("--num-shots", type=int, default=5,
                        help="Number of few-shot examples for base mode (default: 5)")
    parser.add_argument("--sources", type=str, default=None,
                        help="Directory containing fact JSON files (required for --task addition)")
    parser.add_argument("--outdir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--max-answer", type=int, default=1000,
                        help="Maximum answer value")
    parser.add_argument("--report-every", type=int, default=50,
                        help="Print running accuracy every N facts")
    parser.add_argument("--tolerance", type=int, default=0,
                        help="Accept answers within +/- tolerance of expected "
                             "(0 = exact match only)")
    parser.add_argument("--num-trials", type=int, default=4,
                        help="Number of times to ask each question (default: 4)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature for non-greedy trials "
                             "(0 = auto-select 0.5 when num-trials > 1)")
    parser.add_argument("--pass-threshold", type=float, default=0.75,
                        help="Fraction of trials that must be correct to consider "
                             "a fact 'known' (default: 0.75)")

    args = parser.parse_args()

    if args.task == "addition" and args.sources is None:
        parser.error("--sources is required for --task addition")

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

    # ── Multihop task ─────────────────────────────────────────────────────────
    if args.task == "multihop":
        run_multihop(args, model, tokenizer, mode)
        return

    # ── Addition task (default) ───────────────────────────────────────────────
    srcdir = pathlib.Path(args.sources)
    age = load_facts(srcdir / "age_facts.json", "age") if (srcdir / "age_facts.json").exists() else []
    atomic = load_facts(srcdir / "atomic_facts.json", "atomic") if (srcdir / "atomic_facts.json").exists() else []
    static = load_facts(srcdir / "static_facts.json", "static") if (srcdir / "static_facts.json").exists() else []

    all_facts = [(q, a, k) for (q, a, k) in (age + atomic + static) if 0 <= a < args.max_answer]
    print(f"Loaded {len(all_facts)} facts "
          f"(age={len(age)}, atomic={len(atomic)}, static={len(static)})")

    # Show example prompt so user can verify it looks right
    example_q = all_facts[0][0] if all_facts else "What is the atomic number of Helium?"
    if mode == "instruct":
        _, example_prompt = build_prompt_instruct(example_q, tokenizer, num_shots=args.num_shots)
    else:
        _, example_prompt = build_prompt_base(example_q, tokenizer, num_shots=args.num_shots)
    print(f"\nExample prompt:")
    print("-" * 40)
    print(example_prompt)
    print("-" * 40)

    # Evaluate each fact
    print(f"\nEvaluating {len(all_facts)} individual facts...")
    print(f"  Trials per fact: {args.num_trials}")
    print(f"  Pass threshold: {args.pass_threshold:.0%}")
    if args.tolerance > 0:
        print(f"  Tolerance: +/- {args.tolerance}")
    print()

    results = []
    known_count = 0
    total = 0
    start = time.time()

    # Track per-category stats
    category_stats = {}

    for question, expected, kind in tqdm(all_facts, desc="Facts"):
        result = evaluate_fact(model, tokenizer, question, expected,
                               mode=mode, num_shots=args.num_shots,
                               num_trials=args.num_trials,
                               temperature=args.temperature)
        result["kind"] = kind

        # A fact is "known" if correct on >= pass_threshold of trials
        result["known"] = (result["correct_frac"] >= args.pass_threshold)

        # Apply tolerance if specified (on greedy trial for back-compat)
        if args.tolerance > 0 and result["predicted"] is not None:
            result["within_tolerance"] = abs(result["predicted"] - expected) <= args.tolerance
        else:
            result["within_tolerance"] = result["trials"][0]["correct"]

        results.append(result)
        if result["known"]:
            known_count += 1
        total += 1

        # Per-category tracking
        if kind not in category_stats:
            category_stats[kind] = {"known": 0, "total": 0}
        category_stats[kind]["total"] += 1
        if result["known"]:
            category_stats[kind]["known"] += 1

        # Periodic reporting
        if args.report_every > 0 and total % args.report_every == 0:
            acc = known_count / total * 100
            elapsed = time.time() - start
            tqdm.write(f"  [{total}/{len(all_facts)}] {known_count}/{total} known ({acc:.1f}%) "
                       f"[{elapsed:.1f}s]")

    elapsed = time.time() - start

    # Split into known/unknown
    known = [r for r in results if r["known"]]
    unknown = [r for r in results if not r["known"]]

    # Print summary
    print(f"\n{'='*60}")
    print(f"RESULTS: {known_count}/{total} ({known_count/total*100:.1f}%) known "
          f"(>= {args.pass_threshold:.0%} of {args.num_trials} trials correct)")
    greedy_correct = sum(1 for r in results if r["trials"][0]["correct"])
    print(f"         {greedy_correct}/{total} ({greedy_correct/total*100:.1f}%) greedy exact match")
    if args.tolerance > 0:
        n_tol = sum(1 for r in results if r["within_tolerance"])
        print(f"         {n_tol}/{total} ({n_tol/total*100:.1f}%) within tolerance +/- {args.tolerance}")
    print(f"Time: {elapsed:.1f}s ({elapsed/total:.2f}s per fact)")
    print(f"{'='*60}")

    # Per-category breakdown
    print(f"\nBy category:")
    for kind in sorted(category_stats.keys()):
        s = category_stats[kind]
        acc = s["known"] / s["total"] * 100 if s["total"] > 0 else 0
        print(f"  {kind:<10}: {s['known']:>4}/{s['total']:<4} ({acc:.1f}%) known")

    # Show some unknown answers for inspection
    print(f"\nSample unknown facts (first 20):")
    print(f"{'Question':<50} {'Expected':>8} {'Greedy':>8} {'Frac':>6} {'Trials'}")
    print("-" * 110)
    for r in unknown[:20]:
        q_short = r["question"][:47] + "..." if len(r["question"]) > 50 else r["question"]
        trial_strs = [str(t["predicted"]) for t in r["trials"]]
        print(f"{q_short:<50} {r['expected']:>8} {str(r['predicted']):>8} "
              f"{r['correct_frac']:>5.0%} [{', '.join(trial_strs)}]")

    # Save outputs
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # known_facts.json — list of {question, answer, kind} the model knows reliably
    known_facts = [
        {"question": r["question"], "answer": r["expected"], "kind": r["kind"]}
        for r in known
    ]
    (outdir / "known_facts.json").write_text(
        json.dumps(known_facts, indent=2, ensure_ascii=False)
    )

    # unknown_facts.json — same format for unreliable / wrong answers
    unknown_facts = [
        {
            "question": r["question"],
            "answer": r["expected"],
            "kind": r["kind"],
            "correct_frac": r["correct_frac"],
            "greedy_predicted": r["predicted"],
            "trial_predictions": [t["predicted"] for t in r["trials"]],
        }
        for r in unknown
    ]
    (outdir / "unknown_facts.json").write_text(
        json.dumps(unknown_facts, indent=2, ensure_ascii=False)
    )

    # Full detailed results (drop per-trial detail for compact JSONL)
    with (outdir / "fact_eval_full.jsonl").open("w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Summary JSON
    summary = {
        "model": args.model,
        "mode": mode,
        "num_shots": args.num_shots if mode == "base" else args.num_shots,
        "num_trials": args.num_trials,
        "pass_threshold": args.pass_threshold,
        "temperature": args.temperature,
        "total_facts": total,
        "known_count": len(known),
        "unknown_count": len(unknown),
        "known_accuracy": known_count / total if total > 0 else 0,
        "greedy_accuracy": greedy_correct / total if total > 0 else 0,
        "category_stats": {
            kind: {
                "known": s["known"],
                "total": s["total"],
                "accuracy": s["known"] / s["total"] if s["total"] > 0 else 0,
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