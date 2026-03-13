#!/usr/bin/env python3
"""
Test compressed reasoning: give the model K free tokens before "Answer:".

For each test example:
1. Build the prompt (system + question + start of assistant turn)
2. Let the model generate K tokens freely (optionally with digit restriction)
3. Force "\\nAnswer:" into the sequence
4. Let the model generate the answer
5. Check if the answer is correct

Reports accuracy by K value and prints what the model writes in its free tokens.

Usage:
    python scripts/test_compressed_reasoning.py \
        --model outputs/cot_teacher_14b \
        --data-dir data/datasets/2hop_14b \
        --k-values 0,3,5,10 \
        --max-examples 100 \
        --outdir results/compressed_pilot

    # With digit restriction:
    python scripts/test_compressed_reasoning.py \
        --model outputs/cot_teacher_14b \
        --data-dir data/datasets/2hop_14b \
        --k-values 0,3,5,10 \
        --no-digits \
        --max-examples 100 \
        --outdir results/compressed_pilot_nodigits
"""
import argparse
import json
import pathlib
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Project imports
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "scripts"))
from generate_addition_dataset import build_system_prompt, compose_question
from evaluate import load_model_and_tokenizer, parse_integer_answer

USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
DTYPE = torch.bfloat16 if USE_BF16 else torch.float16


def build_prompt_for_generation(
    example: Dict[str, Any],
    tokenizer: Any,
    fewshot_examples: List[Tuple[str, str, int, int, int]],
) -> Tuple[str, List[int]]:
    """Build prompt up to the start of the assistant turn.

    Returns (prompt_text, prompt_ids) — the model generates from here.
    """
    q1 = example["fact1"]
    q2 = example["fact2"]

    messages = [
        {"role": "system", "content": build_system_prompt(fewshot_examples)},
        {"role": "user", "content": compose_question(q1, q2)},
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

    return prompt_text, prompt_ids


def get_digit_token_ids(tokenizer) -> List[int]:
    """Find all token IDs that represent digit characters."""
    digit_ids = set()
    for d in range(10):
        # Try various forms: "0", " 0", etc.
        for prefix in ["", " "]:
            text = f"{prefix}{d}"
            ids = tokenizer.encode(text, add_special_tokens=False)
            for tid in ids:
                decoded = tokenizer.decode([tid]).strip()
                if decoded.isdigit():
                    digit_ids.add(tid)

    # Also scan vocabulary directly for single-digit tokens
    for tid in range(tokenizer.vocab_size):
        try:
            decoded = tokenizer.decode([tid]).strip()
            if decoded and all(c.isdigit() for c in decoded):
                digit_ids.add(tid)
        except Exception:
            continue

    return sorted(digit_ids)


def generate_with_k_free_tokens(
    model: Any,
    tokenizer: Any,
    prompt_ids: List[int],
    k: int,
    device: torch.device,
    digit_token_ids: Optional[List[int]] = None,
    temperature: float = 0.0,
) -> Tuple[str, str, List[int]]:
    """Generate K free tokens, then force Answer: and generate the answer.

    Args:
        model: The language model.
        tokenizer: The tokenizer.
        prompt_ids: Token IDs for the prompt (up to assistant turn start).
        k: Number of free tokens to generate.
        device: GPU device.
        digit_token_ids: If provided, mask these during K free token generation.
        temperature: Sampling temperature (0 = greedy).

    Returns:
        (free_text, answer_text, full_ids)
        free_text: The K tokens the model generated freely.
        answer_text: What the model generated after "Answer:".
        full_ids: Complete token sequence.
    """
    current_ids = list(prompt_ids)

    # Phase 1: Generate K free tokens
    if k > 0:
        ids_tensor = torch.tensor([current_ids], dtype=torch.long, device=device)

        for step in range(k):
            with torch.no_grad():
                outputs = model(input_ids=ids_tensor, use_cache=False)

            logits = outputs.logits[0, -1, :]  # [vocab_size]

            # Mask digit tokens if restricted
            if digit_token_ids is not None:
                logits[digit_token_ids] = float("-inf")

            # Also mask EOS to prevent early stopping
            if tokenizer.eos_token_id is not None:
                logits[tokenizer.eos_token_id] = float("-inf")

            if temperature == 0.0:
                next_token = logits.argmax().item()
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1).item()

            current_ids.append(next_token)
            ids_tensor = torch.tensor([current_ids], dtype=torch.long, device=device)

        free_text = tokenizer.decode(
            current_ids[len(prompt_ids):],
            skip_special_tokens=True,
        )
    else:
        free_text = ""

    # Phase 2: Force "\nAnswer:" into the sequence
    answer_prefix = "\nAnswer:"
    answer_prefix_ids = tokenizer.encode(answer_prefix, add_special_tokens=False)
    current_ids.extend(answer_prefix_ids)

    # Phase 3: Generate the answer (max 10 tokens, greedy)
    ids_tensor = torch.tensor([current_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        answer_output = model.generate(
            input_ids=ids_tensor,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    answer_ids = answer_output[0, len(current_ids):].tolist()
    answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True)

    full_ids = current_ids + answer_ids

    return free_text, answer_text, full_ids


def main():
    parser = argparse.ArgumentParser(
        description="Test compressed reasoning with K free tokens"
    )

    # Model
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--cache-dir", type=str, default=None)

    # Data
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max-examples", type=int, default=None)

    # Generation
    parser.add_argument("--k-values", type=str, default="0,3,5,10",
                        help="Comma-separated K values to test")
    parser.add_argument("--no-digits", action="store_true",
                        help="Mask digit tokens (0-9) during K free generation")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature for free tokens (0 = greedy)")

    # Output
    parser.add_argument("--outdir", type=str, default="results/compressed_pilot")
    parser.add_argument("--show-examples", type=int, default=10,
                        help="Number of example outputs to print per K value")

    args = parser.parse_args()

    k_values = [int(x) for x in args.k_values.split(",")]
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──
    data_dir = pathlib.Path(args.data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    raw_fewshot = manifest["fewshot_examples"]
    fewshot_examples = [
        (e["q1"], e["q2"], e["a1"], e["a2"], e["sum"]) for e in raw_fewshot
    ]

    from datasets import load_dataset
    data_file = data_dir / f"{args.split}.jsonl"
    dataset = load_dataset("json", data_files=str(data_file), split="train")

    # Filter to filler examples with filler_len=0 (one per unique question)
    if "sequence_type" in dataset.column_names:
        dataset = dataset.filter(lambda ex: ex["sequence_type"] != "cot")
    if "filler_len" in dataset.column_names:
        dataset = dataset.filter(lambda ex: ex["filler_len"] == 0)

    if args.max_examples:
        dataset = dataset.select(range(min(args.max_examples, len(dataset))))

    print(f"Loaded {len(dataset)} test examples")

    # ── Load model ──
    print(f"\nLoading model: {args.model}")
    model, tokenizer = load_model_and_tokenizer(
        args.model,
        adapter_path=args.adapter,
        load_in_4bit=args.load_in_4bit,
        use_flash_attn=True,
        trust_remote_code=args.trust_remote_code,
        cache_dir=args.cache_dir,
    )

    # Get device
    if hasattr(model, "hf_device_map") and model.hf_device_map:
        first_device = next(iter(model.hf_device_map.values()))
        device = torch.device(
            f"cuda:{first_device}" if isinstance(first_device, int) else first_device
        )
    else:
        device = next(model.parameters()).device

    # Get digit token IDs if restricting
    digit_token_ids = None
    if args.no_digits:
        digit_token_ids = get_digit_token_ids(tokenizer)
        print(f"Restricting {len(digit_token_ids)} digit tokens during free generation")

    # ── Run evaluation for each K ──
    all_results = {}

    for k in k_values:
        print(f"\n{'='*60}")
        print(f"  K = {k} {'(no digits)' if args.no_digits else '(unrestricted)'}")
        print(f"{'='*60}")

        correct = 0
        total = 0
        examples_shown = 0
        detailed_results = []

        for i in tqdm(range(len(dataset)), desc=f"K={k}"):
            ex = dataset[i]
            expected = ex["answer"]

            # Build prompt
            prompt_text, prompt_ids = build_prompt_for_generation(
                ex, tokenizer, fewshot_examples
            )

            # Generate with K free tokens
            free_text, answer_text, full_ids = generate_with_k_free_tokens(
                model, tokenizer, prompt_ids, k, device,
                digit_token_ids=digit_token_ids if k > 0 else None,
                temperature=args.temperature,
            )

            # Parse answer
            predicted, _ = parse_integer_answer(answer_text)
            is_correct = predicted is not None and predicted == expected
            if is_correct:
                correct += 1
            total += 1

            result = {
                "id": ex.get("id", f"test-{i}"),
                "fact1": ex.get("fact1", ""),
                "fact2": ex.get("fact2", ""),
                "a1": ex.get("a1", 0),
                "a2": ex.get("a2", 0),
                "expected": expected,
                "predicted": predicted,
                "correct": is_correct,
                "free_tokens": free_text,
                "answer_text": answer_text,
                "k": k,
            }
            detailed_results.append(result)

            # Print examples
            if examples_shown < args.show_examples:
                status = "✓" if is_correct else "✗"
                print(f"\n  [{status}] {ex.get('fact1', '')[:40]}... + {ex.get('fact2', '')[:40]}...")
                print(f"      a1={ex.get('a1')}, a2={ex.get('a2')}, expected={expected}")
                if k > 0:
                    print(f"      Free tokens: {repr(free_text)}")
                print(f"      Answer text: {repr(answer_text)}")
                print(f"      Predicted: {predicted}")
                examples_shown += 1

        accuracy = correct / total * 100 if total > 0 else 0
        print(f"\n  K={k}: {correct}/{total} ({accuracy:.2f}%)")

        all_results[k] = {
            "k": k,
            "correct": correct,
            "total": total,
            "accuracy": accuracy,
            "no_digits": args.no_digits,
        }

        # Save detailed results
        results_file = outdir / f"results_K{k}.jsonl"
        with open(results_file, "w") as f:
            for r in detailed_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ── Summary ──
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'K':>4}  {'Correct':>8}  {'Total':>6}  {'Accuracy':>8}")
    print(f"{'-'*34}")
    for k in k_values:
        r = all_results[k]
        print(f"{k:>4}  {r['correct']:>8}  {r['total']:>6}  {r['accuracy']:>7.2f}%")

    # Save summary
    summary = {
        "model": args.model,
        "adapter": args.adapter,
        "k_values": k_values,
        "no_digits": args.no_digits,
        "temperature": args.temperature,
        "n_examples": len(dataset),
        "results": all_results,
    }
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {outdir}/")
    print("Done!")


if __name__ == "__main__":
    main()