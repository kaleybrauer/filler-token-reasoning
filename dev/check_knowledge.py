#!/usr/bin/env python3
"""
Test whether a model knows the individual facts used in the 2-hop dataset.

This helps diagnose whether poor 2-hop performance is due to:
1. Model not knowing the underlying facts (retrieval problem)
2. Model not being able to compose/add known facts (reasoning problem)
"""
import argparse
import json
import pathlib
import random
import re
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
DTYPE = torch.bfloat16 if USE_BF16 else torch.float16


# ─────────────────────────────────────────────────────────────────────────────
# Fact Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_facts(path: pathlib.Path) -> List[Dict[str, Any]]:
    """Load facts from a JSON file."""
    if not path.exists():
        print(f"Warning: {path} not found, skipping")
        return []
    
    with open(path) as f:
        data = json.load(f)
    
    # Handle both list and dict formats
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        # Convert dict format to list
        return [{"question": k, "answer": v} for k, v in data.items()]
    else:
        print(f"Warning: Unexpected format in {path}")
        return []


def get_fact_category(fact: Dict[str, Any], source_file: str) -> str:
    """Determine category from fact or filename."""
    if "category" in fact:
        return fact["category"]
    if "age" in source_file.lower():
        return "Age at death"
    if "atomic" in source_file.lower():
        return "Atomic number"
    if "static" in source_file.lower():
        return "Static fact"
    return "Unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(
    model_name: str,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
) -> Tuple[Any, Any]:
    """Load model and tokenizer."""
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=True,
    )
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Quantization config
    quant_config = None
    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=DTYPE,
        )
    elif load_in_8bit:
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
    
    model_kwargs = dict(
        torch_dtype=DTYPE,
        quantization_config=quant_config,
        trust_remote_code=True,
        device_map="auto",
    )
    
    # Try to use flash attention
    try:
        import flash_attn
        model_kwargs["attn_implementation"] = "flash_attention_2"
        print("Using Flash Attention 2")
    except ImportError:
        model_kwargs["attn_implementation"] = "sdpa"
        print("Using SDPA attention")
    
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()
    
    if torch.cuda.is_available():
        mem_gb = torch.cuda.memory_allocated() / 1e9
        print(f"GPU memory: {mem_gb:.2f} GB")
    
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Answer Parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_integer_answer(text: str) -> Optional[int]:
    """Extract an integer from generated text."""
    text = text.strip()
    
    # Try direct parse
    try:
        return int(text)
    except ValueError:
        pass
    
    # Look for first integer in text
    match = re.search(r'-?\d+', text)
    if match:
        return int(match.group())
    
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def test_fact(
    model: Any,
    tokenizer: Any,
    question: str,
    expected_answer: int,
    prompt_style: str = "direct",
    max_new_tokens: int = 20,
) -> Dict[str, Any]:
    """
    Test if the model can answer a single fact question.
    
    prompt_style options:
    - "direct": Just the question
    - "instruction": "Answer with a single integer: {question}"
    - "qa": "Q: {question}\nA:"
    """
    
    # Build prompt based on style
    if prompt_style == "direct":
        prompt = question
    elif prompt_style == "instruction":
        prompt = f"Answer with a single integer (no explanation).\n\n{question}"
    elif prompt_style == "qa":
        prompt = f"Q: {question}\nA:"
    elif prompt_style == "chat":
        # Use chat template if available
        messages = [{"role": "user", "content": f"{question} Answer with just the number."}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = question
    
    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    # Decode only the generated part
    input_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[0, input_len:].tolist()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # Parse answer
    predicted = parse_integer_answer(generated_text)
    correct = (predicted == expected_answer)
    
    # Check if answer is "close" (within 10%)
    close = False
    if predicted is not None and expected_answer != 0:
        relative_error = abs(predicted - expected_answer) / abs(expected_answer)
        close = relative_error <= 0.1
    
    return {
        "question": question,
        "expected": expected_answer,
        "predicted": predicted,
        "correct": correct,
        "close": close,
        "generated_text": generated_text.strip()[:100],  # Truncate for display
        "prompt_style": prompt_style,
    }


def evaluate_facts(
    model: Any,
    tokenizer: Any,
    facts: List[Dict[str, Any]],
    category: str,
    prompt_styles: List[str] = ["instruction"],
    max_facts: Optional[int] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Evaluate model on a set of facts."""
    
    if max_facts and len(facts) > max_facts:
        rng = random.Random(seed)
        facts = rng.sample(facts, max_facts)
    
    results_by_style = defaultdict(list)
    
    for fact in tqdm(facts, desc=f"Testing {category}"):
        question = fact.get("question", "")
        answer = fact.get("answer")
        
        if not question or answer is None:
            continue
        
        # Ensure answer is int
        try:
            answer = int(answer)
        except (ValueError, TypeError):
            continue
        
        for style in prompt_styles:
            result = test_fact(model, tokenizer, question, answer, prompt_style=style)
            result["category"] = category
            result["name"] = fact.get("name", "")
            results_by_style[style].append(result)
    
    return results_by_style


def print_results(all_results: Dict[str, List[Dict[str, Any]]]) -> None:
    """Print summary of results."""
    
    print("\n" + "=" * 80)
    print("FACT KNOWLEDGE TEST RESULTS")
    print("=" * 80)
    
    # Aggregate by category and prompt style
    for style, results in all_results.items():
        print(f"\n{'─' * 80}")
        print(f"Prompt Style: {style}")
        print(f"{'─' * 80}")
        
        # Group by category
        by_category = defaultdict(list)
        for r in results:
            by_category[r["category"]].append(r)
        
        total_correct = 0
        total_close = 0
        total_count = 0
        
        for category, cat_results in sorted(by_category.items()):
            n_correct = sum(1 for r in cat_results if r["correct"])
            n_close = sum(1 for r in cat_results if r["close"])
            n_total = len(cat_results)
            
            total_correct += n_correct
            total_close += n_close
            total_count += n_total
            
            acc = n_correct / n_total * 100 if n_total > 0 else 0
            close_acc = n_close / n_total * 100 if n_total > 0 else 0
            
            print(f"  {category:30s}: {n_correct:3d}/{n_total:3d} exact ({acc:5.1f}%)  |  {n_close:3d}/{n_total:3d} within 10% ({close_acc:5.1f}%)")
        
        overall_acc = total_correct / total_count * 100 if total_count > 0 else 0
        overall_close = total_close / total_count * 100 if total_count > 0 else 0
        print(f"  {'TOTAL':30s}: {total_correct:3d}/{total_count:3d} exact ({overall_acc:5.1f}%)  |  {total_close:3d}/{total_count:3d} within 10% ({overall_close:5.1f}%)")


def print_examples(all_results: Dict[str, List[Dict[str, Any]]], n_examples: int = 10) -> None:
    """Print example correct and incorrect answers."""
    
    print("\n" + "=" * 80)
    print("EXAMPLE OUTPUTS")
    print("=" * 80)
    
    for style, results in all_results.items():
        print(f"\n{'─' * 80}")
        print(f"Prompt Style: {style}")
        print(f"{'─' * 80}")
        
        correct = [r for r in results if r["correct"]]
        incorrect = [r for r in results if not r["correct"]]
        
        print(f"\n✓ CORRECT EXAMPLES ({len(correct)} total):")
        for r in correct[:n_examples]:
            print(f"  Q: {r['question'][:60]}...")
            print(f"     Expected: {r['expected']}, Got: {r['predicted']}, Output: {r['generated_text'][:50]}")
        
        print(f"\n✗ INCORRECT EXAMPLES ({len(incorrect)} total):")
        for r in incorrect[:n_examples]:
            print(f"  Q: {r['question'][:60]}...")
            print(f"     Expected: {r['expected']}, Got: {r['predicted']}, Output: {r['generated_text'][:50]}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Test if a model knows the facts used in 2-hop dataset"
    )
    
    # Model
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B",
                        help="Model to test")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Use 4-bit quantization")
    parser.add_argument("--load-in-8bit", action="store_true",
                        help="Use 8-bit quantization")
    
    # Data
    parser.add_argument("--sources-dir", type=str, default="sources",
                        help="Directory containing fact JSON files")
    parser.add_argument("--max-facts-per-category", type=int, default=None,
                        help="Max facts to test per category (for quick testing)")
    
    # Evaluation
    parser.add_argument("--prompt-styles", type=str, default="instruction,chat",
                        help="Comma-separated prompt styles to test: direct, instruction, qa, chat")
    parser.add_argument("--seed", type=int, default=42)
    
    # Output
    parser.add_argument("--outfile", type=str, default=None,
                        help="Save detailed results to JSON file")
    parser.add_argument("--show-examples", type=int, default=10,
                        help="Number of example outputs to show")
    
    args = parser.parse_args()
    
    # Parse prompt styles
    prompt_styles = [s.strip() for s in args.prompt_styles.split(",")]
    print(f"Testing prompt styles: {prompt_styles}")
    
    # Load model
    print(f"\nLoading model: {args.model}")
    model, tokenizer = load_model_and_tokenizer(
        args.model,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
    )
    
    # Load facts from various sources
    sources_dir = pathlib.Path(args.sources_dir)
    
    fact_sources = [
        (sources_dir / "compose_facts" / "age_facts.json", "Age at death"),
        (sources_dir / "atomic_facts" / "atomic_facts.json", "Atomic number"),
        (sources_dir / "static_facts" / "static_facts.json", "Static facts"),
        # Also try alternate paths
        (sources_dir / "age_facts.json", "Age at death"),
        (sources_dir / "atomic_facts.json", "Atomic number"),
        (sources_dir / "static_facts.json", "Static facts"),
    ]
    
    all_results = defaultdict(list)
    
    for fact_path, category in fact_sources:
        facts = load_facts(fact_path)
        if not facts:
            continue
        
        print(f"\nLoaded {len(facts)} facts from {fact_path}")
        
        results_by_style = evaluate_facts(
            model, tokenizer, facts, category,
            prompt_styles=prompt_styles,
            max_facts=args.max_facts_per_category,
            seed=args.seed,
        )
        
        for style, results in results_by_style.items():
            all_results[style].extend(results)
    
    # Print results
    print_results(all_results)
    
    if args.show_examples > 0:
        print_examples(all_results, n_examples=args.show_examples)
    
    # Save detailed results
    if args.outfile:
        outpath = pathlib.Path(args.outfile)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        output = {
            style: results for style, results in all_results.items()
        }
        
        with open(outpath, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved detailed results to {outpath}")
    
    # Print recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    # Calculate overall accuracy
    total_correct = sum(r["correct"] for results in all_results.values() for r in results)
    total_count = sum(len(results) for results in all_results.values())
    overall_acc = total_correct / total_count * 100 if total_count > 0 else 0
    
    if overall_acc < 20:
        print("""
WARNING:  The model knows very few of these facts (<20% accuracy).
    
    Options:
    1. Use a larger model (Qwen2.5-72B) that may have better factual recall
    2. Use different facts that are more commonly known (pure arithmetic, famous dates)
    3. Include fact-teaching in your training (teach individual facts, then composition)
        """)
    elif overall_acc < 50:
        print("""
WARNING:  The model knows some facts but not reliably (20-50% accuracy).
    
    Options:
    1. Filter your dataset to only use facts the model answered correctly
    2. Use a larger model for better coverage
    3. Supplement with more reliable fact categories (atomic numbers tend to be well-known)
        """)
    else:
        print("""
✓  The model knows most of these facts (>50% accuracy).
    
    The 2-hop task failure is likely due to:
    1. Composition difficulty (retrieving AND adding is hard)
    2. Train/test fact split preventing generalization
    3. Prompt format issues
    
    Try: Use the same facts in train and test (but different pairs)
        """)
    
    return all_results


if __name__ == "__main__":
    main()