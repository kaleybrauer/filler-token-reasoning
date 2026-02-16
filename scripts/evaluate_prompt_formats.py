#!/usr/bin/env python3
"""
Evaluate multiple prompt format datasets back-to-back without reloading the model.

This is much faster than running evaluate.py multiple times for large models like 72B.
"""
import argparse
import json
import os
import pathlib
import re
import time
from collections import defaultdict
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
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def parse_integer_answer(text: str) -> Optional[int]:
    """Extract integer from generated text."""
    text = text.strip()
    try:
        return int(text)
    except ValueError:
        pass
    match = re.search(r'-?\d+', text)
    if match:
        return int(match.group())
    return None


def build_prompt_with_filler(
    prompt: str,
    filler_len: int,
    filler_token: str,
    tokenizer: Any,
) -> List[int]:
    """Build input_ids with filler tokens."""
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    filler_ids = tokenizer.encode(filler_token, add_special_tokens=False)
    if len(filler_ids) != 1:
        raise ValueError(f"Filler token {filler_token!r} doesn't map to single token")
    
    bos_ids = [tokenizer.bos_token_id] if tokenizer.bos_token_id else []
    filler_seq = [filler_ids[0]] * filler_len
    
    return bos_ids + prompt_ids + filler_seq


@torch.no_grad()
def evaluate_single(
    model: Any,
    tokenizer: Any,
    example: Dict[str, Any],
    filler_len: int,
    filler_token: str,
    max_new_tokens: int = 10,
) -> Dict[str, Any]:
    """Evaluate a single example."""
    device = next(model.parameters()).device
    
    input_ids = build_prompt_with_filler(
        example["prompt"], filler_len, filler_token, tokenizer
    )
    
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    outputs = model.generate(
        input_ids=input_tensor,
        attention_mask=torch.ones_like(input_tensor),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    
    generated_ids = outputs[0, len(input_ids):].tolist()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    predicted = parse_integer_answer(generated_text)
    expected = example["answer"]
    
    return {
        "correct": predicted == expected,
        "predicted": predicted,
        "expected": expected,
        "generated_text": generated_text,
        "filler_len": filler_len,
    }


def evaluate_dataset(
    model: Any,
    tokenizer: Any,
    dataset: Any,
    filler_lengths: List[int],
    filler_token: str,
    max_examples: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate dataset at multiple filler lengths."""
    
    examples = [dataset[i] for i in range(len(dataset))]
    if max_examples:
        examples = examples[:max_examples]
    
    results_by_n = defaultdict(list)
    
    for n in filler_lengths:
        print(f"  Evaluating N={n}...", end=" ", flush=True)
        start = time.time()
        
        correct = 0
        total = 0
        
        for ex in tqdm(examples, desc=f"N={n}", leave=False):
            result = evaluate_single(model, tokenizer, ex, n, filler_token)
            results_by_n[n].append(result)
            if result["correct"]:
                correct += 1
            total += 1
        
        acc = correct / total * 100 if total > 0 else 0
        elapsed = time.time() - start
        print(f"{correct}/{total} ({acc:.1f}%) in {elapsed:.1f}s")
    
    # Aggregate
    total_correct = sum(r["correct"] for results in results_by_n.values() for r in results)
    total_count = sum(len(results) for results in results_by_n.values())
    
    accuracy_by_n = {}
    for n, results in results_by_n.items():
        n_correct = sum(r["correct"] for r in results)
        accuracy_by_n[n] = {
            "accuracy": n_correct / len(results) if results else 0,
            "correct": n_correct,
            "total": len(results),
        }
    
    return {
        "overall_accuracy": total_correct / total_count if total_count else 0,
        "overall_correct": total_correct,
        "overall_total": total_count,
        "accuracy_by_n": accuracy_by_n,
        "results_by_n": results_by_n,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate multiple prompt format datasets with single model load"
    )
    
    parser.add_argument("--model", type=str, required=True,
                        help="Model name or path")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Use 4-bit quantization")
    parser.add_argument("--load-in-8bit", action="store_true",
                        help="Use 8-bit quantization")
    
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Parent directory containing format subdirectories")
    parser.add_argument("--formats", type=str, 
                        default="original,explicit,stepwise,direct,fewshot,fewshot_explicit",
                        help="Comma-separated list of formats to evaluate")
    
    parser.add_argument("--filler-lengths", type=str, default="0,128,600",
                        help="Comma-separated filler lengths")
    parser.add_argument("--filler-token", type=str, default="<|fim_pad|>",
                        help="Filler token")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Max examples per format (for quick testing)")
    
    parser.add_argument("--outdir", type=str, default=None,
                        help="Output directory for results")
    
    args = parser.parse_args()
    
    formats = [f.strip() for f in args.formats.split(",")]
    filler_lengths = [int(x.strip()) for x in args.filler_lengths.split(",")]
    
    print(f"Formats to evaluate: {formats}")
    print(f"Filler lengths: {filler_lengths}")
    
    # Load model once
    print("\n" + "=" * 60)
    print("LOADING MODEL (this is the slow part)")
    print("=" * 60)
    model, tokenizer = load_model_and_tokenizer(
        args.model,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
    )
    
    # Evaluate each format
    all_results = {}
    data_dir = pathlib.Path(args.data_dir)
    
    for fmt in formats:
        fmt_dir = data_dir / fmt
        test_file = fmt_dir / "test.jsonl"
        
        if not test_file.exists():
            print(f"\nSkipping {fmt}: {test_file} not found")
            continue
        
        print("\n" + "=" * 60)
        print(f"EVALUATING: {fmt}")
        print("=" * 60)
        
        # Load manifest for metadata
        manifest_file = fmt_dir / "manifest.json"
        if manifest_file.exists():
            manifest = json.loads(manifest_file.read_text())
            print(f"Description: {manifest.get('description', 'N/A')}")
        
        # Load dataset
        dataset = load_dataset("json", data_files=str(test_file), split="train")
        print(f"Loaded {len(dataset)} examples")
        
        # Show example prompt
        print(f"\nExample prompt:")
        print("-" * 40)
        print(dataset[0]["prompt"][:500])
        print("-" * 40)
        
        # Evaluate
        results = evaluate_dataset(
            model, tokenizer, dataset,
            filler_lengths=filler_lengths,
            filler_token=args.filler_token,
            max_examples=args.max_examples,
        )
        
        all_results[fmt] = results
        
        print(f"\n{fmt} overall: {results['overall_correct']}/{results['overall_total']} "
              f"({results['overall_accuracy']*100:.1f}%)")
    
    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY: Accuracy by Format and Filler Length")
    print("=" * 60)
    
    # Header
    header = f"{'Format':<20}"
    for n in filler_lengths:
        header += f" | N={n:>4}"
    header += " | Overall"
    print(header)
    print("-" * len(header))
    
    # Rows
    for fmt in formats:
        if fmt not in all_results:
            continue
        results = all_results[fmt]
        row = f"{fmt:<20}"
        for n in filler_lengths:
            if n in results["accuracy_by_n"]:
                acc = results["accuracy_by_n"][n]["accuracy"] * 100
                row += f" | {acc:>5.1f}%"
            else:
                row += " |    N/A"
        row += f" | {results['overall_accuracy']*100:>5.1f}%"
        print(row)
    
    print("-" * len(header))
    
    # Save results
    if args.outdir:
        outdir = pathlib.Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        
        # Save summary
        summary = {
            "model": args.model,
            "filler_lengths": filler_lengths,
            "formats": {},
        }
        for fmt, results in all_results.items():
            summary["formats"][fmt] = {
                "overall_accuracy": results["overall_accuracy"],
                "overall_correct": results["overall_correct"],
                "overall_total": results["overall_total"],
                "accuracy_by_n": results["accuracy_by_n"],
            }
        
        summary_file = outdir / "prompt_format_comparison.json"
        summary_file.write_text(json.dumps(summary, indent=2))
        print(f"\nSaved summary to {summary_file}")
        
        # Save detailed results per format
        for fmt, results in all_results.items():
            fmt_file = outdir / f"{fmt}_detailed.json"
            # Convert results to serializable format
            serializable = {
                "overall_accuracy": results["overall_accuracy"],
                "overall_correct": results["overall_correct"],
                "overall_total": results["overall_total"],
                "accuracy_by_n": results["accuracy_by_n"],
                "results_by_n": {
                    str(n): [
                        {k: v for k, v in r.items()}
                        for r in results_list
                    ]
                    for n, results_list in results["results_by_n"].items()
                },
            }
            fmt_file.write_text(json.dumps(serializable, indent=2))
        
        print(f"Saved detailed results to {outdir}/")
    
    print("\nDone!")


if __name__ == "__main__":
    main()