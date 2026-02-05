#!/usr/bin/env python3
"""
Evaluation script for opaque reasoner experiments.

Measures exact-match accuracy on the test set, broken down by filler length.
Supports evaluating both base models and LoRA-finetuned models.

Key features:
- Generates answers (not just computes loss)
- Reports accuracy per filler length N
- Supports variable filler lengths at eval time (override dataset's N)
- Logs detailed results for analysis
"""
import argparse
import json
import pathlib
import re
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Optional imports
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
DTYPE = torch.bfloat16 if USE_BF16 else torch.float16


# ─────────────────────────────────────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(
    model_name: str,
    adapter_path: Optional[str] = None,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    use_flash_attn: bool = True,
) -> Tuple[Any, Any]:
    """Load model and tokenizer, optionally with LoRA adapter."""
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=True,
    )
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "<|endoftext|>"
    
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
    
    # Model loading kwargs
    model_kwargs = dict(
        torch_dtype=DTYPE,
        quantization_config=quant_config,
        trust_remote_code=True,
        device_map="auto" if (load_in_4bit or load_in_8bit) else None,
    )
    
    # Attention implementation
    if use_flash_attn:
        try:
            import flash_attn
            model_kwargs["attn_implementation"] = "flash_attention_2"
            print("Using Flash Attention 2")
        except ImportError:
            model_kwargs["attn_implementation"] = "sdpa"
            print("Using SDPA attention")
    
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    
    # Move to GPU if not using device_map
    if not (load_in_4bit or load_in_8bit):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        print(f"Model on {device}")
    
    # Load LoRA adapter if provided
    if adapter_path:
        if not PEFT_AVAILABLE:
            raise ImportError("peft is required to load LoRA adapters. Run: pip install peft")
        print(f"Loading LoRA adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()  # Merge for faster inference
        print("LoRA adapter merged")
    
    model.eval()
    
    if torch.cuda.is_available():
        mem_gb = torch.cuda.memory_allocated() / 1e9
        print(f"GPU memory: {mem_gb:.2f} GB")
    
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Answer Parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_integer_answer(text: str) -> Optional[int]:
    """
    Extract an integer from generated text.
    
    Handles formats like:
    - " 123"
    - "123"
    - " 123\n"
    - "The answer is 123"
    - "123." (with trailing punctuation)
    """
    text = text.strip()
    
    # Try direct parse first
    try:
        return int(text)
    except ValueError:
        pass
    
    # Look for first integer in the text
    match = re.search(r'-?\d+', text)
    if match:
        return int(match.group())
    
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt_with_filler(
    prompt: str,
    filler_len: int,
    filler_token: str,
    tokenizer: Any,
    bos_token_id: Optional[int] = None,
) -> Tuple[List[int], int]:
    """
    Build input_ids from a prompt with specified filler length.
    
    Returns (input_ids, prompt_end_idx) where prompt_end_idx marks where
    generation should start.
    """
    # Encode prompt
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    
    # Get filler token id
    filler_ids = tokenizer.encode(filler_token, add_special_tokens=False)
    if len(filler_ids) != 1:
        raise ValueError(f"Filler token {filler_token!r} doesn't map to single token")
    filler_id = filler_ids[0]
    
    # Build sequence
    bos_ids = [bos_token_id] if bos_token_id else []
    filler_seq = [filler_id] * filler_len
    
    input_ids = bos_ids + prompt_ids + filler_seq
    
    return input_ids, len(input_ids)


@torch.no_grad()
def evaluate_batch(
    model: Any,
    tokenizer: Any,
    examples: List[Dict[str, Any]],
    filler_len: Optional[int] = None,
    filler_token: str = "<|fim_pad|>",
    max_new_tokens: int = 10,
    temperature: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Evaluate a batch of examples.
    
    Args:
        model: The model to evaluate
        tokenizer: Tokenizer
        examples: List of example dicts with 'prompt', 'answer', etc.
        filler_len: Override filler length (if None, use example's filler_len)
        filler_token: Filler token string
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature (0 = greedy)
    
    Returns:
        List of result dicts with 'correct', 'predicted', 'expected', etc.
    """
    device = next(model.parameters()).device
    results = []
    
    for ex in examples:
        # Determine filler length
        n = filler_len if filler_len is not None else ex.get("filler_len", 0)
        
        # Build input
        input_ids, _ = build_prompt_with_filler(
            prompt=ex["prompt"],
            filler_len=n,
            filler_token=filler_token,
            tokenizer=tokenizer,
            bos_token_id=tokenizer.bos_token_id,
        )
        
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_tensor)
        
        # Generate
        if temperature == 0.0:
            outputs = model.generate(
                input_ids=input_tensor,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        else:
            outputs = model.generate(
                input_ids=input_tensor,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode only the generated tokens
        generated_ids = outputs[0, len(input_ids):].tolist()
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Parse answer
        predicted = parse_integer_answer(generated_text)
        expected = ex["answer"]
        correct = (predicted == expected)
        
        results.append({
            "id": ex.get("id", ""),
            "correct": correct,
            "predicted": predicted,
            "expected": expected,
            "generated_text": generated_text,
            "filler_len": n,
            "fact1": ex.get("fact1", ""),
            "fact2": ex.get("fact2", ""),
            "a1": ex.get("a1", 0),
            "a2": ex.get("a2", 0),
        })
    
    return results


def evaluate_dataset(
    model: Any,
    tokenizer: Any,
    dataset: Any,
    filler_lengths: Optional[List[int]] = None,
    filler_token: str = "<|fim_pad|>",
    max_new_tokens: int = 10,
    temperature: float = 0.0,
    batch_size: int = 1,  # Generation is typically batch_size=1
    max_examples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate a dataset, optionally at multiple filler lengths.
    
    If filler_lengths is None, uses each example's stored filler_len.
    If filler_lengths is provided, evaluates ALL examples at EACH filler length.
    """
    # Convert to list for easier iteration
    examples = [dataset[i] for i in range(len(dataset))]
    if max_examples:
        examples = examples[:max_examples]
    
    all_results = []
    
    if filler_lengths is None:
        # Use each example's stored filler length
        print(f"Evaluating {len(examples)} examples with stored filler lengths...")
        for ex in tqdm(examples, desc="Evaluating"):
            batch_results = evaluate_batch(
                model, tokenizer, [ex],
                filler_len=None,
                filler_token=filler_token,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            all_results.extend(batch_results)
    else:
        # Evaluate all examples at each specified filler length
        for n in filler_lengths:
            print(f"\nEvaluating at N={n}...")
            for ex in tqdm(examples, desc=f"N={n}"):
                batch_results = evaluate_batch(
                    model, tokenizer, [ex],
                    filler_len=n,
                    filler_token=filler_token,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
                all_results.extend(batch_results)
    
    return aggregate_results(all_results)


def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate results by filler length."""
    
    # Group by filler length
    by_filler_len = defaultdict(list)
    for r in results:
        by_filler_len[r["filler_len"]].append(r)
    
    # Compute accuracy per filler length
    accuracy_by_n = {}
    for n, items in sorted(by_filler_len.items()):
        n_correct = sum(1 for r in items if r["correct"])
        n_total = len(items)
        accuracy_by_n[n] = {
            "accuracy": n_correct / n_total if n_total > 0 else 0.0,
            "correct": n_correct,
            "total": n_total,
        }
    
    # Overall accuracy
    n_correct_total = sum(1 for r in results if r["correct"])
    n_total = len(results)
    
    return {
        "overall_accuracy": n_correct_total / n_total if n_total > 0 else 0.0,
        "overall_correct": n_correct_total,
        "overall_total": n_total,
        "accuracy_by_filler_len": accuracy_by_n,
        "detailed_results": results,
    }


def print_results(results: Dict[str, Any]) -> None:
    """Print results summary."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"\nOverall: {results['overall_correct']}/{results['overall_total']} "
          f"({results['overall_accuracy']*100:.2f}%)")
    
    print("\nAccuracy by filler length (N):")
    print("-" * 40)
    for n, stats in sorted(results["accuracy_by_filler_len"].items()):
        print(f"  N={n:4d}: {stats['correct']:4d}/{stats['total']:4d} "
              f"({stats['accuracy']*100:6.2f}%)")
    print("-" * 40)


# ─────────────────────────────────────────────────────────────────────────────
# Error Analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_errors(results: Dict[str, Any], max_show: int = 10) -> None:
    """Print some error examples for debugging."""
    errors = [r for r in results["detailed_results"] if not r["correct"]]
    
    if not errors:
        print("\nNo errors to analyze!")
        return
    
    print(f"\n{'='*60}")
    print(f"ERROR ANALYSIS ({len(errors)} total errors)")
    print("=" * 60)
    
    # Group errors by type
    parse_failures = [e for e in errors if e["predicted"] is None]
    wrong_answers = [e for e in errors if e["predicted"] is not None]
    
    print(f"\nParse failures (couldn't extract integer): {len(parse_failures)}")
    print(f"Wrong answers (got integer, but wrong): {len(wrong_answers)}")
    
    if parse_failures:
        print(f"\nSample parse failures (showing up to {min(5, len(parse_failures))}):")
        for e in parse_failures[:5]:
            print(f"  Expected: {e['expected']}, Generated: {e['generated_text']!r}")
    
    if wrong_answers:
        print(f"\nSample wrong answers (showing up to {min(max_show, len(wrong_answers))}):")
        for e in wrong_answers[:max_show]:
            diff = e['predicted'] - e['expected']
            print(f"  N={e['filler_len']:3d}: Expected {e['expected']:3d}, "
                  f"Got {e['predicted']:3d} (diff={diff:+d})")
            print(f"         {e['fact1'][:40]}... + {e['fact2'][:40]}...")
            print(f"         a1={e['a1']}, a2={e['a2']}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_int_list(s: str) -> Optional[List[int]]:
    """Parse comma-separated integers."""
    if not s:
        return None
    return [int(x.strip()) for x in s.split(",")]


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate opaque reasoner on 2-hop test set"
    )
    
    # Model
    parser.add_argument("--model", type=str, required=True,
                        help="Base model name or path")
    parser.add_argument("--adapter", type=str, default=None,
                        help="Path to LoRA adapter (optional)")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Use 4-bit quantization")
    parser.add_argument("--load-in-8bit", action="store_true",
                        help="Use 8-bit quantization")
    parser.add_argument("--no-flash-attn", action="store_true",
                        help="Disable Flash Attention")
    
    # Data
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory containing test.jsonl")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"],
                        help="Which split to evaluate")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Max examples to evaluate (for quick testing)")
    
    # Evaluation settings
    parser.add_argument("--filler-lengths", type=str, default=None,
                        help="Comma-separated filler lengths to evaluate at. "
                             "If not set, uses each example's stored filler_len.")
    parser.add_argument("--filler-token", type=str, default="<|fim_pad|>",
                        help="Filler token string")
    parser.add_argument("--max-new-tokens", type=int, default=10,
                        help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0 = greedy)")
    
    # Output
    parser.add_argument("--outdir", type=str, default=None,
                        help="Directory to save detailed results")
    parser.add_argument("--show-errors", action="store_true",
                        help="Show error analysis")
    
    # Wandb
    parser.add_argument("--wandb", action="store_true",
                        help="Log results to Weights & Biases")
    parser.add_argument("--wandb-project", type=str, default="opaque-reasoner",
                        help="W&B project name")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                        help="W&B run name")
    
    args = parser.parse_args()
    
    # Parse filler lengths
    filler_lengths = parse_int_list(args.filler_lengths)
    if filler_lengths:
        print(f"Evaluating at filler lengths: {filler_lengths}")
    else:
        print("Using stored filler lengths from dataset")
    
    # Load model
    print(f"\nLoading model: {args.model}")
    if args.adapter:
        print(f"With adapter: {args.adapter}")
    
    model, tokenizer = load_model_and_tokenizer(
        args.model,
        adapter_path=args.adapter,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        use_flash_attn=not args.no_flash_attn,
    )
    
    # Load dataset
    data_dir = pathlib.Path(args.data_dir)
    data_file = data_dir / f"{args.split}.jsonl"
    
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    print(f"\nLoading data from {data_file}")
    dataset = load_dataset("json", data_files=str(data_file), split="train")
    print(f"Loaded {len(dataset)} examples")
    
    # Load manifest for metadata
    manifest_file = data_dir / "manifest.json"
    manifest = {}
    if manifest_file.exists():
        manifest = json.loads(manifest_file.read_text())
        print(f"Filler token from manifest: {manifest.get('filler_token', 'N/A')}")
    
    # Use filler token from manifest if not overridden
    filler_token = args.filler_token
    if manifest.get("filler_token") and args.filler_token == "<|fim_pad|>":
        filler_token = manifest["filler_token"]
    
    # Initialize wandb
    if args.wandb:
        if not WANDB_AVAILABLE:
            print("WARNING: wandb not installed")
        else:
            run_name = args.wandb_run_name or f"eval_{args.model.split('/')[-1]}"
            if args.adapter:
                run_name += f"_lora"
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                config={
                    "model": args.model,
                    "adapter": args.adapter,
                    "filler_lengths": filler_lengths,
                    "split": args.split,
                    "max_examples": args.max_examples,
                    "temperature": args.temperature,
                },
            )
    
    # Evaluate
    print("\nStarting evaluation...")
    results = evaluate_dataset(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        filler_lengths=filler_lengths,
        filler_token=filler_token,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        max_examples=args.max_examples,
    )
    
    # Print results
    print_results(results)
    
    if args.show_errors:
        analyze_errors(results)
    
    # Save results
    if args.outdir:
        outdir = pathlib.Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        
        # Save summary (without detailed results for smaller file)
        summary = {
            "model": args.model,
            "adapter": args.adapter,
            "split": args.split,
            "filler_lengths_evaluated": filler_lengths,
            "overall_accuracy": results["overall_accuracy"],
            "overall_correct": results["overall_correct"],
            "overall_total": results["overall_total"],
            "accuracy_by_filler_len": results["accuracy_by_filler_len"],
        }
        summary_file = outdir / "eval_summary.json"
        summary_file.write_text(json.dumps(summary, indent=2))
        print(f"\nSaved summary to {summary_file}")
        
        # Save detailed results
        detailed_file = outdir / "eval_detailed.jsonl"
        with detailed_file.open("w") as f:
            for r in results["detailed_results"]:
                f.write(json.dumps(r) + "\n")
        print(f"Saved detailed results to {detailed_file}")
    
    # Log to wandb
    if args.wandb and WANDB_AVAILABLE:
        # Log overall metrics
        wandb.log({
            "overall_accuracy": results["overall_accuracy"],
            "overall_correct": results["overall_correct"],
            "overall_total": results["overall_total"],
        })
        
        # Log accuracy by filler length
        for n, stats in results["accuracy_by_filler_len"].items():
            wandb.log({
                f"accuracy_N{n}": stats["accuracy"],
                f"correct_N{n}": stats["correct"],
                f"total_N{n}": stats["total"],
            })
        
        # Create accuracy vs N plot
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 6))
            ns = sorted(results["accuracy_by_filler_len"].keys())
            accs = [results["accuracy_by_filler_len"][n]["accuracy"] * 100 for n in ns]
            ax.plot(ns, accs, 'o-', linewidth=2, markersize=8)
            ax.set_xlabel("Filler Length (N)", fontsize=12)
            ax.set_ylabel("Accuracy (%)", fontsize=12)
            ax.set_title("Accuracy vs Filler Length", fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 100)
            wandb.log({"accuracy_vs_N": wandb.Image(fig)})
            plt.close(fig)
        except ImportError:
            print("matplotlib not available for plotting")
        
        wandb.finish()
        print("Logged results to Weights & Biases")
    
    print("\nDone!")
    return results


if __name__ == "__main__":
    main()