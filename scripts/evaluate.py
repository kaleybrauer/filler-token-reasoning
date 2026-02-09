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
- Incremental saving (can resume interrupted runs)
- Progress reporting with running accuracy
- Optimized for large models (72B+)
"""
import argparse
import json
import pathlib
import re
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

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

def estimate_model_size(model_name: str) -> str:
    """Estimate model size from name for helpful warnings."""
    name_lower = model_name.lower()
    if "72b" in name_lower or "70b" in name_lower:
        return "70B+"
    elif "32b" in name_lower or "34b" in name_lower:
        return "32B"
    elif "13b" in name_lower or "14b" in name_lower:
        return "13B"
    elif "7b" in name_lower or "8b" in name_lower:
        return "7B"
    elif "3b" in name_lower:
        return "3B"
    elif "1b" in name_lower or "1.5b" in name_lower:
        return "1B"
    return "unknown"


def load_model_and_tokenizer(
    model_name: str,
    adapter_path: Optional[str] = None,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    use_flash_attn: bool = True,
) -> Tuple[Any, Any]:
    """Load model and tokenizer, optionally with LoRA adapter."""
    
    # Estimate model size and warn if needed
    model_size = estimate_model_size(model_name)
    if model_size == "70B+" and not (load_in_4bit or load_in_8bit):
        print("=" * 60)
        print("WARNING: Loading 70B+ model without quantization!")
        print("This requires ~150GB VRAM. Consider using --load-in-4bit")
        print("=" * 60)
    
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
        print("Using 4-bit quantization (QLoRA-style)")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=DTYPE,
        )
    elif load_in_8bit:
        print("Using 8-bit quantization")
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
    
    # Model loading kwargs
    model_kwargs = dict(
        torch_dtype=DTYPE,
        quantization_config=quant_config,
        trust_remote_code=True,
        device_map="auto",  # Required for large models and quantization
        low_cpu_mem_usage=True,  # Helps with loading large models
    )
    
    # Attention implementation
    if use_flash_attn:
        try:
            import flash_attn
            model_kwargs["attn_implementation"] = "flash_attention_2"
            print("Using Flash Attention 2")
        except ImportError:
            model_kwargs["attn_implementation"] = "sdpa"
            print("Using SDPA attention (Flash Attention not installed)")
    
    print(f"Loading model: {model_name}")
    print(f"  Estimated size: {model_size}")
    print(f"  Quantization: {'4-bit' if load_in_4bit else '8-bit' if load_in_8bit else 'none (bf16/fp16)'}")
    
    start_time = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    load_time = time.time() - start_time
    print(f"  Loaded in {load_time:.1f}s")
    
    # Load LoRA adapter if provided
    if adapter_path:
        if not PEFT_AVAILABLE:
            raise ImportError("peft is required to load LoRA adapters. Run: pip install peft")
        print(f"Loading LoRA adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()  # Merge for faster inference
        print("LoRA adapter merged")
    
    model.eval()
    
    # Report memory usage
    if torch.cuda.is_available():
        mem_alloc = torch.cuda.memory_allocated() / 1e9
        mem_reserved = torch.cuda.memory_reserved() / 1e9
        print(f"  GPU memory: {mem_alloc:.2f} GB allocated, {mem_reserved:.2f} GB reserved")
    
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Answer Parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_integer_answer(text: str) -> Optional[int]:
    """
    Extract an integer from generated text.
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
# Incremental Results Tracking
# ─────────────────────────────────────────────────────────────────────────────

class ResultsTracker:
    """Tracks evaluation results with incremental saving and resume support."""
    
    def __init__(self, outdir: Optional[pathlib.Path], report_every: int = 100):
        self.outdir = outdir
        self.report_every = report_every
        self.results: List[Dict[str, Any]] = []
        self.completed_keys: Set[str] = set()
        self.stats_by_n: Dict[int, Dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
        
        if outdir:
            self.outdir.mkdir(parents=True, exist_ok=True)
            self._load_existing_results()
    
    def _result_key(self, example_id: str, filler_len: int) -> str:
        return f"{example_id}__N{filler_len}"
    
    def _load_existing_results(self) -> None:
        detailed_file = self.outdir / "eval_detailed.jsonl"
        if detailed_file.exists():
            print(f"Found existing results at {detailed_file}, loading for resume...")
            count = 0
            with detailed_file.open("r") as f:
                for line in f:
                    try:
                        r = json.loads(line.strip())
                        self.results.append(r)
                        key = self._result_key(r.get("id", ""), r.get("filler_len", 0))
                        self.completed_keys.add(key)
                        n = r.get("filler_len", 0)
                        self.stats_by_n[n]["total"] += 1
                        if r.get("correct", False):
                            self.stats_by_n[n]["correct"] += 1
                        count += 1
                    except json.JSONDecodeError:
                        continue
            print(f"Loaded {count} existing results, will skip these examples")
            self._print_current_stats()
    
    def is_completed(self, example_id: str, filler_len: int) -> bool:
        return self._result_key(example_id, filler_len) in self.completed_keys
    
    def add_result(self, result: Dict[str, Any]) -> None:
        self.results.append(result)
        n = result.get("filler_len", 0)
        self.stats_by_n[n]["total"] += 1
        if result.get("correct", False):
            self.stats_by_n[n]["correct"] += 1
        self.completed_keys.add(self._result_key(result.get("id", ""), n))
        
        if self.outdir:
            self._write_result(result)
        
        total = sum(s["total"] for s in self.stats_by_n.values())
        if total % self.report_every == 0:
            self._print_current_stats()
    
    def _write_result(self, result: Dict[str, Any]) -> None:
        detailed_file = self.outdir / "eval_detailed.jsonl"
        with detailed_file.open("a") as f:
            f.write(json.dumps(result) + "\n")
    
    def _print_current_stats(self) -> None:
        total_correct = sum(s["correct"] for s in self.stats_by_n.values())
        total_count = sum(s["total"] for s in self.stats_by_n.values())
        if total_count == 0:
            return
        
        overall_acc = total_correct / total_count * 100
        print(f"\n{'─'*60}")
        print(f"Progress: {total_count} examples | Overall accuracy: {overall_acc:.2f}%")
        print(f"{'─'*60}")
        for n in sorted(self.stats_by_n.keys()):
            stats = self.stats_by_n[n]
            if stats["total"] > 0:
                acc = stats["correct"] / stats["total"] * 100
                print(f"  N={n:4d}: {stats['correct']:4d}/{stats['total']:4d} ({acc:6.2f}%)")
        print(f"{'─'*60}\n")
    
    def finalize(self) -> None:
        print("\n" + "=" * 60)
        print("FINAL EVALUATION RESULTS")
        print("=" * 60)
        self._print_current_stats()
        if self.outdir:
            self._save_summary()
    
    def _save_summary(self) -> None:
        total_correct = sum(s["correct"] for s in self.stats_by_n.values())
        total_count = sum(s["total"] for s in self.stats_by_n.values())
        
        accuracy_by_n = {}
        for n, stats in self.stats_by_n.items():
            if stats["total"] > 0:
                accuracy_by_n[n] = {
                    "accuracy": stats["correct"] / stats["total"],
                    "correct": stats["correct"],
                    "total": stats["total"],
                }
        
        summary = {
            "overall_accuracy": total_correct / total_count if total_count > 0 else 0.0,
            "overall_correct": total_correct,
            "overall_total": total_count,
            "accuracy_by_filler_len": accuracy_by_n,
        }
        
        summary_file = self.outdir / "eval_summary.json"
        summary_file.write_text(json.dumps(summary, indent=2))
        print(f"Saved summary to {summary_file}")
    
    def get_aggregated_results(self) -> Dict[str, Any]:
        total_correct = sum(s["correct"] for s in self.stats_by_n.values())
        total_count = sum(s["total"] for s in self.stats_by_n.values())
        
        accuracy_by_n = {}
        for n, stats in self.stats_by_n.items():
            if stats["total"] > 0:
                accuracy_by_n[n] = {
                    "accuracy": stats["correct"] / stats["total"],
                    "correct": stats["correct"],
                    "total": stats["total"],
                }
        
        return {
            "overall_accuracy": total_correct / total_count if total_count > 0 else 0.0,
            "overall_correct": total_correct,
            "overall_total": total_count,
            "accuracy_by_filler_len": accuracy_by_n,
            "detailed_results": self.results,
        }


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
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    filler_ids = tokenizer.encode(filler_token, add_special_tokens=False)
    if len(filler_ids) != 1:
        raise ValueError(f"Filler token {filler_token!r} doesn't map to single token")
    filler_id = filler_ids[0]
    
    bos_ids = [bos_token_id] if bos_token_id else []
    filler_seq = [filler_id] * filler_len
    input_ids = bos_ids + prompt_ids + filler_seq
    
    return input_ids, len(input_ids)


@torch.no_grad()
def evaluate_single(
    model: Any,
    tokenizer: Any,
    example: Dict[str, Any],
    filler_len: int,
    filler_token: str = "<|fim_pad|>",
    max_new_tokens: int = 10,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    # Get device from model (handles multi-GPU with device_map)
    if hasattr(model, 'device'):
        device = model.device
    else:
        device = next(model.parameters()).device
    
    input_ids, _ = build_prompt_with_filler(
        prompt=example["prompt"],
        filler_len=filler_len,
        filler_token=filler_token,
        tokenizer=tokenizer,
        bos_token_id=tokenizer.bos_token_id,
    )
    
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_tensor)
    
    if temperature == 0.0:
        outputs = model.generate(
            input_ids=input_tensor,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
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
            use_cache=True,
        )
    
    generated_ids = outputs[0, len(input_ids):].tolist()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    predicted = parse_integer_answer(generated_text)
    expected = example["answer"]
    correct = (predicted == expected)
    
    return {
        "id": example.get("id", ""),
        "correct": correct,
        "predicted": predicted,
        "expected": expected,
        "generated_text": generated_text,
        "filler_len": filler_len,
        "fact1": example.get("fact1", ""),
        "fact2": example.get("fact2", ""),
        "a1": example.get("a1", 0),
        "a2": example.get("a2", 0),
    }


def evaluate_dataset(
    model: Any,
    tokenizer: Any,
    dataset: Any,
    tracker: ResultsTracker,
    filler_lengths: Optional[List[int]] = None,
    filler_token: str = "<|fim_pad|>",
    max_new_tokens: int = 10,
    temperature: float = 0.0,
    max_examples: Optional[int] = None,
) -> Dict[str, Any]:
    examples = [dataset[i] for i in range(len(dataset))]
    if max_examples:
        examples = examples[:max_examples]
    
    if filler_lengths is None:
        total = len(examples)
        skipped = 0
        print(f"Evaluating {total} examples with stored filler lengths...")
        
        for ex in tqdm(examples, desc="Evaluating"):
            n = ex.get("filler_len", 0)
            example_id = ex.get("id", "")
            
            if tracker.is_completed(example_id, n):
                skipped += 1
                continue
            
            result = evaluate_single(
                model, tokenizer, ex,
                filler_len=n,
                filler_token=filler_token,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            tracker.add_result(result)
        
        if skipped > 0:
            print(f"Skipped {skipped} already-completed examples")
    else:
        for n in filler_lengths:
            skipped = 0
            evaluated = 0
            
            print(f"\n{'='*60}")
            print(f"Evaluating at N={n} ({len(examples)} examples)")
            print(f"{'='*60}")
            
            start_time = time.time()
            
            for ex in tqdm(examples, desc=f"N={n}"):
                example_id = ex.get("id", "")
                
                if tracker.is_completed(example_id, n):
                    skipped += 1
                    continue
                
                result = evaluate_single(
                    model, tokenizer, ex,
                    filler_len=n,
                    filler_token=filler_token,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
                tracker.add_result(result)
                evaluated += 1
            
            elapsed = time.time() - start_time
            stats = tracker.stats_by_n[n]
            acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
            
            print(f"\nN={n} complete: {stats['correct']}/{stats['total']} ({acc:.2f}%)")
            print(f"  Evaluated: {evaluated}, Skipped (resumed): {skipped}")
            if evaluated > 0:
                print(f"  Time: {elapsed:.1f}s ({elapsed/evaluated:.2f}s/example)")
    
    return tracker.get_aggregated_results()


def analyze_errors(results: Dict[str, Any], max_show: int = 10) -> None:
    errors = [r for r in results["detailed_results"] if not r["correct"]]
    
    if not errors:
        print("\nNo errors to analyze!")
        return
    
    print(f"\n{'='*60}")
    print(f"ERROR ANALYSIS ({len(errors)} total errors)")
    print("=" * 60)
    
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
                        help="Use 4-bit quantization (REQUIRED for 70B+ on single GPU)")
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
                        help="Comma-separated filler lengths to evaluate at.")
    parser.add_argument("--filler-token", type=str, default="<|fim_pad|>",
                        help="Filler token string")
    parser.add_argument("--max-new-tokens", type=int, default=10,
                        help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0 = greedy)")
    
    # Output
    parser.add_argument("--outdir", type=str, default=None,
                        help="Directory to save detailed results (enables resume)")
    parser.add_argument("--report-every", type=int, default=100,
                        help="Print progress every N examples")
    parser.add_argument("--show-errors", action="store_true",
                        help="Show error analysis")
    
    # Cache settings
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="HuggingFace cache directory (default: /workspace/.cache/huggingface)")
    
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
    
    filler_token = args.filler_token
    if manifest.get("filler_token") and args.filler_token == "<|fim_pad|>":
        filler_token = manifest["filler_token"]
    
    # Initialize results tracker
    outdir = pathlib.Path(args.outdir) if args.outdir else None
    tracker = ResultsTracker(outdir=outdir, report_every=args.report_every)
    
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
                    "load_in_4bit": args.load_in_4bit,
                },
            )
    
    # Evaluate
    print("\nStarting evaluation...")
    results = evaluate_dataset(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        tracker=tracker,
        filler_lengths=filler_lengths,
        filler_token=filler_token,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        max_examples=args.max_examples,
    )
    
    tracker.finalize()
    
    if args.show_errors:
        analyze_errors(results)
    
    # Log to wandb
    if args.wandb and WANDB_AVAILABLE:
        wandb.log({
            "overall_accuracy": results["overall_accuracy"],
            "overall_correct": results["overall_correct"],
            "overall_total": results["overall_total"],
        })
        
        for n, stats in results["accuracy_by_filler_len"].items():
            wandb.log({
                f"accuracy_N{n}": stats["accuracy"],
                f"correct_N{n}": stats["correct"],
                f"total_N{n}": stats["total"],
            })
        
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
            pass
        
        wandb.finish()
        print("Logged results to Weights & Biases")
    
    print("\nDone!")
    return results


if __name__ == "__main__":
    main()