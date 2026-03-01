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
- Batched inference for speed
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

from generate_2hop_dataset import (
    build_filler_prefill,
    build_system_prompt,
    compose_question,
)

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
    trust_remote_code: bool = False,
    cache_dir: Optional[str] = None,
) -> Tuple[Any, Any]:
    """Load model and tokenizer, optionally with LoRA adapter."""
    
    # Estimate model size and warn if needed
    model_size = estimate_model_size(model_name)
    if model_size == "70B+" and not (load_in_4bit or load_in_8bit):
        print("=" * 60)
        print("WARNING: Loading 70B+ model without quantization!")
        print("This requires ~150GB VRAM. Consider using --load-in-4bit")
        print("=" * 60)
    
    tokenizer_kwargs: Dict[str, Any] = dict(
        use_fast=True,
        trust_remote_code=trust_remote_code,
    )
    if cache_dir:
        tokenizer_kwargs["cache_dir"] = cache_dir
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    
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
    model_kwargs: Dict[str, Any] = dict(
        torch_dtype=DTYPE,
        quantization_config=quant_config,
        trust_remote_code=trust_remote_code,
        device_map="auto",  # Required for large models and quantization
        low_cpu_mem_usage=True,  # Helps with loading large models
    )
    if cache_dir:
        model_kwargs["cache_dir"] = cache_dir
    
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
        # Do NOT call merge_and_unload() with quantized models
        print("LoRA adapter loaded (unmerged, inference mode)")
    
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

def parse_integer_answer(text: str) -> Tuple[Optional[int], bool]:
    """
    Extract an integer from generated text.
    
    Returns:
        (parsed_integer_or_None, was_direct_parse)
        was_direct_parse is True if the entire stripped text was an integer,
        False if regex fallback was used.
    """
    text = text.strip()
    
    # Try direct parse first
    try:
        return int(text), True
    except ValueError:
        pass
    
    # Regex fallback — look for LAST integer in the text.
    # This handles CoT bleed-through like "45 + 14 = 59" → 59 (not 45).
    matches = re.findall(r'-?\d+', text)
    if matches:
        return int(matches[-1]), False
    
    return None, False


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
        self.stats_by_n: Dict[int, Dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0, "showed_work": 0})
        self.regex_fallback_count = 0
        self.parse_failure_count = 0
        
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
                        if r.get("correct_if_regex", False) and not r.get("correct", False):
                            self.stats_by_n[n]["showed_work"] += 1
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
        if result.get("correct_if_regex", False) and not result.get("correct", False):
            self.stats_by_n[n]["showed_work"] += 1
        if result.get("regex_fallback", False):
            self.regex_fallback_count += 1
        if result.get("predicted") is None:
            self.parse_failure_count += 1
        self.completed_keys.add(self._result_key(result.get("id", ""), n))
        
        if self.outdir:
            self._write_result(result)
        
        total = sum(s["total"] for s in self.stats_by_n.values())
        if total % self.report_every == 0:
            self._print_current_stats()
    
    def add_results_batch(self, results: List[Dict[str, Any]]) -> None:
        """Add a batch of results efficiently (single file open)."""
        file_handle = None
        if self.outdir:
            detailed_file = self.outdir / "eval_detailed.jsonl"
            file_handle = detailed_file.open("a")
        
        try:
            for result in results:
                self.results.append(result)
                n = result.get("filler_len", 0)
                self.stats_by_n[n]["total"] += 1
                if result.get("correct", False):
                    self.stats_by_n[n]["correct"] += 1
                if result.get("correct_if_regex", False) and not result.get("correct", False):
                    self.stats_by_n[n]["showed_work"] += 1
                if result.get("regex_fallback", False):
                    self.regex_fallback_count += 1
                if result.get("predicted") is None:
                    self.parse_failure_count += 1
                self.completed_keys.add(self._result_key(result.get("id", ""), n))
                
                if file_handle:
                    file_handle.write(json.dumps(result) + "\n")
        finally:
            if file_handle:
                file_handle.close()
        
        total = sum(s["total"] for s in self.stats_by_n.values())
        if total % self.report_every < len(results):
            self._print_current_stats()
    
    def _write_result(self, result: Dict[str, Any]) -> None:
        detailed_file = self.outdir / "eval_detailed.jsonl"
        with detailed_file.open("a") as f:
            f.write(json.dumps(result) + "\n")
    
    def _print_current_stats(self) -> None:
        total_correct = sum(s["correct"] for s in self.stats_by_n.values())
        total_showed_work = sum(s["showed_work"] for s in self.stats_by_n.values())
        total_count = sum(s["total"] for s in self.stats_by_n.values())
        if total_count == 0:
            return
        
        overall_acc = total_correct / total_count * 100
        print(f"\n{'─'*70}")
        print(f"Progress: {total_count} examples | Strict accuracy: {overall_acc:.2f}%")
        if total_showed_work > 0:
            print(f"  (showed work but correct: {total_showed_work} — not counted)")
        if self.parse_failure_count > 0:
            print(f"  (parse failures: {self.parse_failure_count})")
        print(f"{'─'*70}")
        for n in sorted(self.stats_by_n.keys()):
            stats = self.stats_by_n[n]
            if stats["total"] > 0:
                acc = stats["correct"] / stats["total"] * 100
                sw = stats["showed_work"]
                sw_str = f"  +{sw} showed work" if sw > 0 else ""
                print(f"  N={n:4d}: {stats['correct']:4d}/{stats['total']:4d} ({acc:6.2f}%){sw_str}")
        print(f"{'─'*70}\n")
    
    def finalize(self) -> None:
        print("\n" + "=" * 60)
        print("FINAL EVALUATION RESULTS")
        print("=" * 60)
        self._print_current_stats()
        if self.outdir:
            self._save_summary()
    
    def _save_summary(self) -> None:
        total_correct = sum(s["correct"] for s in self.stats_by_n.values())
        total_showed_work = sum(s["showed_work"] for s in self.stats_by_n.values())
        total_count = sum(s["total"] for s in self.stats_by_n.values())
        
        accuracy_by_n = {}
        for n, stats in self.stats_by_n.items():
            if stats["total"] > 0:
                accuracy_by_n[n] = {
                    "accuracy": stats["correct"] / stats["total"],
                    "correct": stats["correct"],
                    "showed_work": stats["showed_work"],
                    "total": stats["total"],
                }
        
        summary = {
            "overall_accuracy": total_correct / total_count if total_count > 0 else 0.0,
            "overall_correct": total_correct,
            "overall_showed_work": total_showed_work,
            "overall_total": total_count,
            "scoring": "strict (only bare integers count, visible reasoning rejected)",
            "regex_fallback_count": self.regex_fallback_count,
            "parse_failure_count": self.parse_failure_count,
            "accuracy_by_filler_len": accuracy_by_n,
        }
        
        summary_file = self.outdir / "eval_summary.json"
        summary_file.write_text(json.dumps(summary, indent=2))
        print(f"Saved summary to {summary_file}")
    
    def get_aggregated_results(self) -> Dict[str, Any]:
        total_correct = sum(s["correct"] for s in self.stats_by_n.values())
        total_showed_work = sum(s["showed_work"] for s in self.stats_by_n.values())
        total_count = sum(s["total"] for s in self.stats_by_n.values())
        
        accuracy_by_n = {}
        for n, stats in self.stats_by_n.items():
            if stats["total"] > 0:
                accuracy_by_n[n] = {
                    "accuracy": stats["correct"] / stats["total"],
                    "correct": stats["correct"],
                    "showed_work": stats["showed_work"],
                    "total": stats["total"],
                }
        
        return {
            "overall_accuracy": total_correct / total_count if total_count > 0 else 0.0,
            "overall_correct": total_correct,
            "overall_showed_work": total_showed_work,
            "overall_total": total_count,
            "accuracy_by_filler_len": accuracy_by_n,
            "detailed_results": self.results,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt_with_filler(
    example: Dict[str, Any],
    filler_len: int,
    tokenizer: Any,
    fewshot_examples: List[Tuple[str, str, int, int, int]],
    filler_type: str = "dots",
) -> Tuple[List[int], int]:
    """Build input_ids using chat template with system prompt format.
    
    Layout:
        system: instruction + worked examples (identical for all conditions)
        user: just the question
        assistant: Filler: <filler tokens>\nAnswer:   ← model generates from here
    
    For N=0:
        assistant: Answer:   ← model generates from here
    
    Args:
        example: Dataset example with "fact1" and "fact2" fields.
        filler_len: Number of filler items (0 = no filler).
        tokenizer: The tokenizer (must support apply_chat_template).
        fewshot_examples: Few-shot (q1, q2, a1, a2, sum) tuples loaded from
            the dataset manifest.
        filler_type: "dots" or "counting".
    
    Returns:
        (input_ids, total_length)
    """
    q1 = example["fact1"]
    q2 = example["fact2"]
    
    messages = [
        {"role": "system", "content": build_system_prompt(fewshot_examples)},
        {"role": "user", "content": compose_question(q1, q2)},
    ]
    
    # Apply chat template with generation prompt, then prefill assistant turn
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_text += build_filler_prefill(filler_len, filler_type)
    
    input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    
    return input_ids, len(input_ids)


def _get_model_device(model: Any) -> torch.device:
    """Get the input device for a model (handles device_map='auto')."""
    if hasattr(model, 'hf_device_map'):
        # Multi-GPU: find the device of the first module
        first_device = next(iter(model.hf_device_map.values()))
        if isinstance(first_device, str):
            return torch.device(first_device)
        return torch.device(f"cuda:{first_device}")
    if hasattr(model, 'device'):
        return model.device
    return next(model.parameters()).device


@torch.no_grad()
def evaluate_batch(
    model: Any,
    tokenizer: Any,
    examples: List[Dict[str, Any]],
    filler_lens: List[int],
    fewshot_examples: List[Tuple[str, str, int, int, int]],
    max_new_tokens: int = 10,
    temperature: float = 0.0,
    filler_type: str = "dots",
) -> List[Dict[str, Any]]:
    """
    Evaluate a batch of examples simultaneously.
    
    Args:
        examples: List of example dicts (must have "fact1", "fact2", "answer", etc.)
        filler_lens: Filler length for each example (same length as examples).
        fewshot_examples: Few-shot (q1, q2, a1, a2, sum) tuples loaded from
            the dataset manifest.
        max_new_tokens: Max tokens to generate.
        temperature: Sampling temperature (0 = greedy).
        filler_type: "dots" or "counting".
    
    Returns:
        List of result dicts, one per example.
    """
    device = _get_model_device(model)
    
    # Build all input sequences
    all_input_ids = []
    for ex, n in zip(examples, filler_lens):
        ids, _ = build_prompt_with_filler(
            example=ex,
            filler_len=n,
            tokenizer=tokenizer,
            fewshot_examples=fewshot_examples,
            filler_type=filler_type,
        )
        all_input_ids.append(ids)
    
    # Pad to same length (left-pad for generation)
    max_len = max(len(ids) for ids in all_input_ids)
    pad_id = tokenizer.pad_token_id
    
    padded_ids = []
    attention_masks = []
    input_lengths = []
    
    for ids in all_input_ids:
        seq_len = len(ids)
        pad_len = max_len - seq_len
        padded_ids.append([pad_id] * pad_len + ids)
        attention_masks.append([0] * pad_len + [1] * seq_len)
        input_lengths.append(seq_len)
    
    input_tensor = torch.tensor(padded_ids, dtype=torch.long, device=device)
    attention_mask = torch.tensor(attention_masks, dtype=torch.long, device=device)
    
    # Generate
    generate_kwargs = dict(
        input_ids=input_tensor,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        pad_token_id=pad_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    
    if temperature == 0.0:
        generate_kwargs["do_sample"] = False
    else:
        generate_kwargs["do_sample"] = True
        generate_kwargs["temperature"] = temperature
    
    outputs = model.generate(**generate_kwargs)
    
    # Parse results
    results = []
    for i, (ex, n) in enumerate(zip(examples, filler_lens)):
        # Extract only the generated tokens (after the padded input)
        generated_ids = outputs[i, max_len:].tolist()
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        predicted, was_direct = parse_integer_answer(generated_text)
        expected = ex["answer"]
        # Only count as correct if the model output was JUST the number.
        # If it showed work (e.g. "45 + 14 = 59"), that's visible reasoning,
        # not hidden computation in filler tokens — don't reward it.
        correct = (was_direct and predicted == expected)
        
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
            "regex_fallback": predicted is not None and not was_direct,
            "correct_if_regex": (predicted == expected) if predicted is not None else False,
        })
    
    return results


@torch.no_grad()
def evaluate_single(
    model: Any,
    tokenizer: Any,
    example: Dict[str, Any],
    filler_len: int,
    fewshot_examples: List[Tuple[str, str, int, int, int]],
    max_new_tokens: int = 10,
    temperature: float = 0.0,
    filler_type: str = "dots",
) -> Dict[str, Any]:
    """Evaluate a single example. Delegates to evaluate_batch with batch_size=1."""
    results = evaluate_batch(
        model, tokenizer,
        examples=[example],
        filler_lens=[filler_len],
        fewshot_examples=fewshot_examples,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        filler_type=filler_type,
    )
    return results[0]


def evaluate_dataset(
    model: Any,
    tokenizer: Any,
    dataset: Any,
    tracker: ResultsTracker,
    fewshot_examples: List[Tuple[str, str, int, int, int]],
    filler_lengths: Optional[List[int]] = None,
    max_new_tokens: int = 10,
    temperature: float = 0.0,
    max_examples: Optional[int] = None,
    batch_size: int = 8,
    filler_type: str = "dots",
) -> Dict[str, Any]:
    """
    Evaluate a dataset with batched inference.
    
    Args:
        fewshot_examples: Few-shot (q1, q2, a1, a2, sum) tuples from the manifest.
        batch_size: Number of examples per batch for generation.
        filler_type: "dots" or "counting".
    """
    n_total = min(len(dataset), max_examples) if max_examples else len(dataset)
    
    if filler_lengths is None:
        # Use stored filler lengths — group by filler_len for efficient batching
        print(f"Evaluating {n_total} examples with stored filler lengths (batch_size={batch_size})...")
        
        # Collect examples that need evaluation, grouped by filler_len
        groups: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        skipped = 0
        for idx in range(n_total):
            ex = dataset[idx]
            n = ex.get("filler_len", 0)
            example_id = ex.get("id", "")
            if tracker.is_completed(example_id, n):
                skipped += 1
                continue
            groups[n].append(ex)
        
        if skipped > 0:
            print(f"Skipping {skipped} already-completed examples")
        
        total_remaining = sum(len(v) for v in groups.values())
        pbar = tqdm(total=total_remaining, desc="Evaluating")
        
        for n in sorted(groups.keys()):
            examples_for_n = groups[n]
            for batch_start in range(0, len(examples_for_n), batch_size):
                batch = examples_for_n[batch_start:batch_start + batch_size]
                batch_fillers = [n] * len(batch)
                
                results = evaluate_batch(
                    model, tokenizer, batch,
                    filler_lens=batch_fillers,
                    fewshot_examples=fewshot_examples,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    filler_type=filler_type,
                )
                tracker.add_results_batch(results)
                pbar.update(len(batch))
        
        pbar.close()
    else:
        # Evaluate at each specified filler length
        # Collect all examples that need evaluation
        all_examples = [dataset[idx] for idx in range(n_total)]
        
        for n in filler_lengths:
            print(f"\n{'='*60}")
            print(f"Evaluating at N={n} ({len(all_examples)} examples, batch_size={batch_size})")
            print(f"{'='*60}")
            
            # Filter to non-completed examples
            pending = []
            skipped = 0
            for ex in all_examples:
                example_id = ex.get("id", "")
                if tracker.is_completed(example_id, n):
                    skipped += 1
                else:
                    pending.append(ex)
            
            if skipped > 0:
                print(f"  Skipping {skipped} already-completed examples")
            
            start_time = time.time()
            
            pbar = tqdm(total=len(pending), desc=f"N={n}")
            for batch_start in range(0, len(pending), batch_size):
                batch = pending[batch_start:batch_start + batch_size]
                batch_fillers = [n] * len(batch)
                
                results = evaluate_batch(
                    model, tokenizer, batch,
                    filler_lens=batch_fillers,
                    fewshot_examples=fewshot_examples,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    filler_type=filler_type,
                )
                tracker.add_results_batch(results)
                pbar.update(len(batch))
            pbar.close()
            
            elapsed = time.time() - start_time
            stats = tracker.stats_by_n[n]
            acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
            
            print(f"\nN={n} complete: {stats['correct']}/{stats['total']} ({acc:.2f}%)")
            evaluated = len(pending)
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
    regex_fallbacks = [e for e in errors if e.get("regex_fallback", False)]
    
    print(f"\nParse failures (couldn't extract integer): {len(parse_failures)}")
    print(f"Wrong answers (got integer, but wrong): {len(wrong_answers)}")
    print(f"Regex fallback used (among errors): {len(regex_fallbacks)}")
    
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
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="Allow executing remote code from model repos")
    
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
    parser.add_argument("--filler-type", type=str, default=None,
                        choices=["dots", "counting"],
                        help="Override filler type (default: read from dataset manifest)")
    parser.add_argument("--max-new-tokens", type=int, default=10,
                        help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0 = greedy)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for generation (increase for faster eval, decrease if OOM)")
    
    # Output
    parser.add_argument("--outdir", type=str, default=None,
                        help="Directory to save detailed results (enables resume)")
    parser.add_argument("--report-every", type=int, default=100,
                        help="Print progress every N examples")
    parser.add_argument("--show-errors", action="store_true",
                        help="Show error analysis")
    
    # Cache settings
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="HuggingFace cache directory")
    
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
        trust_remote_code=args.trust_remote_code,
        cache_dir=args.cache_dir,
    )
    
    # Load dataset
    data_dir = pathlib.Path(args.data_dir)
    data_file = data_dir / f"{args.split}.jsonl"
    
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    print(f"\nLoading data from {data_file}")
    dataset = load_dataset("json", data_files=str(data_file), split="train")
    print(f"Loaded {len(dataset)} examples")
    
    # Filter out any CoT examples (only meaningful for filler-based evaluation)
    if "sequence_type" in dataset.column_names:
        n_before = len(dataset)
        dataset = dataset.filter(lambda ex: ex["sequence_type"] != "cot")
        n_cot = n_before - len(dataset)
        if n_cot > 0:
            print(f"Filtered out {n_cot} CoT examples, {len(dataset)} filler examples remaining")
    
    # Load manifest for metadata and few-shot examples
    manifest_file = data_dir / "manifest.json"
    manifest = {}
    if manifest_file.exists():
        manifest = json.loads(manifest_file.read_text())
        manifest_filler_type = manifest.get("filler_type", "dots")
        print(f"Filler type from manifest: {manifest_filler_type}")
    
    # Resolve filler type: CLI override > manifest > default
    if args.filler_type:
        filler_type = args.filler_type
    elif manifest:
        filler_type = manifest_filler_type
    else:
        filler_type = "dots"
    print(f"Using filler type: {filler_type}")
    
    if not manifest_file.exists():
        raise FileNotFoundError(
            f"manifest.json not found in {data_dir}. "
            "Regenerate the dataset with the updated generate_2hop_dataset.py."
        )
    raw_fewshot = manifest.get("fewshot_examples")
    if not raw_fewshot:
        raise KeyError(
            "manifest.json does not contain 'fewshot_examples'. "
            "Regenerate the dataset with the updated generate_2hop_dataset.py."
        )
    fewshot_examples: List[Tuple[str, str, int, int, int]] = [
        (e["q1"], e["q2"], e["a1"], e["a2"], e["sum"]) for e in raw_fewshot
    ]
    print(f"Loaded {len(fewshot_examples)} few-shot examples from manifest")
    
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
                    "filler_type": filler_type,
                    "split": args.split,
                    "max_examples": args.max_examples,
                    "temperature": args.temperature,
                    "load_in_4bit": args.load_in_4bit,
                    "batch_size": args.batch_size,
                },
            )
    
    # Evaluate
    print("\nStarting evaluation...")
    results = evaluate_dataset(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        tracker=tracker,
        fewshot_examples=fewshot_examples,
        filler_lengths=filler_lengths,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        max_examples=args.max_examples,
        batch_size=args.batch_size,
        filler_type=filler_type,
    )
    
    tracker.finalize()
    
    if args.show_errors:
        analyze_errors(results)
    
    # Log to wandb — single log call to avoid creating separate steps
    if args.wandb and WANDB_AVAILABLE:
        log_dict = {
            "overall_accuracy": results["overall_accuracy"],
            "overall_correct": results["overall_correct"],
            "overall_total": results["overall_total"],
        }
        
        for n, stats in results["accuracy_by_filler_len"].items():
            log_dict[f"accuracy_N{n}"] = stats["accuracy"]
            log_dict[f"correct_N{n}"] = stats["correct"]
            log_dict[f"total_N{n}"] = stats["total"]
        
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
            log_dict["accuracy_vs_N"] = wandb.Image(fig)
            plt.close(fig)
        except ImportError:
            pass
        
        wandb.log(log_dict)
        wandb.finish()
        print("Logged results to Weights & Biases")
    
    print("\nDone!")
    return results


if __name__ == "__main__":
    main()