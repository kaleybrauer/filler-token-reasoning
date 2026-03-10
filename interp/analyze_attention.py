"""
Attention Pattern Analysis for Filler-Token Computation

When the model generates the answer token, what is it attending to?
We extract the attention weights from the LAST token position (where the
model decides its output) and map them onto the prompt structure:
system prompt, question, filler region, and "Answer:" prefix.

This script compares two groups:
  - filler_helped: wrong at N=0, correct at N=filler_len
  - filler_didnt_help: wrong at both N=0 and N=filler_len
The difference reveals what the model does differently when filler
computation succeeds.

IMPORTANT: This script requires "eager" attention (not Flash Attention
or SDPA) because those implementations don't return attention weights.
The model loads slightly slower and uses more memory, but we need the
raw attention matrices.

OUTPUT:
  - Per-example attention data (saved to disk for further analysis)
  - Aggregate heatmaps: layers x filler positions for each group
  - Head-level analysis: which heads attend most to filler

Usage:
    python interp/analyze_attention.py \
        --model Qwen/Qwen2.5-72B-Instruct \
        --adapter outputs/lr2e-5_50k \
        --data-dir data/datasets/2hop_add \
        --results-n0 results/lr_2e-5_50k/ \
        --filler-len 128 \
        --load-in-4bit \
        --max-examples 100 \
        --outdir results/attention_analysis_counting
"""

import argparse
import json
import os
import pathlib
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

# ── Imports from project ──
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "scripts"))

from generate_addition_dataset import (
    build_filler_prefill,
    build_system_prompt,
    compose_question,
)

from evaluate import (
    load_model_and_tokenizer,
    build_prompt_with_filler,
    parse_integer_answer,
)

# Optional
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ─────────────────────────────────────────────────────────────────────────────
# Identify filler token positions
# ─────────────────────────────────────────────────────────────────────────────

def locate_filler_region(
    example: Dict[str, Any],
    filler_len: int,
    tokenizer: Any,
    fewshot_examples: List[Tuple[str, str, int, int, int]],
    filler_type: str = "counting",
) -> Dict[str, Any]:
    """
    Tokenize the prompt and identify where each region starts/ends.

    Returns dict with:
        input_ids: full token IDs
        regions: dict mapping region name -> (start, end) token indices
            "system_and_question": everything before filler
            "filler": the filler tokens (e.g., "1 2 3 ... N")
            "answer_prefix": "\\nAnswer:" after filler
        filler_token_ids: the token IDs in the filler region
        total_len: total sequence length
    """
    # Build prompt with NO filler to find where filler would be inserted
    ids_n0, _ = build_prompt_with_filler(
        example, filler_len=0, tokenizer=tokenizer,
        fewshot_examples=fewshot_examples, filler_type=filler_type,
    )

    # Build prompt WITH filler
    ids_full, _ = build_prompt_with_filler(
        example, filler_len=filler_len, tokenizer=tokenizer,
        fewshot_examples=fewshot_examples, filler_type=filler_type,
    )

    # The N=0 prompt ends with "Answer:" tokens
    # The N=filler_len prompt has "Filler: <tokens>\nAnswer:" instead of just "Answer:"
    # Find where they diverge
    n0_len = len(ids_n0)
    full_len = len(ids_full)
    filler_token_count = full_len - n0_len

    # Find the divergence point by comparing from the start
    diverge_at = 0
    for i in range(min(n0_len, full_len)):
        if ids_n0[i] != ids_full[i]:
            diverge_at = i
            break
    else:
        diverge_at = min(n0_len, full_len)

    # The "Answer:" suffix tokens are at the end of both prompts
    # Count matching tokens from the end
    suffix_len = 0
    for i in range(1, min(n0_len, full_len) + 1):
        if ids_n0[-i] == ids_full[-i]:
            suffix_len = i
        else:
            break

    # Regions in the full prompt
    filler_start = diverge_at
    filler_end = full_len - suffix_len
    
    regions = {
        "system_and_question": (0, filler_start),
        "filler": (filler_start, filler_end),
        "answer_prefix": (filler_end, full_len),
    }

    return {
        "input_ids": ids_full,
        "regions": regions,
        "filler_token_ids": ids_full[filler_start:filler_end],
        "total_len": full_len,
        "n0_len": n0_len,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Extract attention from last token
# ─────────────────────────────────────────────────────────────────────────────

def extract_last_token_attention(
    model: Any,
    input_ids: List[int],
    device: torch.device,
) -> np.ndarray:
    """
    Run a forward pass and extract the attention weights FROM the last token
    TO all other tokens, across all layers and heads.

    Returns:
        attention: np.ndarray of shape [n_layers, n_heads, seq_len]
        Each value is how much the last token attends to that position.
    """
    ids_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = model(
            input_ids=ids_tensor,
            output_attentions=True,
            use_cache=False,
        )

    # outputs.attentions is a tuple of (n_layers,) tensors
    # Each tensor shape: [batch=1, n_heads, seq_len, seq_len]
    n_layers = len(outputs.attentions)
    n_heads = outputs.attentions[0].shape[1]
    seq_len = outputs.attentions[0].shape[2]

    # Extract last row from each layer: [n_heads, seq_len]
    attention = np.zeros((n_layers, n_heads, seq_len), dtype=np.float32)
    for layer_idx, attn in enumerate(outputs.attentions):
        attention[layer_idx] = attn[0, :, -1, :].float().cpu().numpy()

    # Explicitly delete the large attention tensors
    del outputs
    torch.cuda.empty_cache()

    return attention


# ─────────────────────────────────────────────────────────────────────────────
# Load evaluation results to classify examples
# ─────────────────────────────────────────────────────────────────────────────

def load_eval_results(results_dir: str) -> Dict[str, Dict]:
    """Load evaluation results keyed by example id.

    Each example id maps to its single eval result. This works whether
    evaluate.py was run with stored filler lengths (one result per id)
    or with --filler-lengths override (multiple results per id, last wins —
    but classify_examples uses pair_id grouping so this is fine).
    """
    results = {}
    results_file = os.path.join(results_dir, "eval_detailed.jsonl")
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"No eval_detailed.jsonl in {results_dir}")

    with open(results_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            results[r.get("id", "")] = r

    print(f"  Loaded {len(results)} result entries from {results_file}")
    if results:
        print(f"  Sample result ids: {list(results.keys())[:3]}")
    else:
        print("  WARNING: results dict is empty!")
    return results


def classify_examples(
    dataset: Any,
    results: Dict[str, Dict],
    filler_len: int,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Classify examples into filler_helped and filler_didnt_help groups.

    Uses pair_id to link the N=0 row and N=filler_len row for the same
    question (they have different example ids but share a pair_id).

    filler_helped: wrong at N=0, correct at N=filler_len
    filler_didnt_help: wrong at N=0, wrong at N=filler_len
    """
    # Group dataset rows by pair_id, then by filler_len
    pair_result_n0: Dict[Any, Dict] = {}   # pair_id → result at filler_len=0
    pair_result_nf: Dict[Any, Dict] = {}   # pair_id → result at filler_len=filler_len
    pair_example_nf: Dict[Any, Dict] = {}  # pair_id → dataset row at filler_len=filler_len

    for idx in range(len(dataset)):
        ex = dataset[idx]
        ex_id = ex.get("id", "")
        pair_id = ex.get("pair_id")
        fl = ex.get("filler_len", 0)
        result = results.get(ex_id)
        if result is None or pair_id is None:
            continue
        if fl == 0:
            pair_result_n0[pair_id] = result
        elif fl == filler_len:
            pair_result_nf[pair_id] = result
            pair_example_nf[pair_id] = ex

    filler_helped = []
    filler_didnt_help = []

    for pair_id, r0 in pair_result_n0.items():
        rf = pair_result_nf.get(pair_id)
        if rf is None:
            continue
        ex = pair_example_nf[pair_id]
        wrong_at_0 = not r0.get("correct", False)
        correct_at_f = rf.get("correct", False)
        if wrong_at_0 and correct_at_f:
            filler_helped.append(ex)
        elif wrong_at_0 and not correct_at_f:
            filler_didnt_help.append(ex)

    n_pairs_with_n0 = len(pair_result_n0)
    n_pairs_with_nf = len(pair_result_nf)
    print(f"  Pairs with N=0 result: {n_pairs_with_n0}, "
          f"with N={filler_len} result: {n_pairs_with_nf}, "
          f"with both: {sum(1 for p in pair_result_n0 if p in pair_result_nf)}")

    return filler_helped, filler_didnt_help


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def plot_attention_heatmap(
    attention_by_group: Dict[str, np.ndarray],
    filler_region: Tuple[int, int],
    outdir: str,
    title_suffix: str = "",
):
    """
    Plot heatmaps of attention to filler positions, averaged across heads.

    attention_by_group: dict mapping group_name -> [n_layers, n_heads, seq_len]
        (averaged across examples in the group)
    """
    if not HAS_MPL:
        print("matplotlib not available, skipping plots")
        return

    filler_start, filler_end = filler_region

    for group_name, attn in attention_by_group.items():
        # Average across heads: [n_layers, seq_len]
        attn_avg = attn.mean(axis=1)

        # Extract just the filler columns
        filler_attn = attn_avg[:, filler_start:filler_end]

        fig, ax = plt.subplots(figsize=(14, 8))
        im = ax.imshow(
            filler_attn,
            aspect="auto",
            cmap="viridis",
            interpolation="nearest",
        )
        ax.set_xlabel("Filler position", fontsize=12)
        ax.set_ylabel("Layer", fontsize=12)
        ax.set_title(
            f"Last-token attention to filler region — {group_name}{title_suffix}",
            fontsize=14,
        )
        plt.colorbar(im, ax=ax, label="Attention weight")
        plt.tight_layout()
        fname = os.path.join(outdir, f"attn_heatmap_{group_name.replace(' ', '_')}.png")
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"Saved {fname}")

    # Difference plot if we have both groups
    if "filler_helped" in attention_by_group and "filler_didnt_help" in attention_by_group:
        diff = (
            attention_by_group["filler_helped"].mean(axis=1)[:, filler_start:filler_end]
            - attention_by_group["filler_didnt_help"].mean(axis=1)[:, filler_start:filler_end]
        )

        fig, ax = plt.subplots(figsize=(14, 8))
        vmax = np.abs(diff).max()
        im = ax.imshow(
            diff,
            aspect="auto",
            cmap="RdBu_r",
            interpolation="nearest",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.set_xlabel("Filler position", fontsize=12)
        ax.set_ylabel("Layer", fontsize=12)
        ax.set_title(
            f"Attention difference (helped − didn't help){title_suffix}",
            fontsize=14,
        )
        plt.colorbar(im, ax=ax, label="Attention weight difference")
        plt.tight_layout()
        fname = os.path.join(outdir, "attn_heatmap_difference.png")
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"Saved {fname}")


def plot_top_heads(
    attention_by_group: Dict[str, np.ndarray],
    filler_region: Tuple[int, int],
    outdir: str,
    top_k: int = 10,
):
    """Find and plot the attention heads that attend most to filler."""
    if not HAS_MPL:
        return

    filler_start, filler_end = filler_region

    for group_name, attn in attention_by_group.items():
        # attn: [n_layers, n_heads, seq_len]
        n_layers, n_heads, seq_len = attn.shape

        # Total attention to filler per head: [n_layers, n_heads]
        filler_attn = attn[:, :, filler_start:filler_end].sum(axis=2)

        # Find top-k heads
        flat_idx = np.argsort(filler_attn.ravel())[-top_k:][::-1]
        top_layers = flat_idx // n_heads
        top_head_ids = flat_idx % n_heads

        print(f"\nTop {top_k} heads attending to filler ({group_name}):")
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        for rank, (ax, li, hi) in enumerate(
            zip(axes.flat, top_layers, top_head_ids)
        ):
            head_attn = attn[li, hi, filler_start:filler_end]
            total_filler_attn = head_attn.sum()
            print(
                f"  #{rank+1}: Layer {li}, Head {hi} — "
                f"total filler attention: {total_filler_attn:.4f}"
            )

            ax.plot(head_attn, linewidth=0.8)
            ax.set_title(f"L{li} H{hi}\n(sum={total_filler_attn:.3f})", fontsize=9)
            ax.set_xlabel("Filler pos", fontsize=8)
            ax.set_ylabel("Attention", fontsize=8)
            ax.tick_params(labelsize=7)

        plt.suptitle(f"Top {top_k} filler-attending heads — {group_name}", fontsize=13)
        plt.tight_layout()
        fname = os.path.join(outdir, f"top_heads_{group_name.replace(' ', '_')}.png")
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"Saved {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyze attention patterns in filler-token computation"
    )

    # Model
    parser.add_argument("--model", type=str, required=True,
                        help="Base model name or path")
    parser.add_argument("--adapter", type=str, default=None,
                        help="Path to LoRA adapter")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--cache-dir", type=str, default=None)

    # Data
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory containing test.jsonl and manifest.json")
    parser.add_argument("--split", type=str, default="test")

    # Results for classification
    parser.add_argument("--results-n0", type=str, required=True,
                        help="Directory with N=0 eval results (results.jsonl)")
    parser.add_argument("--results-nf", type=str, default=None,
                        help="Directory with N=filler_len eval results. "
                             "If not provided, uses --results-n0 (which may "
                             "contain results for multiple N values)")

    # Filler config
    parser.add_argument("--filler-len", type=int, default=128,
                        help="Filler length to analyze")
    parser.add_argument("--filler-type", type=str, default=None,
                        help="Override filler type (default: from manifest)")

    # Limits
    parser.add_argument("--max-examples", type=int, default=50,
                        help="Max examples per group to analyze")

    # Output
    parser.add_argument("--outdir", type=str, default="results/attention_analysis")

    args = parser.parse_args()
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Load data and manifest ──
    data_dir = pathlib.Path(args.data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    filler_type = args.filler_type or manifest.get("filler_type", "counting")
    print(f"Filler type: {filler_type}")

    raw_fewshot = manifest["fewshot_examples"]
    fewshot_examples = [
        (e["q1"], e["q2"], e["a1"], e["a2"], e["sum"]) for e in raw_fewshot
    ]

    from datasets import load_dataset
    data_file = data_dir / f"{args.split}.jsonl"
    dataset = load_dataset("json", data_files=str(data_file), split="train")

    # Filter to filler examples only
    if "sequence_type" in dataset.column_names:
        dataset = dataset.filter(lambda ex: ex["sequence_type"] != "cot")
    print(f"Loaded {len(dataset)} filler examples")

    # ── Load evaluation results ──
    # N=0 and N=filler_len results may live in the same directory (most common)
    # or separate ones. Merge into a single id→result dict.
    print(f"\nLoading results from {args.results_n0}")
    results = load_eval_results(args.results_n0)

    if args.results_nf and args.results_nf != args.results_n0:
        print(f"Loading additional results from {args.results_nf}")
        results.update(load_eval_results(args.results_nf))

    # ── Classify examples ──
    filler_helped, filler_didnt_help = classify_examples(
        dataset, results, args.filler_len
    )
    print(f"\nClassification:")
    print(f"  Filler helped (wrong@0, correct@{args.filler_len}): {len(filler_helped)}")
    print(f"  Filler didn't help (wrong@0, wrong@{args.filler_len}): {len(filler_didnt_help)}")

    if len(filler_helped) == 0:
        print("ERROR: No filler-helped examples found. Check results paths.")
        sys.exit(1)

    # Limit examples
    filler_helped = filler_helped[: args.max_examples]
    filler_didnt_help = filler_didnt_help[: args.max_examples]
    print(f"  Using: {len(filler_helped)} helped, {len(filler_didnt_help)} didn't help")

    # ── Load model ──
    # CRITICAL: Use eager attention to get attention weights
    print("\nLoading model with EAGER attention (required for attention weights)...")
    print("(This is slower than Flash Attention but necessary for this analysis)")

    model, tokenizer = load_model_and_tokenizer(
        args.model,
        adapter_path=args.adapter,
        load_in_4bit=args.load_in_4bit,
        use_flash_attn=False,
        trust_remote_code=args.trust_remote_code,
        cache_dir=args.cache_dir,
        attn_implementation="eager",  # Required to get attention weights back
    )

    # Determine input device: for device_map models, use the embedding layer's device.
    if hasattr(model, "hf_device_map") and model.hf_device_map:
        first_device = next(iter(model.hf_device_map.values()))
        device = torch.device(f"cuda:{first_device}" if isinstance(first_device, int) else first_device)
    else:
        device = next(model.parameters()).device

    # ── Print prompt structure for the first example (informational only) ──
    sample_info = locate_filler_region(
        filler_helped[0], args.filler_len, tokenizer,
        fewshot_examples, filler_type
    )
    print(f"\nPrompt structure (first example):")
    for region_name, (start, end) in sample_info["regions"].items():
        print(f"  {region_name}: tokens {start}-{end} ({end - start} tokens)")
    print(f"  Total length: {sample_info['total_len']}")
    print(f"  Note: filler position varies per example; averaging is filler-position-relative.")

    # ── Extract attention patterns ──
    groups = {
        "filler_helped": filler_helped,
        "filler_didnt_help": filler_didnt_help,
    }

    attention_by_group = {}   # [n_layers, n_heads, filler_len] — filler-position-relative
    pre_attn_by_group = {}    # [n_layers, n_heads] — sum of attention to system+question
    post_attn_by_group = {}   # [n_layers, n_heads] — sum of attention to answer prefix

    for group_name, examples in groups.items():
        if len(examples) == 0:
            print(f"\nSkipping {group_name} (no examples)")
            continue

        print(f"\nExtracting attention for {group_name} ({len(examples)} examples)...")
        all_filler_attn = []
        all_pre_sums = []
        all_post_sums = []

        for i, ex in enumerate(tqdm(examples, desc=group_name)):
            # Locate filler region for THIS example (question lengths vary)
            info = locate_filler_region(
                ex, args.filler_len, tokenizer,
                fewshot_examples, filler_type,
            )
            filler_start, filler_end = info["regions"]["filler"]

            # Extract attention: [n_layers, n_heads, seq_len]
            attn = extract_last_token_attention(model, info["input_ids"], device)

            # Slice to filler-relative coordinates so averaging is aligned
            filler_attn = attn[:, :, filler_start:filler_end]   # [n_layers, n_heads, filler_len]
            pre_sum = attn[:, :, :filler_start].sum(axis=2)      # [n_layers, n_heads]
            post_sum = attn[:, :, filler_end:].sum(axis=2)       # [n_layers, n_heads]

            all_filler_attn.append(filler_attn)
            all_pre_sums.append(pre_sum)
            all_post_sums.append(post_sum)

            # Save per-example data (full attention + per-example regions)
            np.savez_compressed(
                outdir / f"attn_{group_name}_{i}.npz",
                attention=attn.astype(np.float16),
                filler_start=filler_start,
                filler_end=filler_end,
                a1=ex.get("a1", 0),
                a2=ex.get("a2", 0),
                answer=ex.get("answer", 0),
            )

        # Average across examples — all filler slices have shape [n_layers, n_heads, filler_len]
        attention_by_group[group_name] = np.mean(all_filler_attn, axis=0)
        pre_attn_by_group[group_name] = np.mean(all_pre_sums, axis=0)
        post_attn_by_group[group_name] = np.mean(all_post_sums, axis=0)
        print(f"  Mean filler attention shape: {attention_by_group[group_name].shape}")

    # The averaged attention arrays are filler-position-relative.
    # The number of filler TOKENS is not args.filler_len (counting numbers) —
    # "1 2 3 ... 128" tokenizes to many more than 128 tokens.
    # Read the actual token count from the first example's region info.
    filler_token_count = sample_info["regions"]["filler"][1] - sample_info["regions"]["filler"][0]
    filler_region_rel = (0, filler_token_count)

    # ── Save aggregate results ──
    save_dict = {f"attn_{k}": v for k, v in attention_by_group.items()}
    save_dict.update({f"pre_attn_{k}": v for k, v in pre_attn_by_group.items()})
    save_dict.update({f"post_attn_{k}": v for k, v in post_attn_by_group.items()})
    save_dict["filler_len"] = args.filler_len
    np.savez_compressed(outdir / "aggregate_attention.npz", **save_dict)
    print(f"\nSaved aggregate attention to {outdir / 'aggregate_attention.npz'}")

    # ── Summary statistics ──
    print(f"\n{'='*60}")
    print("ATTENTION SUMMARY")
    print(f"{'='*60}")
    for group_name, attn in attention_by_group.items():
        # attn is [n_layers, n_heads, filler_len] — all filler, no pre/post
        filler_sum_per_head = attn.sum(axis=2)  # [layers, heads]
        total_filler = filler_sum_per_head.mean()
        print(f"\n{group_name}:")
        print(f"  Mean attention to filler region: {total_filler:.4f}")
        print(f"  Mean attention to system+question: {pre_attn_by_group[group_name].mean():.4f}")
        print(f"  Mean attention to answer prefix: {post_attn_by_group[group_name].mean():.4f}")

        # Which layers attend most to filler?
        layer_filler_attn = filler_sum_per_head.mean(axis=1)  # [layers]
        top_layers = np.argsort(layer_filler_attn)[-5:][::-1]
        print(f"  Top 5 layers by filler attention:")
        for li in top_layers:
            print(f"    Layer {li}: {layer_filler_attn[li]:.4f}")

    # ── Visualize ──
    plot_attention_heatmap(
        attention_by_group, filler_region_rel, str(outdir),
        title_suffix=f" (N={args.filler_len})",
    )
    plot_top_heads(attention_by_group, filler_region_rel, str(outdir))

    # ── Save config ──
    config = {
        "model": args.model,
        "adapter": args.adapter,
        "filler_len": args.filler_len,
        "filler_type": filler_type,
        "n_filler_helped": len(filler_helped),
        "n_filler_didnt_help": len(filler_didnt_help),
        "filler_region": list(filler_region_rel),
        "filler_token_count": filler_token_count,
        "total_seq_len": sample_info["total_len"],
    }
    with open(outdir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nAll results saved to {outdir}")
    print("Done!")


if __name__ == "__main__":
    main()