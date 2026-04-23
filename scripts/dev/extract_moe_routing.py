"""
extract_moe_routing.py

Extract MoE router decisions from DeepSeek V3 to compare expert routing
between baseline and filler conditions. Batched for GPU efficiency.

For each MoE layer, captures top-8 expert indices at:
- Filler positions (dots_100 only): how diverse is routing across filler?
- Answer position (both conditions): does filler change which experts are active?

Usage:
    source /workspace/config/probing_env.sh
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python scripts/dev/extract_moe_routing.py \
        --model-path /workspace/models/deepseek-v3-awq \
        --max-examples 500 \
        --batch-size 4 \
        --output-dir results/routing
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

import transformers.activations as _act
if not hasattr(_act, "PytorchGELUTanh"):
    _act.PytorchGELUTanh = _act.GELUTanh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from extract.extract_hidden_states import find_filler_boundaries, load_model
from prompt_utils import build_messages


class MoERouterExtractor:
    """Hook into MoEGate modules to capture routing decisions."""

    def __init__(self, model, moe_layer_indices: List[int]):
        self.model = model
        self.moe_layer_indices = moe_layer_indices
        self._captured: Dict[int, dict] = {}
        self._hooks = []

    def _make_gate_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            topk_idx, topk_weight = output
            # Gate flattens (batch, seq_len) → (batch*seq_len). Reshape back.
            bsz, seq_len, _ = input[0].shape
            self._captured[layer_idx] = {
                "topk_idx": topk_idx.view(bsz, seq_len, -1).detach().cpu(),
                "topk_weight": topk_weight.view(bsz, seq_len, -1).detach().cpu(),
            }
        return hook_fn

    def register_hooks(self):
        self.remove_hooks()
        for layer_idx in self.moe_layer_indices:
            gate = self.model.model.layers[layer_idx].mlp.gate
            hook = gate.register_forward_hook(self._make_gate_hook(layer_idx))
            self._hooks.append(hook)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._captured.clear()


def detect_moe_layers(model) -> List[int]:
    moe_layers = []
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer.mlp, "gate"):
            moe_layers.append(i)
    return moe_layers


def compute_diversity(topk_idx: np.ndarray) -> dict:
    n_unique = int(len(np.unique(topk_idx)))
    flat = topk_idx.astype(np.int32).flatten()
    counts = np.bincount(flat, minlength=256)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    entropy = float(-np.sum(probs * np.log2(probs)))
    return {"n_unique": n_unique, "entropy": entropy}


def expert_overlap(experts_a: np.ndarray, experts_b: np.ndarray) -> float:
    set_a = set(experts_a.astype(int))
    set_b = set(experts_b.astype(int))
    return len(set_a & set_b) / len(set_a | set_b) if set_a | set_b else 0.0


def tokenize_and_pad(tokenizer, texts, device):
    """Tokenize a list of texts, left-pad to max length."""
    all_ids = [tokenizer(t, return_tensors="pt")["input_ids"][0] for t in texts]
    max_len = max(len(ids) for ids in all_ids)
    pad_id = tokenizer.pad_token_id or 0

    padded = torch.full((len(all_ids), max_len), pad_id, dtype=torch.long)
    mask = torch.zeros((len(all_ids), max_len), dtype=torch.long)
    pad_lens = []

    for b, ids in enumerate(all_ids):
        pl = max_len - len(ids)
        padded[b, pl:] = ids
        mask[b, pl:] = 1
        pad_lens.append(pl)

    return padded.to(device), mask.to(device), pad_lens, all_ids


def run_batched_condition(
    model, tokenizer, extractor, moe_layers, problems, few_shot,
    mode, k, batch_size, device,
):
    """Run a condition in batches, return per-example routing at answer + filler positions."""
    results = []

    for batch_start in tqdm(range(0, len(problems), batch_size),
                            desc=f"{mode}_k{k}",
                            total=(len(problems) + batch_size - 1) // batch_size):
        batch_problems = problems[batch_start:batch_start + batch_size]
        batch_indices = list(range(batch_start, batch_start + len(batch_problems)))

        # Build texts
        texts = []
        for prob in batch_problems:
            msgs = build_messages(few_shot[:5], prob, mode, k)
            texts.append(tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            ))

        padded_ids, attn_mask, pad_lens, orig_ids = tokenize_and_pad(
            tokenizer, texts, device
        )

        # Find filler boundaries and positions per example
        per_example_meta = []
        for b, prob in enumerate(batch_problems):
            seq_len = len(orig_ids[b])
            pl = pad_lens[b]

            filler_start, filler_end = -1, -1
            if k > 0:
                _, filler_start, filler_end = find_filler_boundaries(
                    tokenizer, orig_ids[b].unsqueeze(0), k
                )

            per_example_meta.append({
                "seq_len": seq_len,
                "pad_len": pl,
                "filler_start": filler_start,
                "filler_end": filler_end,
                "answer_pos_padded": pl + seq_len - 1,
            })

        # Forward pass
        extractor._captured = {}
        with torch.no_grad():
            model(input_ids=padded_ids, attention_mask=attn_mask)

        # Extract per-example routing
        for b, (prob, meta) in enumerate(zip(batch_problems, per_example_meta)):
            pl = meta["pad_len"]
            answer_pos = meta["answer_pos_padded"]

            example_routing = {}
            for layer_idx in moe_layers:
                cap = extractor._captured[layer_idx]
                # Answer position routing
                answer_experts = cap["topk_idx"][b, answer_pos].numpy().astype(np.int16)
                answer_weights = cap["topk_weight"][b, answer_pos].numpy().astype(np.float16)

                entry = {
                    "answer_experts": answer_experts,
                    "answer_weights": answer_weights,
                }

                # Filler routing (if applicable)
                if meta["filler_start"] >= 0:
                    fs = pl + meta["filler_start"]
                    fe = pl + meta["filler_end"]
                    filler_experts = cap["topk_idx"][b, fs:fe+1].numpy().astype(np.int16)
                    entry["filler_experts"] = filler_experts

                example_routing[layer_idx] = entry

            results.append({
                "prob_idx": batch_indices[b],
                "fact_value": prob["fact_value"],
                "x": prob["x"],
                "answer": prob["answer"],
                "routing": example_routing,
            })

        extractor._captured = {}
        del padded_ids, attn_mask
        torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/workspace/models/deepseek-v3-awq")
    parser.add_argument("--dataset", type=Path,
                        default=Path("data/1hop_addition_dataset.json"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/routing"))
    parser.add_argument("--max-examples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--k", type=int, default=100)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.dataset) as f:
        dataset = json.load(f)
    few_shot = dataset["few_shot_facts"]
    problems = dataset["examples"][:args.max_examples]

    model, tokenizer = load_model(args.model_path)
    first_device = next(model.parameters()).device

    moe_layers = detect_moe_layers(model)
    print(f"Detected {len(moe_layers)} MoE layers: {moe_layers[0]}-{moe_layers[-1]}")

    extractor = MoERouterExtractor(model, moe_layers)
    extractor.register_hooks()

    t0 = time.time()

    # Run both conditions batched
    print("\n--- dots_100 ---")
    filler_results = run_batched_condition(
        model, tokenizer, extractor, moe_layers, problems, few_shot,
        "dots", args.k, args.batch_size, first_device,
    )

    print("\n--- baseline ---")
    baseline_results = run_batched_condition(
        model, tokenizer, extractor, moe_layers, problems, few_shot,
        "baseline", 0, args.batch_size, first_device,
    )

    extractor.remove_hooks()
    elapsed = time.time() - t0
    print(f"\n{len(problems)} examples × 2 conditions in {elapsed:.0f}s")

    # --- Compute and print summary ---
    print(f"\n{'='*70}")
    print("FILLER ROUTING DIVERSITY (unique experts / entropy across 100 filler positions)")
    print(f"{'='*70}")
    print(f"  {'Layer':<8} {'Unique/256':>12} {'Entropy':>10}")
    print(f"  {'-'*32}")
    for l in moe_layers:
        if l % 5 != 0 and l not in [3, 53, 58, 60]:
            continue
        uniques = []
        entropies = []
        for ex in filler_results:
            if "filler_experts" in ex["routing"][l]:
                d = compute_diversity(ex["routing"][l]["filler_experts"])
                uniques.append(d["n_unique"])
                entropies.append(d["entropy"])
        if uniques:
            print(f"  L{l:<6} {np.mean(uniques):>12.1f} {np.mean(entropies):>10.2f}")

    print(f"\n{'='*70}")
    print("ANSWER POSITION: expert overlap between baseline and dots_100 (Jaccard)")
    print(f"{'='*70}")
    print(f"  {'Layer':<8} {'Mean Jaccard':>14} {'Std':>8}")
    print(f"  {'-'*32}")
    for l in moe_layers:
        if l % 5 != 0 and l not in [3, 53, 58, 60]:
            continue
        jaccards = []
        for f_ex, b_ex in zip(filler_results, baseline_results):
            j = expert_overlap(
                f_ex["routing"][l]["answer_experts"],
                b_ex["routing"][l]["answer_experts"],
            )
            jaccards.append(j)
        print(f"  L{l:<6} {np.mean(jaccards):>14.3f} {np.std(jaccards):>8.3f}")

    # Save raw results
    # Convert numpy arrays to lists for JSON
    def convert(results):
        out = []
        for ex in results:
            r = {k: v for k, v in ex.items() if k != "routing"}
            r["routing"] = {}
            for l, entry in ex["routing"].items():
                r["routing"][str(l)] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in entry.items()
                }
            out.append(r)
        return out

    with open(args.output_dir / "filler_routing.json", "w") as f:
        json.dump(convert(filler_results), f)
    with open(args.output_dir / "baseline_routing.json", "w") as f:
        json.dump(convert(baseline_results), f)

    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
