"""
extract_hidden_states.py

Extract hidden states from DeepSeek V3 during filler token processing
for linear probing experiments.

Usage:
    source /workspace/config/probing_env.sh
    python probing/scripts/extract_hidden_states.py \
        --model-path /workspace/models/deepseek-v3-awq \
        --dataset probing/data/1hop_addition_dataset.json \
        --output-dir probing/extracted_states \
        --conditions baseline dots_250 \
        --max-problems 400

Prerequisites:
    - Model downloaded (setup_vllm.sh handles this)
    - Token boundaries confirmed (see token boundary verification output)
"""

import argparse
import json
import pickle
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

# Patch: autoawq imports PytorchGELUTanh which was renamed in older transformers
import transformers.activations as _act
if not hasattr(_act, "PytorchGELUTanh"):
    _act.PytorchGELUTanh = _act.GELUTanh

# Reuse prompt construction from the dataset generator (supports all filler types)
from generate_1hop_dataset import build_system_message, build_user_turn


def build_messages_for_condition(few_shot_items, target, filler_type, k, rng=None):
    """Build chat messages for any filler condition.

    Uses generate_1hop_dataset's prompt helpers (which support all filler types)
    but matches the assistant turn format ("Answer: {answer}") used by the
    eval script and existing extractions.
    """
    system_msg = build_system_message(filler_type, k)
    messages = [{"role": "system", "content": system_msg}]

    for fs in few_shot_items:
        user_content = build_user_turn(
            fs["fact_phrase"], fs["x"], filler_type, k, rng=rng
        )
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": f"Answer: {fs['answer']}"})

    user_content = build_user_turn(
        target["fact_phrase"], target["x"], filler_type, k, rng=rng
    )
    messages.append({"role": "user", "content": user_content})

    return messages


# ==============================================================================
# Configuration
# ==============================================================================

FILLER_FRACTIONS = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]

# Absolute filler token offsets for fine-grained extraction
FILLER_ABSOLUTE_POSITIONS = None  # Set via --filler-positions; uses fractions if None

CONDITIONS = {
    "baseline": (0, "dots"),
    "dots_50": (50, "dots"),
    "dots_250": (250, "dots"),
    "dots_500": (500, "dots"),
    "counting_150": (150, "counting"),
    "counting_scrambled_150": (150, "counting-scrambled"),
}


# ==============================================================================
# Token boundary detection
# ==============================================================================

def find_filler_boundaries(
    tokenizer,
    input_ids: torch.Tensor,
    k: int,
) -> Tuple[int, int]:
    """
    Find the start and end token positions of the TARGET question's filler.

    Works for any filler type (dots, counting, scrambled counting).
    Scans backward from the last "Answer" token to find filler content
    boundaries using the "Filler: <content>\\n\\nAnswer:" structure.

    Returns:
        filler_start: index of first filler content token in target question
        filler_end: index of last filler content token in target question (inclusive)
    """
    if k == 0:
        return -1, -1

    ids = input_ids[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(ids)
    seq_len = len(tokens)

    # Find the last "Answer" token (marks end of filler region)
    last_answer_pos = -1
    for i in range(seq_len - 1, -1, -1):
        if tokens[i] == "Answer":
            last_answer_pos = i
            break

    if last_answer_pos == -1:
        raise ValueError("Could not find 'Answer' token in sequence")

    # Scan backward from Answer, skip newline/whitespace-only tokens to find filler_end
    filler_end = -1
    for i in range(last_answer_pos - 1, -1, -1):
        decoded = tokenizer.decode([ids[i]])
        if decoded.strip():  # has non-whitespace content
            filler_end = i
            break

    if filler_end == -1:
        raise ValueError("Could not find filler content before 'Answer' token")

    # Scan backward from filler_end to find filler_start
    # Stop when we hit ":" (the colon of "Filler:")
    # Note: space tokens ("Ġ" → " ") between counting numbers must NOT stop the scan
    filler_start = filler_end
    for i in range(filler_end - 1, -1, -1):
        decoded = tokenizer.decode([ids[i]])
        stripped = decoded.strip()
        if stripped == ":":
            break
        filler_start = i

    return filler_start, filler_end


def compute_extraction_positions(
    filler_start: int,
    filler_end: int,
    seq_len: int,
    fractions: List[float] = FILLER_FRACTIONS,
    absolute_positions: Optional[List[int]] = None,
) -> Dict[str, int]:
    """
    Compute the token positions to extract hidden states from.
    Returns dict mapping descriptive name -> token index.

    If absolute_positions is provided, uses those as filler token offsets
    (e.g. [1, 5, 10, 25, 50] extracts at the 1st, 5th, ... filler token).
    Otherwise uses fractions of the filler region.
    """
    positions = {}

    # Last token before filler (question end)
    positions["question_end"] = filler_start - 1

    filler_len = filler_end - filler_start + 1

    if absolute_positions is not None:
        for k in absolute_positions:
            idx = filler_start + k - 1  # k=1 means first filler token
            if idx <= filler_end:
                positions[f"filler_k{k}"] = idx
    else:
        for frac in fractions:
            idx = filler_start + int(frac * (filler_len - 1))
            idx = min(idx, filler_end)
            positions[f"filler_{frac:.2f}"] = idx

    # Last token before generation
    positions["answer_prompt"] = seq_len - 1

    return positions


def compute_baseline_positions(seq_len: int) -> Dict[str, int]:
    """For baseline (no filler): question_end and answer_prompt."""
    return {
        "question_end": seq_len - 2,  # token before the final special token
        "answer_prompt": seq_len - 1,
    }


# ==============================================================================
# Hidden state extraction
# ==============================================================================

class HiddenStateExtractor:
    """
    Registers forward hooks on transformer layers to capture
    residual stream hidden states at specified token positions.

    Hook point: model.model.layers[i]
    Output format: DeepseekV2DecoderLayer returns (hidden_states, residual)
    We capture output[0] = hidden_states (post-layer residual stream).
    Shape: (batch=1, seq_len, 7168)
    """

    def __init__(self, model, layer_indices: List[int]):
        self.model = model
        self.layer_indices = layer_indices
        self._captured = {}
        self._hooks = []

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            self._captured[layer_idx] = hidden[0].detach().cpu()
        return hook_fn

    def register_hooks(self):
        self.remove_hooks()
        for layer_idx in self.layer_indices:
            module = self.model.model.layers[layer_idx]
            hook = module.register_forward_hook(self._make_hook(layer_idx))
            self._hooks.append(hook)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self._captured = {}

    def extract(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        positions: Dict[str, int],
    ) -> Dict[str, Dict[int, np.ndarray]]:
        """
        Run one forward pass and extract hidden states at specified positions.

        Returns:
            {position_name: {layer_idx: np.ndarray of shape (7168,)}}
        """
        self._captured = {}

        with torch.no_grad():
            self.model(input_ids=input_ids, attention_mask=attention_mask)

        result = {}
        for pos_name, pos_idx in positions.items():
            result[pos_name] = {}
            for layer_idx in self.layer_indices:
                vec = self._captured[layer_idx][pos_idx]  # shape: (7168,)
                result[pos_name][layer_idx] = vec.to(torch.float16).numpy()

        self._captured = {}
        torch.cuda.empty_cache()

        return result


# ==============================================================================
# Model loading
# ==============================================================================

def load_tokenizer(model_path: str):
    """
    Load the tokenizer from a model directory.

    Uses PreTrainedTokenizerFast to load tokenizer.json directly, bypassing
    the tokenizer_config.json "tokenizer_class": "LlamaTokenizer" override.
    AutoTokenizer loads as LlamaTokenizer (sentencepiece) which aggressively
    merges tokens (e.g. 250 dots → 2 tokens). The fast tokenizer respects
    the pre_tokenizer rules in tokenizer.json and gives correct tokenization
    (e.g. 250 dots → 250 tokens).
    """
    import json
    from pathlib import Path
    from transformers import PreTrainedTokenizerFast

    model_dir = Path(model_path)
    tokenizer_file = model_dir / "tokenizer.json"
    config_file = model_dir / "tokenizer_config.json"

    # Read chat_template and special tokens from tokenizer_config.json
    with open(config_file) as f:
        config = json.load(f)

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_file),
        chat_template=config.get("chat_template"),
        bos_token=config.get("bos_token"),
        eos_token=config.get("eos_token"),
    )
    return tokenizer


def load_model(model_path: str):
    """
    Load the AWQ model via transformers with device_map="auto".

    Note: with vLLM 0.8.5 installed, transformers is ~4.48 which does NOT
    have the slow _move_missing_keys_from_meta_to_device bug (that's 5.x).
    Loading should take ~5-10 minutes on 3x H200.
    """
    from transformers import AutoModelForCausalLM

    print(f"Loading tokenizer from {model_path}...")
    tokenizer = load_tokenizer(model_path)

    n_gpus = torch.cuda.device_count()
    print(f"Loading model on {n_gpus} GPUs...")
    for i in range(n_gpus):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU {i}: {name}, {mem:.0f} GB")

    # Set max_memory per GPU to prevent CPU offloading.
    # AWQ quantizer rejects device maps with CPU.
    max_memory = {
        i: f"{int(torch.cuda.get_device_properties(i).total_memory * 0.80 / 1e9)}GiB"
        for i in range(n_gpus)
    }
    print(f"  max_memory: {max_memory}")

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        max_memory=max_memory,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.eval()
    elapsed = time.time() - t0
    print(f"Model loaded in {elapsed:.0f}s")

    for i in range(n_gpus):
        alloc = torch.cuda.memory_allocated(i) / 1e9
        print(f"  GPU {i}: {alloc:.1f} GB allocated")

    return model, tokenizer


# ==============================================================================
# Answer extraction from generation
# ==============================================================================

def get_model_answer(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Tuple[str, Optional[int]]:
    """Generate the model's answer and parse the numeric result."""
    with torch.no_grad():
        gen_output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=20,
            do_sample=False,
        )

    new_tokens = gen_output[0][input_ids.shape[1]:]
    raw_response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    m = re.search(r"Answer:\s*(-?\d+)", raw_response)
    if m:
        return raw_response, int(m.group(1))
    m = re.search(r"(-?\d+)", raw_response)
    if m:
        return raw_response, int(m.group(1))
    return raw_response, None


# ==============================================================================
# Main extraction pipeline
# ==============================================================================

def run_extraction(
    model,
    tokenizer,
    problems: list,
    few_shot: list,
    conditions: Dict[str, Tuple[int, str]],
    layer_indices: List[int],
    output_dir: Path,
    skip_existing: bool = True,
    also_generate: bool = True,
    filler_positions: Optional[List[int]] = None,
):
    """
    For each (condition, problem), extract hidden states and save.

    Directory structure:
        output_dir/
            {condition_name}/
                prob_0000.pkl
                ...
            metadata.json
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    extractor = HiddenStateExtractor(model, layer_indices)
    extractor.register_hooks()

    # Find the device for input tensors (first parameter's device)
    first_param = next(model.parameters())
    input_device = first_param.device

    # Save metadata
    metadata = []
    for i, p in enumerate(problems):
        metadata.append({
            "idx": i,
            "fact_phrase": p["fact_phrase"],
            "fact_value": p["fact_value"],
            "x": p["x"],
            "answer": p["answer"],
            "kind": p["kind"],
        })
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    for cond_name, (k, filler_type) in conditions.items():
        cond_dir = output_dir / cond_name
        cond_dir.mkdir(exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Condition: {cond_name} (k={k}, filler_type={filler_type})")
        print(f"{'='*60}")

        correct_count = 0
        total_count = 0
        t_start = time.time()

        for prob_idx, problem in enumerate(tqdm(problems, desc=cond_name)):
            save_path = cond_dir / f"prob_{prob_idx:04d}.pkl"

            if skip_existing and save_path.exists():
                continue

            # Per-example deterministic RNG (for scrambled filler reproducibility)
            example_rng = random.Random(prob_idx)

            # Build prompt (supports all filler types, matches eval format)
            messages = build_messages_for_condition(
                few_shot[:5], problem, filler_type, k, rng=example_rng
            )
            full_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(full_text, return_tensors="pt")
            input_ids = inputs["input_ids"].to(input_device)
            attention_mask = inputs["attention_mask"].to(input_device)
            seq_len = input_ids.shape[1]

            # Find positions
            filler_start, filler_end = None, None
            if k > 0:
                filler_start, filler_end = find_filler_boundaries(
                    tokenizer, input_ids, k
                )
                positions = compute_extraction_positions(
                    filler_start, filler_end, seq_len,
                    absolute_positions=filler_positions,
                )
            else:
                positions = compute_baseline_positions(seq_len)

            # Sanity check bounds
            for pos_name, pos_idx in positions.items():
                assert 0 <= pos_idx < seq_len, (
                    f"Position '{pos_name}' = {pos_idx} out of bounds "
                    f"(seq_len={seq_len}) for problem {prob_idx}"
                )

            # Extract hidden states (forward pass 1)
            states = extractor.extract(input_ids, attention_mask, positions)

            # Get model's answer (forward pass 2)
            model_response, model_answer, is_correct = None, None, None
            if also_generate:
                model_response, model_answer = get_model_answer(
                    model, tokenizer, input_ids, attention_mask
                )
                is_correct = (model_answer == problem["answer"])
                correct_count += int(is_correct)
                total_count += 1

            # Save
            result = {
                "problem_idx": prob_idx,
                "condition": cond_name,
                "k": k,
                "fact_value": problem["fact_value"],
                "x": problem["x"],
                "answer": problem["answer"],
                "positions": positions,
                "states": states,
                "model_response": model_response,
                "model_answer": model_answer,
                "model_correct": is_correct,
                "seq_len": seq_len,
                "filler_start": filler_start,
                "filler_end": filler_end,
            }

            with open(save_path, "wb") as f:
                pickle.dump(result, f)

            del input_ids, attention_mask, states
            torch.cuda.empty_cache()

        elapsed = time.time() - t_start
        if total_count > 0:
            print(f"\n  Accuracy: {correct_count}/{total_count} = {correct_count/total_count:.1%}")
        print(f"  Time: {elapsed:.0f}s ({elapsed/len(problems):.1f}s per problem)")

    extractor.remove_hooks()
    print("\nExtraction complete!")


# ==============================================================================
# Verification utility
# ==============================================================================

def verify_extraction(output_dir: str, condition_name: str, n_check: int = 3):
    """Load a few extracted files and print diagnostics."""
    cond_dir = Path(output_dir) / condition_name
    files = sorted(cond_dir.glob("prob_*.pkl"))[:n_check]

    for f in files:
        with open(f, "rb") as fp:
            data = pickle.load(fp)

        print(f"\n--- {f.name} ---")
        print(f"  fact_value={data['fact_value']}, x={data['x']}, answer={data['answer']}")
        print(f"  Model answer: {data['model_answer']} "
              f"({'correct' if data['model_correct'] else 'wrong'})")
        print(f"  Seq length: {data['seq_len']}")
        print(f"  Filler: [{data['filler_start']}, {data['filler_end']}]")
        print(f"  Positions: {data['positions']}")

        for pos_name, layer_dict in data["states"].items():
            n_layers = len(layer_dict)
            sample_layer = list(layer_dict.keys())[0]
            shape = layer_dict[sample_layer].shape
            print(f"  States['{pos_name}']: {n_layers} layers, shape={shape}")

        all_vecs = []
        for pos_name in data["states"]:
            for layer_idx in data["states"][pos_name]:
                vec = data["states"][pos_name][layer_idx]
                all_vecs.append(vec)
                if np.any(np.isnan(vec)):
                    print(f"  WARNING: NaN in states[{pos_name}][{layer_idx}]")
                if np.all(vec == 0):
                    print(f"  WARNING: All zeros in states[{pos_name}][{layer_idx}]")

        all_vecs = np.stack(all_vecs)
        print(f"  Value range: [{all_vecs.min():.4f}, {all_vecs.max():.4f}], "
              f"mean|x|={np.abs(all_vecs).mean():.4f}")


# ==============================================================================
# Entry point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/workspace/models/deepseek-v3-awq")
    parser.add_argument("--dataset", type=Path,
                        default=Path("probing/data/1hop_addition_dataset.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("probing/extracted_states"))
    parser.add_argument("--layers", default="all",
                        help="'all', 'every4', or comma-separated indices")
    parser.add_argument("--conditions", nargs="+",
                        default=["baseline", "dots_250"])
    parser.add_argument("--max-problems", type=int, default=None)
    parser.add_argument("--no-skip-existing", action="store_true",
                        help="Re-extract even if output files exist")
    parser.add_argument("--no-generate", action="store_true",
                        help="Skip answer generation (faster, no accuracy tracking)")
    parser.add_argument("--verify-only", action="store_true",
                        help="Just verify existing extractions, don't run")
    parser.add_argument("--filler-positions", type=str, default=None,
                        help="Comma-separated absolute filler token offsets "
                             "(e.g. '1,5,10,25,50'). Overrides fraction-based positions.")
    args = parser.parse_args()

    # Load dataset
    with open(args.dataset) as f:
        dataset = json.load(f)
    few_shot = dataset["few_shot_facts"]
    problems = dataset["examples"]
    if args.max_problems:
        problems = problems[:args.max_problems]
    print(f"Loaded {len(problems)} problems, {len(few_shot)} few-shot facts")

    # Select conditions
    selected = {k: CONDITIONS[k] for k in args.conditions}
    print(f"Conditions: {selected}")

    # Determine layers
    num_layers = 61  # DeepSeek V3
    if args.layers == "all":
        layer_indices = list(range(num_layers))
    elif args.layers == "every4":
        layer_indices = list(range(0, num_layers, 4))
    else:
        layer_indices = [int(x) for x in args.layers.split(",")]
    print(f"Extracting from {len(layer_indices)} layers")

    if args.verify_only:
        for cond_name in selected:
            print(f"\nVerifying {cond_name}:")
            verify_extraction(str(args.output_dir), cond_name)
        return

    # Parse filler positions
    filler_pos = None
    if args.filler_positions:
        filler_pos = [int(x) for x in args.filler_positions.split(",")]
        print(f"Using absolute filler positions: {filler_pos}")

    # Storage estimate
    if filler_pos:
        n_positions = len(filler_pos) + 2  # absolute positions + question_end + answer_prompt
    else:
        n_positions = len(FILLER_FRACTIONS) + 2  # filler fracs + question_end + answer_prompt
    bytes_per_problem = n_positions * len(layer_indices) * 7168 * 2  # fp16
    total_gb = len(problems) * len(selected) * bytes_per_problem / 1e9
    print(f"Estimated storage: {total_gb:.1f} GB")

    # Load model
    model, tokenizer = load_model(args.model_path)

    # Run extraction
    run_extraction(
        model=model,
        tokenizer=tokenizer,
        problems=problems,
        few_shot=few_shot,
        conditions=selected,
        layer_indices=layer_indices,
        output_dir=args.output_dir,
        skip_existing=not args.no_skip_existing,
        also_generate=not args.no_generate,
        filler_positions=filler_pos,
    )

    # Verify
    for cond_name in selected:
        verify_extraction(str(args.output_dir), cond_name)


if __name__ == "__main__":
    main()
