"""
run_experiment_grid.py (logit inspection)

For each (example, source_layer, source_pos, template, target_layer):
  1. Tokenize the inspection prompt
  2. Single forward pass with hook patching source activation into residual stream
  3. Extract logits at last position
  4. Record rank and probability of A1, A2, sum tokens, plus top-K

Fully self-contained. Supports resuming from partial CSV.

Usage:
    python run_experiment_grid.py
    python run_experiment_grid.py --output results/patchscope_grid.csv
    python run_experiment_grid.py --num-examples 10
"""

import argparse
import csv
import os
import sys
import time
import pickle
import glob
import numpy as np
import torch
import torch.nn.functional as F


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = "/workspace/filler-token-reasoning/data/extracted_states_2fact_allpos"
CONDITION = "dots_10"
DEFAULT_OUTPUT = "results/patchscope_grid.csv"
DEFAULT_NUM_EXAMPLES = 50
TOP_K = 50

SOURCE_CONFIGS = [
    (50, "pos_001", "logit_lens_decodes_A1"),
    (51, "pos_006", "logit_lens_kinda_A1"),
    (51, "pos_005", "logit_lens_kinda_A2"),
    (53, "pos_005", "logit_lens_kinda_A2_2"),
    (58, "pos_015", "late_layer"),
    # (20, "pos_001", "early_negative_control"),
]

TEMPLATES = [
    "The number is",
    "The answer is",
    "I am thinking about",
    "The value stored here is",
    "The second number in the addition is",
    "The atomic number of the second element is",
    "The first number in the addition is",
    "The sum is",
]

TARGET_LAYER_OFFSETS = [-20, -10, 0, 5]


# ============================================================
# MODEL LOADING
# ============================================================

def load_model_and_tokenizer():
    """Load model and tokenizer using the project's existing loader."""
    try:
        sys.path.insert(0, "/workspace/filler-token-reasoning/scripts")
        from extract_hidden_states import load_model
        model, tokenizer = load_model("/workspace/models/deepseek-v3-awq")
        print("Loaded model via extract_hidden_states.load_model()")
        return model, tokenizer
    except (ImportError, Exception) as e:
        print(f"Could not use extract_hidden_states.load_model(): {e}")
        sys.exit(1)


# ============================================================
# TOKEN LOOKUP
# ============================================================

def build_number_token_map(tokenizer, max_val=200):
    """
    Map integer values 0..max_val to token IDs.
    Prefers space-prefixed tokens (e.g. " 42") since those follow "is" in templates.
    """
    num_to_token_id = {}
    for n in range(max_val + 1):
        space_ids = tokenizer.encode(f" {n}", add_special_tokens=False)
        if len(space_ids) == 1:
            num_to_token_id[n] = space_ids[0]
        else:
            bare_ids = tokenizer.encode(str(n), add_special_tokens=False)
            if len(bare_ids) == 1:
                num_to_token_id[n] = bare_ids[0]
            else:
                num_to_token_id[n] = space_ids[0] if space_ids else bare_ids[0]
    return num_to_token_id


# ============================================================
# DATA LOADING
# ============================================================

def load_correct_examples(data_dir, condition):
    """Load all correct examples from pickle files."""
    condition_dir = os.path.join(data_dir, condition)
    files = sorted(glob.glob(os.path.join(condition_dir, "*.pkl")))
    correct = []
    for f in files:
        with open(f, "rb") as fh:
            ex = pickle.load(fh)
        if isinstance(ex, list):
            for e in ex:
                if e.get("model_correct", False):
                    correct.append(e)
        elif isinstance(ex, dict):
            if ex.get("model_correct", False):
                correct.append(ex)
    return correct


def select_diverse_examples(examples, n):
    """Greedy farthest-point sampling in (A1, A2) space."""
    if n >= len(examples):
        return examples

    pairs = np.array([(ex["fact_value_1"], ex["fact_value_2"]) for ex in examples])
    selected = []
    median = np.median(pairs, axis=0)
    dists = np.linalg.norm(pairs - median, axis=1)
    selected.append(int(np.argmin(dists)))

    for _ in range(n - 1):
        min_dists = np.full(len(examples), np.inf)
        for si in selected:
            d = np.linalg.norm(pairs - pairs[si], axis=1)
            min_dists = np.minimum(min_dists, d)
        min_dists[selected] = -1
        selected.append(int(np.argmax(min_dists)))

    result = [examples[i] for i in selected]
    a1s = [ex["fact_value_1"] for ex in result]
    a2s = [ex["fact_value_2"] for ex in result]
    print(f"Selected {n} examples:")
    print(f"  A1 range: {min(a1s)}-{max(a1s)}, unique: {len(set(a1s))}")
    print(f"  A2 range: {min(a2s)}-{max(a2s)}, unique: {len(set(a2s))}")
    return result


def get_activation(example, pos_key, layer):
    """Extract activation vector (7168,) from saved states."""
    vec = example["states"][pos_key][layer]
    if isinstance(vec, np.ndarray):
        vec = torch.from_numpy(vec)
    return vec.to(torch.float16)


# ============================================================
# PATCHSCOPE LOGIT INSPECTION
# ============================================================

def run_patchscope_logits(model, tokenizer, source_activation, inspection_prompt,
                          target_layer, patch_position=-1):
    """
    Single forward pass with patched activation. Returns logits at last position.

    Args:
        model: HuggingFace CausalLM
        tokenizer: tokenizer
        source_activation: (7168,) tensor — vector to patch in
        inspection_prompt: str — the prompt to process
        target_layer: int — layer to patch at (0-indexed)
        patch_position: int — token position to patch (-1 = last)

    Returns:
        logits: (vocab_size,) tensor at last token position
    """
    if isinstance(source_activation, np.ndarray):
        source_activation = torch.from_numpy(source_activation)
    source_activation = source_activation.to(torch.float16)

    inputs = tokenizer(inspection_prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).to(model.device)
    seq_len = input_ids.shape[1]

    if patch_position < 0:
        patch_position = seq_len + patch_position

    def patch_hook(module, args):
        hidden_states = args[0]
        # Only patch on the full-sequence pass, not single-token KV-cache steps
        if hidden_states.shape[1] == 1:
            return args
        device = hidden_states.device
        activation = source_activation.to(device)
        hidden_states = hidden_states.clone()
        hidden_states[0, patch_position, :] = activation
        return (hidden_states,) + args[1:]

    handle = model.model.layers[target_layer].register_forward_pre_hook(patch_hook)
    try:
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[0, -1, :]
    finally:
        handle.remove()

    return logits


def analyze_logits(logits, a1, a2, total, num_to_token_id, tokenizer, top_k=50):
    """
    Compute ranks/probs for target values and extract top-K tokens.

    Returns dict with: A1_rank, A1_prob, A2_rank, A2_prob, sum_rank, sum_prob,
                        diff_rank, diff_prob, product_rank, product_prob,
                        top_1_token, top_1_prob, top_1_number,
                        top_k_tokens, top_k_probs
    """
    probs = F.softmax(logits.float(), dim=-1)
    vocab_size = probs.shape[0]

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)

    # Build rank lookup: token_id -> rank (0-indexed)
    rank_lookup = torch.zeros(vocab_size, dtype=torch.long, device=logits.device)
    rank_lookup[sorted_indices] = torch.arange(vocab_size, device=logits.device)

    result = {}

    # Target values
    for name, value in [("A1", a1), ("A2", a2), ("sum", total)]:
        token_id = num_to_token_id.get(value)
        if token_id is not None:
            result[f"{name}_rank"] = rank_lookup[token_id].item()
            result[f"{name}_prob"] = probs[token_id].item()
        else:
            result[f"{name}_rank"] = -1
            result[f"{name}_prob"] = 0.0

    # Derived values
    for name, value in [("diff", abs(a1 - a2))]:
        token_id = num_to_token_id.get(value)
        if token_id is not None:
            result[f"{name}_rank"] = rank_lookup[token_id].item()
            result[f"{name}_prob"] = probs[token_id].item()
        else:
            result[f"{name}_rank"] = -1
            result[f"{name}_prob"] = 0.0

    # Top-K
    top_k_ids = sorted_indices[:top_k].cpu().tolist()
    top_k_probs_list = sorted_probs[:top_k].cpu().tolist()
    top_k_tokens = [tokenizer.decode([tid]) for tid in top_k_ids]

    result["top_k_tokens"] = ";".join(top_k_tokens)
    result["top_k_probs"] = ";".join(f"{p:.6f}" for p in top_k_probs_list)
    result["top_1_token"] = top_k_tokens[0]
    result["top_1_prob"] = top_k_probs_list[0]

    try:
        result["top_1_number"] = int(top_k_tokens[0].strip())
    except ValueError:
        result["top_1_number"] = ""

    return result


# ============================================================
# GRID LOGIC
# ============================================================

def build_grid(examples):
    grid = []
    for ex_idx, ex in enumerate(examples):
        for source_layer, source_pos, desc in SOURCE_CONFIGS:
            for template in TEMPLATES:
                for offset in TARGET_LAYER_OFFSETS:
                    target_layer = max(0, min(60, source_layer + offset))
                    grid.append({
                        "example_idx": ex_idx,
                        "problem_idx": ex.get("problem_idx", ex_idx),
                        "A1": ex["fact_value_1"],
                        "A2": ex["fact_value_2"],
                        "sum": ex["answer"],
                        "source_layer": source_layer,
                        "source_pos": source_pos,
                        "source_desc": desc,
                        "template": template,
                        "target_layer": target_layer,
                    })
    return grid


def make_row_key(row):
    return (
        int(row["problem_idx"]),
        int(row["source_layer"]),
        row["source_pos"],
        row["template"],
        int(row["target_layer"]),
    )


def load_completed(output_path):
    completed = set()
    if not os.path.exists(output_path):
        return completed
    with open(output_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            completed.add(make_row_key(row))
    return completed


CSV_COLUMNS = [
    "example_idx", "problem_idx", "A1", "A2", "sum",
    "source_layer", "source_pos", "source_desc",
    "template", "target_layer",
    "A1_rank", "A1_prob",
    "A2_rank", "A2_prob",
    "sum_rank", "sum_prob",
    "diff_rank", "diff_prob",
    "top_1_token", "top_1_prob", "top_1_number",
    "top_k_tokens", "top_k_probs",
]


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Run patchscoping grid (logit inspection)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--num-examples", type=int, default=DEFAULT_NUM_EXAMPLES)
    args = parser.parse_args()

    output_path = args.output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Load examples
    print("Loading correct examples...")
    all_correct = load_correct_examples(DATA_DIR, CONDITION)
    print(f"Total correct: {len(all_correct)}")
    examples = select_diverse_examples(all_correct, args.num_examples)

    # Build grid
    grid = build_grid(examples)
    print(f"\nTotal grid size: {len(grid)} forward passes")
    print(f"  = {len(examples)} examples"
          f" x {len(SOURCE_CONFIGS)} source configs"
          f" x {len(TEMPLATES)} templates"
          f" x {len(TARGET_LAYER_OFFSETS)} target offsets")

    # Resume check
    completed = load_completed(output_path)
    remaining = [row for row in grid if make_row_key(row) not in completed]
    print(f"\nAlready completed: {len(completed)}")
    print(f"Remaining: {len(remaining)}")

    if not remaining:
        print("All done! Nothing to run.")
        return

    est_seconds = len(remaining) * 1.0
    print(f"Estimated time: {est_seconds/60:.0f} minutes ({est_seconds/3600:.1f} hours)")

    # Load model
    print("\nLoading model...")
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer()
    print(f"Model loaded in {time.time()-t0:.0f}s")

    # Build number -> token_id map
    print("Building number token map...")
    num_to_token_id = build_number_token_map(tokenizer, max_val=200)
    for test_val in [1, 8, 42, 99, 150]:
        tid = num_to_token_id.get(test_val)
        decoded = tokenizer.decode([tid]) if tid else "N/A"
        print(f"  {test_val} -> token_id={tid}, decoded={repr(decoded)}")

    # Show template tokenizations
    print("Template tokenizations:")
    for template in TEMPLATES:
        toks = tokenizer.convert_ids_to_tokens(
            tokenizer.encode(template, add_special_tokens=False))
        print(f"  \"{template}\" -> {len(toks)+1} tokens (incl BOS)")

    # Open CSV for appending
    file_exists = os.path.exists(output_path) and os.path.getsize(output_path) > 0
    csvfile = open(output_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csvfile, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    if not file_exists:
        writer.writeheader()
        csvfile.flush()

    # Run grid
    n_done = 0
    n_total = len(remaining)
    t_start = time.time()

    for row in remaining:
        ex = examples[row["example_idx"]]
        activation = get_activation(ex, row["source_pos"], row["source_layer"])

        try:
            logits = run_patchscope_logits(
                model=model,
                tokenizer=tokenizer,
                source_activation=activation,
                inspection_prompt=row["template"],
                target_layer=row["target_layer"],
                patch_position=-1,
            )

            scored = analyze_logits(
                logits=logits,
                a1=row["A1"],
                a2=row["A2"],
                total=row["sum"],
                num_to_token_id=num_to_token_id,
                tokenizer=tokenizer,
                top_k=TOP_K,
            )
            row.update(scored)

        except Exception as e:
            row["A1_rank"] = -1
            row["A2_rank"] = -1
            row["sum_rank"] = -1
            row["diff_rank"] = -1
            row["top_1_token"] = f"ERROR: {e}"
            print(f"  ERROR on row {n_done}: {e}")

        writer.writerow(row)
        n_done += 1

        if n_done % 100 == 0 or n_done == n_total:
            csvfile.flush()
            elapsed = time.time() - t_start
            rate = n_done / elapsed
            eta = (n_total - n_done) / rate if rate > 0 else 0

            a1r = row.get("A1_rank", "?")
            a2r = row.get("A2_rank", "?")
            sr = row.get("sum_rank", "?")
            top1 = row.get("top_1_token", "?")

            print(f"  [{n_done}/{n_total}] "
                  f"{rate:.1f}/s, ETA {eta/60:.0f}min | "
                  f"A1={row['A1']}, A2={row['A2']} | "
                  f"L{row['source_layer']}/{row['source_pos']}->L{row['target_layer']} | "
                  f"ranks: A1={a1r}, A2={a2r}, S={sr} | "
                  f"top1={repr(top1)}")

    csvfile.close()
    elapsed = time.time() - t_start
    print(f"\nDone. {n_done} forward passes in {elapsed/60:.1f} minutes "
          f"({n_done/elapsed:.1f}/s).")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()