"""
patchscope_single.py

Core patchscoping engine. Takes a saved activation vector (7168,) and patches it
into an inspection prompt's residual stream at a specified (layer, token position),
then generates a completion.

Usage:
    # As a module
    from patchscope_single import run_patchscope, load_model_and_tokenizer

    text = run_patchscope(
        model=model,
        tokenizer=tokenizer,
        source_activation=vec,        # (7168,) numpy or tensor
        inspection_prompt="The number is",
        target_layer=50,
        patch_position=-1,            # -1 = last token (default)
        max_new_tokens=20,
    )

    # As a standalone test
    python patchscope_single.py
"""

import sys
import torch
import numpy as np
import pickle
import os
import glob

# ============================================================
# CONFIG — edit these for standalone testing
# ============================================================
DATA_DIR = "/workspace/filler-token-reasoning/data/extracted_states_2fact_allpos"
CONDITION = "dots_10"

# Test parameters
TEST_SOURCE_POS = "pos_001"   # offset 1 from question_end
TEST_SOURCE_LAYER = 50
TEST_INSPECTION_PROMPT = "The number is"
TEST_TARGET_LAYER = 50
TEST_MAX_NEW_TOKENS = 20


# ============================================================
# MODEL LOADING
# ============================================================

MODEL_PATH = "/workspace/models/deepseek-v3-awq"

def load_model_and_tokenizer(model_path=MODEL_PATH):
    """Load model and tokenizer using the project's existing loader."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
        from extract_hidden_states import load_model
        model, tokenizer = load_model(model_path)
        print(f"Loaded model from {model_path}")
        return model, tokenizer
    except (ImportError, Exception) as e:
        print(f"Could not use extract_hidden_states.load_model(): {e}")
        print("Please load the model manually and pass it to run_patchscope().")
        sys.exit(1)


# ============================================================
# ACTIVATION LOADING
# ============================================================

def load_correct_examples(data_dir, condition, max_examples=None):
    """Load correct examples from pickle files. Returns list of example dicts."""
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
        
        if max_examples and len(correct) >= max_examples:
            break
    
    return correct[:max_examples] if max_examples else correct


def get_activation(example, pos_key, layer):
    """Extract a single activation vector from an example.
    
    Args:
        example: dict with 'states' key
        pos_key: e.g. 'pos_001'
        layer: int, e.g. 50
    
    Returns:
        torch.Tensor of shape (7168,), float16
    """
    vec = example["states"][pos_key][layer]
    if isinstance(vec, np.ndarray):
        vec = torch.from_numpy(vec)
    return vec.to(torch.float16)


# ============================================================
# PATCHSCOPING
# ============================================================

def run_patchscope(
    model,
    tokenizer,
    source_activation,
    inspection_prompt,
    target_layer,
    patch_position=-1,
    max_new_tokens=20,
    temperature=0.0,
    verbose=False,
):
    """
    Run patchscoping: inject source_activation into inspection_prompt at target_layer,
    then generate a completion.
    
    Args:
        model: HuggingFace CausalLM
        tokenizer: corresponding tokenizer
        source_activation: (d_model,) tensor or numpy array — the vector to patch in
        inspection_prompt: str — the prompt to run through the model
        target_layer: int — which layer to patch at (0-indexed)
        patch_position: int — which token position to patch at (-1 = last token)
        max_new_tokens: int — how many tokens to generate after patching
        temperature: float — 0.0 for greedy
        verbose: bool — print debug info
    
    Returns:
        str — the generated completion (excluding the inspection prompt)
    """
    # Prepare the activation tensor
    if isinstance(source_activation, np.ndarray):
        source_activation = torch.from_numpy(source_activation)
    source_activation = source_activation.to(torch.float16)
    
    # Tokenize the inspection prompt
    inputs = tokenizer(inspection_prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    seq_len = input_ids.shape[1]
    
    # Resolve patch position
    if patch_position < 0:
        patch_position = seq_len + patch_position  # -1 -> last token
    
    if verbose:
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        print(f"Inspection prompt: {repr(inspection_prompt)}")
        print(f"Tokens ({seq_len}): {tokens}")
        print(f"Patching at: layer {target_layer}, position {patch_position}")
        print(f"Source activation: shape={source_activation.shape}, "
              f"dtype={source_activation.dtype}, "
              f"norm={source_activation.float().norm():.2f}")
    
    # --- Hook setup ---
    # We patch the residual stream at the INPUT to the target layer.
    # In HuggingFace transformers, model.model.layers[L] receives hidden_states
    # as its first positional argument. A forward pre-hook can modify this.
    
    hook_fired = {"count": 0}
    
    def patch_hook(module, args):
        """Forward pre-hook that replaces hidden_states at patch_position."""
        # args[0] is hidden_states: (batch, seq_len, d_model)
        hidden_states = args[0]
        current_seq_len = hidden_states.shape[1]
        
        # During generation with KV cache, after the first forward pass,
        # hidden_states will have seq_len=1 (just the new token).
        # We only want to patch during the initial forward pass.
        if current_seq_len == 1:
            return args
        
        # Patch: overwrite the activation at patch_position
        device = hidden_states.device
        activation = source_activation.to(device)
        
        # Clone to avoid in-place modification issues
        hidden_states = hidden_states.clone()
        hidden_states[0, patch_position, :] = activation
        
        hook_fired["count"] += 1
        
        # Return modified args — reconstruct the full args tuple
        return (hidden_states,) + args[1:]
    
    # Register the hook on the target layer
    target_module = model.model.layers[target_layer]
    handle = target_module.register_forward_pre_hook(patch_hook)
    
    try:
        # Generate
        with torch.no_grad():
            if temperature == 0.0:
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )
            else:
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.95,
                )
        
        # Decode only the generated part
        generated_ids = output_ids[0, seq_len:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        if verbose:
            print(f"Hook fired: {hook_fired['count']} time(s)")
            print(f"Generated: {repr(generated_text)}")
    
    finally:
        # Always remove the hook
        handle.remove()
    
    return generated_text


# ============================================================
# STANDALONE TEST
# ============================================================

def run_test():
    """
    Quick test: load one correct example, patch its activation into an
    inspection prompt, and print the result. Also run without patching
    for comparison.
    """
    print("=" * 60)
    print("PATCHSCOPE SINGLE — STANDALONE TEST")
    print("=" * 60)
    
    # Load one correct example
    print("\nLoading examples...")
    correct = load_correct_examples(DATA_DIR, CONDITION, max_examples=5)
    if not correct:
        print("No correct examples found!")
        sys.exit(1)
    
    ex = correct[0]
    a1 = ex["fact_value_1"]
    a2 = ex["fact_value_2"]
    s = ex["answer"]
    print(f"Using example: A1={a1}, A2={a2}, sum={s}")
    
    # Get the activation
    activation = get_activation(ex, TEST_SOURCE_POS, TEST_SOURCE_LAYER)
    print(f"Source activation: {TEST_SOURCE_POS}, layer {TEST_SOURCE_LAYER}")
    print(f"  shape={activation.shape}, norm={activation.float().norm():.2f}")
    
    # Load model
    print("\nLoading model...")
    model, tokenizer = load_model_and_tokenizer()
    
    # --- Baseline: run inspection prompt WITHOUT patching ---
    print(f"\n{'='*40}")
    print("BASELINE (no patching)")
    print(f"{'='*40}")
    inputs = tokenizer(TEST_INSPECTION_PROMPT, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=TEST_MAX_NEW_TOKENS, do_sample=False)
    baseline_text = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True)
    print(f"Prompt: {repr(TEST_INSPECTION_PROMPT)}")
    print(f"Completion: {repr(baseline_text)}")
    
    # --- Patchscoped: run with patching ---
    print(f"\n{'='*40}")
    print("PATCHSCOPED")
    print(f"{'='*40}")
    result = run_patchscope(
        model=model,
        tokenizer=tokenizer,
        source_activation=activation,
        inspection_prompt=TEST_INSPECTION_PROMPT,
        target_layer=TEST_TARGET_LAYER,
        patch_position=-1,
        max_new_tokens=TEST_MAX_NEW_TOKENS,
        verbose=True,
    )
    
    # --- Check if result contains expected values ---
    print(f"\n{'='*40}")
    print("RESULT CHECK")
    print(f"{'='*40}")
    print(f"Ground truth: A1={a1}, A2={a2}, sum={s}")
    print(f"Generated:    {repr(result)}")
    print(f"Contains A1 ({a1})?  {str(a1) in result}")
    print(f"Contains A2 ({a2})?  {str(a2) in result}")
    print(f"Contains sum ({s})? {str(s) in result}")
    
    # --- Run a few more examples for quick sanity check ---
    print(f"\n{'='*40}")
    print("QUICK SWEEP (same config, more examples)")
    print(f"{'='*40}")
    for i, ex in enumerate(correct[:5]):
        a1 = ex["fact_value_1"]
        a2 = ex["fact_value_2"]
        s = ex["answer"]
        act = get_activation(ex, TEST_SOURCE_POS, TEST_SOURCE_LAYER)
        
        gen = run_patchscope(
            model=model,
            tokenizer=tokenizer,
            source_activation=act,
            inspection_prompt=TEST_INSPECTION_PROMPT,
            target_layer=TEST_TARGET_LAYER,
            max_new_tokens=TEST_MAX_NEW_TOKENS,
        )
        
        # flag which values appear
        flags = []
        if str(a1) in gen:
            flags.append("A1")
        if str(a2) in gen:
            flags.append("A2")
        if str(s) in gen:
            flags.append("SUM")
        flag_str = ",".join(flags) if flags else "NONE"
        
        print(f"  ex {i}: A1={a1:>3}, A2={a2:>3}, sum={s:>3} | "
              f"gen={repr(gen[:60]):62s} | matches: {flag_str}")


if __name__ == "__main__":
    run_test()