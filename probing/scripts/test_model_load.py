#!/usr/bin/env python3
"""Test loading DeepSeek V3 in 4-bit quantization on available GPUs.

Usage:
    source /workspace/config/probing_env.sh
    uv run python probing/scripts/test_model_load.py
"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "deepseek-ai/DeepSeek-V3"
MODEL_PATH = "/workspace/models/deepseek-v3"


def load_model_4bit(model_path: str = MODEL_PATH):
    """Load DeepSeek V3 in 4-bit quantization across GPUs."""
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )

    print(f"Loading model in 4-bit from {model_path}...")
    print(f"GPUs available: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}, {props.total_memory / 1e9:.1f} GB")

    # Reserve some VRAM for activations/KV cache; offload remainder to CPU
    max_memory = {
        i: f"{int(torch.cuda.get_device_properties(i).total_memory * 0.92 / 1e9)}GiB"
        for i in range(torch.cuda.device_count())
    }
    max_memory["cpu"] = "64GiB"
    print(f"Max memory config: {max_memory}")

    import os
    from pathlib import Path as _Path
    offload_dir = "/workspace/.offload"
    _Path(offload_dir).mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory=max_memory,
        torch_dtype=torch.bfloat16,
        offload_folder=offload_dir,
    )
    model.eval()
    load_time = time.time() - t0

    print(f"\nModel loaded in {load_time:.1f}s")
    print(f"Model type: {type(model).__name__}")
    print(f"Config hidden_size: {model.config.hidden_size}")
    print(f"Config num_hidden_layers: {model.config.num_hidden_layers}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Memory usage
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1e9
        res = torch.cuda.memory_reserved(i) / 1e9
        print(f"  GPU {i}: {alloc:.1f} GB allocated, {res:.1f} GB reserved")

    return model, tokenizer


def test_generation(model, tokenizer):
    """Quick sanity check: generate a short response."""
    messages = [
        {"role": "system", "content": "Answer with just the number."},
        {"role": "user", "content": "What is 2 + 2?"},
    ]

    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    print(f"\nTest generation (input length: {inputs['input_ids'].shape[1]} tokens)...")
    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
        )
    gen_time = time.time() - t0

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    print(f"Response: {response!r} ({gen_time:.2f}s)")


def test_hook(model):
    """Verify we can hook into transformer layers and extract hidden states."""
    captured = {}

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        captured["shape"] = hidden.shape
        captured["dtype"] = hidden.dtype
        captured["device"] = hidden.device

    # Hook into the first layer
    layer_0 = model.model.layers[0]
    handle = layer_0.register_forward_hook(hook_fn)

    # Run a dummy forward pass
    dummy = torch.tensor([[1, 2, 3]], device=model.device)
    with torch.no_grad():
        model(input_ids=dummy)

    handle.remove()

    print(f"\nHook test on layer 0:")
    print(f"  Output shape: {captured['shape']}")
    print(f"  Output dtype: {captured['dtype']}")
    print(f"  Output device: {captured['device']}")
    print(f"  Hidden dim: {captured['shape'][-1]}")


if __name__ == "__main__":
    model, tokenizer = load_model_4bit()
    test_generation(model, tokenizer)
    test_hook(model)
    print("\nAll tests passed!")
