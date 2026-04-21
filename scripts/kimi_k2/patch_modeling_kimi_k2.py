"""
patch_modeling_kimi_k2.py

Apply patches required by Kimi K2 (DeepseekV3ForCausalLM) under transformers 4.57+.

Patches:
  1. Remove `from transformers.utils.import_utils import is_torch_fx_available`
  2. Make the `if is_torch_fx_available():` guard unconditional
  3. `_init_weights`: skip modules without `.weight` (compressed-tensors uses `weight_packed`)
  4. DynamicCache: `seen_tokens` / `get_max_length()` / `get_usable_length()` are gone in 4.57;
     replace with `get_seq_length()` and add hasattr guards.
  5. Apply to both the source file and the HF cache copy, then delete .pyc.
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


PATCHES: list[tuple[str, str, str]] = [
    # (description, old_regex, new_string). Uses re.sub on full file text.
    (
        "remove is_torch_fx_available import",
        r"^from transformers\.utils\.import_utils import is_torch_fx_available\n",
        "",
    ),
    (
        "make is_torch_fx_available guard unconditional",
        r"if is_torch_fx_available\(\):\s*\n(\s+)_prepare_4d_causal_attention_mask = torch\.fx\.wrap\(_prepare_4d_causal_attention_mask\)\s*\n",
        r"\1_prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)\n",
    ),
    (
        "seen_tokens -> getattr fallback",
        r"past_key_values\.seen_tokens",
        "getattr(past_key_values, 'seen_tokens', past_key_values.get_seq_length())",
    ),
    (
        "get_max_length -> None fallback",
        r"past_key_values\.get_max_length\(\)",
        "(past_key_values.get_max_length() if hasattr(past_key_values, 'get_max_length') else None)",
    ),
    (
        "get_usable_length -> get_seq_length",
        r"past_key_values\.get_usable_length\(\s*([^)]*?)\s*\)",
        r"past_key_values.get_seq_length()",
    ),
]


INIT_WEIGHTS_PATCH_OLD = """    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()"""

INIT_WEIGHTS_PATCH_NEW = """    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            if hasattr(module, "weight"):
                module.weight.data.normal_(mean=0.0, std=std)
            if getattr(module, "bias", None) is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            if hasattr(module, "weight"):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()"""


def apply_patches(path: Path, verbose: bool = True) -> None:
    text = path.read_text()
    original = text

    for desc, pattern, repl in PATCHES:
        new_text, n = re.subn(pattern, repl, text, flags=re.MULTILINE)
        if verbose:
            print(f"  [{n}x] {desc}")
        text = new_text

    if INIT_WEIGHTS_PATCH_OLD in text:
        text = text.replace(INIT_WEIGHTS_PATCH_OLD, INIT_WEIGHTS_PATCH_NEW)
        if verbose:
            print("  [1x] _init_weights hasattr guard")
    elif INIT_WEIGHTS_PATCH_NEW in text:
        if verbose:
            print("  [skip] _init_weights already patched")
    else:
        if verbose:
            print("  [WARN] _init_weights block not found exactly; inspect manually")

    if text == original:
        if verbose:
            print(f"  no changes applied to {path}")
        return

    backup = path.with_suffix(path.suffix + ".bak")
    if not backup.exists():
        shutil.copy2(path, backup)
        if verbose:
            print(f"  backed up to {backup.name}")
    path.write_text(text)
    if verbose:
        print(f"  wrote patched {path}")


def delete_pyc(dir_path: Path) -> None:
    for pyc in dir_path.rglob("*.pyc"):
        pyc.unlink()
        print(f"  rm {pyc}")


def find_hf_cache_copy(model_path: Path) -> Path | None:
    """Locate the HF cache copy of modeling_deepseek.py for trust_remote_code loads."""
    hf_home = Path("/workspace/.cache/huggingface/modules/transformers_modules")
    if not hf_home.exists():
        return None
    # HF replaces non-alphanumerics with underscores; match any that contains 'kimi'
    candidates = sorted(hf_home.glob("*kimi*/modeling_deepseek.py"))
    return candidates[0] if candidates else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model-path",
        default="/workspace/models/kimi-k2-awq",
        type=Path,
    )
    args = ap.parse_args()

    source = args.model_path / "modeling_deepseek.py"
    print(f"Source: {source}")
    if not source.exists():
        raise SystemExit(f"not found: {source}")
    apply_patches(source)
    delete_pyc(args.model_path)

    cache = find_hf_cache_copy(args.model_path)
    if cache:
        print(f"\nHF cache copy: {cache}")
        apply_patches(cache)
        delete_pyc(cache.parent)
    else:
        print("\nNo HF cache copy yet (will patch after first load if needed).")


if __name__ == "__main__":
    main()
