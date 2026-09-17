"""
models.py — per-model paths and shapes for the J-lens analyses: DeepSeek-V3 and the Qwen3.5 comparison.

Every analysis script takes --model (default "v3", which reproduces the original V3 runs). Qwen3.5 states
are the extract_qwen35_states.py outputs copied back from the GPU box into outputs/jlens_qwen35/<tag>/; its
unembedding is that run's export (checked by G-UNEMBED on the model), with the Hub range-read copy as a
fallback before the transfer.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
J = REPO / "outputs/jlens"
Q = J / "qwen35"
QS = Path(os.environ.get("JLENS_QWEN35_DIR") or REPO / "outputs/jlens_qwen35")   # override for dry runs only


def _qwen(tag, label, lens, n_layers, d_model, alt_lenses=None):
    unembed = QS / tag / "unembed"
    return dict(label=label, tokenizer="Qwen/Qwen3.5-122B-A10B", tokenizer_kind="auto", bos=False,
                unembed_dir=unembed if (unembed / "lm_head_weight.npy").exists() else Q / tag / "unembed_hub",
                lens=lens, alt_lenses=alt_lenses or {}, halves=None, n_layers=n_layers, d_model=d_model,
                vocab_cache=J / "vocab_scripts_qwen35.json",
                states={k: QS / tag / f"{k}_states.pt" for k in ("wikitext", "wiki_zh", "wiki_en", "concept")},
                saturated=[])


MODELS = {
    "v3": dict(label="DeepSeek-V3-0324 (AWQ int4)", tokenizer="/workspace/models/deepseek-v3-awq", tokenizer_kind="v3",
               bos=True, unembed_dir=REPO / "data/model_weights/deepseek_v3", lens=J / "lens_v3_filtered40.pt",
               # PENULT_RUNBOOK.md: a few prompts fitted to target layer 59, and the same prompts' target-60 mean
               alt_lenses={"target59": J / "lens_v3_target59.pt",
                           "target60_same": J / "lens_v3_target60_same_prompts.pt"},
               halves=("n50", "n50b"), n_layers=61, d_model=7168, vocab_cache=J / "vocab_scripts.json",
               states={"wikitext": J / "wikitext_states.pt", "wiki_zh": J / "wiki_zh_states.pt",
                       "wiki_en": J / "wiki_en_states.pt", "concept": J / "concept_states.pt"},
               saturated=[(215, 69)]),
    "qwen35_122b": _qwen("qwen35_122b", "Qwen3.5-122B-A10B (bf16)", Q / "lenses/dallinmj/qwen35_122b_bf16_lens_fp32.pt", 48, 3072),
    "qwen35_397b_fp8": _qwen("qwen35_397b_fp8", "Qwen3.5-397B-A17B (FP8)", Q / "lenses/dallinmj/qwen35_397b_fp8_lens_fp32.pt", 60, 4096,
                             alt_lenses={"n24": Q / "lenses/praxagent-org/jlens/wikitext/qwen35_397b.pt"}),
}


def get(name: str) -> dict:
    if name not in MODELS:
        raise SystemExit(f"unknown model {name!r}; known: {sorted(MODELS)}")
    return MODELS[name]


def load_tokenizer_for(name: str):
    m = get(name)
    if m["tokenizer_kind"] == "v3":          # AutoTokenizer mis-tokenizes V3 (merges dots)
        import sys
        sys.path.insert(0, str(REPO / "scripts"))
        from extract.extract_hidden_states import load_tokenizer
        return load_tokenizer(m["tokenizer"])
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(m["tokenizer"])


def open_lens(name: str, which: str):
    """'logit' -> None; 'shipped' -> the model's lens; a V3 subset name (n50, n50b, ...) or an alt-lens key."""
    m = get(name)
    if which == "logit":
        return None
    from score_lazy import LazyLens
    if which == "shipped":
        return LazyLens(m["lens"])
    if which in m["alt_lenses"]:
        return LazyLens(m["alt_lenses"][which])
    if name == "v3":
        from clean_floor import build_subset
        return build_subset(J, J / "per_prompt", 40.0, which)
    raise SystemExit(f"{name} has no lens {which!r}")
