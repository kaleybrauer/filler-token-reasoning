#!/usr/bin/env python3
"""Phase B: validate the vLLM V1 capture plumbing on tiny-random/kimi-k2.5 (9.5 MB).

Same architecture as the real checkpoint (KimiK25ForConditionalGeneration over a
DeepseekV2-style text model), so this exercises everything except the weights:
worker_extension_cls injection, collective_rpc, the language_model.model.layers
path, the (hidden, residual) tuple shape, the first-forward-per-layer guard, and
that prefix caching / chunked prefill are actually off.

Usage:
    /root/.venvs/k25/bin/python scripts/kimi_k25/smoke_tiny_k25.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HOOK_DIR = Path(__file__).resolve().parent
if str(_HOOK_DIR) not in sys.path:
    sys.path.insert(0, str(_HOOK_DIR))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/workspace/models/tiny-kimi-k2.5")
    ap.add_argument("--max-model-len", type=int, default=512)
    args = ap.parse_args()

    cfg = json.loads((Path(args.model_path) / "config.json").read_text())
    tcfg = cfg.get("text_config", cfg)
    n_layers = tcfg["num_hidden_layers"]
    hidden = tcfg["hidden_size"]
    print(f"tiny model: {n_layers} layers, hidden {hidden}")

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        tensor_parallel_size=1,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        enforce_eager=True,
        trust_remote_code=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        worker_extension_cls="kimi_k25_hooks.K25CaptureExt",
        gpu_memory_utilization=0.25,
    )

    failures: list[str] = []

    infos = llm.collective_rpc("k25_install_hooks")
    print(f"install_hooks -> {infos}")
    if infos[0]["n_layers"] != n_layers:
        failures.append(f"hooked {infos[0]['n_layers']} layers, expected {n_layers}")

    tok = llm.get_tokenizer()
    ids = tok.encode("Filler: . . . . .\n\nAnswer:", add_special_tokens=False)
    seq_len = len(ids)
    positions = list(range(max(0, seq_len - 6), seq_len))
    print(f"prompt {seq_len} tokens, capturing {len(positions)} positions {positions}")

    llm.collective_rpc("k25_set_positions", args=(positions, seq_len))
    outs = llm.generate([TokensPrompt(prompt_token_ids=ids)],
                        SamplingParams(temperature=0, max_tokens=5, detokenize=False),
                        use_tqdm=False)
    payloads = llm.collective_rpc("k25_pop")
    p0 = payloads[0]

    print(f"pop -> rank={p0['rank']} n_captured={p0['n_captured']} errors={p0['errors']}")
    if p0["errors"]:
        failures.append(f"hook errors: {p0['errors']}")
    if p0["n_captured"] != n_layers:
        failures.append(f"captured {p0['n_captured']} layers, expected {n_layers}")

    states = p0["states"]
    if states:
        for li, arr in sorted(states.items()):
            if arr.shape != (len(positions), hidden):
                failures.append(f"L{li} shape {arr.shape} != {(len(positions), hidden)}")
            if arr.dtype != np.float16:
                failures.append(f"L{li} dtype {arr.dtype} != float16")
            if not np.isfinite(arr.astype(np.float32)).all():
                failures.append(f"L{li} has non-finite values")
        a0 = states[0]
        print(f"  L0 shape={a0.shape} dtype={a0.dtype} absmax={np.abs(a0.astype(np.float32)).max():.4g}")

    # decode steps must NOT overwrite the prefill capture: pop() cleared state, so a
    # second pop with no arming must come back empty
    empty = llm.collective_rpc("k25_pop")[0]
    if empty["n_captured"] != 0:
        failures.append(f"stale capture after pop: {empty['n_captured']} layers")

    n_removed = llm.collective_rpc("k25_remove_hooks")[0]
    print(f"removed {n_removed} hooks")

    print("\n" + "=" * 60)
    if failures:
        print("TINY SMOKE FAILED:")
        for f in failures:
            print(f"  FAIL {f}")
        return 1
    print("TINY SMOKE PASSED — capture plumbing works on vLLM 0.25.1 V1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
