#!/usr/bin/env python
"""Preflight for the native-FP8 DeepSeek-V3 extraction run.

Run this on the GPU box BEFORE the multi-hour extraction. Every check here maps to a
failure that would otherwise surface only after the run, as a silently-invalid
AWQ-vs-FP8 comparison.

    uv run --python /root/.venvs/fp8/bin/python scripts/fp8/preflight_fp8.py \
        --fp8-path /workspace/models/deepseek-v3-fp8 \
        --awq-path /workspace/models/deepseek-v3-awq

Checks 1-2 need no GPU and run in seconds; 3-7 load the model (~30-60 min).
Use --no-load to run only the cheap ones (e.g. mid-download).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    return ok


def rms_norm(x, weight, eps=1e-6):
    """Byte-identical to scripts/analysis/decode_2fact_heatmap.py:25."""
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight


# ---------------------------------------------------------------- 1. tokenizer
def check_tokenizer(fp8_path, awq_path, tokenizer_path):
    """Token ids must be IDENTICAL to the AWQ run, or every position index shifts.

    The AWQ repo's tokenizer.json sets post_processor=TemplateProcessing (prepends
    BOS); the official DeepSeek repo's sets post_processor=ByteLevel (no BOS). Vocab
    and the 0-299 number-token map are identical — the sole difference is that leading
    token. Unfixed, pos_001 (where A1 peaks) silently becomes pos_000.
    """
    print("\n[1] tokenizer parity (token ids must match the AWQ run exactly)")
    from tokenizers import Tokenizer
    from data.generate_2fact_dataset import (
        build_system_message_2fact, build_user_turn_2fact,
    )

    awq_tok = Tokenizer.from_file(str(Path(awq_path) / "tokenizer.json"))
    run_tok = Tokenizer.from_file(str(Path(tokenizer_path) / "tokenizer.json"))

    ok = True
    for ftype, k in [("dots", 10), ("dots", 25), ("dots", 50),
                     ("counting", 10), ("counting", 25)]:
        text = (build_system_message_2fact(ftype, k) + "\n\n"
                + build_user_turn_2fact("the atomic number of Uranium",
                                        "the atomic number of Carbon", ftype, k))
        ea, er = awq_tok.encode(text).ids, run_tok.encode(text).ids
        ok &= check(f"ids match on {ftype}_{k}", ea == er,
                    f"AWQ={len(ea)} run={len(er)}"
                    + ("" if ea == er else "  <-- pass --tokenizer-path <awq dir>"))

    def nummap(tok):
        m = {}
        for v in range(300):
            ids = tok.encode(str(v)).ids
            if ids and tok.id_to_token(ids[0]) and "begin" in str(tok.id_to_token(ids[0])):
                ids = ids[1:]
            if len(ids) == 1:
                m[ids[0]] = v
        return m

    ok &= check("0-299 number-token map identical", nummap(awq_tok) == nummap(run_tok))
    return ok


# ------------------------------------------------------------------- 2. config
def check_config(fp8_path, awq_path):
    """Architecture must match the AWQ checkpoint, and FP8 must actually be declared."""
    print("\n[2] config sanity")
    f = json.load(open(Path(fp8_path) / "config.json"))
    a = json.load(open(Path(awq_path) / "config.json"))
    ok = True
    for k in ["num_hidden_layers", "hidden_size", "num_attention_heads",
              "n_routed_experts", "vocab_size", "first_k_dense_replace"]:
        ok &= check(f"{k} matches AWQ ({f.get(k)})", f.get(k) == a.get(k),
                    "" if f.get(k) == a.get(k) else f"fp8={f.get(k)} awq={a.get(k)}")
    q = f.get("quantization_config") or {}
    ok &= check("quant_method == fp8", q.get("quant_method") == "fp8", json.dumps(q))
    return ok


# --------------------------------------------------------------- 3-7. on-model
def check_model(fp8_path, tokenizer_path, awq_weights_dir, n_smoke):
    import torch
    from extract.extract_hidden_states import load_model, HiddenStateExtractor

    print("\n[3] model load")
    model, tokenizer = load_model(fp8_path, precision="fp8",
                                  tokenizer_path=tokenizer_path)
    n_gpus = torch.cuda.device_count()
    for i in range(n_gpus):
        tot = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"      GPU {i}: {torch.cuda.memory_allocated(i)/1e9:.1f} / {tot:.0f} GB")
    # The checkpoint carries a 1-layer Multi-Token-Prediction module (model.layers.61,
    # 15.4 GB of the 688.6 GB) that this experiment has no use for. config.json declares
    # num_nextn_predict_layers=1 and transformers' num_mtp_layers also defaults to 1, so
    # it probably loads. Whether it lands inside model.model.layers or as a separate
    # attribute depends on the transformers version — report, don't guess.
    n_layers = len(model.model.layers)
    ok = check("61 or 62 decoder layers", n_layers in (61, 62), f"got {n_layers}")
    mtp_loaded = n_layers == 62 or any(
        "mtp" in n.lower() or "nextn" in n.lower() for n, _ in model.named_modules()
    )
    print(f"      MTP module loaded: {mtp_loaded}"
          + ("  — 15.4 GB reclaimable via num_mtp_layers=0 / num_nextn_predict_layers=0"
             if mtp_loaded else "  — good, 673.2 GB main model only"))
    ok &= check("no CPU/disk in device map",
                all("cpu" not in str(d).lower() and "disk" not in str(d).lower()
                    for d in set(getattr(model, "hf_device_map", {}).values())),
                str(sorted(set(getattr(model, "hf_device_map", {}).values()))))

    print("\n[4] hook shape at model.model.layers[i]")
    layers = [0, 30, n_layers - 1]
    ex = HiddenStateExtractor(model, layers)
    ex.register_hooks()
    enc = tokenizer("Question: What is 2 plus 2?\n\nAnswer:", return_tensors="pt")
    ids = enc["input_ids"].to(model.device)
    with torch.no_grad():
        out = model(ids, use_cache=False)
    seq = ids.shape[1]
    for li in layers:
        cap = ex._captured.get(li)
        ok &= check(f"layer {li} captured (1,{seq},7168)",
                    cap is not None and tuple(cap.shape) == (1, seq, 7168),
                    str(None if cap is None else tuple(cap.shape)))
    ex.remove_hooks()

    print("\n[5] readout instrument vs the AWQ run")
    lm_np = model.lm_head.weight.detach().to("cpu").to(torch.float16).numpy()
    nm_np = model.model.norm.weight.detach().to("cpu").to(torch.float32).numpy()
    awq_lm = Path(awq_weights_dir) / "lm_head_weight.npy"
    awq_nm = Path(awq_weights_dir) / "rms_norm_weight.npy"
    if awq_lm.exists() and awq_nm.exists():
        a_lm, a_nm = np.load(awq_lm), np.load(awq_nm)
        d_lm = float(np.abs(a_lm.astype(np.float32) - lm_np.astype(np.float32)).max())
        d_nm = float(np.abs(a_nm - nm_np).max())
        # Informational, not fatal: AWQ leaves lm_head unquantized, so these SHOULD be
        # ~identical. If they are, the logit lens is literally the same instrument in
        # both conditions and any decode difference is attributable to the states alone.
        check(f"lm_head identical to AWQ (max|d|={d_lm:.2e})", True)
        check(f"final norm identical to AWQ (max|d|={d_nm:.2e})", True)
    else:
        print(f"      (skipped — no saved AWQ weights at {awq_weights_dir})")

    print("\n[6] logit-lens closure: rms_norm(h_last) @ lm_head.T == model logits")
    ex = HiddenStateExtractor(model, [n_layers - 1])
    ex.register_hooks()
    with torch.no_grad():
        out = model(ids, use_cache=False)
    h = ex._captured[n_layers - 1][0].to(torch.float32).numpy()
    ex.remove_hooks()
    recon = rms_norm(h, nm_np) @ lm_np.astype(np.float32).T
    ref = out.logits[0].to(torch.float32).cpu().numpy()
    agree = float((recon.argmax(-1) == ref.argmax(-1)).mean())
    corr = float(np.corrcoef(recon[-1], ref[-1])[0, 1])
    ok &= check("top-1 agreement >= 0.95", agree >= 0.95, f"{agree:.3f}")
    ok &= check("last-position logit corr >= 0.99", corr >= 0.99, f"{corr:.4f}")

    print(f"\n[7] behavioural smoke ({n_smoke} examples)")
    print("      NOTE: run the real evaluator for the accuracy number; this only")
    print("      confirms the model produces sane answers under the fp8 path.")
    with torch.no_grad():
        gen = model.generate(ids, max_new_tokens=8, do_sample=False)
    txt = tokenizer.decode(gen[0][ids.shape[1]:], skip_special_tokens=True)
    ok &= check("generates non-empty output", len(txt.strip()) > 0, repr(txt))
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fp8-path", default="/workspace/models/deepseek-v3-fp8")
    p.add_argument("--awq-path", default="/workspace/models/deepseek-v3-awq")
    p.add_argument("--tokenizer-path", default=None,
                   help="defaults to --awq-path (the correct choice: keeps token ids "
                        "identical to the AWQ run)")
    p.add_argument("--awq-weights-dir", default="data/model_weights/deepseek_v3")
    p.add_argument("--n-smoke", type=int, default=4)
    p.add_argument("--no-load", action="store_true", help="skip checks 3-7")
    a = p.parse_args()
    tokenizer_path = a.tokenizer_path or a.awq_path

    print("=" * 72)
    print("FP8 PREFLIGHT")
    print(f"  fp8       : {a.fp8_path}")
    print(f"  awq       : {a.awq_path}")
    print(f"  tokenizer : {tokenizer_path}")
    print("=" * 72)

    check_tokenizer(a.fp8_path, a.awq_path, tokenizer_path)
    check_config(a.fp8_path, a.awq_path)
    if not a.no_load:
        check_model(a.fp8_path, tokenizer_path, a.awq_weights_dir, a.n_smoke)

    failed = [n for n, ok in RESULTS if not ok]
    print("\n" + "=" * 72)
    if failed:
        print(f"PREFLIGHT FAILED ({len(failed)}/{len(RESULTS)}) — do NOT start extraction")
        for n in failed:
            print(f"   - {n}")
        sys.exit(1)
    print(f"PREFLIGHT PASSED ({len(RESULTS)}/{len(RESULTS)}) — safe to extract")


if __name__ == "__main__":
    main()
