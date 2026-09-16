"""
extract_qwen35_states.py — Qwen3.5 residuals for the DeepSeek-V3 comparison. Written for a GPU box that does
not share /workspace: everything goes under --outdir/--tag and is copied back for the CPU analysis.

Forward passes only, one model load, for Qwen/Qwen3.5-122B-A10B (bf16 weights) or Qwen/Qwen3.5-397B-A17B-FP8
(official FP8 weights). Steps, each skipped when its output already exists (so a crashed run resumes):

  unembed     lm_head weight (fp16) and the final norm's EFFECTIVE per-dimension multiplier as .npy. Qwen3.5's
              RMSNorm computes (1 + weight) * norm(x); the convention is detected by probing the module. Then
              G-UNEMBED: the offline readout rebuilt from those arrays must reproduce the model's own logits on
              final-layer residuals (top-1 agreement 1.0, correlation > 0.999), or the run stops.
  regime      FP8 model only. The lens was fit with bf16 matmuls on the FP8 weights, not with FP8 kernels.
              --regime-prompts WikiText windows run through both paths (the bf16 arm dequantizes each weight
              block-wise and uses the eager expert loop); states at 25 layers from both, agreement statistics.
  paragraphs  WikiText 200-299, Wikipedia zh 0-99 and en 0-99: 128-token windows, residuals at positions 16..127,
              all layers — the format of extract_wikitext_states.py.
  concepts    the 581 concepts usable in both models (concept_prompts_qwen35.json): all layers at the concept token,
              each template's own final state, duplicate token sequences run once — the format of
              extract_concept_states.py.

States are stored fp16. Values beyond the fp16 range are clamped and every clamped (prompt, position, layer)
is recorded under "fp16_clamped", so the analysis can exclude them.

Conventions follow jlens: from_hf(model, tokenizer) (layout model.language_model), lm.encode, and
ActivationRecorder on the decoder layers. Qwen3.5 has no BOS token and jlens's force_bos cannot add one, so
neither the lens fit nor this extraction has one.

    python scripts/jlens/extract_qwen35_states.py --model /nvme/models/Qwen3.5-122B-A10B --tag qwen35_122b --outdir /nvme/out
"""
from __future__ import annotations

import argparse, hashlib, json, os, platform, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import jlens  # noqa: E402  (must precede REPO/scripts on sys.path)
from jlens.hooks import ActivationRecorder  # noqa: E402
if not hasattr(jlens, "from_hf"):
    raise SystemExit(f"scripts/jlens shadowed the jlens library: {jlens.__file__}")

REPO = Path(__file__).resolve().parents[2]
CORPORA = {"wikitext": ("scripts/jlens/prompts_wikitext.json", 200),
           "wiki_zh": ("scripts/jlens/prompts_wiki_zh.json", 0),
           "wiki_en": ("scripts/jlens/prompts_wiki_en.json", 0)}
MAX_SEQ, SKIP_FIRST = 128, 16
FP16_MAX = 65504.0


# ---------------------------------------------------------------- bf16 matmuls on FP8 weights (regime arm)

BF16_GEMM = {"on": False}


def dequantize_fp8_blocks(weight: torch.Tensor, scale_inv: torch.Tensor, block_size, out_dtype) -> torch.Tensor:
    """float8 weight [..., rows, cols] times its per-block scale grid [..., ceil(rows/bm), ceil(cols/bn)]."""
    w = weight.to(torch.float32)
    s = scale_inv.to(torch.float32)
    if s.numel() == 1 or block_size is None:
        return (w * s).to(out_dtype)
    bm, bn = block_size
    s = s.repeat_interleave(bm, dim=-2).repeat_interleave(bn, dim=-1)
    return (w * s[..., : w.shape[-2], : w.shape[-1]]).to(out_dtype)


def install_bf16_gemm_patch():
    """While BF16_GEMM['on'], FP8 linears and the eager FP8 expert loop multiply in the input dtype with the
    dequantized weight instead of calling the FP8 kernels."""
    import transformers.integrations.finegrained_fp8 as ff
    if getattr(ff, "_jlens_bf16_patch", False):
        return ff
    orig_linear, orig_grouped, orig_expert = ff.FP8Linear.forward, ff.FP8GroupedLinear.forward, ff.FP8Experts.linear

    def linear_forward(self, input):
        if BF16_GEMM["on"] and self.weight.element_size() == 1:
            return F.linear(input, dequantize_fp8_blocks(self.weight, self.weight_scale_inv, self.block_size,
                                                         input.dtype), self.bias)
        return orig_linear(self, input)

    def grouped_forward(self, x):
        if BF16_GEMM["on"] and self.weight.element_size() == 1:
            W = dequantize_fp8_blocks(self.weight, self.weight_scale_inv, self.block_size, x.dtype)
            input_shape, hidden_dim = x.shape[:-2], x.shape[-1]
            w = W.view(self.n_groups, -1, hidden_dim).transpose(1, 2)
            y = torch.bmm(x.reshape(-1, self.n_groups, hidden_dim).transpose(0, 1), w).transpose(0, 1)
            y = y.reshape(*input_shape, self.n_groups, -1)
            if self.has_bias:
                y.add_(self.bias.view(self.n_groups, -1))
            return y
        return orig_grouped(self, x)

    def expert_linear(self, input, weight, weight_scale_inv, activation_scale=None):
        if BF16_GEMM["on"] and weight.element_size() == 1:
            return F.linear(input, dequantize_fp8_blocks(weight, weight_scale_inv, self.block_size, input.dtype), None)
        return orig_expert(self, input, weight, weight_scale_inv, activation_scale)

    ff.FP8Linear.forward, ff.FP8GroupedLinear.forward, ff.FP8Experts.linear = linear_forward, grouped_forward, expert_linear
    ff._jlens_bf16_patch = True
    return ff


# ---------------------------------------------------------------- model, readout

def load_model(path, max_memory_gib=None):
    from transformers import AutoTokenizer
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForConditionalGeneration
    tok = AutoTokenizer.from_pretrained(path)
    n = torch.cuda.device_count()
    if max_memory_gib:
        caps = [int(x) for x in str(max_memory_gib).split(",")]
        caps = caps * n if len(caps) == 1 else caps
    else:
        caps = [int(torch.cuda.get_device_properties(i).total_memory / 2 ** 30 * 0.9) for i in range(n)]
    t0 = time.time()
    model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
        path, dtype=torch.bfloat16, device_map="auto", max_memory={i: f"{c}GiB" for i, c in enumerate(caps)}).eval()
    placement = {str(v) for v in getattr(model, "hf_device_map", {}).values()}
    if placement & {"cpu", "disk"}:
        raise SystemExit(f"weights spilled to {placement & {'cpu', 'disk'}} with caps {caps} GiB: use more GPU memory")
    print(f"loaded {path} in {time.time() - t0:.0f}s on {n} GPUs, caps {caps} GiB", flush=True)
    return model, tok


def effective_norm_weight(norm):
    """The per-dimension multiplier the final norm applies after RMS scaling, found by probing the module."""
    w = norm.weight.detach().float().cpu()
    eps = float(getattr(norm, "eps", getattr(norm, "variance_epsilon", 1e-6)))
    x = torch.randn(4, w.numel(), dtype=torch.float32)
    with torch.no_grad():
        y = norm(x.to(norm.weight.device)).float().cpu()
    base = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    errs = {name: float((base * g - y).abs().max() / y.abs().max()) for name, g in (("w", w), ("1+w", 1.0 + w))}
    best = min(errs, key=errs.get)
    if errs[best] > 1e-4:
        raise SystemExit(f"final norm matches neither w nor 1+w: relative errors {errs}")
    return ((1.0 + w) if best == "1+w" else w).numpy().astype(np.float32), best, eps, errs


def record(lm, ids, layers):
    with torch.no_grad(), ActivationRecorder(lm.layers, at=layers) as rec:
        lm.forward(ids)
        return {L: rec.activations[L][0].detach() for L in layers}


def to_fp16(t, clamped, where):
    """Clamp to the fp16 range before the cast; log where it happened."""
    t = t.float()
    over = t.abs() > FP16_MAX
    if bool(over.any()):
        clamped.append({**where, "n": int(over.sum())})
        t = t.clamp(-FP16_MAX, FP16_MAX)
    return t.half().cpu()


def paragraphs_of(name):
    path, lo = CORPORA[name]
    corpus = json.loads((REPO / path).read_text())
    return corpus, lo


def provenance(args, model, tok):
    import transformers
    cfg = REPO / "scripts/jlens/concept_prompts_qwen35.json"
    return {"model": str(args.model), "tag": args.tag, "commit_hash": getattr(model.config, "_commit_hash", None),
            "torch": torch.__version__, "transformers": transformers.__version__,
            "jlens_file": jlens.__file__, "python": platform.python_version(),
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "quantization": getattr(model.config, "quantization_config", None) and dict(model.config.quantization_config),
            "bos": tok.bos_token, "concept_prompts_sha256": hashlib.sha256(cfg.read_bytes()).hexdigest(),
            "date": time.strftime("%Y-%m-%d %H:%M:%S %Z")}


def save(obj, path):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)
    print(f"wrote {path} ({path.stat().st_size / 2 ** 30:.2f} GiB)", flush=True)


# ---------------------------------------------------------------- steps

def step_unembed(lm, out, prov):
    d = out / "unembed"
    if (d / "check.json").exists():
        chk = json.loads((d / "check.json").read_text())
        if not chk["passed"]:
            raise SystemExit(f"existing G-UNEMBED failed: {chk}")
        return chk
    d.mkdir(parents=True, exist_ok=True)
    g, convention, eps, errs = effective_norm_weight(lm._final_norm)
    W = lm._lm_head.weight.detach().to(torch.float16).cpu().numpy()
    np.save(d / "lm_head_weight.npy", W)
    np.save(d / "rms_norm_weight.npy", g)
    corpus, lo = paragraphs_of("wikitext")
    last = lm.n_layers - 1
    probe = []
    for p in corpus["prompts"][lo:lo + 8]:
        probe.append(record(lm, lm.encode(p, max_length=MAX_SEQ), [last])[last][-1].float().cpu())
    h = torch.stack(probe)
    offline = (h / torch.sqrt(h.pow(2).mean(-1, keepdim=True) + eps) * torch.from_numpy(g)) @ torch.from_numpy(W).float().T
    with torch.no_grad():
        online = lm.unembed(h).float().cpu()
        # and the same readout in fp32 through the model's own final norm and lm_head: bf16 logits are rounded to
        # 2^-7 relative steps, which can tie a near-tied top-2 and flip top-1 with no error in the readout itself
        head_w = lm._lm_head.weight
        normed = lm._final_norm(h.to(lm._final_norm.weight.device)).float()
        online_fp32 = F.linear(normed.to(head_w.device), head_w.float()).cpu()

    def agreement(ref):
        top1 = float((offline.argmax(-1) == ref.argmax(-1)).float().mean())
        corr = float(np.mean([np.corrcoef(offline[i].numpy(), ref[i].numpy())[0, 1] for i in range(len(h))]))
        max_abs = float((offline - ref).abs().max())
        return {"top1_agreement": top1, "logit_corr": round(corr, 6), "max_abs_diff": round(max_abs, 4),
                "max_abs_diff_relative": round(max_abs / float(ref.abs().max()), 6),
                "passed": top1 == 1.0 and corr > 0.999}

    bf16, fp32 = agreement(online), agreement(online_fp32)
    chk = {"n_checked": len(h), "norm_convention": convention, "norm_probe_rel_err": errs, "rms_norm_eps": eps,
           **{k: v for k, v in bf16.items() if k != "passed"}, "bf16_passed": bf16["passed"], "fp32_reference": fp32,
           "passed": bf16["passed"] or fp32["passed"],
           "passed_via": "bf16" if bf16["passed"] else "fp32_reference" if fp32["passed"] else None, "provenance": prov}
    (d / "check.json").write_text(json.dumps(chk, indent=1, default=str))
    print(f"G-UNEMBED: {chk}", flush=True)
    if not chk["passed"]:
        raise SystemExit("G-UNEMBED failed: the offline readout does not reproduce the model's logits")
    return chk


def step_regime(model, lm, out, n_prompts, prov):
    path = out / "regime.pt"
    if path.exists():
        return
    ff = install_bf16_gemm_patch()
    if not any(isinstance(m, (ff.FP8Linear, ff.FP8Experts)) for m in model.modules()):
        print("regime check skipped: no FP8 modules in this model", flush=True)
        return
    corpus, lo = paragraphs_of("wikitext")
    prompts = corpus["prompts"][lo:lo + n_prompts]
    last = lm.n_layers - 1
    panel = sorted({int(round(x)) for x in np.linspace(0, last, 25)})
    text_cfg = model.config.get_text_config()
    default_impl = getattr(text_cfg, "_experts_implementation", None)

    def run():
        H = torch.empty(len(prompts), MAX_SEQ - SKIP_FIRST, len(panel), lm.d_model, dtype=torch.float16)
        top1, clamped = [], []
        t0 = time.perf_counter()
        for i, p in enumerate(prompts):
            acts = record(lm, lm.encode(p, max_length=MAX_SEQ), sorted(set(panel) | {last}))
            for j, L in enumerate(panel):
                H[i, :, j] = to_fp16(acts[L][SKIP_FIRST:], clamped, {"prompt": lo + i, "layer": L})
            with torch.no_grad():
                top1.append(lm.unembed(acts[last][SKIP_FIRST:]).argmax(-1).cpu())
        return H, torch.stack(top1), clamped, (time.perf_counter() - t0) / len(prompts)

    HA, tA, cA, sA = run()
    print(f"regime arm A (FP8 kernels, experts '{default_impl}'): {sA:.2f}s/prompt", flush=True)
    BF16_GEMM["on"] = True
    model.set_experts_implementation("eager")
    try:
        HB, tB, cB, sB = run()
    finally:
        BF16_GEMM["on"] = False
        model.set_experts_implementation(default_impl or "grouped_mm")
    print(f"regime arm B (bf16 matmuls on dequantized FP8 weights, eager experts): {sB:.2f}s/prompt", flush=True)
    a, b = HA.float(), HB.float()
    cos = (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1) + 1e-12)            # [n, T, layers]
    rel = (a - b).norm(dim=-1) / (a.norm(dim=-1) + 1e-12)
    stats = {"layers": panel, "cos_mean": cos.mean((0, 1)).tolist(), "cos_p01": torch.quantile(cos.flatten(0, 1), 0.01, dim=0).tolist(),
             "rel_diff_mean": rel.mean((0, 1)).tolist(), "final_top1_agreement": float((tA == tB).float().mean()),
             "secs_per_prompt": {"fp8": sA, "bf16": sB}, "default_experts_implementation": default_impl}
    print(f"regime: final top-1 agreement {stats['final_top1_agreement']:.3f}; mean cos by layer "
          + " ".join(f"{L}:{c:.4f}" for L, c in zip(panel, stats["cos_mean"])), flush=True)
    save({"arm_fp8": HA, "arm_bf16": HB, "top1_fp8": tA, "top1_bf16": tB, "clamped_fp8": cA, "clamped_bf16": cB,
          "prompt_span": [lo, lo + len(prompts)], "skip_first": SKIP_FIRST, "stats": stats, "provenance": prov}, path)


def step_paragraphs(lm, out, n_prompts, chk, prov):
    layers = list(range(lm.n_layers))
    for name in CORPORA:
        path = out / f"{name}_states.pt"
        if path.exists():
            continue
        corpus, lo = paragraphs_of(name)
        prompts = corpus["prompts"][lo:lo + n_prompts]
        T = MAX_SEQ - SKIP_FIRST
        H = torch.empty(len(prompts), T, len(layers), lm.d_model, dtype=torch.float16)
        kept, seq_lens, clamped = [], [], []
        t0 = time.perf_counter()
        for i, p in enumerate(prompts):
            ids = lm.encode(p, max_length=MAX_SEQ)
            if int(ids.shape[-1]) < MAX_SEQ:
                raise SystemExit(f"{name} prompt {lo + i} tokenised to {int(ids.shape[-1])} < {MAX_SEQ}")
            acts = record(lm, ids, layers)
            for L in layers:
                H[i, :, L] = to_fp16(acts[L][SKIP_FIRST:], clamped, {"prompt": lo + i, "layer": L})
            kept.append(ids[0, SKIP_FIRST:].cpu()); seq_lens.append(int(ids.shape[-1]))
            if (i + 1) % 10 == 0:
                el = time.perf_counter() - t0
                print(f"  {name} {i + 1}/{len(prompts)}  {el / (i + 1):.1f}s/prompt", flush=True)
        save({"model_path": str(prov["model"]), "corpus": corpus["source"], "prompt_span": [lo, lo + len(prompts)],
              "n_layers": lm.n_layers, "layers": layers, "d_model": lm.d_model, "rms_norm_eps": chk["rms_norm_eps"],
              "norm_convention": chk["norm_convention"], "max_seq_len": MAX_SEQ, "skip_first": SKIP_FIRST,
              "force_bos": False, "seq_lens": seq_lens, "input_ids": torch.stack(kept), "H": H,
              "fp16_clamped": clamped, "unembed_check": {k: chk[k] for k in ("top1_agreement", "logit_corr", "passed")},
              "provenance": prov}, path)
        print(f"{name}: {len(clamped)} (prompt, layer) cells had values clamped to the fp16 range", flush=True)


def step_concepts(lm, out, spec, max_concepts, chk, prov):
    path = out / "concept_states.pt"
    if path.exists():
        return
    prompts = spec["prompts"]
    if max_concepts:
        prompts = [p for p in prompts if p["concept"] < max_concepts]
    layers = list(range(lm.n_layers))
    uniq, dup_of = {}, []
    for i, p in enumerate(prompts):
        ids = lm.encode(p["text"])
        if int(ids[0, -1]) != p["expect_last_id"] or int(ids.shape[-1]) != p["n_tokens"]:
            raise SystemExit(f"concept prompt {i} tokenised differently on this box: {ids[0].tolist()}")
        dup_of.append(uniq.setdefault(tuple(ids[0].tolist()), i))
    run = sorted(set(dup_of))
    print(f"concepts: {len(run)} distinct token sequences ({len(prompts) - len(run)} duplicates mapped)", flush=True)
    H = torch.empty(len(prompts), len(layers), lm.d_model, dtype=torch.float16)
    clamped = []
    t0 = time.perf_counter()
    for k, i in enumerate(run):
        acts = record(lm, lm.encode(prompts[i]["text"]), layers)
        # device_map="auto" leaves each layer's state on its own GPU; torch.stack needs one device
        H[i] = to_fp16(torch.stack([acts[L][-1].to(acts[layers[0]].device) for L in layers]), clamped, {"prompt": i})
        if (k + 1) % 200 == 0:
            el = time.perf_counter() - t0
            print(f"  concepts {k + 1}/{len(run)}  {el / (k + 1):.2f}s/prompt  eta {(len(run) - k - 1) * el / (k + 1) / 60:.1f} min", flush=True)
    for i, j in enumerate(dup_of):
        if j != i:
            H[i] = H[j]
    template_states = {}
    for lang, tpls in spec["templates"].items():
        for ti, tpl in enumerate(tpls, start=1):
            acts = record(lm, lm.encode(tpl), layers)
            template_states[f"{lang}:{ti}"] = {"text": tpl, "H": to_fp16(
                torch.stack([acts[L][-1].to(acts[layers[0]].device) for L in layers]), clamped, {"template": f"{lang}:{ti}"})}
    sub = {**spec, "prompts": prompts}
    if max_concepts:
        sub["concepts"] = spec["concepts"][:max_concepts]
    save({"model_path": str(prov["model"]), "spec": sub, "n_layers": lm.n_layers, "layers": layers, "d_model": lm.d_model,
          "rms_norm_eps": chk["rms_norm_eps"], "norm_convention": chk["norm_convention"], "force_bos": False,
          "dup_of": dup_of, "template_states": template_states, "H": H, "fp16_clamped": clamped,
          "unembed_check": {k: chk[k] for k in ("top1_agreement", "logit_corr", "passed")}, "provenance": prov}, path)
    print(f"concepts: {len(clamped)} states had values clamped to the fp16 range", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local checkpoint directory (or hub id)")
    ap.add_argument("--tag", required=True, help="e.g. qwen35_122b, qwen35_397b_fp8")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--steps", default="unembed,regime,paragraphs,concepts")
    ap.add_argument("--concepts", type=Path, default=REPO / "scripts/jlens/concept_prompts_qwen35.json")
    ap.add_argument("--max-memory-gib", default=None, help="per-GPU cap(s); default 90%% of each card")
    ap.add_argument("--regime-prompts", type=int, default=32)
    ap.add_argument("--n-paragraphs", type=int, default=100)
    ap.add_argument("--max-concepts", type=int, default=None, help="dry runs only")
    args = ap.parse_args()
    steps = args.steps.split(",")
    out = args.outdir / args.tag
    out.mkdir(parents=True, exist_ok=True)

    # fail fast, before the load: inputs must tokenise as they did when built
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    spec = json.loads(args.concepts.read_text())
    bad = [p for p in spec["prompts"] if (lambda ids: ids[-1] != p["expect_last_id"] or len(ids) != p["n_tokens"])(tok(p["text"])["input_ids"])]
    if bad:
        raise SystemExit(f"{len(bad)} concept prompts tokenise differently here, e.g. {bad[0]}")
    for name in CORPORA:
        corpus, lo = paragraphs_of(name)
        short = [i for i, t in enumerate(corpus["prompts"][lo:lo + args.n_paragraphs]) if len(tok(t)["input_ids"]) < MAX_SEQ]
        if short:
            raise SystemExit(f"{name}: paragraphs {short[:5]} are shorter than {MAX_SEQ} tokens")
    print("tokenizer pre-checks passed (concept prompts and paragraph lengths)", flush=True)

    model, tok = load_model(args.model, args.max_memory_gib)
    lm = jlens.from_hf(model, tok)
    prov = provenance(args, model, tok)
    print(f"{lm}  bos={tok.bos_token}", flush=True)
    chk = step_unembed(lm, out, prov)
    if "regime" in steps:
        try:     # the regime check informs the analysis; it must never block the extraction itself
            step_regime(model, lm, out, args.regime_prompts, prov)
        except Exception as e:
            BF16_GEMM["on"] = False
            (out / "regime_error.json").write_text(json.dumps({"error": f"{type(e).__name__}: {e}"}, indent=1))
            print(f"REGIME CHECK FAILED ({type(e).__name__}: {e}); continuing with the extraction", flush=True)
    if "paragraphs" in steps:
        step_paragraphs(lm, out, args.n_paragraphs, chk, prov)
    if "concepts" in steps:
        step_concepts(lm, out, spec, args.max_concepts, chk, prov)
    (out / "run.json").write_text(json.dumps({"steps": steps, "provenance": prov, "finished": time.strftime("%Y-%m-%d %H:%M:%S")},
                                             indent=1, default=str))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
