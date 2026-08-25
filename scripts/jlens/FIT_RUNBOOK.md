# Fitting a Jacobian lens for DeepSeek V3 — run plan

**Status:** RUNNING as of 2026-08-19 on 3x H200. Implemented in this directory:
`v3_autograd.py` (the patches + loader), `gates.py` (G1-G5, `--then-fit` chains straight
into the fit on the same model load), `fit_v3.py` (the fit), `apply_lens.py` +
`--jlens` flags in the decode scripts (application), `eval_lens_quality.py` (the paper's
own lens-quality evals), `report_jlens_vs_logit.py` (head-to-head).

**Purpose: Kaley's own research interest and learning.** This is NOT a reviewer rebuttal — no
reviewer has raised the lens. So scope it by what is interesting to know, and note up front
that **a null result (J-lens ≈ logit lens) is a perfectly good outcome**, not a failed run.

**Decision 2026-08-19 (Kaley): match the Anthropic paper's fitting exactly.** That means
`source_layers=None` — *every* layer below the target, not a band. An earlier plan to fit
only the decode band (>= L30) was rejected: min-layer 30 is a *logit-lens* convention (the
logit lens reads nothing useful early), so importing it here would beg the question the
lens is meant to answer — where V3's workspace band actually is. Cost note: reverse mode
returns every source layer from one backward, so extra layers *above* the minimum are free;
only the minimum sets the price (the backward traverses `60 - min_layer` blocks).

---

## 1. What is being built

One matrix `J_L` (7168 × 7168) per layer, answering: *if the residual stream at layer L is
nudged, how does the final layer's residual move?* — averaged over prompts and positions.
The readout is then

```
lens_L(h) = softmax( W_U · RMSNorm( J_L · h ) )
```

The current logit lens is exactly the special case `J_L = I`. **Nothing is trained**: `J_L` is a
closed-form running mean of gradients — no loss, no optimiser, no learned parameters. That is
why it converges in ~60 prompts rather than needing a training run.

Method: Anthropic, *Verbalizable Representations Form a Global Workspace in Language Models*
(transformer-circuits.pub/2026/workspace); reference implementation
`github.com/anthropics/jacobian-lens` (`jlens`).

**No pre-fitted lens exists for DeepSeek V3, or for any Kimi model** (HF-wide check 2026-08-18;
the only DeepSeek-family lens is for V4-Flash, a different architecture). This has to be fitted.

---

## 2. Why V3 and not Kimi

- V3's extracted states are the **residual stream** (transformers path) — the object a J-lens
  transports. Directly usable.
- **Kimi K2's saved states are per-layer MLP writes**, not the stream (see
  `scripts/kimi_k2/REEXTRACT_RUNBOOK.md`). Applying a stream-fitted `J` to them stacks two
  errors. Kimi K2 is out until its re-extraction lands.
- Kimi K2.5's states are correct, but K2.5 ships native INT4 with **no bf16 release**, and its
  quantized kernels have no autograd backward — strictly harder than V3.

---

## 3. Do we need bf16? No.

The fit needs gradients w.r.t. **activations**, not weights: for a linear layer
`grad_x = grad_y @ W`, which needs W's *values*, not its dtype or its gradient. AWQ already
ships this — `WQLinearMMFunction.backward` (`awq/modules/linear/gemm.py:87-112`) dequantizes
the weight (`awq_ext.dequantize_weights_cuda`, :98) and returns `grad_output.bmm(weights.T)`
(:110). That is an **exact** activation gradient for the dequantized model, which is the model
being studied — not an approximation of a bf16 model we never run.

(praxagent's 397B recipe used bf16 only because it fit in 8×H200. V3-671B in bf16 is ~1.3 TB,
so staying quantized is the only route and costs nothing in gradient correctness.)

What actually gates the fit: (a) every op in the path needs a backward, (b) no `torch.no_grad()`
blocking the graph, (c) memory for the reverse-mode graph. All three are addressed below.

---

## 3b. GPU sizing — 3× H200 is the FLOOR, not the target

The existing V3 extraction runs on 3× H200, but that is **forward-only**. The fit additionally
pays for retained activations across the band, gradient buffers (scaling with `dim_batch`), and
AWQ dequantization temporaries inside the MoE backward. Arithmetic with V3-AWQ ≈ 350 GB weights:

| GPUs | total VRAM | free after weights | per-GPU headroom |
|---|---|---|---|
| 3× H200 | 423 GB | ~73 GB | ~24 GB |
| 4× H200 | 564 GB | ~214 GB | ~53 GB |
| 8× H200 | 1128 GB | ~778 GB | ~97 GB |

**3× H200 should be enough — plan for 3.** (4× is hard to get; 8× would match praxagent's
known-working 397B setup but is not required.) Estimated working set for the backward:

- retained activations, 31-layer band, 128 tokens, d=7168 (residual + MLA intermediates +
  8 active experts × 2048): **~800 MB**
- gradient buffers scale with `dim_batch` → **single-digit GB** at B=16
- AWQ dequant temporary is created *inside* `WQLinearMMFunction.backward` and is a local:
  one expert's gate+up+down ≈ **88 MB fp16**, and `moe_infer` loops experts so autograd replays
  them one at a time → peak a few hundred MB, **not** 256×88 MB = 22 GB. (That figure is the
  pessimistic bound if every expert were materialised at once; treat it as the thing G4 rules
  out, not the expectation.)

Binding constraint on 3× is therefore not capacity but that weights need ~117 GB/GPU, leaving
**~24 GB free per GPU**, borne by whichever GPU holds the layer being differentiated.
**Fragmentation across hundreds of sequential 88 MB allocations is the likelier failure than
running out** — `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` targets exactly this.

### MEASURED 2026-08-19 on 3x H200 (140.4 GiB each)

- **`max_memory` caps must total ~360 GiB or the load fails.** 327.7 GiB of weights + 11.2 GiB
  of buffers, and accelerate also reserves per-device slack for the largest atomic block.
  Uniform 112 GiB (336 total) and skewed 130/108/108 (346) both fell back to a CPU device map,
  which the AWQ quantizer rejects outright (`ValueError: ... device_map that contains a CPU or
  disk device`). The extraction path's uniform **120 GiB works**; `load_v3_awq` falls back to
  it automatically. Attempts to skew the map toward the band GPUs are pointless: `device_map=
  "auto"` balances anyway.
- Resulting placement: weights **103.8 / 106.7 / 119.6 GiB**, i.e. free **~36 / ~34 / ~21 GiB**.
  Blocks 0-~26 on cuda:0, ~27-38 on cuda:1, ~39-60 on cuda:2. **cuda:2 is the binding
  constraint and does not change with `min_layer`** — lowering the minimum adds retained
  activations to the two cards that have headroom.
- Load takes **~13 min** (36 shards, ~21 s each, mfs-bound). `gates.py --then-fit` exists so
  gates and the fit share one load.

The knob that actually matters is `dim_batch`: it does not change the estimator or the total
backward FLOPs, but the per-pass costs that *do* scale with the number of passes — AWQ
re-dequantising every weight it traverses, and ~50k kernel launches from the 256-expert Python
loop — are amortised over `dim_batch` cotangent directions. Take the largest that fits.
Ordered fallbacks if G4 complains: smaller `dim_batch` (wall-clock ~linear) → gradient
checkpointing → 64-token sequences (a deviation from the paper's 128).

## 4. Environment

Same as the V3 extraction path (see `project_deepseek_v3_loading` notes):

- Model: `/workspace/models/deepseek-v3-awq`
- `transformers>=4.48,<5` (5.x routes AWQ through gptqmodel, whose kernels don't build here).
  Installed here: **4.57.6 / torch 2.6.0+cu124 / autoawq 0.2.7.post3 + autoawq-kernels**
- `jacobian-lens` cloned to `/workspace/jacobian-lens`, installed with
  **`uv pip install --no-deps -e`**: its pyproject demands `transformers>=5.5`, which would
  break the AWQ path, but the only 5.x-era API it touches is `config.get_text_config()`,
  present since 4.4x.
- ⚠ **`scripts/jlens/` shadows the `jlens` library.** Python treats our directory as a
  namespace package, so `import jlens` after `REPO_ROOT/scripts` joins `sys.path` silently
  resolves to *it* and dies with `module 'jlens' has no attribute 'from_hf'` — after the
  13-minute load. Every entry point imports `jlens` first and asserts on `jlens.__file__`.
- `AutoModelForCausalLM.from_pretrained(device_map="auto", max_memory={i: "120GiB"})`
  — 80% cap prevents CPU offload, which the AWQ quantizer rejects
- `attn_implementation="eager"` (praxagent's recipe; needed for a clean graph)
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- Load takes ~16 min on 3× H200 (36 shards)
- Tokenizer: `load_tokenizer()` from `scripts/extract/extract_hidden_states.py` —
  `PreTrainedTokenizerFast` on `tokenizer.json`; **AutoTokenizer mis-tokenizes** (merges dots)
- Existing source patches still required: `PytorchGELUTanh` compat shim,
  `is_torch_fx_available` removal, `DynamicCache` compat — in **both** the source and the
  HF modules cache copy of `modeling_deepseek.py`

---

## 5. The four V3-specific changes

1. **Enable AWQ autograd.** The forward builds a graph under `if self.training:`
   (`gemm.py:258`) but wraps the same call in `torch.no_grad()` in eval (:270). So set
   `.training = True` on the `WQLinear_GEMM` **leaves** (attention MLA projections + expert
   MLPs). A flag flip, not a code change.
2. **Unblock the MoE.** `modeling_deepseek.py` has `@torch.no_grad()` on `moe_infer` (:535) —
   the only one in the file. Strip it, **and** rewrite its two in-place ops out-of-place
   (`new_x[idxs] = outs` :601, `.mul_(topk_weight)` :605) or autograd's version counter errors.
   Discrete top-k (argsort/scatter on indices) stays constant — correct straight-through;
   gradient flows via the selected experts and the gate weights.
3. **Keep the MoE container in EVAL.** `DeepseekV3MoE.forward` has
   `if not self.training: y = moe_infer` with **no else** (:528) → switching it to train mode
   raises `UnboundLocalError`. `MoEGate.forward` additionally asserts `not self.training`.
   Patch `moe_infer`; do not flip either container.
4. **Replace the AWQ backward's `.repeat`.** `WQLinearMMFunction.backward` builds
   `weights.T.unsqueeze(0).repeat(B,1,1)` before `torch.bmm` — for `o_proj` that is 235 MB
   x `dim_batch` of transient allocation per call. `torch.matmul(grad_output, weights.T)`
   broadcasts instead and is the same value (gate G1b). Not required for correctness; it is
   what makes a large `dim_batch` affordable.

With those, `jlens.fit` should run unmodified (`from_hf` → `HFLensModel`, layout auto-detects
`Layout("model")`).

---

## 6. Gates — run these BEFORE the expensive fit

Each is cheap and each kills a different failure mode.

**G1 — AWQ backward is correct.** Gradcheck one layer: compare `WQLinearMMFunction.backward`
against a finite difference of its **own fused forward**. Note the dequant flag differs between
paths (`1` at :98 vs `0` at :50) — **confirm the layout/transpose empirically, don't assume.**

**G2 — MLA attention differentiates.** Confirm no `no_grad`/`detach` in the MLA path; check a
gradient actually arrives at `h_L` from `h_final` through one attention block.

**G3 — MoE gradient flows.** After the `moe_infer` patch, verify a non-zero gradient reaches the
routed expert inputs and the gate.

**G4 — memory probe.** One backward, one source layer, `dim_batch=1`, then 16. Record peak
memory before committing to the full run. **This is the real unknown** — AWQ's backward
dequantizes weights, and in a MoE layer 128 tokens routing 8-of-256 may materialise a large
fraction of a layer's experts in fp16 transiently. If this blows up, the mitigations are
gradient checkpointing, chunking the expert backward, or a smaller `dim_batch`.

**G5 — sanity of the fitted object** (after a 2–3 prompt smoke fit):
- a late-layer `J` should approach `~I` (the logit-lens limit)
- `jlens.apply` returns both lens logits and model logits — check the transport
  reconstruction against the model's own logits

---

## 7. The fit

**Run it as the paper does — `jlens.fit` unmodified, library defaults.** `scripts/jlens/
fit_v3.py` with no `--min-layer`/`--source-layers` passes `source_layers=None`, which is the
reference default (every layer below the target).

- **Corpus:** WikiText-103 via the repo's **own** helper, `jlens.examples.load_wikitext_prompts`
  (`min_chars=600`), cached to `scripts/jlens/prompts_wikitext.json` by `prepare_prompts.py` so
  the fit never touches the network. The paper says only "a pretraining-like corpus", so this
  is the closest documented match, and it is recorded in the lens's `.json` sidecar.
- **n:** the paper uses **1000** sequences of 128 tokens; that is the target here. The fit is
  checkpointed every 2 prompts and the checkpoint is itself a usable lens at whatever n it has
  reached (`apply_lens.load_jlens` divides the running sum by `n_done`), so the run can be
  stopped at any point and the achieved n reported. For calibration: the README says quality
  saturates fast and ~100 is usable; xiangchensong measured the curve flat from n=60 (a refit
  at n=1000 moved their headline by 0.0000); praxagent shipped n=24.
- **Positions:** `skip_first=16` (attention sinks), drop the last; per-prompt Jacobian is the
  mean over remaining source positions with cotangents at every target position ≥ source.
- **NO `normalize_per_prompt`.** It is not in the Anthropic reference implementation — it is
  xiangchensong's fork's addition (per-prompt `J` scaled to unit Frobenius before averaging,
  which improved *their* pass@1 by 73%). A plain running mean is what the paper does, so a
  plain running mean is what we do. Revisit only if `fit`'s per-prompt `max||J||/sqrt(d)` log
  turns out to be wildly heavy-tailed — and flag it before changing anything.
- **`dim_batch`:** a memory/speed knob, not a method parameter — the estimator and the total
  backward FLOPs are independent of it, and V3's routing is per-token top-k with no capacity
  limit, so replicating the prompt along the batch axis cannot perturb which experts fire.
  (Expect fp16-noise differences across values, not bit-identity: GEMM tiling changes.)
- **Cost anchor:** ~10 min/prompt for a 397B-A17B MoE on 8xH200, full depth. V3 is ~2.2x the
  active parameters, so full depth here should be slower; G4 measures it.
- **Parallelism:** fitting is embarrassingly parallel over disjoint prompt slices —
  run slices and combine with `JacobianLens.merge()`. Not useful on one box: the weights
  already need all three cards, so slices would have to run sequentially anyway.
- **Storage:** fp16 on disk (entries are O(1); fp16's mantissa beats bf16 here), load as fp32.
  205 MB per layer fp32 → a 60-layer fit checkpoint is **12.3 GiB per write**, and `fit` holds
  the running sum plus the per-prompt Jacobians (~25 GB) in RAM.
- **Layer index convention — RESOLVED, no off-by-one.** `jlens.ActivationRecorder` hooks
  `model.layers[L]` and takes `output[0]`; `scripts/extract/extract_hidden_states.py` hooks
  `model.model.layers[i]` and stores `output[0]` (`HiddenStateExtractor`, :374-395). Same
  object, same index. `L` means "output of block L" in the saved states and in the lens alike.

---

## 8. Applying it — forward-only, no re-extraction

`JacobianLens.transport(h, layer)` is `h @ J.T` and acts on **raw** residuals, so a fitted lens
applies to the already-extracted V3 numpy states. Same states in both arms ⇒ a clean head-to-head
against the logit lens, with the entire downstream pipeline unchanged.

DONE — `--jlens <path>` on both consumers, applying `h <- h @ J_L.T` immediately before the
RMSNorm: `scripts/decode/extract_residual_fingerprints.py` (both the CPU and GPU paths) and
`scripts/analysis/decode_2fact_heatmap.py` (outputs get a `_jlens` suffix so nothing is
clobbered). `scripts/jlens/apply_lens.py` loads a finished lens, a **fit checkpoint** (so a
run in progress can be inspected — the running sum is divided by `n_done`), or an `.npz`.
Layers the lens does not cover are skipped, and both arms are then restricted to the same set.

**Verified**: with a synthetic `J = I` lens, both consumers reproduce the logit-lens output
bit-for-bit (fingerprint ids identical, values max-abs-diff 0.0; every shared heatmap cell
identical). With the flag unset, behaviour is unchanged.

Then aggregate → judge exactly as now. Report **accuracies for each arm, not deltas.**

Note the target layer is excluded by construction (`source_layers` must be `< target`), and at
L60 the two lenses coincide anyway — `h_60` *is* the final residual, so `J_60` would be `I`.
The existing logit-lens arm to compare against is
`results/unsupervised_decode_2fact_allpos/decode_2fact_dots_10.json` (n=244 correct, 17
positions, layers 0-60): **A1 78.7% @pos_001 L50, A2 57.8% @pos_005 L51, sum 89.8% @pos_016 L60.**

---

## 9. What to actually look at

The question worth the GPU time is not "does the logit lens survive" — the causal KV-transplant
evidence is already lens-independent. It is:

1. **Does V3 have a workspace band at all, and does it coincide with where filler content
   decodes?** Nobody has measured this on a 671B MLA-MoE. Real result either way.
2. Does J-space surface content at filler positions that the logit lens misses?
3. The lens-comparison heatmap then falls out as a by-product.

Prior signals point **opposite ways**, so genuinely open: our own offline ridge/Belrose
comparison found principled lenses *worse* at mid-layer operands (logit lens A1 85% @pos_001 L43
vs ridge 55%), while praxagent's real J-lens on Qwen-397B recovered hidden two-hop bridge
entities at median rank 43/248,320 vs 620 for identity transport. Neither used our task.

Use the established metric: the per-(layer, position) number-token argmax heatmap from
`decode_2fact_heatmap.py`. **Do not** use full-vocab single-token rank at pooled dots positions
— A1 lives at pos_001, not in the dots content, and that error previously produced a misleading
"transport surfaces the operand" conclusion.

---

## 10. Fallback

If G1–G4 fail, **Route B** is the independent cross-check and needs no patches at all: a
forward-only finite-difference estimator (perturb source positions by `eps·v`, sum `Δh_final`
over valid targets, divide by count). Validated 2026-07-22 on a tiny model — reproduces `jlens`'
autograd `J` to 1e-13 in the linear limit; fp16-robust for decoded **tokens** at large eps
(eps=0.1 → top-5 agreement 1.00), though the matrix itself has a ~7e-3 floor with no U-plateau.
Two estimators agreeing would be strong faithfulness evidence; it is slower for a full `J`
(d probes/prompt), so use it as a check, not the primary route.

---

## 11. Known unknowns

- **Peak memory of the MoE backward** — G4. The one thing that decides afternoon vs wall.
- Fragmentation/allocator behaviour at this scale; a paper calculation cannot settle it.
- Whether real AWQ GPU kernels are deterministic enough for Route B (the CPU proxy showed 0
  noise; real kernels may add ~1e-3, raising the eps floor). G1 reports it.
- ~~Layer-index alignment~~ — **RESOLVED**, see §7: same hook, same object, no off-by-one.
- Whether the fp16 backward stays finite over a 60-block span. G4 counts non-finite entries
  and reports `grad_max_abs`. If it overflows, the exact fix is to scale the cotangent by
  `s` and divide the resulting rows by `s` (linear in the cotangent, so the estimator is
  unchanged) — but that means forking `jacobian_for_prompt`, i.e. no longer running the
  reference implementation unmodified. Flag it before doing it.

---

## 12. What "matching the paper" does and does not cover

Matched (all verified in the cloned reference, not assumed): the estimator itself
(`jlens.fit` unmodified), `source_layers=None` (every layer below target), target = final
block, `skip_first=16`, last position dropped, 128-token sequences, 1000-sequence target,
plain unnormalised mean, `force_bos=True`, fp16 storage, and the readout being the model's
own final norm + unembedding.

Deviations, all forced by the model and none touching the estimator:

1. **The weights are AWQ int4, not bf16** — the one substantive difference. The gradient is
   *exact for the weights being run* (AWQ's backward dequantises and returns
   `grad_output @ W.T`), but the paper's models were not 4-bit. Unavoidable: V3-671B in bf16
   is ~1.3 TB, and V3 has no unquantized release at all — it was trained and shipped FP8.
2. `moe_infer` rewritten out-of-place with `@torch.no_grad()` stripped — identical arithmetic.
3. The AWQ backward's `.repeat` replaced by a broadcasting matmul — same value (G1b).
4. `dim_batch` raised for throughput — documented as not affecting the estimator.
5. `attn_implementation="eager"` — exact attention either way.

Ours, not theirs, and not to be presented as the paper's: the metric we apply the lens *to*
(the per-(layer, position) number-token argmax heatmap on 2-fact). `eval_lens_quality.py`
exists so the lens is *also* scored by the paper's own metric on the paper's own sets.

**Convergence is read from the reference's own two in-fit diagnostics**, which `jlens.fit`
logs every prompt (`fitting.py:349`): `max||J||/sqrt(d)` flags heavy-tailed outlier prompts,
and `max_d_mean` = max_l ||J_i - J_n|| / ((n+1)*||J_n||) is the relative step of the running
mean, documented to fall ~1/n once settled. Read them together: a high `max_d_mean` sitting
next to a high norm is one outlier prompt, not non-convergence. Lens *quality* is
`eval_lens_quality.py`'s pass@k on the six shipped eval sets.

Do **not** add a homemade stopping rule on top. One was written and removed (`convergence.py`,
readout-argmax agreement between successive checkpoints): it measured the size of each step,
which is set by which prompts happened to land rather than by how converged the lens is, and it
was reported without an error bar — which would have shown it could not resolve the ~0.013
per-step change it claimed to track against a ~0.042 noise floor.
