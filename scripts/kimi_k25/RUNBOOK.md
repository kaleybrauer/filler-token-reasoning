# Kimi K2.5 — system-of-equations (varbind) run plan

Prepared 2026-07-28 on the CPU/analysis instance. Everything below is verified from the HF
repo metadata, the vLLM source, and the local tokenizer — no GPU was used.

**Decisions (Kaley, 2026-07-28):** 8×H200 TP=8 · 8 conditions × 500 (match the V3 varbind set)
· instant mode only (thinking disabled everywhere) · hidden-state extraction, **no transplants**
· **the GPU pod does not mount `/workspace`** → this runbook is self-contained: clone the repo,
download weights onto the pod's local disk, ship artifacts out over the network.

---

## 0. Established facts

| Fact | Value |
|---|---|
| Checkpoint | `moonshotai/Kimi-K2.5` — **already 4-bit**: compressed-tensors `pack-quantized`, INT4, group_size 32, symmetric |
| Not quantized (stay BF16) | `self_attn.*`, `shared_experts.*`, dense `mlp.{gate,up,down}_proj`, `lm_head`, `vision_tower`, `mm_projector` |
| Size / shards | **595.2 GB**, 64 safetensors, 208 550 tensors, ungated, Modified-MIT |
| Architecture | `KimiK25ForConditionalGeneration` (multimodal wrapper) over a DeepseekV3-style text model: 61 layers, hidden 7168, 64 heads, MLA (kv_lora 512 / q_lora 1536), 384 experts (8 active), vocab 163840 — **same shape as Kimi K2**, plus a 400 M MoonViT tower we never touch |
| Weight keys | `language_model.model.layers.N.…`, `language_model.lm_head.weight` + `language_model.model.norm.weight` (both BF16, **shard 62 of 64**) |
| Thinking control | chat template kwarg. `thinking=False` renders `<|im_assistant|>assistant<|im_middle|><think></think>` (tokens **163606, 163607**); default renders a bare open `<think>` |
| Few-shot turns | history assistant turns always render `<think></think>` — consistent with instant mode |
| Tokenizer | `tiktoken.model` is **byte-identical to the Kimi K2 copy already on this volume** (md5 `d94182879fdcd560c79fdc346afc20b6`); `tokenization_kimi.py` differs only by the media tokens |
| Number tokens | bare ints **0–999 are single tokens**, 1000+ are not; space-prefixed numbers (`' 93'`) are *not* tokens → `decode_varbind_heatmap.py --max-num-token 1000` ports unchanged |
| `Filler` label | tokenizes `[' F', 'iller']` — same as DeepSeek V3, so `find_filler_boundaries` works unchanged |
| Engine | vLLM **≥ 0.15.0** (pin `0.25.1`; 0.26.0 is 3 days old). `KimiK25ForConditionalGeneration` implements `SupportsPP`, `SupportsMultiModal`, `SupportsQuant` |
| Official recipe | `vllm serve $MODEL -tp 8 --trust-remote-code --reasoning-parser kimi_k2 --mm-encoder-tp-mode data` on 8×H200 |
| Alternatives (rejected) | `nvidia/Kimi-K2.5-NVFP4` 590.9 GB (Blackwell), `amd/Kimi-K2.5-W4A8` 532.8 GB (ROCm), `QuantTrio/Kimi-K2.5-E304` 476.2 GB and `Ex0bit/…REAP-530B` 309.6 GB (**expert-pruned = different model**), GGUF forks (no extraction path) |

**GPU math** (595.2 GB weights; H200 = 141 GB, proven ceiling ≈130 GB of weights/GPU from the
K2 W4A16 run): 8 GPUs → 74 GB/GPU ✅ · 6 (TP2×PP3) → 99 GB ✅ · 5 (PP5) → 119 GB ✅ ·
4 → 149 GB ❌. TP must divide 64 heads ⇒ TP ∈ {1,2,4,8}; PP is free over 61 layers.
**Chosen: 8×H200, TP=8** — every worker then holds all 61 layers (post-all-reduce hidden
states are replicated), which keeps the hook code trivial and prefill fastest.

---

## 1. Phase A — CPU prep (do all of this before renting anything)

**A1. Config-only model dir.** Pull the ~3 MB of non-weight files into
`/workspace/models/kimi-k2.5-configonly/` (`config.json`, `chat_template.jinja`,
`tokenizer_config.json`, `tokenization_kimi.py`, `tiktoken.model`, `generation_config.json`,
`preprocessor_config.json`, `configuration_*.py`, `modeling_*.py`). This makes every
prompt/boundary check runnable offline.

**A2. `scripts/kimi_k25/prompt_k25.py` — instant-mode prompt builder.**
Render the shared varbind scaffold with the Kimi markers and thinking disabled:

```
<|im_system|>system<|im_middle|>{system}<|im_end|>
<|im_user|>user<|im_middle|>{few-shot user}<|im_end|>
<|im_assistant|>assistant<|im_middle|><think></think>{few-shot answer}<|im_end|>
…
<|im_user|>user<|im_middle|>{vars + Question:}<|im_end|>
<|im_assistant|>assistant<|im_middle|><think></think>Filler: {filler}

Answer:
```

Use `apply_chat_template(msgs, add_generation_prompt=True, thinking=False, tokenize=False)`
and append the `Filler:`/`Answer:` scaffold, so the template — not us — decides the markers.
The scaffold from `question_end` onward must stay byte-identical to the V3 varbind prompts.

**A3. Offline gates (assert, don't eyeball):**
1. assistant header token ids end `[…, 163601 (<|im_middle|>), 163606 (<think>), 163607 (</think>)]`;
2. no occurrence of 163606 that is not immediately followed by 163607 anywhere in the prompt;
3. filler span per condition: `dots_k` → k tokens, `counting_k` → 2k−1 tokens (see
   `project_filler_token_counts`), and `find_filler_boundaries` recovers it;
4. prompt lengths per condition (V3 ran ~600–900 tokens; set `max_model_len=2048`);
5. ints 0–999 single-token (done — keep the assertion in the test).

**A4. `scripts/kimi_k25/extract_hidden_states_vllm.py` + `hooks.py` — the real port.**
The V0 path used by `scripts/kimi_k2/extract_hidden_states_vllm.py`
(`VLLM_USE_V1=0`, `llm_engine.model_executor.driver_worker.model_runner.model`) **does not
exist in vLLM ≥0.15**. Replacement:

* `LLM(..., worker_extension_cls="kimi_k25_hooks.CaptureExt")`, driven by
  `llm.collective_rpc("install_hooks" | "pop" | "clear")`. The extension class must be
  importable inside the worker processes (put its dir on `PYTHONPATH`, no closures).
* Hook `model.language_model.model.layers` (note the extra `language_model.`).
* **Capture `output[0] + output[1]`, not `output[0]`.** vLLM's `DeepseekV2DecoderLayer.forward`
  returns `(hidden_states, residual)` in the *split* convention — `output[0]` is the MLP write,
  the residual stream after layer L is the sum (verified in vLLM `v0.8.5` and `main`). The HF
  `modeling_deepseek.py` path used for V3 adds the residual inside the layer, which is why
  `output[0]` was right there and is wrong here. See §5 for the audit of the existing Kimi/Qwen3
  extractions, which use `output[0]`.
* `enable_prefix_caching=False` **and** `enable_chunked_prefill=False`. Both default on in V1;
  either one silently truncates or skips the prefill we are trying to capture.
* Keep the "first forward per layer after `clear()`" guard (decode steps overwrite otherwise).
* Slice to the target positions **inside the worker** and return fp16 arrays
  (~52 MB/example at dots_50) — never ship the full 61×seq×7168 tensor (786 MB) over RPC.
* With TP=8 take worker 0's result. (If a PP config is ever used, merge by global layer index
  via `model.language_model.model.start_layer`.)
* Output pkl schema identical to `data/extracted_states_varbind_allpos/` (positions, states,
  boundaries, model_response, model_correct, `intermediate`), so every downstream script works.

**A5. `scripts/kimi_k25/eval_varbind_vllm.py`** — batched accuracy sweep. No varbind vLLM
evaluator exists yet; model it on `scripts/kimi_k2/eval_accuracy_vllm.py` (one big
`llm.generate` call per condition, ~0.08 s/example).

**A6. Residual-convention audit (CPU, free, uses data already on this volume).** Take an
existing Kimi K2 pkl + `data/model_weights/kimi_k2/{lm_head,rms_norm}_weight.npy`, apply
RMSNorm→lm_head to the **last layer** at `answer_prompt`, and check the argmax equals the token
the model actually generated. True residual stream ⇒ near-100% match; MLP-delta-only ⇒ it fails.
This is the same gate we then run live on K2.5 (C3.4), and it tells us whether the published
Kimi K2 / Qwen3-32B numbers were computed on the residual stream or on layer writes.

**A7. Push the branch** (Kaley — no commits or pushes from the agent) so the GPU pod can clone.
`data/chained_var_binding_dataset.json` is force-tracked in git (md5
`b3d33393b8973667d3043a4dc2c7d0b8`) → the pod gets the exact dataset from the clone.

## 2. Phase B — tiny-model smoke (1 cheap GPU, ~20 min, ~$1)

`tiny-random/kimi-k2.5` is 9.5 MB with the **same architecture** (`KimiK25ForConditionalGeneration`,
2 layers, hidden 8, 32 experts, unquantized). Point `--tokenizer` at the config-only dir.
It validates: vLLM version loads `kimi_k25`, `worker_extension_cls` + `collective_rpc` plumbing,
layer path, capture-count == n_layers, prefix caching off, pkl schema, no `<think>` in output.
Everything except the real weights. Can also be run as the first 20 minutes of the big pod.

## 3. Phase C — GPU pod (8×H200)

**Pod requirements:** 8×H200 141 GB, **1 TB disk** (595 GB weights + ~120 GB states + 2.4 GB
lm_head + ~30 GB HF download scratch + ~25 GB venv/logs ≈ 775 GB, leaving ~225 GB headroom),
CUDA 12.8+ driver, plenty of RAM. Point `HF_HOME` at the big disk so the xet chunk cache and
`.incomplete` files don't fill the container root. Bring: HF token, GitHub token (if the repo
is private), and an upload target (HF private repo recommended). Upload the states directory
directly — do **not** tar it first, that would double the space.

Per-condition state sizes (V3 measurements × 1.18 for K2.5's longer position span):
baseline ~1 GB · dots_5 ~6 · dots_10 ~8.5 · dots_25 ~17 · dots_50 ~28 · counting_5 ~9 ·
counting_10 ~14.5 · counting_25 ~28 ⇒ ~112 GB, call it 120 GB with pickle overhead.

| t | step | detail |
|---|---|---|
| 0:00 | download + env in parallel | `hf download moonshotai/Kimi-K2.5 --local-dir /models/kimi-k2.5` under `nohup`; meanwhile `uv venv` + `uv pip install vllm==0.25.1 tiktoken blobfile setuptools` and `git clone` the repo |
| 0:15 | tiny smoke (Phase B) | while weights download |
| ~0:45 | load real model | `tensor_parallel_size=8, quantization="compressed-tensors", dtype="bfloat16", enforce_eager=True, max_model_len=2048, gpu_memory_utilization=0.92, enable_prefix_caching=False, enable_chunked_prefill=False, trust_remote_code=True`; `NCCL_NVLS_ENABLE=0` if NCCL hangs |
| +10 min | preflight gates | see below — **stop the run if any fails** |
| +20 min | behavioral sweep | 8 conditions × 500, batched (~10 min) |
| +3.5–5 h | extraction | 8 conditions × 500, all positions, 61 layers, fp16 → ~92 GB (V3's varbind set is 92 GB and the dims are identical) |
| +30–60 min | ship artifacts | upload to HF private repo, verify, then terminate |

**Preflight gates (C3):**
1. `verify_no_thinking.py` — generate 5 free-running completions; assert no `<think>` token
   (163606) in any output and that the rendered prompt carries the closed pair.
2. filler span == expected for each condition (the only guard that boundaries are right).
3. captured layer count == 61 on the first example; no NaN/Inf; shape (n_pos, 7168).
4. **last-layer logit-lens gate**: RMSNorm→lm_head on the captured L60 state at `answer_prompt`
   ⇒ argmax == the token vLLM actually generated. This is what proves the capture convention.
5. save `language_model.lm_head.weight` + `language_model.model.norm.weight` from **shard 62**
   to `data/model_weights/kimi_k25/{lm_head_weight,rms_norm_weight}.npy` (2.4 GB) — needed by
   every decode script, and small enough to ship out first.

**Conditions:** `baseline`, `dots_{5,10,25,50}`, `counting_{5,10,25}` × 500 examples,
matching `data/extracted_states_varbind_allpos/`. `CONDITIONS` in
`scripts/extract/extract_hidden_states.py` already has these entries.

**Ship-out priority:** ① accuracy grid JSON + logs (KB) ② lm_head/norm npy (2.4 GB)
③ raw states (~92 GB). If the link is slow, also run `decode_varbind_heatmap.py` on the pod so
the headline ladder survives even if ③ never lands.

**Rough cost:** 6–8 h on 8×H200 ≈ $180–250 at typical RunPod rates.

## 4. Phase D — analysis (back here, CPU)

1. `scripts/analysis/decode_varbind_heatmap.py --max-num-token 1000` → the 5-target ladder
   (x, c₁·x, y, c₂·y, answer). Both models are 61 layers, so the comparison to V3's
   x@L33 → c₁·x@L38 → y@L44 → c₂·y@L51 → answer@L60 is directly apples-to-apples.
2. `extract_residual_fingerprints.py` → `aggregate_residuals_all_settings.py` →
   `llm_decode_batch.py --task varbind` (+ `--shuffle` control). ~8 conditions on 8 cores is
   slow (V3 took ~2 h on 32 cores for 6 conditions) — consider a cheap CPU-heavy pod.
3. Headline comparisons: does the +24–31 pt depth-1 filler uplift replicate on a
   natively-reasoning model with reasoning suppressed, and does the never-written `y` / `c₂·y`
   still decode at 90–99% top-10?

## 5. Risks & open items

| Risk | Mitigation |
|---|---|
| vLLM V1 hook API unfamiliar | Phase B tiny-model smoke; gates 3 + 4 |
| **Prefix caching / chunked prefill silently corrupt captures** | both explicitly disabled + capture-count gate |
| Capture convention (`output[0]` vs `output[0]+output[1]`) | A6 audit offline, gate 4 live |
| Reasoning leaks into the answer | closed `<think></think>` + teacher-forced `Answer:` scaffold + generation scan |
| INT4 group-32 Marlin kernels on Hopper | the official recipe is 8×H200 with this exact checkpoint; fails loudly at load |
| Download interrupted / disk full | `hf download` resumes; verify 64 shards against the HF blob sizes before loading |
| No shared volume | clone repo from GitHub, weights from HF, artifacts out via HF private repo |
| Answers > 999 are multi-token | same limitation as V3; exact-match metric unchanged |

**Open items:** (a) confirm the upload target for the ~92 GB of states; (b) decide whether to
re-run the K2/Qwen3 residual-convention audit results into the paper if A6 shows a mismatch;
(c) `--mm-encoder-tp-mode data` is irrelevant for text-only prompts but harmless to pass.
