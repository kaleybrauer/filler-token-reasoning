# DeepSeek-V3 vs Qwen3.5 — plan (written 2026-09-15)

Two questions, one shared pipeline.

**Q1. Is V3's workspace profile characteristic of frontier-scale multilingual MoEs, or is V3
unusual?** Compare V3 with Qwen3.5-397B-A17B (same scale class, multilingual MoE, different lab,
different architecture and training lineage) and with Qwen3.5-122B-A10B (same lineage, same
tokenizer, same 500 fit passages, smaller) which separates lineage from scale.

**Q2. Is the mid-depth representation language-neutral, and what sets the lexical language of its readout?** (Open question; nothing so far constrains the geometry — see the confound below.) For a fixed semantic state, is the
mid-depth geometry language-neutral, language-conditioned, or hybrid (shared meaning component +
language component)? And what determines which lexical language the readout shows?

Both questions use the same cached residuals; Q2 adds one matched bilingual input set.

---

## 0. Materials found (2026-09-15)

| | DeepSeek-V3-0324 (ours) | Qwen3.5-397B-A17B | Qwen3.5-122B-A10B |
|---|---|---|---|
| layers / d_model | 61 / 7168 | 60 / 4096 | 48 / 3072 |
| experts (active) | 256 (8) | 512 (10) | 256 (8) |
| attention | MLA, all full | hybrid: Gated DeltaNet, full attention every 4th layer | same |
| vocab | 129,280 (27% Han) | 248,320 (Han share: measure) | same tokenizer |
| BOS | yes (id 0) | `bos_token_id: null` | same |
| weights on disk | AWQ int4, 328 GiB | official FP8, 378 GiB | bf16, 233 GiB |
| lens | ours, n=100, T=40 pre-filter, halves A/B | `dallinmj/Qwen3.5-Jacobian-Lenses`, n=500, fp32, layers 0-58; also `praxagent-org/jacobian-lens-qwen3.5-397b-a17b`, n=24 | `dallinmj/...`, n=500, fp32, layers 0-46 |
| lens fit compute | AWQ int4 fwd, exact bwd, dim_batch 64 | FP8 weights with **bf16 grouped GEMMs**, dim_batch 16 | bf16, dim_batch 16 |

Notes that matter for reproduction:
- The dallinmj lenses are plain means (no pre-filter) over the same 500 WikiText passages
  (articles >=200 chars, `random.Random(0)` shuffle, first 128 tokens, "BOS enabled", one
  sequence per article). No per-prompt files -> no half-fit error bars for Qwen. The praxagent
  n=24 397B lens is an independent second fit: a crude between-fit check for lens-only measures.
- Qwen3.5 has no BOS token; the lens README says "BOS enabled". Before extracting, reproduce the
  encoder exactly (`jlens.encode` with `force_bos` on a BOS-less tokenizer should be a no-op —
  verify on one passage against the lens repo's stated recipe; ask the author if ambiguous).
- Qwen3.5 `RMSNorm.weight` is initialised to zeros -> the layer computes `(1 + weight) * norm(x)`.
  Fold `1 + w`, not `w`, into the unembedding. (V3 folds `w`.) Verify with G-UNEMBED as for V3.
- `Qwen3_5MoeForConditionalGeneration` wraps the LM under `model.language_model`; hook
  `model.language_model.layers[L]` output[0] (jlens `Layout(path="model.language_model")`).
- The 397B fit used bf16 compute on FP8 weights. Standard FP8 inference (W8A8) is a different
  numerical regime. Expect the same kind of regime sensitivity we measured on V3 (single Jacobians
  batch-dependent, averaged readouts stable) — see project notes RESOLVED 2026-09-14.

---

## 1. Q1 — the direct comparison

Reproduce the V3 analysis series for series, on the same held-out text (our WikiText 200-299 and
the one-dump Wikipedia zh/en corpora already built), over normalised depth:

1. Figure 28 (a)-(d): next-token top-k, kurtosis percentiles, top-1 autocorrelation, J-space
   dimensionality. (`workspace_readouts.py`, `workspace_signatures.py`)
2. CKA layer bands. (`cka_layers.py`)
3. Panel (b) taken apart (`bilingual_diagnostics.py`, `plot_bilingual.py`): full-vocab,
   Latin-only, Han-only, size-matched random subsets, offset-removed kurtosis; script-offset eta^2;
   offset-removed top-k script shares; translation pairs with the other-prompt null.
4. Lens health, because Qwen's lenses are unfiltered plain means: per-layer top-singular share
   s1/||J||_F, whether the dominant direction is a script axis (`script_axis.py`), giant signatures
   (rank-1 structure, punctuation/EOS readout directions), between-fit agreement n=500 vs n=24.

Pre-specified comparisons (decide before looking):
- Onset depth from (a), (c), (d); peak depth and height of within-script kurtosis p99; CKA band.
- Translation-pair curve and the Chinese-input crossover depth (Latin -> Han majority).
- Noise floor: V3's half-fit ribbon. A V3-vs-Qwen difference counts only if it exceeds the ribbon
  AND is not explained by (i) vocab composition (use within-script / offset-removed measures),
  (ii) lens-fit differences (n, no pre-filter — check item 4), (iii) architecture (Qwen's hybrid
  attention; 122B vs 397B shows whether the profile is a lineage property).
- Three outcomes: V3 ~ Qwen-397B ~ Qwen-122B (frontier multilingual MoE profile); V3 differs from
  both while the two Qwens agree (V3 unusual — then check whether the V3 lens fit explains it);
  all three differ (no single profile; report per-model with error bars where we have them).

---

## 2. Q2 — does the workspace have a language?

### The confound to design around
The unembedding maps translations near each other by itself: in V3, centred unembedding rows of
72,827 single-token English/Chinese dictionary pairs have median cosine 0.08 (random 0.00), and
nearest-Han-neighbour retrieval from the unembedding alone recovers a true translation top-1 52%
of the time, top-10 75%. So readout co-occurrence of ' dog' and '狗' is expected from W_U for ANY
residual vector that ranks ' dog' high. Readout-based evidence cannot separate the hypotheses.
Tests below live in residual space with matched inputs; the readout is used only for the
"which lexical language appears" question, with W_U-aware controls.

### Matched bilingual inputs (one GPU extraction per model)
A. **Templated concept pairs (single position, fixed concept).** ~500-1000 concepts from the
   CC-CEDICT single-token pair list (both tokens single in each model's vocab; intersect V3 and
   Qwen vocabularies so the same concept set is used across models). A few neutral templates per
   language that end at the concept token, e.g. "The word I am thinking of is X" / "我想到的词是X",
   plus the concept token alone after a fixed prefix. Residual at the concept position and at the
   next position, all layers. Same concept x {en, zh} x templates.
B. **Natural parallel sentences.** FLORES-200 dev (eng_Latn / zho_Hans, ~1000 sentences,
   professional translations; fallback WMT news-commentary zh-en). Residual at every position
   (mean over positions after the first few) and at the last token, all layers.
C. Already have: our zh and en Wikipedia corpora (unmatched content) and the paper's
   lens-eval-multilingual set (V3 states cached).

### Tests (CPU, on cached states; run per model, per layer, raw residual AND J-transported)
1. **Shared component: cross-lingual retrieval.** For each en state, nearest neighbour among zh
   states of the same set (cosine, after per-language centring). Accuracy vs depth. Also without
   per-language centring. High mid-depth retrieval = a shared semantic component exists (needed by
   neutral and hybrid; conditioned predicts poor retrieval without a learned alignment).
2. **Language component: separability and its rank.** AUC of en-vs-zh along the mean-difference
   direction d_lang = mu_zh - mu_en; and a linear probe. Then the spectrum of the between-language
   difference (PCA of paired differences h_zh(c) - h_en(c) over concepts c): how many directions
   carry the language separability, and what share of total variance they hold, vs depth.
   - Neutral: separability collapses at mid depth (AUC -> ~0.5), language share -> ~0.
   - Hybrid: separability stays high but low-rank (1-few directions), small variance share,
     and removing that subspace leaves retrieval intact or improved.
   - Conditioned: separability high AND spread across many directions; removing a low-rank
     subspace does not remove it; retrieval needs more than centring.
3. **Decomposition check for hybrid.** Project out the top-k language directions (k = 1, 4, 16)
   and re-run test 1; report retrieval vs k. Also: does h_lang direction at layer L match across
   concept sets A and B and across our unmatched Wikipedia corpora (cosine between d_lang
   estimates)? A single stable direction across inputs is the hybrid signature.
4. **What sets the readout's lexical language.** Using the lens (J-lens and logit lens):
   (i) for each concept, rank(' dog') vs rank('狗') in the readout of the en state and of the zh
   state, vs depth — the base-rate-free version of today's "English lean"; (ii) regress the
   Latin-vs-Han readout balance on the projection onto d_lang; (iii) the lens-steering test:
   read out h_en + alpha * d_lang for a grid of alpha — does the lexical language flip while the
   concept's translation stays top? Controls: the same with W_U-similar random directions, and
   the unembedding-only prediction (rank change expected from W_U geometry alone).
   Interpretation: language re-entering "near the output" = the readout balance is set by
   d_lang projection which is small mid-depth and grows late.
5. **Causal follow-up (GPU, later, optional).** Add alpha * d_lang at layer L during a real forward
   on English text and measure whether the model's continuation switches language (and whether
   content is preserved). This is the model-level test of "language re-enters near the output";
   the lens tests above are descriptive.

### Cross-model reading
Run tests 1-4 on V3, Qwen-397B, Qwen-122B (+ a cheap CPU-runnable small model, e.g. Qwen3.5-4B
or Gemma-3-4B, for an English-dominant contrast if wanted). "What sets the language" is then:
depth of the crossover, language-subspace share, and which hypothesis fits — against training
lineage (V3 vs Qwen) and scale (122B vs 397B).

---

## 3. Compute and order of work

CPU first (no GPU needed, start now, after the running chain finishes):
1. Make the analysis scripts model-generic: a registry {model: lens path/format, unembedding
   tensor + norm weight (+1 convention), tokenizer, states path, n_layers, d}. Today they hard-code
   V3 paths (`WEIGHTS`, `vocab_scripts.json`, `STATES`).
2. Download the dallinmj lenses (1.7 + 3.8 GB) and praxagent lens (1.9 GB); Qwen `lm_head` and
   final norm from two shards (6.4 + 2.5 GiB for 397B; 6.0 + 5.0 for 122B). Convert unembedding
   to the .npy form the scripts read. Vocab script classes for the Qwen tokenizer.
3. Lens-only measures for all Qwen lenses: panel (d), CKA, s1 share, script axis, giant signatures,
   n=500 vs n=24 agreement. Answers part of Q1 before any GPU time.
4. Build input sets A and B (tokenise with both tokenizers; keep concepts single-token in both).

GPU session 1 — Qwen3.5-122B bf16 (2-3x H200, exact regime): ~1-2 h. Forward only.
   Extract: WikiText 200-299, Wikipedia zh/en (100 each, 128 tok, all 48 layers: ~3.4 GB per
   corpus fp16), sets A and B, G-UNEMBED gate (with the 1+w norm fold).
GPU session 2 — Qwen3.5-397B FP8 (4x H200 recommended; 3x is 423 GiB vs 378 GiB weights — tight):
   same extraction; ~2-3 h. Standard FP8 path; note the regime caveat. If an 8x H200 box is ever
   available, bf16-dequantised weights (~756 GiB) reproduce the fit's compute path exactly.
GPU session 3 — V3 (3x H200, ~40 min): sets A and B only (everything else is cached).
   Sessions 1-3 are independent; order by availability. Session 3 can share a box with 1.

CPU after each session: run items 1-4 of section 1 and tests 1-4 of section 2 for that model.
When the outputs land in `outputs/jlens_qwen35/<tag>/`:
`python scripts/jlens/run_qwen35_analysis.py qwen35_122b qwen35_397b_fp8` runs
`check_qwen35_transfer.py` (hard gate) and then every analysis above with `--model`, resumably.

---

## 4. Caveats to carry into the write-up
- No within-model error bars for Qwen (single fits). V3's half-fit ribbon is the only floor; the
  n=500 vs n=24 397B lenses give a between-fit check for lens-only measures.
- Heavy tails: the Qwen lenses are unfiltered plain means. If item 1.4 shows giant signatures,
  early-layer Qwen readouts carry the same caveat as V3's pre-filter analysis, and we cannot fix
  it without per-prompt refits.
- Vocabulary-level statistics (full-vocab kurtosis, script shares) are not comparable across
  tokenizers; use within-script, offset-removed, and pair-rank measures.
- Forward regime differs from fit regime for every model (batch, kernels, FP8 path).
- "Held-out" is exact only for V3. Our WikiText 200-299 are middle paragraphs from the same
  WikiText-103 train split as the dallinmj passages (article openings, 500 of ~28k articles):
  expected article overlap is about 2 of 100, not verified. The praxagent card names WikiText-103
  (seed 0) but not the split.
- Depth normalisation hides architecture: Qwen's hybrid attention may itself shift the band.
- Hypothesis tests 1-4 are descriptive (lens/geometry). Only test 5 is causal.
