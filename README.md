# Decoding Hidden Computation in Filler Tokens

Investigates whether filler tokens (e.g., `. . . . .` or `1 2 3 4 5`) inserted between a question and an answer give large language models additional computation depth and what the model is actually computing during that span. The tasks span two kinds of hidden value: facts the model must *retrieve* (addition, letter-position) and a system of equations task which must be computed fully in-context. This repo contains scripts for experiments on instruction-tuned MoE models: hidden-state decoding via logit lens (with an LLM judge over residual top tokens), KV-cache transplant interventions, and attention-pattern analysis.


## Tasks

Four tasks share the same trailing scaffold (5 few-shot examples held out of any target pool, then a target question, then `Filler:` filler tokens, then `Answer:`):

| task | prompt template | what the model computes |
|---|---|---|
| 1-fact addition | "What is [fact phrase] plus [X]?" | look up A, return A + X |
| 2-fact addition | "What is [fact phrase 1] plus [fact phrase 2]?" | look up A₁ and A₂, return A₁ + A₂ |
| letter-position | "What is the [Nth] letter of the chemical element with atomic number [Z]?" or "What is the last letter of the name of the capital of [region]?" | look up the entity (element name from atomic number, or capital from a country, state, province, or territory), then return the requested letter — `second`/`third`/`last` for elements (95 elements × 3 positions), always the last letter for capitals |
| system of equations | five variable definitions, each a literal (`zab = 45`) or an expression over an earlier one (`xoc = twice the number for zab plus 14`), then "What is [c₂] times the number for [term] [±] [k₂]?" | resolve the reference chain to a value y that is never written in the prompt, then return c₂·y ± k₂ |

The first three tasks test **retrieval** of a stored fact. *System of equations* (a depth-1 chained variable-binding task, ported from the multi-hop repo) tests **transient computation**: the queried value y is defined only through a reference chain — `x → c₁·x → y` — and never appears in the prompt, so the model must construct it in the forward pass. It uses a task-specific system message and nonsense CVC terms instead of the addition/letter-position system prompt.

Filler types: `dots` (`. . .`), `counting` (`1 2 3 ...`), `alphabet` (`a b c ...`). Accuracy evals sweep k = 0–200, where k=0 is the no-filler `baseline`. Hidden-state extraction uses the named conditions in `CONDITIONS` (`scripts/extract/extract_hidden_states.py`) — `dots_{1,3,5,10,25,50,100}`, `counting_{1,3,5,10,25,50}`, `alphabet_{10,25,100}`, `baseline` — a subset, since each condition costs a full cache. Note that `counting_K` renders as 2K−1 tokens rather than K, so counting tops out at lower K than dots. Datasets and held-out few-shot pools live in `data/`.

## Datasets

We include data files for 2-fact addition, the two letter-position domains, and the system-of-equations task (its nonsense CVC terms carry no real-world facts, so it ships in full). The 1-fact addition dataset must be regenerated locally because it draws facts from Ryan Greenblatt's [`compose_facts`](https://github.com/rgreenblatt/compose_facts) repo which are not to be pushed to github so they can continue to be used as test questions for future LLMs.

| dataset | file | provided? | source / regenerate with |
|---|---|---|---|
| 1-fact addition | `data/1hop_addition_dataset.json` | no | `scripts/data/generate_1hop_dataset.py` (needs a `known_facts.json` from a model-specific knowledge check) |
| 2-fact addition | `data/2fact_addition_dataset.json` | yes | `scripts/data/generate_2fact_dataset.py` |
| System of equations (variable binding) | `data/chained_var_binding_dataset.json` | yes | `scripts/data/generate_varbind_dataset.py` (500 examples, seed 42; `--chain-len 2` builds the depth-2 negative control) |
| System of equations — easy variant | `data/chained_var_binding_easy_dataset.json` | yes | same generator with `--max-coef 2 --const-max 30`: coefficients {2} not {2, 3}, constants 1–30 not 1–50, so answers land in 8–460 and are **all single-token** (the original spans 0–1013 and has two multi-token answers). This is the dataset the Kimi K2.5 run uses; both are 500 examples + 8 few-shot, seed 42, `chain_len=1`, `num_terms=5`, and are independently sampled rather than matched pairs |
| Letter-position (elements) | `data/element_letter_positions.json` | yes | `scripts/data/generate_letterpos_dataset.py` |
| Letter-position (capitals) | `data/capital_letter_position.json` | yes | hand-curated; no generator |
| Element multilingual aliases | `data/element_aliases.json` | yes | `scripts/data/build_element_aliases.py` (Wikidata) |
| Capital multilingual aliases | `data/capital_aliases.json` | yes | `scripts/data/build_capital_aliases.py` (Wikidata + manual fixes) |

The fact pool for 1-fact comes from Ryan Greenblatt's [`compose_facts`](https://github.com/rgreenblatt/compose_facts) repo (atomic numbers, ages at death, hand-curated static counts). To regenerate the 1-fact dataset end-to-end:

```bash
# 1. Download upstream fact JSONs
python finetuning/scripts/fetch_facts.py --outdir data/sources

# 2. Filter to facts the target model reliably knows
python finetuning/scripts/knowledge_check.py \
    --model /workspace/models/deepseek-v3-awq \
    --sources data/sources \
    --outdir data/facts/knowledge_deepseek_v3

# 3. Generate the 1-fact addition dataset
python scripts/data/generate_1hop_dataset.py \
    --known-facts data/facts/knowledge_deepseek_v3/known_facts.json \
    --output data/1hop_addition_dataset.json
```

## Models

The headline pair is two instruction-tuned MoE models, run from publicly-released 4-bit checkpoints on Hugging Face. Everything under [`release/`](release/) comes from these two:

| Hugging Face repo | parameters | quantization | extraction stack |
|---|---|---|---|
| [`cognitivecomputations/DeepSeek-V3-0324-AWQ`](https://huggingface.co/cognitivecomputations/DeepSeek-V3-0324-AWQ) | 671B total / 37B activated | AWQ 4-bit, group 128 | HF transformers + autoawq, `scripts/extract/` |
| [`RedHatAI/Kimi-K2-Instruct-quantized.w4a16`](https://huggingface.co/RedHatAI/Kimi-K2-Instruct-quantized.w4a16) | 1T total / 32B activated | W4A16 (compressed-tensors), group 128 | vLLM TP=4, `scripts/kimi_k2/` |

Two further checkpoints are used as controls, each with its own extraction fork and runbook:

| checkpoint | why | scripts |
|---|---|---|
| [`deepseek-ai/DeepSeek-V3-0324`](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324) native block-FP8 (689 GB, 163 shards) | precision control — does the decode result survive at canonical precision, or is it an AWQ artifact? Scoped to 2-fact `dots_10`, the headline condition. Pre-registered success criteria in [`scripts/fp8/RUNBOOK.md`](scripts/fp8/RUNBOOK.md) | `scripts/fp8/` (`download_fp8.sh`, `preflight_fp8.py`) |
| `moonshotai/Kimi-K2.5` | cross-model replication of the system-of-equations ladder with reasoning suppressed (`thinking=False`); also 1-fact and 2-fact | `scripts/kimi_k25/` (+ `RUNBOOK.md`, `HANDOFF.md`, `SESSION_STATE.md`) |

Download each checkpoint to any local directory and pass the path to scripts via `--model-path`.

Kimi K2.5 needs a separate fork rather than the K2 one because vLLM ≥ 0.15 moved the hook path into the workers (`worker_extension_cls` + `collective_rpc`), K2.5 wraps its text model as `language_model.model.layers`, and `enable_prefix_caching` / `enable_chunked_prefill` default on in V1 and silently corrupt the prefill being captured. See `scripts/kimi_k25/RUNBOOK.md` for the capture-convention audit (`audit_residual_convention.py`) that confirms the hook returns the residual stream rather than the MLP write.

## Analysis pipeline

The three interpretability analyses share the model, dataset, prompt scaffold, and filler-boundary detection logic, but each runs its own forward passes — hidden-state decoding caches per-(layer, position) states to disk for offline decoding, while KV transplant and attention analysis capture the quantities they need in-memory during the run. The accuracy evals in §4 establish the behavioural effect the other three are explaining.

### 1. Hidden-state decoding (all four tasks)

Four stages, the first on GPU and the rest CPU-only:

```bash
# (a) Cache hidden states at every (layer, position) for each example.
# --dataset-type selects the prompt builder (default 1hop); --all-positions caches every
# token from question_end through answer_prompt (without it, only the filler positions).
# Add `baseline` to --conditions for the no-filler control.
python scripts/extract/extract_hidden_states.py \
    --model-path /workspace/models/deepseek-v3-awq \
    --dataset data/2fact_addition_dataset.json --dataset-type 2fact --all-positions \
    --conditions dots_10 dots_25 dots_50 counting_10 counting_25 counting_50 \
    --output-dir data/extracted_states_2fact_allpos

# (b) Per-(layer, position) residual fingerprints (P_e − mean_e P_e, top-30 tokens per setting)
# --extraction-dir points at the per-condition subdir (prob_*.pkl); --lm-head/--rms-norm
# are the saved readout weights for the model that produced the states.
python scripts/decode/extract_residual_fingerprints.py \
    --extraction-dir data/extracted_states_2fact_allpos/dots_10 \
    --model-path /workspace/models/deepseek-v3-awq \
    --lm-head data/model_weights/deepseek_v3/lm_head_weight.npy \
    --rms-norm data/model_weights/deepseek_v3/rms_norm_weight.npy \
    --output outputs/deepseek_aggregated/fingerprints_dots_10.npz

# (c) Aggregate residuals across (layer, position) into per-example top-50
python scripts/decode/aggregate_residuals_all_settings.py \
    --fingerprints outputs/deepseek_aggregated/fingerprints_dots_10.npz \
    --model-path /workspace/models/deepseek-v3-awq \
    --output outputs/deepseek_aggregated/aggregated_dots_10.json

# (d) LLM-as-judge over the aggregated top-50 tokens (Claude Haiku 4.5 + Sonnet 4.6)
# --aggregated-dir holds aggregated_<cond>.json; pass matching --conditions.
python scripts/decode/llm_decode_batch.py \
    --aggregated-dir outputs/deepseek_aggregated \
    --conditions dots_10 \
    --task 2fact --prompt neutral
```

Stage (d) defaults to both judges (`--models haiku sonnet`) and writes next to `--aggregated-dir` unless `--outdir` is given. Convenience drivers for the two letter-position domains: `scripts/run_letterpos_fingerprints.sh` (elements) and `scripts/run_capitalpos_fingerprints.sh` (capitals). Multilingual coverage of the per-example top-K tokens (substring match against EN + ZH name tables) is `scripts/analysis/multilingual_coverage.py`, with reference tables in `data/element_aliases.json` and `data/capital_aliases.json`.

**Pooled read-out (alternative to stages b–c).** Rather than one residual fingerprint per (layer, position), `scripts/decode/pooled/` pools full-vocab softmax across the settings that survive a global-agreement + greedy-dedup selection, then takes per-example top-K: `pool_decode_global_dedup.py` (or `pool_decode_global_dedup_1hop.py` for 1-fact) selects the settings, `pool_top_tokens.py` writes the pooled top-K — optionally numeric tokens only, which rules out the judge reading entity names straight off the tokens — and `llm_decode_pool.py` / `pool_decode_topk.py` score them (the latter covers 2-fact and varbind, with an option to mask each example's externally visible values). `scripts/decode/single_setting/` holds the earlier one-(layer, position) variants used to find those settings unsupervised: `cross_example_consistency.py` (pairwise AMI between settings), `per_position_layer_select.py` and `discover_variables.py` (neighbour-layer consistency), `unsupervised_decode_filler.py`.

**Residual ablation.** `extract_residual_fingerprints.py --no-residual` ranks by raw `P_e(token | s)` instead of the cross-example-mean-subtracted residual, which is the ablation reported in `release/tables/table_residual_ablation.tex` (DeepSeek V3 only).

**System of equations.** The same four stages run for the variable-binding task. Extract with `--dataset data/chained_var_binding_dataset.json --dataset-type varbind` over `--conditions dots_10 dots_25 dots_50 counting_5 counting_10 counting_25` (counting tops out lower than the addition tasks because `counting_K` is 2K−1 tokens), point the fingerprint/aggregate stages at the resulting `data/extracted_states_varbind_allpos`, then judge with `--task varbind --varbind-dataset data/chained_var_binding_dataset.json --prompt neutral`. The judge ranks every integer in the residual top tokens; scoring then checks each serial intermediate (x, c₁·x, y, c₂·y, answer) against that ranked list.

**Logit-lens heatmaps (addition tasks).** A direct, supervised view of the same extraction cache: at every (layer, position) decode by RMSNorm + lm\_head + argmax over single-token integers 0–299, then compute the fraction of examples where the prediction matches each known target (`A₁`, `A₂`, `A₁+A₂` for 2-fact; `A`, `X`, `A+X` for 1-fact) exactly or within ±5. Produces the (layer, position) heatmaps used in the paper to localize the staged "look up, look up, sum" computation.

```bash
# Per-(layer, position) decode, written to a JSON keyed by (pos, layer).
# --condition takes multiple values; --output-dir defaults to results/unsupervised_decode_2fact.
python scripts/analysis/decode_2fact_heatmap.py --condition dots_50 \
    --output-dir results/unsupervised_decode_2fact_allpos

# Same for the model-incorrect subset (writes decode_2fact_<cond>_incorrect.json)
python scripts/analysis/decode_2fact_heatmap.py --condition dots_50 --incorrect-only \
    --output-dir results/unsupervised_decode_2fact_allpos

# Paper-ready 3x2 figure (rows: A1 / A2 / A1+A2; cols: correct / wrong)
python plotting/plot_decode_heatmap_correct_vs_wrong.py \
    --correct-json results/unsupervised_decode_2fact_allpos/decode_2fact_dots_50.json \
    --incorrect-json results/unsupervised_decode_2fact_allpos/decode_2fact_dots_50_incorrect.json \
    --metric exact \
    --outfile-stem plotting/plots/decode_2fact_dots_50_correct_vs_wrong_exact
```

`scripts/analysis/decode_1fact_heatmap.py` is the 1-fact sibling (targets `A`, `X`, `A+X`; pass `--extraction-dir`, since its defaults point elsewhere). `scripts/analysis/decode_2fact_baseline_heatmap.py` does the no-filler baseline, which has only two positions (`question_end`, `answer_prompt`) — it is deliberately torch-free (tokenizers backend, lm\_head pre-sliced to the number-token rows) so it runs in a minimal CPU venv.

For the system-of-equations task, `scripts/analysis/decode_varbind_heatmap.py` produces the analogous (layer, position) map, decoding the full serial chain — x → c₁·x → y → c₂·y → answer — to show where each never-written intermediate first appears. Three further views of that map: `varbind_overlay_heatmap.py` collapses the five panels into one, colouring each cell by *which* target decodes most strongly (so the serial chain reads as a vertical colour sweep down the layers); `varbind_paper_figure.py` composes the baseline and filler maps side by side as proportional colour blends; and `varbind_conditional_failure.py` conditions per example rather than in aggregate, separating "the bound value y was wrong" from "the arithmetic on a correct y diverged". The model also has two chain-of-thought conditions that verbalise the chain; `scripts/extract/extract_varbind_cot.py` extracts their hidden states (`--mode teacher_forced | free_gen`), `scripts/analysis/decode_varbind_cot_heatmap.py` decodes them, and `scripts/analysis/varbind_cot_paper_figure.py` renders the filler-vs-chain-of-thought decode maps side by side — testing whether the depth ladder stacked across *layers* at one position restructures into a ladder across *positions* once the chain is verbalised.

### 2. KV-cache transplant (1-fact addition, 2-fact addition, and system of equations)

Splices the donor's filler-region KV cache into the target's KV state at every layer and decodes the next token; reports the rank of the donor's answer in the target's predictions before and after the swap. One run covers one condition — `--filler-k` and `--filler-type` select it, so sweeping conditions means one invocation each.

**1-fact.** Pairs are matched on the literal addend (same X, different A, with |ΔA| ≥ `--min-a-diff`; the code calls this "Y-matching"), so the donor answer is A_donor + X. `--mode` chooses what gets spliced: `filler` (the filler tokens only), `filler_with_label` (the `Filler:` label plus the tokens, stopping before `Answer:`), or `question` (the question tokens — the control).

```bash
python scripts/intervention/filler_kv_transplant.py \
    --model-path /workspace/models/deepseek-v3-awq \
    --dataset data/1hop_addition_dataset.json \
    --filler-k 100 --filler-type dots --mode filler \
    --max-pairs 500 \
    --output-dir results/kv_transplant_v4
```

**2-fact.** `scripts/intervention/twofact_kv_transplant.py` makes the test *position-resolved*: are the filler positions the heatmaps flag as encoding A₁ (vs A₂) the ones that causally carry that addend? Pairs are **fact-matched** — the addend not being swapped is held to the same element across donor and target, so the donor's answer is on-manifold for the target and adoption is directly comparable to the 1-fact numbers. Two directions (`vary_a1` holds A₂, `vary_a2` holds A₁), and three scopes per direction: `whole` (all filler KV rows — the headline causal number), `<a>_pos` (positions whose decode for that addend clears θ, read from the logit-lens heatmaps via `--heatmap-dir`), and `no_<a>_pos` (the complement — the control). Run over θ ∈ {0.15, 0.20, 0.30} for robustness.

```bash
# Optional but recommended: precompute the model-correct index so both-correct
# pair sampling is not mostly rejections.
python scripts/intervention/make_2fact_correct_idx.py dots_10 dots_50

python scripts/intervention/twofact_kv_transplant.py \
    --conditions dots_50 --directions vary_a1,vary_a2 --thetas 0.15,0.20,0.30 \
    --max-pairs 500 \
    --correct-idx-json data/twofact_correct_idx_dots_50.json \
    --output-root results/twofact_kv_transplant

python scripts/analysis/twofact_transplant_analyze.py --root results/twofact_kv_transplant
python plotting/plot_twofact_transplant.py
```

`scripts/intervention/run_twofact_transplant.sh` drives both conditions in one model load. The pure helpers (pair finding, candidate construction, threshold scoping) are CPU-unit-tested via `--self-test`; the MLA cache surgery is shared with the varbind transplant.

**System of equations.** `scripts/intervention/varbind_kv_transplant.py` asks whether the filler positions causally carry the never-written intermediate y: it splices a donor problem's filler-region KV into the target and checks whether the answer shifts *through* y. `scripts/intervention/run_varbind_transplant_sweep.sh` drives the filler-k sweep (with a batched-vs-sequential equivalence gate — `--verify-batched N` before committing to `--batched`), and `scripts/analysis/varbind_transplant_analyze.py` merges the runs, filters to both-correct pairs, and reports rank shifts with bootstrap 95% CIs.

```bash
python scripts/intervention/varbind_kv_transplant.py \
    --filler-k 25 --filler-type dots \
    --max-pairs 300 --self-check 5 \
    --output-dir results/varbind_kv_transplant_dots25
```

Other interventions on the same scaffold:

- `attention_knockout.py` — zero attention *to* the filler positions (at the answer position only, or at every position) and measure the accuracy cost. Unlike deleting the filler, this holds sequence length, positional encoding, and the filler KV entries themselves fixed, so it isolates whether the model must *read* them.
- `filler_patching.py` — cross-example activation patching (rather than KV splicing) within one condition; the `answer_prompt`-only vs contiguous-from-k₁ contrast separates "the final hidden state is sufficient" from "the model reads filler KV".
- `baseline_patching.py` — cross-*condition*: inject filler pre-answer states into no-filler targets, testing whether filler's whole role is building that one hidden state.
- `scripts/patchscope/` — patch a saved activation into an inspection prompt's residual stream and read out the rank and probability of A₁/A₂/sum over a (source layer × source position × template × target layer) grid (`patchscope_single.py`, `run_experiment_grid.py`, `analyze_results.py`).

### 3. Attention analysis (1-fact addition)

Aggregates per-head attention weights into a segment-level breakdown (system / few-shot / question / filler region / answer-label) across 100 examples per condition, plus a no-filler baseline.

```bash
python scripts/extract/extract_filler_attention.py \
    --model-path /workspace/models/deepseek-v3-awq \
    --max-examples 100 \
    --output-dir results/filler_attention_v2

python scripts/extract/extract_answer_attention.py \
    --model-path /workspace/models/deepseek-v3-awq \
    --max-examples 100 \
    --output-dir results/attention/baseline
```

The "filler region" is the filler header tokens plus the filler tokens themselves, treated as one segment. Both scripts default to `data/1hop_addition_dataset.json`; `extract_filler_attention.py` reads the dataset's `few_shot_facts` field, so it is 1-fact-specific as written.

### 4. Accuracy evals

vLLM-based, one script per task. The addition evals sweep filler length in a single run: `evaluate_{1hop,2fact}_vllm.py --filler-type dots|counting|alphabet --filler-k 0,1,5,10,25,50,100,200` (k=0 is the no-filler baseline). `evaluate_letterpos_vllm.py` instead takes explicit `--conditions baseline dots_100 counting_200 …` and serves both letter-position domains via `--dataset`. These produce the accuracy-vs-k curves and the per-example correctness the decode analyses condition on — `save_categories.py` writes per-example category + correctness labels from the extraction pkls, and `make_2fact_correct_idx.py` writes the model-correct index the 2-fact transplant samples from.

`evaluate_truncation.py` is the one variant worth calling out: it truncates *only the target problem's* filler to k tokens while few-shot examples keep their condition's full length and the system message still describes the full length. That separates "the model needs the compute" from "the model is reading the prompt's description of the filler".

## Released artifacts

The pre-computed pipeline outputs that the paper depends on are committed under [`release/`](release/), so reviewers can verify or extend results without re-running ~10⁴ judge API calls or the GPU extraction. 224 JSON files + 6 LaTeX tables, ~143 MB total. Everything here is DeepSeek V3 or Kimi K2.

| dir | files | what |
|---|---|---|
| `release/top_tokens/` | 45 | per-example aggregated top-50 residual tokens (input to the LLM judge), one file per (model × task × condition) |
| `release/judge_outputs/` | 138 | per-example Claude Haiku 4.5 / Sonnet 4.6 judge responses, one file per (model × task × condition × judge × prompt) |
| `release/judge_outputs_shuffled/` | 22 | the shuffled-tokens control at `dots_10`: each example's top-50 tokens swapped for a *different* example's from the same (model, task, condition), then scored against the original example's ground truth. Accuracy collapsing means the judge is reading example-specific signal rather than a task-level prior |
| `release/accuracy_tables/` | 11 | aggregated judge top-K hit counts per (condition, judge), one file per (model, task, prompt) |
| `release/logit_lens_heatmaps/` | 8 | per-(layer, position) decode-match fractions for the 2-fact heatmaps |
| `release/tables/` | 6 | paper-appendix LaTeX tables built from the dirs above: `table_{1fact,2fact}.tex`, `table_letterpos_{neutral,leading}.tex`, `table_residual_ablation.tex`, `table_shuffled_control.tex` |

Filename convention: `<model>_<task>_<condition>[_<judge>][_<prompt>][_incorrect].json`. The schema for each subdirectory and the gotchas (e.g., the `prob` field is a residual sum, not a probability; 2-fact `top_k` values are dicts) are documented in [`release/README.md`](release/README.md).

The four core dirs regenerate deterministically from the working `outputs/` and `results/` trees; the control and the tables have their own scripts (the control makes fresh judge calls, so it is *not* deterministic):

```bash
python scripts/release/stage_release.py                    # top_tokens, judge_outputs, accuracy_tables, logit_lens_heatmaps
python scripts/release/run_shuffled_control.py             # judge_outputs_shuffled/  (hits the Anthropic API)
python scripts/release/generate_appendix_tables.py         # tables/table_{1fact,2fact,letterpos_*}.tex
python scripts/release/generate_shuffled_control_table.py  # tables/table_shuffled_control.tex
```

The appendix tables also carry a rule-based "direct match" column that scores top tokens against ground truth without a judge (numeric equality for the addition tasks, EN/ZH alias substring match for letter-position). Both that and the judge columns use ground truth, so neither is more supervised than the other — they differ only in matching method.

## Repository structure

```
scripts/
  prompt_utils.py    # shared prompt scaffold + answer extraction (imported everywhere)
  data/              # dataset generation + alias tables
                     #   generate_{1hop,2fact,letterpos,varbind}_dataset.py
                     #   build_{element,capital}_aliases.py (Wikidata SPARQL + cleanup)
  extract/           # hidden state and attention extraction (GPU; DeepSeek)
                     #   extract_hidden_states.py  (--dataset-type {1hop,2fact,letterpos,varbind})
                     #   extract_varbind_cot.py    (varbind chain-of-thought conditions)
                     #   extract_filler_attention.py / extract_answer_attention.py
                     #   extract_kv_latents.py
  decode/            # residual fingerprints + aggregation + LLM judge
                     #   extract_residual_fingerprints.py  (--no-residual = the ablation)
                     #   aggregate_residuals_all_settings.py
                     #   llm_decode_batch.py     (--task {2fact,1fact,letterpos,capitalpos,varbind})
    pooled/          #   pooled-softmax read-out (pool_decode_global_dedup.py,
                     #   pool_top_tokens.py, pool_decode_topk.py, llm_decode_pool.py)
    single_setting/  #   earlier one-(layer, position) variants (cross_example_consistency.py,
                     #   per_position_layer_select.py, discover_variables.py)
  intervention/      # causal interventions
                     #   filler_kv_transplant.py     (1-fact)
                     #   twofact_kv_transplant.py    (2-fact, position-resolved + fact-matched)
                     #   varbind_kv_transplant.py    (system of equations)
                     #   make_2fact_correct_idx.py   (model-correct index for pair sampling)
                     #   attention_knockout.py
                     #   filler_patching.py / baseline_patching.py
  patchscope/        # patch activations into an inspection prompt and read out A1/A2/sum
                     #   patchscope_single.py, run_experiment_grid.py, analyze_results.py
  eval/              # vLLM-based accuracy evals
                     #   evaluate_{1hop,2fact,letterpos}_vllm.py
                     #   evaluate_truncation.py, save_categories.py
  analysis/          # post-hoc analyses
                     #   decode_{1fact,2fact}_heatmap.py  (logit-lens decode, addition tasks)
                     #   decode_2fact_baseline_heatmap.py (no-filler control; torch-free)
                     #   decode_varbind_heatmap.py        (system of equations)
                     #   decode_varbind_cot_heatmap.py    (varbind CoT conditions)
                     #   varbind_{overlay_heatmap,paper_figure,conditional_failure}.py
                     #   {twofact,varbind}_transplant_analyze.py (merge + bootstrap CIs)
                     #   varbind_cot_paper_figure.py      (filler vs CoT decode maps)
                     #   multilingual_coverage.py         (substring match for letterpos)
                     #   layer_attribution.py
  release/           # stage_release.py, run_shuffled_control.py, generate_*_table*.py
  kimi_k2/           # Kimi K2-specific extraction (vLLM, TP=4)
  kimi_k25/          # Kimi K2.5 fork (vLLM V1 worker hooks) + RUNBOOK / HANDOFF / SESSION_STATE
  fp8/               # native-FP8 DeepSeek precision control (download + preflight + RUNBOOK)
  dev/, tests/       # archived / experimental (incl. dev/sae/) / unit tests

plotting/            # all figure generation (Matplotlib)
data/                # datasets, held-out few-shot pools, alias tables
results/             # analysis outputs (JSON + plots)
logs/                # script logs
release/             # committed paper artifacts (see "Released artifacts")
finetuning/          # Qwen2.5-72B LoRA fine-tuning (separate experiment)
```

## Setup

Python ≥ 3.10 and a CUDA-capable GPU node are required for any extraction or intervention script. CPU is sufficient for the residual-decode aggregation, the LLM-judge driver, and all plotting.

### 1. Install dependencies

The project uses [uv](https://github.com/astral-sh/uv) for environment management:

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# In the repo root:
uv sync                                    # creates .venv and installs everything in pyproject.toml
uv pip install flash-attn --no-build-isolation   # optional: only if you want Flash Attention 2
```

`uv sync` pins `torch==2.6.0+cu124` from the PyTorch index plus `transformers`, `accelerate`, `bitsandbytes`, `autoawq`, `vllm`, `anthropic`, `matplotlib`, etc. Activate the env with `source .venv/bin/activate` or prefix commands with `uv run`.

This env covers the DeepSeek V3 (AWQ) and Kimi K2 pipelines. The two control checkpoints need their own environments and cannot share this one: Kimi K2.5 requires vLLM ≥ 0.25 with torch 2.11 (the V1 worker-hook path), and the native-FP8 DeepSeek run needs an FP8-capable stack instead of autoawq. See `scripts/kimi_k25/RUNBOOK.md` and `scripts/fp8/RUNBOOK.md`.

### 2. Download model checkpoints

The two headline checkpoints are ~330–510 GB on disk. Use `huggingface-cli` (installed by `uv sync`) and pass the local path to scripts via `--model-path`:

```bash
huggingface-cli download cognitivecomputations/DeepSeek-V3-0324-AWQ \
    --local-dir /path/to/deepseek-v3-awq

huggingface-cli download RedHatAI/Kimi-K2-Instruct-quantized.w4a16 \
    --local-dir /path/to/kimi-k2-w4a16
```

Both are public; no `HF_TOKEN` required. The controls are much larger — native-FP8 DeepSeek V3 is 689 GB / 163 shards (`bash scripts/fp8/download_fp8.sh [dest]`, resumable) and the Kimi K2.5 checkpoint is ~595 GB / 64 shards. Point `HF_HOME` at a large volume before downloading either; a default-sized root disk will fill.

### 3. Environment variables

Only one is strictly required:

| variable | when needed |
|---|---|
| `ANTHROPIC_API_KEY` | LLM-as-judge step (`scripts/decode/llm_decode_batch.py`, `scripts/release/run_shuffled_control.py`). Alternatively pass `--key-path` to a file holding the key — it defaults to `/workspace/keys/anthropic_api_key` |
| `HF_HOME` (optional) | redirect HuggingFace cache (model downloads, tokenizer assets) to a non-default location |
| `HF_TOKEN` (optional) | only needed if you swap in a gated model |

### 4. Hardware

3–4 × NVIDIA H200 for the headline experiments: DeepSeek V3 fits across 3 H200s in 4-bit AWQ; Kimi K2 needs 4 under W4A16 with vLLM TP=4. The unquantized controls need a full 8-GPU node (native-FP8 DeepSeek V3 at 689 GB, Kimi K2.5 at ~595 GB). Cumulative hidden-state extraction cache for the four-task DeepSeek suite is ~680 GB on local SSD (1-fact 183, 2-fact 246, elements 70, capitals 89, system-of-equations 92). Smaller setups can run a subset of conditions; every step after extraction — residual decode, aggregation, the judge driver, all plotting — is CPU-only.
