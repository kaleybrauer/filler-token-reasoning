# Paper artifact release

~200 JSON files, ~125 MB. Re-staged from `outputs/` and `results/` by `../scripts/release/stage_release.py` — see the parent `../README.md` for the upstream pipeline that produces these.

## Directory layout

| dir | what it contains |
|---|---|
| [`top_tokens/`](#top_tokens) | Per-example aggregated top-50 residual tokens (input to the LLM judge) |
| [`judge_outputs/`](#judge_outputs) | Per-example Claude Haiku 4.5 / Sonnet 4.6 judge responses |
| [`accuracy_tables/`](#accuracy_tables) | Aggregated judge accuracy by (condition × judge × top-K) |
| [`logit_lens_heatmaps/`](#logit_lens_heatmaps) | Per-(layer, position) decode-match fractions for the 2-fact heatmaps |

## Filename convention

`<model>_<task>_<condition>[_<judge>][_<prompt>][_incorrect].json`:

- **model**: `deepseek_v3` or `kimi_k2`
- **task**: `1fact`, `2fact`, `letterpos`, or `capitalpos`
- **condition**: `dots_<k>`, `counting_<k>`. 1-fact additionally includes exploratory `alphabet_10/25`, `counting_5`, `dots_100`.
- **judge**: `haiku` (Claude Haiku 4.5) or `sonnet` (Claude Sonnet 4.6)
- **prompt**: judge framing — `neutral` (task-agnostic), `chemistry` (letterpos), `geography` (capitalpos)
- **`_incorrect`**: only on logit-lens heatmaps — the model-incorrect subset (otherwise the model-correct subset is implied)

---

## `top_tokens/`

45 files, ~75 MB. Per-example top-50 residual tokens after aggregating across all (layer, position) settings.

```json
{
  "idx": <int>,
  "top_tokens": [
    {"id": <int>, "str": <str>, "prob": <float>, "n_settings": <int>},
    ...   // length 50, ranked descending by `prob`
  ],
  // ground-truth fields (subset depending on task):
  "fact_value_1": <int>, "fact_value_2": <int>, "answer": <int>,    // 2-fact
  "fact_value": <int>, "answer": <int>,                              // 1-fact
  "element": <str>, "atomic_number": <int>,                          // letterpos
  "intermediate": <str>                                              // capitalpos
}
```

⚠ **`prob` is a misnomer** — it is the **summed residual score** `Σ_s [P_e(token | s) − mean_e P_e(token | s)]` across (layer, position) settings `s` where the token was in the per-setting top-K, **not** a probability. `n_settings` is the count of settings that placed the token in their top-K.

Pipeline: `scripts/decode/extract_residual_fingerprints.py` → `scripts/decode/aggregate_residuals_all_settings.py`.

## `judge_outputs/`

138 files, ~43 MB. Each file contains the judge responses for one (model, task, condition, judge, prompt) combination.

```json
{
  "idx": <int>,
  "pred": <str|int|[int,int]|null>,    // primary guess; null = call/parse failure
  "backups": [<str|int>, ...],          // ranked alternatives
  "got": <bool>,                        // top-1 match against ground truth
  "top_k": {                            // top-K hit at K=1,2,3,5,10
    "1": <bool>, "2": <bool>, "3": <bool>, "5": <bool>, "10": <bool>
  },
  "raw": <str|null>,                    // full judge text response
  // ground-truth fields (mirrored from top_tokens)
}
```

⚠ **2-fact has a different `top_k` shape.** Each value is a dict `{a1: bool, a2: bool, both: bool}` rather than a bare bool, because we score whether each addend is recovered separately. Records also carry `got_a1` / `got_a2` / `both` (each top-1).

`raw=null` records are judge call/parse failures (1 across the entire release: idx=272 in `deepseek_v3_letterpos_counting_50_sonnet_neutral.json`, marked with `judge_refusal=true` and a `note` field).

Pipeline: `scripts/decode/llm_decode_batch.py` (prompt definitions are in that file).

## `accuracy_tables/`

11 files, <1 MB. Aggregated top-K hit counts per (condition, judge), one file per (model, task, prompt).

```json
{
  "<cond>_<judge>": {"n": <int>, "summary": {"1": <int>, "2": <int>, "3": <int>, "5": <int>, "10": <int>}},
  ...
}
```

`summary["K"]` is the top-K hit count out of `n`. These are deterministic re-aggregations of the corresponding `judge_outputs/` files; the headline accuracy numbers in the paper come from these.

## `logit_lens_heatmaps/`

8 files, ~5 MB. Per-(layer, position) decode-match fractions for the DeepSeek V3 2-fact heatmaps in the paper.

```json
{
  "_condition": <str>,
  "_n": <int>,
  "_layers": [<int>, ...],
  "_positions": [<str>, ...],
  "_boundaries": {"question_end_offset": ..., "filler_start_offset": ...,
                  "filler_end_offset": ..., "answer_prompt_offset": ...},
  "<pos>": {
    "<layer>": {
      "frac_A1_exact": <float>, "frac_A1_within5": <float>,
      "frac_A2_exact": <float>, "frac_A2_within5": <float>,
      "frac_A1A2_exact": <float>, "frac_A1A2_within5": <float>,
      "frac_A1_or_A2_within5": <float>
    },
    ...
  },
  ...
}
```

Pipeline: `scripts/analysis/decode_2fact_heatmap.py`.

---

## Reproducing

To regenerate this directory from the working `outputs/` and `results/` trees:

```bash
python scripts/release/stage_release.py
```
