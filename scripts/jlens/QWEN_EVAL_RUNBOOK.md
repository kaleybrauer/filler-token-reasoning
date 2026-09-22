# Qwen3.5-122B: the paper's eval sets, J-lens against logit lens

**Status:** code written and CPU-tested 2026-09-19 (the scorer reproduces the V3 numbers on the V3 states; the
extraction step is new and gets a 3-item smoke run on the box before the full run). **The model is already on the
volume** (`/workspace/models/Qwen3.5-122B-A10B`, downloaded 2026-09-19 03:08 UTC: 39 shards, 233 GiB, index and
tokenizer present, verified against the index). **GPU: 2× H200 or more, forward only, about 1 h.** Skip §1.3.

## 0. Why

The post says Qwen3.5's J-lens readouts are far sharper than its logit lens's (entropy) and contain translation
pairs the logit lens's don't, but has no head-to-head on the paper's eval sets for Qwen: those states exist only
for V3. This run extracts them for the 122B (the model where the logit lens is most uniform) and scores both
lenses with the wrong-answer control. One number per set, either direction is useful.

## 1. Before launching

1. **Box:** 2× H200 minimum (233 GiB of bf16 weights; `device_map="auto"` spreads them). 3–4 is what the
   September-15 extraction used and gives headroom. No GPU is needed until the download is done.
2. **Disk:** the model needs ~235 GiB. Put it on the box's container disk if the volume can't hold it; the
   volume only needs the output (`outputs/jlens_qwen35/qwen35_122b/eval_states.pt`, ~140 MB). The Kimi K2 weights
   were removed from `/workspace/models` on 2026-09-19 to make room; check the portal figure, not `df`.
3. **Download** (already done; only if the directory is missing or incomplete; HF token is in `probing_env.sh`, do not echo it):
   ```bash
   source /workspace/config/probing_env.sh
   /workspace/venv/bin/python -c "from huggingface_hub import snapshot_download; \
     snapshot_download('Qwen/Qwen3.5-122B-A10B', local_dir='/workspace/models/Qwen3.5-122B-A10B')"
   ```
   Expect 1–3 h depending on bandwidth. Set `MODEL=` to wherever it lands if not there.
4. **Environment:** the same venv the September-15 Qwen extraction used (`/workspace/venv`, transformers 5.16,
   torch 2.14); `bash logs/bootstrap_jlens.sh` on a fresh box. `import jlens` must resolve to the library, not
   `scripts/jlens` (the script checks and exits if not).
5. **Nothing else on the GPUs:** `nvidia-smi` empty.

## 2. Smoke run (3 items per set, ~15 min including the load)

```bash
MAX_ITEMS=3 setsid nohup bash logs/qwen_eval.sh >/dev/null 2>&1 </dev/null &
tail -f logs/qwen_eval.log
```
**The unembed step only measures when `outputs/jlens_qwen35/qwen35_122b/unembed/check.json` is absent**; if a
stored check exists (there is one from 2026-09-15) it is reused silently and no line is printed. To re-measure on
this box, move it aside first: `mv outputs/jlens_qwen35/qwen35_122b/unembed outputs/jlens_qwen35/qwen35_122b/unembed_20260915`.
Expected in the log, in order: `loaded ... on N GPUs`; a `G-UNEMBED: {...}` line containing `'passed': True` (it
re-derives the norm convention, `1+w` for Qwen3.5, and compares the offline readout with the model's own logits:
top-1 agreement 1.0, correlation > 0.999; the script exits before extracting if it fails); five
`evals <set>: (3, 48, 3072) seq_len a-b` lines; `evals: 0 cells clamped`; `DONE`; `QWEN_EVAL_EXIT=0`.
Then **delete the smoke output** so the full run does not skip it:
```bash
rm outputs/jlens_qwen35/qwen35_122b/eval_states.pt
```

## 3. Full run (~30–40 min after the load)

```bash
setsid nohup bash logs/qwen_eval.sh >/dev/null 2>&1 </dev/null &
tail -f logs/qwen_eval.log
```
453 items over five sets, each one forward pass with all 48 layers recorded at the last prompt token. Expect
`evals multihop: (93, 48, 3072)`, `order-ops: (55, ...)`, `association: (102, ...)`, `multilingual: (107, ...)`,
`typo: (96, ...)`, then `DONE` and `QWEN_EVAL_EXIT=0`. Output: `outputs/jlens_qwen35/qwen35_122b/eval_states.pt`
(~140 MB) and an updated `run.json`. Release the box once the file is there; everything else is CPU.

## 4. What not to do

- Do not run any other step (`paragraphs`, `concepts`, `regime`): those states exist and the step skips
  existing files, but `regime` reloads and costs time.
- Do not delete `/workspace/models/deepseek-v3-awq`.
- Do not edit `scripts/jlens/extract_qwen35_states.py` on the box; if the step fails, copy the traceback verbatim.

## 5. CPU steps afterwards (sandbox, ~20 min)

```bash
PY=/root/.venvs/jlens-cpu/bin/python
$PY scripts/jlens/score_evals_model.py --model qwen35_122b          # J-lens vs logit lens, layers 0-46, + control
```
Writes `outputs/jlens_qwen35/qwen35_122b/analysis/evals_jlens_vs_logit.json`. The scorer skips intermediates with
no single-token variant on Qwen's tokenizer (every multi-digit number: 9 of multihop's 103 and 22 of order-ops'
110; it prints `scored a/b`), and both arms use the 47 layers the released lens holds. The wrong-answer columns
say whether a win is block credit. For V3 on the same code: `--model v3 --layers 0:60`.

## 6. Files

| Path | What |
|---|---|
| `scripts/jlens/extract_qwen35_states.py` | `--steps evals` (new): the five eval sets at the last prompt token, `eval_states.pt` in the V3 layout; `--max-items` for the smoke run |
| `logs/qwen_eval.sh` | launcher (`MODEL=`, `MAX_ITEMS=`) |
| `scripts/jlens/score_evals_model.py` | model-agnostic scorer: J-lens vs logit lens on the lens's layers, wrong-answer control, single-token rule with skipping |
| `/workspace/models/qwen35-122b-tokenizer/` | tokenizer files only (downloaded 2026-09-19 for the CPU checks) |
