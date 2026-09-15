# Qwen3.5 extraction runbook (standalone GPU box, results copied back)

For the DeepSeek-V3 vs Qwen3.5 comparison (`QWEN35_PLAN.md`). Forward passes only: residual states of
Qwen3.5-122B-A10B (bf16) and Qwen3.5-397B-A17B-FP8 on the same held-out text and concept prompts used for V3,
the unembedding for the offline readout, and an FP8-vs-bf16 regime check. No lens fitting, no gradients.
The box does NOT share /workspace: clone the repo from GitHub, work on local disk, send `out/` back.

## Ground rules
- Don't print API keys or tokens. Don't commit or push from this box.
- Don't delete anything you didn't create here. Kill processes only by explicit PID.
- Run the two models one after the other: together they exceed 4x H200.

## 0. Hardware (2 min)
```bash
nvidia-smi --query-gpu=name,memory.total --format=csv
df -h                      # pick the largest LOCAL disk and set WORK to it below
```
Needs >= ~560 GB total GPU memory (4x H200 known-good; the 397B FP8 weights are 378 GiB) and >= 900 GB free
local disk (611 GiB of weights, ~35 GB of outputs, the environment). FP8 needs Hopper/Ada/Blackwell GPUs.

## 1. Code and environment (10-15 min)
```bash
export WORK=/path/to/local/disk            # e.g. /nvme or /root/work
git clone https://github.com/kaleybrauer/filler-token-reasoning.git $WORK/filler-token-reasoning
cd $WORK/filler-token-reasoning && git checkout c2df06b   # the commit with the extraction script, inputs and this runbook
command -v uv || { curl -LsSf https://astral.sh/uv/install.sh | sh; source $HOME/.local/bin/env; }
uv venv $WORK/venv --python 3.11 && source $WORK/venv/bin/activate
uv pip install torch==2.14.0
uv pip install transformers==5.16.1 accelerate safetensors numpy "huggingface_hub[hf_transfer]"
uv pip install --no-deps "git+https://github.com/anthropics/jacobian-lens@581d398613e5602a5af361e1c34d3a92ea82ba8e"
uv pip install kernels || echo "optional hub kernels unavailable: the torch fallback for the linear-attention layers is used"
python -c "import torch, transformers, jlens; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count(), transformers.__version__, jlens.__file__)"
```
Expect `2.14.0 True 4 5.16.1 .../site-packages/jlens/__init__.py`. transformers must be 5.16.1 (the version the
inputs were built and dry-run with). jlens must import from site-packages, NOT from the repo's `scripts/jlens`.

## 2. Weights to local disk (20-60 min, can run while you do step 1)
```bash
export HF_HOME=$WORK/hf HF_HUB_ENABLE_HF_TRANSFER=1
hf download Qwen/Qwen3.5-122B-A10B --local-dir $WORK/models/Qwen3.5-122B-A10B
hf download Qwen/Qwen3.5-397B-A17B-FP8 --local-dir $WORK/models/Qwen3.5-397B-A17B-FP8
python -c "from huggingface_hub import HfApi; a=HfApi(); [print(m, a.model_info(m).sha) for m in ('Qwen/Qwen3.5-122B-A10B','Qwen/Qwen3.5-397B-A17B-FP8')]"
```
Both repos are public. Record the two revision hashes for the report.

## 3. Qwen3.5-122B-A10B, bf16 (~1-2 h)
```bash
cd $WORK/filler-token-reasoning && mkdir -p logs
nohup python scripts/jlens/extract_qwen35_states.py --model $WORK/models/Qwen3.5-122B-A10B \
    --tag qwen35_122b --outdir $WORK/out > logs/qwen35_122b.log 2>&1 &
tail -f logs/qwen35_122b.log
```
Expected, in order:
- `tokenizer pre-checks passed (concept prompts and paragraph lengths)` within a minute. If not, stop and report.
- `loaded ... on 4 GPUs`, then `HFLensModel(Qwen3_5MoeForConditionalGeneration, n_layers=48, d_model=3072)  bos=None`
- `G-UNEMBED: {... 'norm_convention': '1+w', ..., 'passed': True ...}`. If it fails the run stops: report the line.
- `regime check skipped: no FP8 modules in this model` (expected for the bf16 model)
- paragraph progress every 10 prompts for `wikitext`, `wiki_zh`, `wiki_en`, each ending in `wrote .../<name>_states.pt`
  and a line with the number of fp16-clamped cells (report it)
- `concepts: 4929 distinct token sequences (300 duplicates mapped)`, progress every 200, `wrote .../concept_states.pt`
- `DONE`

## 4. Qwen3.5-397B-A17B-FP8 (~2-3 h)
Only after step 3 prints `DONE` (check with `ps` that its python process is gone).
```bash
nohup python scripts/jlens/extract_qwen35_states.py --model $WORK/models/Qwen3.5-397B-A17B-FP8 \
    --tag qwen35_397b_fp8 --outdir $WORK/out > logs/qwen35_397b.log 2>&1 &
tail -f logs/qwen35_397b.log
```
Expected as in step 3 with `n_layers=60, d_model=4096`, except the regime check runs before the paragraphs:
- `regime arm A (FP8 kernels, experts 'grouped_mm'): N s/prompt`
- `regime arm B (bf16 matmuls on dequantized FP8 weights, eager experts): N s/prompt` (slower: expected)
- `regime: final top-1 agreement X; mean cos by layer ...` then `wrote .../regime.pt`
- If you see `REGIME CHECK FAILED (...)`, the extraction continues by design: report the message.
If loading reports `weights spilled to cpu`, relaunch with `--max-memory-gib 135` and report.
If a run dies midway, relaunch the same command: finished outputs are skipped.

## 5. Verify (2 min)
```bash
python - <<'PY'
import json, os, torch
root = os.path.expandvars("$WORK/out")
for tag, (L, d) in {"qwen35_122b": (48, 3072), "qwen35_397b_fp8": (60, 4096)}.items():
    chk = json.load(open(f"{root}/{tag}/unembed/check.json"))
    print(tag, "G-UNEMBED", chk["passed"], chk["norm_convention"], chk["top1_agreement"], chk["logit_corr"])
    for name in ("wikitext", "wiki_zh", "wiki_en"):
        S = torch.load(f"{root}/{tag}/{name}_states.pt", map_location="cpu", mmap=True, weights_only=False)
        print("  ", name, tuple(S["H"].shape), "expect", (100, 112, L, d), "clamped", len(S["fp16_clamped"]),
              "finite", bool(torch.isfinite(S["H"][:, :, L // 2].float()).all()))
    C = torch.load(f"{root}/{tag}/concept_states.pt", map_location="cpu", mmap=True, weights_only=False)
    print("   concepts", tuple(C["H"].shape), "expect", (5229, L, d), "dups", sum(j != i for i, j in enumerate(C["dup_of"])), "clamped", len(C["fp16_clamped"]))
R = f"{root}/qwen35_397b_fp8/regime.pt"
if os.path.exists(R):
    st = torch.load(R, map_location="cpu", weights_only=False)["stats"]
    print("regime: final top-1 agreement", st["final_top1_agreement"], "| mean cos min over layers", min(st["cos_mean"]))
PY
du -sh $WORK/out
```

## 6. Send the results back (~35 GB)
```bash
cd $WORK && tar cf qwen35_out.tar out && runpodctl send qwen35_out.tar
```
Report the one-time code `runpodctl send` prints. The CPU session receives it with
`runpodctl receive <code>` into `/workspace/filler-token-reasoning/outputs/jlens_qwen35/`.
Fallback if `runpodctl` is unavailable: `rclone` or `rsync` to wherever Kaley chooses.

## 7. Report back
GPU model and count; the versions line from step 1; the two model revision hashes; load time and
s/prompt for each model; both G-UNEMBED lines; the regime lines (both arms' s/prompt, final top-1 agreement,
mean cos by layer) or the regime error; fp16-clamped counts; the verification output; `du -sh` of `out/`;
the `runpodctl` code. Then the instance can be released.
