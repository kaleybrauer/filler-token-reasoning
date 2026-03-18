#!/usr/bin/env bash
# Setup script for DeepSeek V3 AWQ evaluation with vLLM.
# Detects CUDA driver version and GPU count, installs matching vLLM + torch.
#
# Usage:
#   source /workspace/config/probing_env.sh
#   bash probing/scripts/setup_vllm.sh

set -euo pipefail

VENV="/root/.venvs/filler-probing"
MODEL_DIR="/workspace/models/deepseek-v3-awq"
MODEL_REPO="cognitivecomputations/DeepSeek-V3-0324-AWQ"
UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/.cache/uv}"
export UV_CACHE_DIR

# --- Detect hardware ---
DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
DRIVER_MAJOR=$(echo "$DRIVER_VERSION" | cut -d. -f1)
CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
NUM_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)

echo "GPUs:    ${NUM_GPUS}x ${GPU_NAME}"
echo "Driver:  ${DRIVER_VERSION} (CUDA ${CUDA_VERSION})"

# --- Determine vLLM version ---
# Driver 570+ → CUDA 12.8 → vLLM 0.8.5 works, 0.17.1 also works (but needs QKV patch)
# Driver 550+ → CUDA 12.4 → vLLM 0.8.5 + torch 2.6.0+cu124
# vLLM 0.8.5 is preferred: no QKV fusion patch needed, stable PP support via V0 engine.
VLLM_VERSION="0.8.5"

if [ "$DRIVER_MAJOR" -ge 550 ]; then
    TORCH_VERSION="2.6.0"
    TORCH_INDEX="https://download.pytorch.org/whl/cu124"
    echo "Plan:    vLLM ${VLLM_VERSION} + torch ${TORCH_VERSION} (cu124)"
else
    echo "ERROR: Driver ${DRIVER_VERSION} is too old (need >= 550). Upgrade your RunPod instance."
    exit 1
fi

# --- Determine parallelism ---
# 128 attention heads → TP must divide 128.
# PP is used when TP alone can't cover all GPUs or when model doesn't fit per-GPU.
if [ "$NUM_GPUS" -ge 4 ]; then
    TP=4
    PP=1
elif [ "$NUM_GPUS" -eq 2 ]; then
    TP=2
    PP=1
else
    TP=1
    PP="$NUM_GPUS"
fi
echo "Parallel: TP=${TP}, PP=${PP}"

# --- Ensure venv exists ---
if [ ! -f "$VENV/bin/python" ]; then
    echo "Creating venv..."
    uv sync --project /workspace/filler-token-reasoning/probing
fi

# --- Install vLLM + torch ---
echo ""
echo "Installing vLLM ${VLLM_VERSION} + torch ${TORCH_VERSION}+cu124..."

uv pip install --python "$VENV/bin/python" \
    --index-strategy unsafe-best-match \
    --extra-index-url "$TORCH_INDEX" \
    "vllm==${VLLM_VERSION}" "torch==${TORCH_VERSION}+cu124"

# Verify
TORCH_CUDA=$("$VENV/bin/python" -c "import torch; print(torch.version.cuda)")
echo "Installed torch CUDA: ${TORCH_CUDA}"

# --- Download model if needed ---
echo ""
if [ -d "$MODEL_DIR" ] && [ "$(ls "$MODEL_DIR"/*.safetensors 2>/dev/null | wc -l)" -ge 36 ]; then
    echo "Model already downloaded at $MODEL_DIR"
else
    echo "Downloading AWQ model (~350GB, ~10 min)..."
    "$VENV/bin/python" -m huggingface_hub.commands.huggingface_cli download \
        "$MODEL_REPO" --local-dir "$MODEL_DIR"
fi

# --- Summary ---
echo ""
echo "============================================"
echo "Setup complete!"
echo "============================================"
echo "  vLLM:    ${VLLM_VERSION}"
echo "  torch:   ${TORCH_VERSION}+cu124"
echo "  GPUs:    ${NUM_GPUS}x ${GPU_NAME}"
echo "  TP=${TP}, PP=${PP}"
echo "  Model:   ${MODEL_DIR}"
echo ""
echo "Run evaluation:"
echo "  cd /workspace/filler-token-reasoning"
echo "  $VENV/bin/python probing/scripts/evaluate_1hop_vllm.py \\"
echo "    --model-path $MODEL_DIR \\"
echo "    --tensor-parallel-size $TP \\"
echo "    --pipeline-parallel-size $PP \\"
echo "    --max-examples 200 --filler-k 250"
