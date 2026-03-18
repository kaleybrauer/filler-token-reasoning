#!/bin/bash
# Download pre-quantized DeepSeek V3 GPTQ model
# Run: source /workspace/config/probing_env.sh && bash probing/scripts/download_gptq_model.sh

set -e

MODEL_REPO="ISTA-DASLab/DeepSeek-V3-0324-GPTQ-4b-128g-experts"
OUTPUT_DIR="/workspace/models/deepseek-v3-gptq"

echo "Downloading ${MODEL_REPO} to ${OUTPUT_DIR}..."
echo "This is ~346GB and may take 30-60 minutes."

# Use huggingface-cli for resumable downloads
huggingface-cli download "${MODEL_REPO}" \
    --local-dir "${OUTPUT_DIR}" \
    --local-dir-use-symlinks False

echo "Download complete. Model at ${OUTPUT_DIR}"
echo ""
echo "To run evaluation:"
echo "  source /workspace/config/probing_env.sh"
echo "  uv run --project /workspace/filler-token-reasoning/probing python scripts/evaluate_1hop.py --max-examples 200 --filler-k 250"
