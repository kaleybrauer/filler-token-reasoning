# Decoding Hidden Computation in Filler Tokens

This project investigates whether filler tokens (e.g., `. . . . .`) inserted between a question and answer give models additional serial computation depth and what the models compute during that computation.

## Task Setup

1-hop addition: "What is [fact phrase] plus [X]?" where [fact phrase] maps to a known integer A (e.g., "the atomic number of silicon" = 14). The model must retrieve A and compute A+X.

Filler tokens are inserted between the question and the answer position:
```
Question: What is the atomic number of silicon plus 62?

Filler: . . . . . . . . . .

Answer:
```
Few-shot examples (5) show the same format. The model outputs just the number. Filler types: dots (`. . .`), counting (`1 2 3 ...`), alphabet (`a b c ...`).

## Repository Structure

```
scripts/                     # Core analysis scripts
  generate_1hop_dataset.py   # Dataset generation
  prompt_utils.py            # Shared prompt construction
  extract_hidden_states.py   # Extract residual stream states (GPU)
  extract_filler_attention.py # Extract filler attention patterns (GPU)
  extract_answer_attention.py # Extract answer-position attention (GPU)
  evaluate_1hop_vllm.py      # Evaluate accuracy with vLLM (GPU)
  evaluate_truncation.py     # Filler truncation curves (GPU)
  attention_knockout.py      # Block filler attention via mask (GPU)
  filler_kv_transplant.py    # KV cache transplant between examples (GPU)
  filler_patching.py         # Activation patching (GPU)
  baseline_patching.py       # Cross-condition patching (GPU)
  unsupervised_decode_filler.py   # Unsupervised A decode pipeline (CPU)
  unsupervised_decode_per_position.py  # Per-position decode (CPU)
  layer_attribution.py       # Per-layer logit contribution (CPU)
  save_categories.py         # Save per-example categories
  scripts/dev/               # Archived/experimental scripts
  scripts/tests/             # Unit tests

plotting/                    # All visualization scripts
data/                        # Dataset and extracted states
  1hop_addition_dataset.json # Not committed to repo, needs to be generated
  extracted_states/          # Pre-extracted hidden states (pkl)
  extracted_kv_latents/      # Pre-extracted KV latents (pkl)
results/                     # Analysis outputs (JSON + plots)
logs/                        # Script logs
finetuning/                  # Qwen2.5-72B LoRA fine-tuning
```

## Unsupervised Decode Pipeline

The core interpretability result: decode intermediate values from filler hidden states without any labels.

```bash
python scripts/unsupervised_decode_filler.py \
    --condition dots_10 dots_25 dots_100 counting_5 counting_25 counting_50
```

Pipeline:
1. Average hidden states across filler token positions
2. At each layer, apply RMSNorm and the unembedding matrix (logit lens)
3. Argmax over number tokens (0-999)
4. Select the best layer via cross-condition consistency (do different filler types decode to the same value for the same example?)

## Setup

Requires Python 3.10+ and CUDA for GPU scripts. CPU scripts need only numpy/scipy.

```bash
uv sync
source /workspace/config/probing_env.sh
```

Model: DeepSeek V3 (671B MoE, AWQ quantized) at `/workspace/models/deepseek-v3-awq`.

