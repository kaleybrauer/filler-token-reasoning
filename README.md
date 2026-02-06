# filler-token-reasoning

Train and analyze language models whose accuracy improves as we add **non-semantic filler tokens** to the prompt. The filler span is treated as a token-space "workspace" to later probe and decode.

## Models

We use Qwen2.5-7B for pilot, Qwen2.5-72B for scale.

## Data

We use Ryan Greenblatt's `compose_facts` generator to create multi-hop data locally.

## Quickstart
- Download facts: `python scripts/fetch_compose_facts_sources.py --outdir data/sources`
- Create data locally (not in repo): `python generate_2hop_dataset.py --sources ./data/sources --outdir ./data/datasts/2hop`
- Baseline eval: `python evaluate.py --model Qwen/Qwen2.5-7B --data-dir ./data/datasets/2hop --filler-lengths 0,32,128,300,600,1000 --outdir ./results/baseline`
- LoRA train: `python scripts/train_lora.py --model Qwen/Qwen2.5-7B --data-dir ./data/datasets/2hop --outdir  ./outputs/qwen2p5-7b-qlora --batch-size 4`
- Fine-tuned eval: `python evaluate.py --model Qwen/Qwen2.5-7B --adapter ./checkpoints/lora_r16 --data-dir ./data/datasets/2hop --filler-lengths 0,32,128,300,600,1000 --outdir ./results/finetuned --wandb`