# filler-token-reasoning

Train and analyze language models whose accuracy improves as we add **non-semantic filler tokens** to the prompt. The filler span is treated as a token-space "workspace" to later probe and decode.

## Models

We use Qwen2.5-7B for pilot, Qwen2.5-72B for scale.

## Data

We use Ryan Greenblatt's `compose_facts` generator to create multi-hop data locally.

## Quickstart
- Create data locally (not in repo): `python scripts/gen_2hop.py --out /path/to/data`
- Preprocess: `python scripts/preprocess.py --data /path/to/data --out /path/to/processed`
- Eval baseline acc vs N: `python scripts/eval_baseline.py --model Qwen/Qwen2.5-7B --data /path/to/processed`
