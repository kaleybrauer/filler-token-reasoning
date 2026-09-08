"""
eval_lens_quality.py — the paper's own lens-quality evals, on our fitted V3 lens.

Runs the evaluation sets shipped with the jacobian-lens repo
(data/evaluations/lens-eval-*.json, "§methods-comparison") with their published
metric, comparing two arms on identical states:

    J-lens     lens_L(h) = unembed( J_L h )
    logit lens lens_L(h) = unembed( h )        (the J_L = I special case,
                                                jlens' own use_jacobian=False)

Metric (from data/evaluations/README.md): readout at a single position — the
token immediately preceding `target`, i.e. the last prompt token — across
layers; pass@k = mean over items of the fraction of `intermediates` whose
min-over-layers lens rank <= k.

Two sets matter most for us:
  lens-eval-multihop  93 items, hidden bridge entity (praxagent's 397B headline
                      was this shape: J-lens rank 43 vs 620 for identity)
  lens-eval-order-ops 55 items, hidden arithmetic intermediate — the closest
                      published analogue of our 2-fact addition task

Scoring note: `intermediates` are words, not token ids. An intermediate counts
at the best (lowest) rank over its surface variants {w, " "+w, lowercased},
each taken at its FIRST token. Reported alongside the numbers, since the paper
does not pin the variant convention.

Only the lens's fitted layers are scored, and BOTH arms are restricted to the
same layer set — a band-only fit therefore reports "min over the fitted band",
not "min over all layers".

Usage:
    python scripts/jlens/eval_lens_quality.py --lens outputs/jlens/lens_v3.pt \
        --sets multihop,order-ops
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

# NOTE: `jlens` (the library) must be imported BEFORE REPO_ROOT/scripts joins
# sys.path. This file lives in scripts/jlens/, and Python treats that directory
# as a namespace package named `jlens`, which would shadow the library.
import jlens  # noqa: E402
from jlens.lens import JacobianLens  # noqa: E402
if "filler-token-reasoning" in (jlens.__file__ or "scripts/jlens namespace pkg"):
    raise SystemExit(f"scripts/jlens shadowed the jlens library: {jlens.__file__}")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from v3_autograd import load_v3_awq  # noqa: E402

EVAL_DIR = Path("/workspace/jacobian-lens/data/evaluations")


def variant_token_ids(tokenizer, word: str) -> list[int]:
    """Token id of each SINGLE-TOKEN surface variant of `word`.

    `intermediates` are words, not ids; the paper picks them to be single-token
    words and credits an intermediate at its best rank over surface variants
    {w, " "+w, lower(w), " "+lower(w)}. Only variants that tokenize to ONE token
    are credited. Taking the first token of a multi-token variant is wrong on
    DeepSeek's tokenizer: " 5" -> [" ", "5"], so the bare space would be credited
    for every numeric intermediate, and "squared" -> ["s", "quared"] would credit
    "s". That inflated pass@1 for any lens whose readout is whitespace (found
    2026-09-08: a giant-dominated lens scored 54 numeric rank-1 "hits" at layer 3
    through the space token). Words with no single-token variant are skipped and
    counted by the caller. Keep in sync with the twin in the other script.
    """
    ids = []
    for surface in {word, " " + word, word.lower(), " " + word.lower()}:
        enc = tokenizer.encode(surface, add_special_tokens=False)
        if len(enc) == 1:
            ids.append(enc[0])
    return sorted(set(ids))


def min_rank_over_layers(lens_logits: dict[int, torch.Tensor], token_ids: list[int]) -> int:
    """Best (lowest) full-vocab rank of any variant at any scored layer."""
    best = None
    for logits in lens_logits.values():
        row = logits[0]  # single readout position
        for tid in token_ids:
            rank = int((row > row[tid]).sum().item()) + 1
            best = rank if best is None else min(best, rank)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", type=Path, required=True)
    ap.add_argument("--model-path", default="/workspace/models/deepseek-v3-awq")
    ap.add_argument("--sets", default="multihop,order-ops")
    ap.add_argument("--eval-dir", type=Path, default=EVAL_DIR)
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--ks", default="1,5,10,50,100")
    ap.add_argument("--max-memory-gib", default="130,122,108")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "outputs/jlens/lens_eval.json")
    args = ap.parse_args()

    ks = [int(k) for k in args.ks.split(",")]
    lens = JacobianLens.load(str(args.lens))
    print(f"{lens}  <- {args.lens}")

    model, tokenizer = load_v3_awq(args.model_path, max_memory_gib=args.max_memory_gib)
    lens_model = jlens.from_hf(model, tokenizer)  # forward-only: no patches needed
    layers = lens.source_layers

    report = {"lens": str(args.lens), "layers_scored": [layers[0], layers[-1]],
              "n_layers_scored": len(layers), "sets": {}}

    for slug in args.sets.split(","):
        path = args.eval_dir / f"lens-eval-{slug}.json"
        items = json.loads(path.read_text())["items"][: args.max_items]
        print(f"\n=== {slug}: {len(items)} items ===")

        ranks = {"jlens": [], "logit": []}
        per_item = []
        t0 = time.perf_counter()
        for i, item in enumerate(items):
            row = {"name": item["name"], "intermediates": {}}
            arms = {}
            for arm, use_j in (("jlens", True), ("logit", False)):
                lens_logits, _, _ = lens.apply(
                    lens_model, item["prompt"], layers=layers,
                    positions=[-1], max_seq_len=512, use_jacobian=use_j,
                )
                arms[arm] = lens_logits
            for word in item["intermediates"]:
                tids = variant_token_ids(tokenizer, word)
                row["intermediates"][word] = {}
                for arm in arms:
                    r = min_rank_over_layers(arms[arm], tids)
                    ranks[arm].append(r)
                    row["intermediates"][word][arm] = r
            per_item.append(row)
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(items)}  {(time.perf_counter() - t0) / (i + 1):.1f}s/item")

        summary = {}
        for arm, rs in ranks.items():
            rs_t = torch.tensor(rs, dtype=torch.float64)
            summary[arm] = {
                "n": len(rs),
                "median_rank": float(rs_t.median()),
                "mean_log10_rank": round(float(torch.log10(rs_t).mean()), 3),
                **{f"pass@{k}": round(float((rs_t <= k).double().mean()), 4) for k in ks},
            }
            print(f"  {arm}: {summary[arm]}")
        report["sets"][slug] = {"n_items": len(items), "summary": summary,
                                "per_item": per_item}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
