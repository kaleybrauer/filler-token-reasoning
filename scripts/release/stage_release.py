"""
Stage paper artifacts into release/ from outputs/ and results/.

Run from repo root:
    python scripts/release/stage_release.py

Produces a self-contained release/ directory with four flat subdirs:

  release/
    top_tokens/<model>_<task>_<cond>.json
        ← outputs/<model_task_dir>/aggregated_<cond>.json

    judge_outputs/<model>_<task>_<cond>_<judge>_<prompt>.json
        ← outputs/<model_task_dir>/llm_decode_<cond>_<judge>_<prompt>[_<task>].json

    accuracy_tables/llm_judge_<model>_<task>_<prompt>.json
        (merged across any per-task vs no-suffix split in the source)

    logit_lens_heatmaps/deepseek_v3_2fact_<cond>[_incorrect].json
        ← results/unsupervised_decode_2fact_allpos/decode_2fact_<cond>[_incorrect].json

"""
from __future__ import annotations

import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RELEASE = REPO / "release"

# Source dir → (canonical model name, canonical task name)
DIR_TO_MODEL_TASK = {
    "outputs/deepseek_aggregated":           ("deepseek_v3", "2fact"),
    "outputs/deepseek_1fact_aggregated":     ("deepseek_v3", "1fact"),
    "outputs/deepseek_letterpos_aggregated": ("deepseek_v3", "letterpos"),
    "outputs/deepseek_capitalpos_aggregated": ("deepseek_v3", "capitalpos"),
    "outputs/kimi_k2_2fact_aggregated":      ("kimi_k2",    "2fact"),
    "outputs/kimi_k2_letterpos_aggregated":  ("kimi_k2",    "letterpos"),
    "outputs/kimi_k2_capitalpos_aggregated": ("kimi_k2",    "capitalpos"),
}

# llm_decode_<cond>_<judge>_<prompt>[_<task>].json
JUDGE_PAT = re.compile(
    r'^llm_decode_(?P<cond>.+?)_(?P<judge>haiku|sonnet)_(?P<prompt>[a-z]+?)'
    r'(?:_(?P<task>\w+))?\.json$'
)


def fresh_dir(p: Path) -> None:
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True)


def stage_top_tokens() -> int:
    out = RELEASE / "top_tokens"
    fresh_dir(out)
    n = 0
    for src_dir, (model, task) in DIR_TO_MODEL_TASK.items():
        for f in sorted((REPO / src_dir).glob("aggregated_*.json")):
            cond = f.stem.removeprefix("aggregated_")
            shutil.copy2(f, out / f"{model}_{task}_{cond}.json")
            n += 1
    return n


def stage_judge_outputs() -> int:
    out = RELEASE / "judge_outputs"
    fresh_dir(out)
    n = 0
    for src_dir, (model, task) in DIR_TO_MODEL_TASK.items():
        for f in sorted((REPO / src_dir).glob("llm_decode_*.json")):
            if "summary" in f.name:
                continue
            m = JUDGE_PAT.match(f.name)
            if not m:
                continue
            cond = m.group("cond")
            judge = m.group("judge")
            prompt = m.group("prompt")
            shutil.copy2(f, out / f"{model}_{task}_{cond}_{judge}_{prompt}.json")
            n += 1
    return n


def stage_accuracy_tables() -> int:
    """Re-derive merged judge accuracy tables from the per-record files.

    The original outputs/ has both `llm_decode_summary_<prompt>.json` (older,
    pre-task-suffix layout) and `llm_decode_summary_<prompt>_<task>.json`. We
    rebuild a single canonical summary per (model, task, prompt) so the release
    has one file per group.
    """
    out = RELEASE / "accuracy_tables"
    fresh_dir(out)
    n = 0
    for src_dir, (model, task) in DIR_TO_MODEL_TASK.items():
        # group by prompt: prompt -> {<cond>_<judge>: {n, summary}}
        grouped: dict[str, dict] = defaultdict(dict)
        for f in sorted((REPO / src_dir).glob("llm_decode_*.json")):
            if "summary" in f.name:
                continue
            m = JUDGE_PAT.match(f.name)
            if not m:
                continue
            cond = m.group("cond")
            judge = m.group("judge")
            prompt = m.group("prompt")
            records = json.load(open(f))
            agg = {str(K): 0 for K in (1, 2, 3, 5, 10)}
            for r in records:
                tk = r.get("top_k", {})
                for K in (1, 2, 3, 5, 10):
                    v = tk.get(str(K), tk.get(K))
                    if v is True or v == 1:
                        agg[str(K)] += 1
                    elif isinstance(v, dict) and v.get("both"):
                        # 2fact records: top_k[K] = {a1: bool, a2: bool, both: bool}
                        agg[str(K)] += 1
            grouped[prompt][f"{cond}_{judge}"] = {"n": len(records), "summary": agg}
        for prompt, entries in grouped.items():
            target = out / f"llm_judge_{model}_{task}_{prompt}.json"
            json.dump(entries, open(target, "w"), indent=2, sort_keys=True)
            n += 1
    return n


def stage_logit_lens_heatmaps() -> int:
    out = RELEASE / "logit_lens_heatmaps"
    fresh_dir(out)
    src = REPO / "results" / "unsupervised_decode_2fact_allpos"
    n = 0
    for f in sorted(src.glob("decode_2fact_*.json")):
        # decode_2fact_<cond>[_incorrect].json -> deepseek_v3_2fact_<cond>[_incorrect].json
        new_name = "deepseek_v3_2fact_" + f.stem.removeprefix("decode_2fact_") + ".json"
        shutil.copy2(f, out / new_name)
        n += 1
    return n


def main() -> None:
    if RELEASE.exists():
        print(f"Wiping existing {RELEASE}/")
    RELEASE.mkdir(exist_ok=True)
    n_tt = stage_top_tokens()
    n_ju = stage_judge_outputs()
    n_ac = stage_accuracy_tables()
    n_hm = stage_logit_lens_heatmaps()
    print(f"\nStaged:")
    print(f"  release/top_tokens/         {n_tt:>4} files")
    print(f"  release/judge_outputs/      {n_ju:>4} files")
    print(f"  release/accuracy_tables/    {n_ac:>4} files")
    print(f"  release/logit_lens_heatmaps/{n_hm:>4} files")
    print(f"  TOTAL                       {n_tt + n_ju + n_ac + n_hm:>4} files")
    total_bytes = sum(
        f.stat().st_size for f in RELEASE.rglob("*.json") if f.is_file()
    )
    print(f"  Total size: {total_bytes / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
