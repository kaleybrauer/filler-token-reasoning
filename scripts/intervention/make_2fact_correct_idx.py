"""Build the model-correct index for a 2-fact condition from the extracted-states pkls.

Each pkl stores `problem_idx` + `model_correct` (authoritative — same prompt path the
transplant rebuilds). We only need those two fields, but pickle is all-or-nothing, so we
parallel-load to hide the MooseFS I/O. Output: data/twofact_correct_idx_<cond>.json (sorted
list of model-correct problem_idx) — used by twofact_kv_transplant.py --correct-idx-json to
enrich both-correct pair sampling (final both-correct is still re-checked on the fly).

Usage:
    python scripts/intervention/make_2fact_correct_idx.py dots_10 dots_50
"""
import json
import pickle
import sys
from pathlib import Path
from multiprocessing import Pool

ROOT = Path(__file__).resolve().parents[2]
EXTRACT_DIR = ROOT / "data" / "extracted_states_2fact_allpos"


def _one(f):
    with open(f, "rb") as fp:
        d = pickle.load(fp)
    return int(d["problem_idx"]), bool(d["model_correct"])


def main():
    conds = sys.argv[1:] or ["dots_10", "dots_50"]
    for cond in conds:
        files = sorted((EXTRACT_DIR / cond).glob("prob_*.pkl"))
        if not files:
            print(f"{cond}: NO pkls found at {EXTRACT_DIR / cond}")
            continue
        with Pool(16) as p:
            res = p.map(_one, files)
        correct = sorted(idx for idx, ok in res if ok)
        out = ROOT / "data" / f"twofact_correct_idx_{cond}.json"
        json.dump(correct, open(out, "w"))
        print(f"{cond}: {len(correct)}/{len(res)} correct -> {out}")


if __name__ == "__main__":
    main()
