"""
twofact_kv_transplant.py

Causal test for the 2-fact addition task: are the filler positions the heatmaps
flag as encoding A1 (vs A2) the positions that causally CARRY that addend?

Design — FACT-MATCH the addend you are NOT swapping (Kaley 2026-06-29). Hold one
addend's element fixed across donor and target, vary the other:

  vary_a1 (hold A2):  donor = "A1_d + A2",  target = "A1_t + A2"   (same A2 element)
  vary_a2 (hold A1):  donor = "A1 + A2_d",  target = "A1 + A2_t"   (same A1 element)

Because the held addend is the SAME element, the donor's answer is ON-MANIFOLD for
the target, so donor-answer adoption is directly comparable to the 1-fact transplant.
Two candidates: keep = target["answer"], donor = donor["answer"].

Per-direction scopes (a position counts for an addend iff it DECODES that addend
above theta — overlap allowed, since fact-matching makes the held addend inert):
  whole       : all filler KV rows (offsets 1..filler_end). Headline causal number.
  <a>_pos     : positions with <addend>-decode >= theta  (a = a1 for vary_a1, a2 for vary_a2)
  no_<a>_pos  : the complement (filler \ <a>_pos) -> the control. Its residual is
                "causal A1 outside the decoded-A1 positions" (distributed / sum).
Run at theta in {0.15, 0.20, 0.30} for robustness.

Throughput: group-by-group PREFILL-REUSE — within a fact-matched group every example
is prefilled (and normal-answer'd) once and reused across that group's pairs; the
transplant clones the cache, so reuse is safe and memory is bounded to one group.

Pure helpers (compute_candidates_2fact, classify_2fact, find_factmatched_pairs,
load_decode_table, sites_at_threshold, offset_pairs) avoid torch and are CPU-unit-
tested via --self-test. MLA cache surgery is reused from varbind_kv_transplant.

Usage (GPU):
    source /workspace/config/probing_env.sh
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python scripts/intervention/twofact_kv_transplant.py \
        --conditions dots_50 --directions vary_a1,vary_a2 --thetas 0.15,0.20,0.30 \
        --max-pairs 500 --output-root results/twofact_kv_transplant

CPU self-test:
    python scripts/intervention/twofact_kv_transplant.py --self-test
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# AWQ import safety (transformers renamed PytorchGELUTanh -> GELUTanh). Guarded so the
# module still imports on a CPU box without transformers (for --self-test).
try:
    import transformers.activations as _act  # noqa: E402
    if not hasattr(_act, "PytorchGELUTanh"):
        _act.PytorchGELUTanh = _act.GELUTanh
except Exception:
    pass

A1K, A2K, ANSK, IDXK = "fact_value_1", "fact_value_2", "answer", "idx"
RANK_KEYS = ["keep", "donor"]
CLASSIFY_ORDER = ["keep", "donor", "a1_donor", "a2_donor", "a1_target", "a2_target"]


# ---------------------------------------------------------------------------
# Pure helpers (no torch) — CPU-testable
# ---------------------------------------------------------------------------

def compute_candidates_2fact(donor, target):
    return {
        "keep": target[ANSK],
        "donor": donor[ANSK],
        "a1_donor": donor[A1K], "a2_donor": donor[A2K],
        "a1_target": target[A1K], "a2_target": target[A2K],
    }


def classify_2fact(answer, cands):
    if answer is None:
        return "parse_failed"
    for name in CLASSIFY_ORDER:
        if answer == cands[name]:
            return name
    return "novel"


def find_factmatched_pairs(examples, direction, n_pairs, seed=42, correct_idx=None):
    """Ordered (donor_idx, target_idx) sharing the HELD addend's value, differing in
    the VARIED addend. direction in {vary_a1, vary_a2}. Restricted to correct_idx."""
    hold, vary = (A2K, A1K) if direction == "vary_a1" else (A1K, A2K)
    pool = [e for e in examples if correct_idx is None or e[IDXK] in correct_idx]
    groups = defaultdict(list)
    for e in pool:
        groups[e[hold]].append(e)
    cand = []
    for g in groups.values():
        if len(g) < 2:
            continue
        for d in g:
            for t in g:
                if d[IDXK] != t[IDXK] and d[vary] != t[vary]:
                    cand.append((d[IDXK], t[IDXK]))
    rng = random.Random(seed)
    rng.shuffle(cand)
    return cand[:n_pairs]


def load_decode_table(heatmap_path):
    """Per filler-offset A1/A2/SUM decode = MAX over layers of the logit-lens
    exact-match hit-rate (fraction of correct examples whose number-token argmax == the
    value). Returns {table: {off: {A1,A2,SUM}}, filler_start_offset, filler_end_offset}."""
    d = json.load(open(heatmap_path))
    layers = [str(l) for l in d["_layers"]]
    b = d["_boundaries"]
    fs_off, fe_off = b["filler_start_offset"], b["filler_end_offset"]
    table = {}
    for p in d["_positions"]:
        if p == "question_end":
            off = 0
        elif p.startswith("pos_"):
            off = int(p.split("_")[1])
        else:
            continue
        if off < 1 or off > fe_off:
            continue
        cell = d[p]

        def mx(key):
            vs = [float(cell[L][key]) for L in layers if L in cell]
            return max(vs) if vs else 0.0

        table[off] = {"A1": mx("frac_A1_exact"), "A2": mx("frac_A2_exact"), "SUM": mx("frac_A1A2_exact")}
    return {"table": table, "filler_start_offset": fs_off, "filler_end_offset": fe_off}


def sites_at_threshold(table, fe_off, addend, theta):
    """Filler offsets whose `addend` ('A1'/'A2') decode >= theta (overlap allowed)."""
    return sorted([o for o in range(1, fe_off + 1) if table[o][addend] >= theta])


def offset_pairs(qend_donor, qend_target, offsets):
    return [(qend_donor + o, qend_target + o) for o in offsets]


# ---------------------------------------------------------------------------
# GPU section
# ---------------------------------------------------------------------------

def build_2fact_messages(few_shot, target, filler_type, k, rng):
    from extract.extract_hidden_states import build_messages_for_condition
    return build_messages_for_condition(few_shot[:5], target, filler_type, k,
                                        rng=rng, dataset_type="2fact")


def _summarize(cond, direction, results, config_labels, np):
    both = [r for r in results if r["T_correct"] and r["D_correct"]]
    use = both if both else results
    print(f"\n{'='*72}\nSUMMARY  {cond}/{direction}  (n={len(results)}, both-correct n={len(both)})\n{'='*72}")
    for label in config_labels:
        outs = [r["configs"][label]["outcome"] for r in use]
        n = max(len(outs), 1)
        adopt = outs.count("donor") / n
        shifts = [r["rank_norm"]["donor"] - r["configs"][label]["rank_transplant"]["donor"]
                  for r in use
                  if r["rank_norm"]["donor"] is not None
                  and r["configs"][label]["rank_transplant"]["donor"] is not None]
        med = float(np.median(shifts)) if shifts else float("nan")
        rn = [r["rank_norm"]["donor"] for r in use if r["rank_norm"]["donor"] is not None]
        rt = [r["configs"][label]["rank_transplant"]["donor"] for r in use
              if r["configs"][label]["rank_transplant"]["donor"] is not None]
        rnm = float(np.median(rn)) if rn else float("nan")
        rtm = float(np.median(rt)) if rt else float("nan")
        ns = use[0]["configs"][label]["n_swapped"]
        print(f"  {label:14s} n_swap≈{ns:>3}  donor_adopted={adopt:5.1%}  "
              f"| donor rank {rnm:.0f}->{rtm:.0f} (med shift {med:+.0f})")


def run(args):
    import torch
    import numpy as np
    from tqdm import tqdm
    from extract.extract_hidden_states import load_model, find_filler_boundaries
    from intervention.filler_kv_transplant import prefill_and_cache, generate_from_cache
    from intervention.varbind_kv_transplant import apply_transplant, _truncate_cache, _rank

    with open(args.dataset) as f:
        dataset = json.load(f)
    few_shot = dataset["few_shot_facts"]
    examples = dataset["examples"]
    by_idx = {e[IDXK]: e for e in examples}

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    directions = [d.strip() for d in args.directions.split(",") if d.strip()]
    thetas = [float(x) for x in args.thetas.split(",")]

    model, tokenizer = load_model(args.model_path)
    device = next(model.parameters()).device

    def tok_id(v):
        ids = tokenizer.encode(str(v), add_special_tokens=False)
        return ids[0] if len(ids) == 1 else None

    def ids_for(ex, filler_type, k):
        msgs = build_2fact_messages(few_shot, ex, filler_type, k, random.Random(ex[IDXK]))
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return tokenizer(text, return_tensors="pt")["input_ids"].to(device)

    def prep_example(idx, filler_type, k):
        """Prefill + normal answer + rank-baseline logits for one example (reused)."""
        ids = ids_for(by_idx[idx], filler_type, k)
        q, _, _ = find_filler_boundaries(tokenizer, ids, k)
        cache, _ = prefill_and_cache(model, tokenizer, ids, torch.ones_like(ids))
        last = ids[0, -1].item()
        _, ans_norm, logits_norm = generate_from_cache(
            model, tokenizer, last, _truncate_cache(cache), args.max_new_tokens)
        return {"cache": cache, "q": q, "last": last, "ans_norm": ans_norm,
                "logits_norm": logits_norm, "correct": ans_norm == by_idx[idx][ANSK],
                "n_tok": ids.shape[1]}

    for cond in conditions:
        filler_type, k = cond.split("_")[0], int(cond.split("_")[1])
        assert filler_type == "dots", "this experiment is dots-only per the design"
        dt = load_decode_table(args.heatmap_dir / f"decode_2fact_{cond}.json")
        table, fe_off = dt["table"], dt["filler_end_offset"]
        whole_offsets = list(range(1, fe_off + 1))

        cpath = (Path(args.correct_idx_json) if args.correct_idx_json
                 else Path(f"data/twofact_correct_idx_{cond}.json"))
        correct_idx = set(json.load(open(cpath))) if cpath.exists() else None
        print(f"\n[{cond}] whole=1..{fe_off}; {len(correct_idx or [])} model-correct examples")
        for th in thetas:
            print(f"  theta={th}: a1_pos={sites_at_threshold(table,fe_off,'A1',th)} | "
                  f"a2_pos={sites_at_threshold(table,fe_off,'A2',th)}")

        # Self-transplant sanity (MLA surgery gate): swap a target's whole filler with ITS OWN.
        if args.self_check > 0:
            sample = list(correct_idx)[:args.self_check] if correct_idx else [e[IDXK] for e in examples[:args.self_check]]
            ok = 0
            for ci, idx0 in enumerate(sample):
                e = prep_example(idx0, filler_type, k)
                if ci == 0:
                    print(f"  [sanity] prompt {e['n_tok']} tok; q_end={e['q']}")
                ct = apply_transplant(e["cache"], e["cache"], offset_pairs(e["q"], e["q"], whole_offsets))
                _, a_self, _ = generate_from_cache(model, tokenizer, e["last"], ct, args.max_new_tokens)
                ok += (a_self == e["ans_norm"])
                del ct, e["cache"]
                torch.cuda.empty_cache()
            print(f"  self-transplant preserved the answer: {ok}/{len(sample)}")
            if ok < len(sample):
                print("  WARNING: self-transplant changed the answer — MLA cache surgery may be malformed.")

        for direction in directions:
            hold_key = A2K if direction == "vary_a1" else A1K
            addend = "A1" if direction == "vary_a1" else "A2"
            tag = "a1" if direction == "vary_a1" else "a2"
            CONFIGS = [("whole", whole_offsets)]
            for th in thetas:
                pos = sites_at_threshold(table, fe_off, addend, th)
                nopos = [o for o in range(1, fe_off + 1) if o not in set(pos)]
                CONFIGS.append((f"{tag}_pos@{th}", pos))
                CONFIGS.append((f"no_{tag}_pos@{th}", nopos))
            config_labels = [c[0] for c in CONFIGS]

            pairs = find_factmatched_pairs(examples, direction, args.max_pairs,
                                           seed=args.seed, correct_idx=correct_idx)
            outdir = args.output_root / f"{cond}_{direction}"
            outdir.mkdir(parents=True, exist_ok=True)
            meta = {"condition": cond, "direction": direction, "thetas": thetas,
                    "configs": {lab: offs for lab, offs in CONFIGS}, "n_pairs": len(pairs),
                    "filler_end_offset": fe_off}
            # group pairs by the held addend value -> prefill each example once per group
            buckets = defaultdict(list)
            for (idx_D, idx_T) in pairs:
                buckets[by_idx[idx_D][hold_key]].append((idx_D, idx_T))
            print(f"\n=== {cond} / {direction}: {len(pairs)} pairs over {len(buckets)} fact-matched groups -> {outdir} ===")

            results = []
            pbar = tqdm(total=len(pairs), desc=f"{cond}/{direction}")
            for held_val, bpairs in buckets.items():
                uniq = {x for p in bpairs for x in p}
                excache = {idx: prep_example(idx, filler_type, k) for idx in uniq}
                for (idx_D, idx_T) in bpairs:
                    D, T, eD, eT = by_idx[idx_D], by_idx[idx_T], excache[idx_D], excache[idx_T]
                    cands = compute_candidates_2fact(D, T)
                    cand_tok = {key: tok_id(cands[key]) for key in RANK_KEYS}
                    rank_norm = {key: _rank(eT["logits_norm"], cand_tok[key]) for key in RANK_KEYS}
                    per_config = {}
                    for label, offs in CONFIGS:
                        ipairs = offset_pairs(eD["q"], eT["q"], offs)
                        cache_t = apply_transplant(eT["cache"], eD["cache"], ipairs)
                        _, ans, logits_t = generate_from_cache(model, tokenizer, eT["last"], cache_t, args.max_new_tokens)
                        per_config[label] = {
                            "n_swapped": len(ipairs), "answer": ans,
                            "outcome": classify_2fact(ans, cands),
                            "rank_transplant": {key: _rank(logits_t, cand_tok[key]) for key in RANK_KEYS},
                        }
                        del cache_t
                    results.append({
                        "idx_D": idx_D, "idx_T": idx_T, "candidates": cands,
                        "T_correct": eT["correct"], "D_correct": eD["correct"],
                        "rank_norm": rank_norm, "configs": per_config,
                    })
                    torch.cuda.empty_cache()
                    pbar.update(1)
                    if len(results) % 10 == 0 or len(results) == len(pairs):
                        json.dump({"meta": meta, "results": results},
                                  open(outdir / "twofact_transplant_results.json", "w"), indent=2)
                for e in excache.values():
                    del e["cache"]
                del excache
                torch.cuda.empty_cache()
            pbar.close()

            _summarize(cond, direction, results, config_labels, np)
            json.dump({"meta": meta, "results": results},
                      open(outdir / "twofact_transplant_results.json", "w"), indent=2)
            print(f"saved {outdir}/twofact_transplant_results.json")


# ---------------------------------------------------------------------------
# CPU self-test
# ---------------------------------------------------------------------------

def self_test():
    root = Path(__file__).resolve().parents[2]
    print("== candidates / classify ==")
    D = {A1K: 90, A2K: 52, ANSK: 142, IDXK: 1}
    T = {A1K: 30, A2K: 52, ANSK: 82, IDXK: 2}
    c = compute_candidates_2fact(D, T)
    assert c["keep"] == 82 and c["donor"] == 142
    assert classify_2fact(142, c) == "donor" and classify_2fact(82, c) == "keep"
    assert classify_2fact(90, c) == "a1_donor" and classify_2fact(7, c) == "novel"
    print("   ok")

    print("== find_factmatched_pairs (dots_50) ==")
    ds = json.load(open(root / "data/2fact_addition_dataset.json"))
    ex = ds["examples"]
    by = {e[IDXK]: e for e in ex}
    cidx = set(json.load(open(root / "data/twofact_correct_idx_dots_50.json")))
    for direction in ["vary_a1", "vary_a2"]:
        pairs = find_factmatched_pairs(ex, direction, 500, correct_idx=cidx)
        hold, vary = (A2K, A1K) if direction == "vary_a1" else (A1K, A2K)
        assert all(by[d][hold] == by[t][hold] and by[d][vary] != by[t][vary] for d, t in pairs)
        assert all(d in cidx and t in cidx and by[d][ANSK] != by[t][ANSK] for d, t in pairs)
        assert len(set(pairs)) == len(pairs)
        # group sizes (for prefill-reuse): unique examples << pairs
        uniq = {x for p in pairs for x in p}
        print(f"   {direction}: {len(pairs)} pairs, {len(uniq)} unique examples")

    print("== load_decode_table / sites_at_threshold (dots_50) ==")
    dt = load_decode_table(root / "results/unsupervised_decode_2fact_allpos/decode_2fact_dots_50.json")
    fe = dt["filler_end_offset"]
    for th in [0.15, 0.20, 0.30]:
        a1 = sites_at_threshold(dt["table"], fe, "A1", th)
        a2 = sites_at_threshold(dt["table"], fe, "A2", th)
        print(f"   theta={th}: a1_pos(n={len(a1)})={a1}")
        print(f"   theta={th}: a2_pos(n={len(a2)})={a2}")
    assert sites_at_threshold(dt["table"], fe, "A1", 0.20) == [1, 3, 6, 10, 11, 17, 18, 21, 28]
    assert sites_at_threshold(dt["table"], fe, "A2", 0.20) == [11, 13, 16, 17, 18, 28, 30, 34]
    assert offset_pairs(100, 200, [1, 6]) == [(101, 201), (106, 206)]
    print("\nALL SELF-TESTS PASSED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--model-path", default="/workspace/models/deepseek-v3-awq")
    ap.add_argument("--dataset", type=Path, default=Path("data/2fact_addition_dataset.json"))
    ap.add_argument("--conditions", default="dots_50")
    ap.add_argument("--directions", default="vary_a1,vary_a2")
    ap.add_argument("--thetas", default="0.15,0.20,0.30")
    ap.add_argument("--output-root", type=Path, default=Path("results/twofact_kv_transplant"))
    ap.add_argument("--heatmap-dir", type=Path, default=Path("results/unsupervised_decode_2fact_allpos"))
    ap.add_argument("--max-pairs", type=int, default=500)
    ap.add_argument("--max-new-tokens", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--correct-idx-json", default=None)
    ap.add_argument("--self-check", type=int, default=5)
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    run(args)


if __name__ == "__main__":
    main()
