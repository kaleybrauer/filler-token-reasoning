"""
varbind_kv_transplant.py

Causal test for the chained variable-binding task: are the FILLER positions
causally carrying the never-written intermediate y, such that the answer is
computed *through* it?

Design — a DISSOCIATION the addition transplant can't do. Pick receiver R and
donor D with DIFFERENT y AND a DIFFERENT question op (c₂/k₂; require c₂_R≠c₂_D
for clean separation). Transplant D's filler KV into R and adjudicate the
generated answer between three precomputed candidates:

    keep         = c₂_R·y_R ± k₂_R  (= answer_R)          transplant had no effect
    through_y    = c₂_R·y_D ± k₂_R                          filler carried y_D, R's op re-applied  <-- the result
    carry_answer = c₂_D·y_D ± k₂_D  (= answer_D)            filler carried the answer

`through_y` appears in NEITHER prompt and is NEITHER example's answer, so emitting
it can only mean the model read the donor's never-written y out of the filler and
ran the receiver's never-written question op on it: causal + constructed +
computed-through-the-intermediate, in one number.

Two pair modes (--pair-mode):
  op_varied (default): the dissociation above — R and D have DIFFERENT question ops,
      so through_y is a distinct candidate.
  op_fixed: R and D share the question op and differ only in y, so the donor's answer
      is ON-MANIFOLD for the receiver. This is the direct structural analog of the
      1-fact transplant (vary the looked-up/intermediate value, hold the downstream op),
      so it isolates retrieval-vs-construction / recomputability as the sole axis vs
      1-fact. through_y collapses into carry_answer (identical value) and is dropped;
      the clean binary is keep vs carry_answer.

Headline metric = median rank-improvement of the `through_y` token under
transplant (graded; robust even at modest argmax adoption — cf. the addition task:
13% adoption but +72 rank shift), plus the argmax outcome distribution.

Graded N-sweep (this replaces a separate attention knockout): transplant D's
filler at N ∈ {1,3,6,12,all} randomly-chosen positions; gradual rise ⇒ redundant
(reinforcement), cliff ⇒ a critical position. first-N vs last-N localizes
y-carrying vs answer-carrying filler positions (causal mirror of the depth ladder).

The pure helpers (apply_op, compute_candidates, classify, find_varbind_pairs,
select_swap_indices) avoid torch and are CPU-unit-tested; torch / model load /
cache surgery are imported lazily inside the GPU functions.

Usage (GPU):
    source /workspace/config/probing_env.sh
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python scripts/intervention/varbind_kv_transplant.py \
        --filler-k 25 --max-pairs 300 --output-dir results/varbind_kv_transplant
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Torch-free prompt construction.
from data.generate_varbind_dataset import (  # noqa: E402
    build_system_message_varbind, build_user_turn_varbind,
)

# (label, N, mode, layer_range) — N=None whole filler; layer_range=None all layers
# or (lo,hi) inclusive. KEY: through_y (answer RECOMPUTED from y) is only possible
# where the donor's y is injected WITHOUT the donor's answer — otherwise the answer
# sits in the swapped filler and you just copy it (carry_answer). The decode
# localizes the answer to the LAST filler tokens (+ answer_prompt) and to the top
# layers (~L60), while y is smeared across earlier positions and forms by ~L44. So
# through_y is testable by swapping EARLY filler positions and/or only layers up to
# ~47 (y present, answer not). "full"/"last" inject the answer -> carry_answer
# (causality baseline). The result is the outcome MAP over (positions, layers).
DEFAULT_SWEEP = [
    ("full", None, "all", None),            # causality baseline -> expect carry_answer
    ("y_layers", None, "all", (0, 47)),     # inject y, not the answer -> through_y test (layer lever)
    ("first6", 6, "first", None),           # early filler (y, not answer) -> through_y test (position lever)
    ("first12", 12, "first", None),
    ("last12", 12, "last", None),           # late filler (answer) -> expect carry_answer
    ("n3", 3, "random", None), ("n6", 6, "random", None), ("n12", 12, "random", None),  # redundancy curve
]
CANDS = ["keep", "through_y", "carry_answer", "raw_yD"]


# ---------------------------------------------------------------------------
# Pure helpers (no torch) — CPU-testable
# ---------------------------------------------------------------------------

def apply_op(coef, op, const, y):
    """Apply a varbind question/chain operation: coef*y ± const."""
    v = coef * y
    return v + const if op == "plus" else v - const


def compute_candidates(R, D, pair_mode="op_varied"):
    """The candidate answer values for a receiver R / donor D pair.

    op_varied: through_y (= R's op on D's y) is distinct from carry_answer (= D's answer).
    op_fixed: R and D share the question op, so R's-op-on-yD == D's answer; through_y is
        set to None (dropped) to avoid mislabeling donor-answer matches. Clean binary =
        keep vs carry_answer (the 1-fact-matched control).
    """
    yD = D["queried_value"]
    through_y = apply_op(R["coefficient"], R["operation"], R["constant"], yD)   # c₂_R·y_D ± k₂_R
    if pair_mode == "op_fixed":
        through_y = None
    return {
        "keep": R["answer"],            # = c₂_R·y_R ± k₂_R
        "through_y": through_y,
        "carry_answer": D["answer"],    # = c₂_D·y_D ± k₂_D
        "raw_yD": yD,
    }


def classify(answer, cands):
    """Which candidate the generated integer matches (priority order)."""
    if answer is None:
        return "parse_failed"
    for name in ("keep", "through_y", "carry_answer", "raw_yD"):
        if answer == cands[name]:
            return name
    return "novel"


def find_varbind_pairs(examples, n_pairs, seed=42, require_c2_differ=True,
                       max_token=1000, correct_idx=None, pair_mode="op_varied"):
    """Sample transplant pairs (i=receiver, j=donor).

    op_varied (default): DIFFERENT y AND different question op (and c₂ if
        require_c2_differ for max through_y/carry_answer separation), with
        keep/through_y/carry_answer all distinct and through_y a single-token int
        (<max_token) — the dissociation design.
    op_fixed: SAME question op (c₂,op,k₂), different y -> the donor's answer is
        on-manifold for the receiver (1-fact-matched control). through_y collapses
        into carry_answer and is dropped; require keep != carry_answer and
        carry_answer a single-token int (<max_token).

    correct_idx: optional set of example `idx` to restrict both R and D to.
    """
    rng = random.Random(seed)
    n = len(examples)
    pairs, seen, attempts = [], set(), 0
    cap = n_pairs * 500
    while len(pairs) < n_pairs and attempts < cap:
        attempts += 1
        i, j = rng.randrange(n), rng.randrange(n)
        if i == j or (i, j) in seen:
            continue
        R, D = examples[i], examples[j]
        if correct_idx is not None and not (R["idx"] in correct_idx and D["idx"] in correct_idx):
            continue
        if D["queried_value"] == R["queried_value"]:
            continue
        op_R = (R["coefficient"], R["operation"], R["constant"])
        op_D = (D["coefficient"], D["operation"], D["constant"])
        if pair_mode == "op_fixed":
            if op_R != op_D:                  # SAME question op -> donor answer on-manifold
                continue
        else:
            if op_R == op_D:                  # DIFFERENT question op (the dissociation)
                continue
            if require_c2_differ and R["coefficient"] == D["coefficient"]:
                continue
        c = compute_candidates(R, D, pair_mode=pair_mode)
        if pair_mode == "op_fixed":
            if not (0 <= c["carry_answer"] < max_token) or c["keep"] == c["carry_answer"]:
                continue
        else:
            if not (0 <= c["through_y"] < max_token):
                continue
            if len({c["keep"], c["through_y"], c["carry_answer"]}) != 3:
                continue
        seen.add((i, j))
        pairs.append((i, j))
    return pairs


def select_swap_indices(fs_t, fe_t, fs_d, fe_d, n=None, mode="all", rng=None):
    """List of (donor_abs_idx, target_abs_idx) filler positions to swap, aligned
    by filler-relative index over the common length. n=None -> all; else pick n
    by mode ('first'/'last'/'random')."""
    L = min(fe_t - fs_t + 1, fe_d - fs_d + 1)
    rel = list(range(L))
    if n is not None and n < L:
        if mode == "first":
            rel = rel[:n]
        elif mode == "last":
            rel = rel[-n:]
        else:  # random
            rel = sorted((rng or random.Random(0)).sample(rel, n))
    return [(fs_d + r, fs_t + r) for r in rel]


def build_varbind_messages(few_shot, target, filler_type, k, rng):
    """Chat messages for a varbind filler condition — mirrors
    extract_hidden_states.build_messages_for_condition's varbind branch exactly."""
    msgs = [{"role": "system", "content": build_system_message_varbind(filler_type, k)}]
    for fs in few_shot:
        msgs.append({"role": "user", "content": build_user_turn_varbind(
            fs["definitions"], fs["question"], filler_type, k, rng=rng)})
        msgs.append({"role": "assistant", "content": str(fs["answer"])})
    msgs.append({"role": "user", "content": build_user_turn_varbind(
        target["definitions"], target["question"], filler_type, k, rng=rng)})
    return msgs


# ---------------------------------------------------------------------------
# GPU section (torch imported lazily so the module loads on a CPU box)
# ---------------------------------------------------------------------------

def apply_transplant(target_cache, donor_cache, index_pairs, truncate_last=True, layers=None):
    """Swap the given (donor_idx, target_idx) positions of the KV cache (clone
    receiver, overwrite the selected positions with donor's). `layers`: optional
    iterable of layer indices to restrict the swap to (e.g. (0..47) injects the
    donor's y before its answer forms, so the answer must be RECOMPUTED -> the
    through_y test); None = all layers. Returns a fresh DynamicCache; truncate_last
    drops the final entry so generation re-runs the last token."""
    import torch  # noqa: F401
    from transformers.cache_utils import DynamicCache
    swap_layers = set(range(len(target_cache))) if layers is None else set(layers)
    cache = DynamicCache()
    for layer_idx in range(len(target_cache)):
        tk, tv = target_cache[layer_idx]
        dk, dv = donor_cache[layer_idx]
        nk, nv = tk.clone(), tv.clone()
        if layer_idx in swap_layers:
            for d_idx, t_idx in index_pairs:
                nk[:, :, t_idx, :] = dk[:, :, d_idx, :]
                nv[:, :, t_idx, :] = dv[:, :, d_idx, :]
        if truncate_last:
            nk, nv = nk[:, :, :-1, :], nv[:, :, :-1, :]
        cache.update(nk, nv, layer_idx=layer_idx)
    return cache


def _truncate_cache(cache):
    import torch  # noqa: F401
    from transformers.cache_utils import DynamicCache
    c = DynamicCache()
    for layer_idx in range(len(cache)):
        k, v = cache[layer_idx]
        c.update(k[:, :, :-1, :], v[:, :, :-1, :], layer_idx=layer_idx)
    return c


def _rank(logits_arr, tok_id):
    if tok_id is None:
        return None
    logit = float(logits_arr[tok_id])
    return int((logits_arr > logit).sum()) + 1


# ---------------------------------------------------------------------------
# Batched config decode (within-pair). All sweep configs share the SAME receiver
# prompt, so they have identical cache length -> we stack the B config-variants in
# the batch dimension and decode them in ONE pass instead of B sequential decodes.
# No padding needed (common prefix). Numerically independent per row (attention is
# row-local, MoE routing per token), so results match the sequential path up to
# kernel float noise -- gated by --verify-batched before trusting.
# ---------------------------------------------------------------------------

def apply_transplant_batched(target_cache, donor_cache, idx_pairs_list, layers_list,
                             truncate_last=True):
    """Build one KV cache with batch dim B=len(idx_pairs_list); row r = receiver cache
    with donor positions swapped per config r. layers_list[r]: None (all layers) or a
    list of layer indices to restrict row r's swap to. Returns a fresh DynamicCache."""
    import torch  # noqa: F401
    from transformers.cache_utils import DynamicCache
    B = len(idx_pairs_list)
    n_layers = len(target_cache)
    swap_sets = [set(range(n_layers)) if L is None else set(L) for L in layers_list]
    cache = DynamicCache()
    for layer_idx in range(n_layers):
        tk, tv = target_cache[layer_idx]            # (1, H, S, D)
        dk, dv = donor_cache[layer_idx]
        nk = tk.repeat(B, 1, 1, 1)                  # repeat allocates fresh, writable
        nv = tv.repeat(B, 1, 1, 1)
        for r in range(B):
            if layer_idx in swap_sets[r]:
                for d_idx, t_idx in idx_pairs_list[r]:
                    nk[r, :, t_idx, :] = dk[0, :, d_idx, :]
                    nv[r, :, t_idx, :] = dv[0, :, d_idx, :]
        if truncate_last:
            nk, nv = nk[:, :, :-1, :], nv[:, :, :-1, :]
        cache.update(nk, nv, layer_idx=layer_idx)
    return cache


def generate_from_cache_batched(model, tokenizer, last_token_id, past_key_values,
                                max_new_tokens=6):
    """Batched analogue of filler_kv_transplant.generate_from_cache. All B rows share
    one last token and a common cache length. Returns raws[B], answers[B],
    first_logits (B, vocab) numpy."""
    import torch
    from prompt_utils import extract_answer
    device = next(model.parameters()).device
    B = past_key_values[0][0].shape[0]
    cache_len = past_key_values[0][0].shape[2]
    input_ids = torch.full((B, 1), last_token_id, dtype=torch.long, device=device)
    attn_mask = torch.ones(B, cache_len + 1, dtype=torch.long, device=device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attn_mask,
                        past_key_values=past_key_values, use_cache=True)
    first_logits = outputs.logits[:, -1, :].float().cpu().numpy()   # (B, vocab)
    logits = outputs.logits[:, -1, :]
    past = outputs.past_key_values
    generated = [[] for _ in range(B)]
    finished = [False] * B
    eos = tokenizer.eos_token_id
    for _ in range(max_new_tokens):
        next_tokens = logits.argmax(dim=-1)                         # (B,)
        for b in range(B):
            if not finished[b]:
                tid = next_tokens[b].item()
                if tid == eos:
                    finished[b] = True
                else:
                    generated[b].append(tid)
        if all(finished):
            break
        cur_len = past[0][0].shape[2] + 1
        step_mask = torch.ones(B, cur_len, dtype=torch.long, device=device)
        with torch.no_grad():
            outputs = model(input_ids=next_tokens.unsqueeze(1), attention_mask=step_mask,
                            past_key_values=past, use_cache=True)
        logits = outputs.logits[:, -1, :]
        past = outputs.past_key_values
    raws = [tokenizer.decode(g, skip_special_tokens=True).strip() for g in generated]
    answers = [extract_answer(r) for r in raws]
    return raws, answers, first_logits


def _per_config_sequential(model, tokenizer, last_R, cache_R, cache_D, sweep,
                           idx_list, layer_list, cands, cand_tok, max_new_tokens):
    from intervention.filler_kv_transplant import generate_from_cache
    per_config = {}
    for bi, (label, n, mode, lr) in enumerate(sweep):
        cache_t = apply_transplant(cache_R, cache_D, idx_list[bi], layers=layer_list[bi])
        raw, ans, logits_t = generate_from_cache(
            model, tokenizer, last_R, cache_t, max_new_tokens)
        per_config[label] = {
            "n_swapped": len(idx_list[bi]), "answer": ans, "raw": raw,
            "outcome": classify(ans, cands),
            "rank_transplant": {k: _rank(logits_t, cand_tok[k]) for k in CANDS},
        }
        del cache_t
    return per_config


def _per_config_batched(model, tokenizer, last_R, cache_R, cache_D, sweep,
                        idx_list, layer_list, cands, cand_tok, max_new_tokens):
    batched_cache = apply_transplant_batched(cache_R, cache_D, idx_list, layer_list)
    raws, anss, logits_b = generate_from_cache_batched(
        model, tokenizer, last_R, batched_cache, max_new_tokens)
    per_config = {}
    for bi, (label, n, mode, lr) in enumerate(sweep):
        per_config[label] = {
            "n_swapped": len(idx_list[bi]), "answer": anss[bi], "raw": raws[bi],
            "outcome": classify(anss[bi], cands),
            "rank_transplant": {k: _rank(logits_b[bi], cand_tok[k]) for k in CANDS},
        }
    del batched_cache
    return per_config


def run(args):
    import torch
    import numpy as np
    from tqdm import tqdm
    from extract.extract_hidden_states import load_model, find_filler_boundaries
    from intervention.filler_kv_transplant import prefill_and_cache, generate_from_cache

    with open(args.dataset) as f:
        dataset = json.load(f)
    few_shot = dataset.get("few_shot_examples") or dataset["few_shot_facts"]
    examples = dataset["examples"]

    correct_idx = None
    if args.correct_idx_json and Path(args.correct_idx_json).exists():
        correct_idx = set(json.load(open(args.correct_idx_json)))
        print(f"restricting to {len(correct_idx)} model-correct examples")

    pairs = find_varbind_pairs(examples, args.max_pairs, seed=args.seed,
                               require_c2_differ=not args.allow_same_c2,
                               correct_idx=correct_idx, pair_mode=args.pair_mode)
    print(f"{len(pairs)} {args.pair_mode} pairs (filler_k={args.filler_k}, {args.filler_type})")
    if len(pairs) < args.max_pairs:
        print(f"  NOTE: requested {args.max_pairs} but only {len(pairs)} available "
              f"(pool exhausted for pair_mode={args.pair_mode}).")

    sweep = [("full", None, "all", None)] if args.sweep == "full" else DEFAULT_SWEEP
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model(args.model_path)
    device = next(model.parameters()).device

    # Self-transplant sanity: swapping a receiver's filler with ITS OWN must be a
    # no-op. If it changes the answer, the (MLA) cache surgery is malformed — stop.
    if args.self_check > 0:
        ok = 0
        for k in range(min(args.self_check, len(examples))):
            e0 = examples[k]
            m = build_varbind_messages(few_shot[:5], e0, args.filler_type, args.filler_k,
                                       random.Random(e0.get("idx", k)))
            t = tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            idz = tokenizer(t, return_tensors="pt")["input_ids"].to(device)
            _, fsz, fez = find_filler_boundaries(tokenizer, idz, args.filler_k)
            if k == 0:
                print(f"  [sanity] detected filler span = {fez - fsz + 1} positions "
                      f"(expected {args.filler_k}; prompt len {idz.shape[1]} tokens)")
            cz, _ = prefill_and_cache(model, tokenizer, idz, torch.ones_like(idz))
            last = idz[0, -1].item()
            _, a_norm, _ = generate_from_cache(model, tokenizer, last, _truncate_cache(cz), args.max_new_tokens)
            ct = apply_transplant(cz, cz, select_swap_indices(fsz, fez, fsz, fez))
            _, a_self, _ = generate_from_cache(model, tokenizer, last, ct, args.max_new_tokens)
            ok += (a_self == a_norm)
            del cz, ct
            torch.cuda.empty_cache()
        print(f"self-transplant preserved the answer: {ok}/{min(args.self_check, len(examples))}")
        if ok < min(args.self_check, len(examples)):
            print("WARNING: self-transplant changed the answer — MLA cache surgery may be "
                  "malformed. Inspect before trusting results.")

    def tok_id(v):
        if v is None:                       # op_fixed drops through_y -> no token / rank
            return None
        ids = tokenizer.encode(str(v), add_special_tokens=False)
        return ids[0] if len(ids) == 1 else None

    _vb_stats = {"total": 0, "answer_match": 0, "rank_match": 0, "mismatch": []}
    if args.verify_batched:
        pairs = pairs[:args.verify_batched]
        print(f"VERIFY mode: running {len(pairs)} pairs in BOTH sequential and batched, "
              f"comparing per-config answers/ranks (no full run).")

    results = []
    for pi, (i, j) in enumerate(tqdm(pairs, desc="transplant")):
        R, D = examples[i], examples[j]
        rng = random.Random(R.get("idx", i))
        msgs_R = build_varbind_messages(few_shot[:5], R, args.filler_type, args.filler_k, rng)
        msgs_D = build_varbind_messages(few_shot[:5], D, args.filler_type, args.filler_k,
                                        random.Random(D.get("idx", j)))
        text_R = tokenizer.apply_chat_template(msgs_R, tokenize=False, add_generation_prompt=True)
        text_D = tokenizer.apply_chat_template(msgs_D, tokenize=False, add_generation_prompt=True)
        ids_R = tokenizer(text_R, return_tensors="pt")["input_ids"].to(device)
        ids_D = tokenizer(text_D, return_tensors="pt")["input_ids"].to(device)
        mask_R = torch.ones_like(ids_R)
        mask_D = torch.ones_like(ids_D)

        _, fs_R, fe_R = find_filler_boundaries(tokenizer, ids_R, args.filler_k)
        _, fs_D, fe_D = find_filler_boundaries(tokenizer, ids_D, args.filler_k)

        cache_R, _ = prefill_and_cache(model, tokenizer, ids_R, mask_R)
        cache_D, _ = prefill_and_cache(model, tokenizer, ids_D, mask_D)
        last_R = ids_R[0, -1].item()
        last_D = ids_D[0, -1].item()

        # Normal answers (controls + correctness) for R and D.
        _, ansR_norm, logits_norm = generate_from_cache(
            model, tokenizer, last_R, _truncate_cache(cache_R), args.max_new_tokens)
        _, ansD_norm, _ = generate_from_cache(
            model, tokenizer, last_D, _truncate_cache(cache_D), args.max_new_tokens)

        cands = compute_candidates(R, D, pair_mode=args.pair_mode)
        cand_tok = {k: tok_id(v) for k, v in cands.items()}
        rank_norm = {k: _rank(logits_norm, cand_tok[k]) for k in CANDS}

        # Build swap indices ONCE (consumes rng in sweep order — identical to the
        # original sequential consumption, so seq and batched see the same swaps).
        idx_list, layer_list = [], []
        for label, n, mode, lr in sweep:
            idx_list.append(select_swap_indices(fs_R, fe_R, fs_D, fe_D, n=n, mode=mode, rng=rng))
            layer_list.append(None if lr is None else list(range(lr[0], lr[1] + 1)))

        if args.verify_batched and pi < args.verify_batched:
            pc_seq = _per_config_sequential(model, tokenizer, last_R, cache_R, cache_D,
                                            sweep, idx_list, layer_list, cands, cand_tok,
                                            args.max_new_tokens)
            pc_bat = _per_config_batched(model, tokenizer, last_R, cache_R, cache_D,
                                         sweep, idx_list, layer_list, cands, cand_tok,
                                         args.max_new_tokens)
            for label, _, _, _ in sweep:
                _vb_stats["total"] += 1
                if pc_seq[label]["answer"] == pc_bat[label]["answer"]:
                    _vb_stats["answer_match"] += 1
                else:
                    _vb_stats["mismatch"].append(
                        (pi, label, pc_seq[label]["answer"], pc_bat[label]["answer"]))
                if pc_seq[label]["rank_transplant"] == pc_bat[label]["rank_transplant"]:
                    _vb_stats["rank_match"] += 1
            per_config = pc_seq                      # record the trusted sequential result
        elif args.batched:
            per_config = _per_config_batched(model, tokenizer, last_R, cache_R, cache_D,
                                             sweep, idx_list, layer_list, cands, cand_tok,
                                             args.max_new_tokens)
        else:
            per_config = _per_config_sequential(model, tokenizer, last_R, cache_R, cache_D,
                                                sweep, idx_list, layer_list, cands, cand_tok,
                                                args.max_new_tokens)
        torch.cuda.empty_cache()

        results.append({
            "pair_idx": pi, "idx_R": R.get("idx", i), "idx_D": D.get("idx", j),
            "candidates": cands,
            "R_correct": ansR_norm == R["answer"], "D_correct": ansD_norm == D["answer"],
            "ansR_norm": ansR_norm, "ansD_norm": ansD_norm,
            "rank_norm": rank_norm, "configs": per_config,
        })
        del cache_R, cache_D
        torch.cuda.empty_cache()

        if (pi + 1) % 10 == 0 or pi == len(pairs) - 1:
            json.dump(results, open(args.output_dir / "varbind_transplant_results.json", "w"), indent=2)

    if args.verify_batched:
        t = _vb_stats["total"]
        am, rm = _vb_stats["answer_match"], _vb_stats["rank_match"]
        print(f"\n{'='*64}\nBATCHED-vs-SEQUENTIAL EQUIVALENCE  ({len(results)} pairs x "
              f"{len(sweep)} configs = {t} comparisons)\n{'='*64}")
        print(f"  answer (argmax) match: {am}/{t} = {am/t:.1%}" if t else "  (no comparisons)")
        print(f"  full rank-dict match : {rm}/{t} = {rm/t:.1%}" if t else "")
        if _vb_stats["mismatch"]:
            print(f"  answer mismatches ({len(_vb_stats['mismatch'])}):")
            for pi_, label, a_seq, a_bat in _vb_stats["mismatch"][:40]:
                print(f"    pair{pi_} [{label}] seq={a_seq} batched={a_bat}")
        verdict = "PASS — batched == sequential" if am == t else "MISMATCH — do NOT use batched"
        print(f"\n  VERDICT: {verdict}")
        return

    # ---- summary ----
    both = [r for r in results if r["R_correct"] and r["D_correct"]]
    print(f"\n{'='*64}\nSUMMARY  (n={len(results)}, both-correct n={len(both)})\n{'='*64}")
    use = both if both else results
    for label, _, _, _ in sweep:
        outs = [r["configs"][label]["outcome"] for r in use]
        n = len(outs)
        frac = {o: outs.count(o) / n for o in set(outs)}
        # median rank improvement of through_y (normal_rank - transplant_rank; + = better)
        shifts = [r["rank_norm"]["through_y"] - r["configs"][label]["rank_transplant"]["through_y"]
                  for r in use
                  if r["rank_norm"]["through_y"] is not None
                  and r["configs"][label]["rank_transplant"]["through_y"] is not None]
        med = float(np.median(shifts)) if shifts else float("nan")
        ty = frac.get("through_y", 0.0); ca = frac.get("carry_answer", 0.0); kp = frac.get("keep", 0.0)
        print(f"  {label:8s} n_swap≈{use[0]['configs'][label]['n_swapped']:>3}  "
              f"through_y={ty:5.1%}  carry_answer={ca:5.1%}  keep={kp:5.1%}  "
              f"| median through_y rank shift={med:+.0f}")
    json.dump(results, open(args.output_dir / "varbind_transplant_results.json", "w"), indent=2)
    print(f"\nsaved {args.output_dir}/varbind_transplant_results.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/workspace/models/deepseek-v3-awq")
    ap.add_argument("--dataset", type=Path, default=Path("data/chained_var_binding_dataset.json"))
    ap.add_argument("--output-dir", type=Path, default=Path("results/varbind_kv_transplant"))
    ap.add_argument("--filler-k", type=int, default=25)
    ap.add_argument("--filler-type", default="dots", choices=["dots", "counting"])
    ap.add_argument("--max-pairs", type=int, default=300)
    ap.add_argument("--sweep", choices=["default", "full"], default="default",
                    help="'default' = the 8-config exploratory sweep; 'full' = full-swap only "
                         "(well-powered headline top-up; the supporting configs are already run).")
    ap.add_argument("--max-new-tokens", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--self-check", type=int, default=5,
                    help="Self-transplant sanity gate: swap N receivers' filler with their OWN "
                         "(must be a no-op). Must pass N/N before trusting results. 0 disables.")
    ap.add_argument("--batched", action="store_true",
                    help="Decode the sweep configs in ONE batched pass per pair (within-pair "
                         "batching; ~2-3x faster). Verify with --verify-batched first.")
    ap.add_argument("--verify-batched", type=int, default=0,
                    help="Run N pairs in BOTH sequential and batched modes and report "
                         "per-config answer/rank equivalence, then exit (no full run).")
    ap.add_argument("--pair-mode", choices=["op_varied", "op_fixed"], default="op_varied",
                    help="op_varied = dissociation (different question op; tests through_y). "
                         "op_fixed = 1-fact-matched control (same question op, vary y only; "
                         "carry_answer on-manifold, through_y N/A).")
    ap.add_argument("--allow-same-c2", action="store_true",
                    help="Drop the c₂_R≠c₂_D requirement (op_varied only; weaker separation).")
    ap.add_argument("--correct-idx-json", default=None,
                    help="JSON list of model-correct example idx to restrict pairs to.")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
