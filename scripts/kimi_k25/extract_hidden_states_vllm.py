#!/usr/bin/env python3
"""All-position hidden-state extraction for Kimi K2.5 on the varbind task (vLLM V1).

Produces pickles with the same layout as data/extracted_states_varbind_allpos/, so
every downstream decode script works unchanged.

Differences from scripts/kimi_k2/extract_hidden_states_vllm.py, all forced by
vLLM >= 0.15 and by K2.5's architecture:

  * The V0 hook path (`llm_engine.model_executor.driver_worker.model_runner.model`)
    is gone. Hooks are installed inside the workers through
    `worker_extension_cls` + `collective_rpc` (see kimi_k25_hooks.py).
  * Layers live at `language_model.model.layers` — K2.5 wraps the text model.
  * The hook captures `output[0] + output[1]` (residual stream), not `output[0]`
    (MLP write). See kimi_k25_hooks.py and RUNBOOK §A6.
  * `enable_prefix_caching` and `enable_chunked_prefill` are forced False. Both
    default on in V1 and both silently corrupt the prefill being captured — the
    few-shot prefix is shared by all 500 examples, so prefix caching would skip
    exactly the forward we want.
  * Prompts come from prompt_k25.build_prompt_k25 (thinking=False, filler in the
    user turn).

Usage:
    scripts/kimi_k25/run_extract.sh            # full run, all 8 conditions
    # or, for the pre-run capture-convention check:
    … extract_hidden_states_vllm.py --conditions dots_10 --max-problems 20 \
        --output-dir data/extracted_states_varbind_allpos_kimi_k25_smoke
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
# The worker extension must be importable by qualified name inside every worker
# process, so its directory goes on sys.path AND on PYTHONPATH (run_extract.sh).
_HOOK_DIR = Path(__file__).resolve().parent
if str(_HOOK_DIR) not in sys.path:
    sys.path.insert(0, str(_HOOK_DIR))

from extract.extract_hidden_states import (  # noqa: E402
    CONDITIONS,
    compute_all_positions,
    compute_baseline_positions,
    find_filler_boundaries,
    problem_metadata,
)
from kimi_k25.prompt_k25 import (  # noqa: E402
    THINK_OPEN,
    build_prompt_k25,
    check_prompt_tail,
    check_thinking_suppressed,
    check_v3_scaffold,
)

RUN_CONDITIONS = ["baseline", "dots_5", "dots_10", "dots_25", "dots_50",
                  "counting_5", "counting_10", "counting_25"]
N_LAYERS = 61
HIDDEN = 7168
RMS_EPS = 1e-5  # K2.5 uses 1e-5; K2 and V3 use 1e-6


# ---------------------------------------------------------------------------
# logit lens (CPU) for the live capture-convention gate
# ---------------------------------------------------------------------------

class LogitLens:
    """RMSNorm -> lm_head on a captured state. lm_head is cast to fp32 ONCE."""

    def __init__(self, weights_dir: Path):
        self.lm_head = np.load(weights_dir / "lm_head_weight.npy").astype(np.float32)
        self.norm_w = np.load(weights_dir / "rms_norm_weight.npy").astype(np.float32)
        print(f"  logit lens: lm_head {self.lm_head.shape}, norm {self.norm_w.shape}, "
              f"eps={RMS_EPS}")

    def top_token(self, h: np.ndarray) -> int:
        x = h.astype(np.float32)
        x = x / np.sqrt((x * x).mean() + RMS_EPS) * self.norm_w
        return int(np.argmax(self.lm_head @ x))

    def layer_profile(self, states: dict, pos_name: str, target: int) -> dict:
        """argmax per layer at one position — the diagnostic that actually
        separates residual stream from layer writes (RUNBOOK §A6)."""
        out = {}
        for li in sorted(states[pos_name].keys()):
            out[li] = int(self.top_token(states[pos_name][li]) == target)
        return out


# ---------------------------------------------------------------------------

def parse_answer(text: str):
    m = re.search(r"Answer:\s*(-?\d+)", text)
    if not m:
        m = re.search(r"(-?\d+)", text)
    return int(m.group(1)) if m else None


def build_llm(args):
    from vllm import LLM

    print(f"\nLoading vLLM TP={args.tensor_parallel_size} from {args.model_path} …")
    t0 = time.time()
    kwargs = dict(
        model=str(args.model_path),
        tokenizer=str(args.model_path),
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,            # hooks must see eager forwards
        trust_remote_code=True,
        enable_prefix_caching=False,   # MUST stay False — shared few-shot prefix
        enable_chunked_prefill=False,  # MUST stay False — would split the prefill
        worker_extension_cls="kimi_k25_hooks.K25CaptureExt",
        disable_custom_all_reduce=True,
    )
    if args.quantization:
        kwargs["quantization"] = args.quantization
    llm = LLM(**kwargs)
    print(f"Loaded in {time.time() - t0:.0f}s")
    return llm


def install_hooks(llm) -> None:
    infos = llm.collective_rpc("k25_install_hooks")
    n = {i["n_layers"] for i in infos}
    print(f"  hooks installed on {len(infos)} worker(s); n_layers={n}; "
          f"layer_cls={infos[0]['layer_cls']}; model_cls={infos[0]['model_cls']}")
    if n != {N_LAYERS}:
        raise SystemExit(f"GATE FAIL: expected {N_LAYERS} layers per worker, got {n}")


def run_one(llm, ids: list[int], positions: dict, sampling):
    """Arm hooks, run one request, return (states, generated_text, token_ids)."""
    from vllm.inputs import TokensPrompt

    pos_list = [positions[p] for p in positions]
    llm.collective_rpc("k25_set_positions", args=(pos_list, len(ids)))
    outs = llm.generate([TokensPrompt(prompt_token_ids=ids)], sampling, use_tqdm=False)
    payloads = llm.collective_rpc("k25_pop")

    errors = [e for p in payloads for e in p["errors"]]
    if errors:
        raise SystemExit("CAPTURE FAIL:\n  " + "\n  ".join(errors[:10]))
    counts = {p["n_captured"] for p in payloads}
    if counts != {N_LAYERS}:
        raise SystemExit(f"CAPTURE FAIL: layers captured per rank = {counts}, "
                         f"expected {{{N_LAYERS}}}")

    p0 = payloads[0]
    if p0.get("data") is None:
        raise SystemExit(f"CAPTURE FAIL: rank-0 payload carried no data (rank={p0['rank']}, "
                         f"n_captured={p0['n_captured']})")
    layers = p0["layers"]
    n_layers, n_pos, hidden = p0["shape"]
    if n_pos != len(positions) or n_layers != N_LAYERS or hidden != HIDDEN:
        raise SystemExit(
            f"CAPTURE FAIL: worker returned shape {p0['shape']} but expected "
            f"({N_LAYERS}, {len(positions)}, {HIDDEN}) — positions requested="
            f"{len(pos_list)}, seq_len={len(ids)}")
    arr = np.frombuffer(p0["data"], dtype=np.float16).reshape(n_layers, n_pos, hidden)
    # regroup to {pos_name: {layer_idx: vec}} — the reference pkl schema
    states = {name: {li: arr[i, j] for i, li in enumerate(layers)}
              for j, name in enumerate(positions)}
    o = outs[0].outputs[0]
    return states, o.text.strip(), list(o.token_ids)


# ---------------------------------------------------------------------------

def live_gates(llm, tok, problems, few_shot, args, dataset_type: str = "varbind") -> dict:
    """Gates 2 and 4 plus the layer profile, on a handful of examples."""
    from vllm import SamplingParams

    print("\n" + "=" * 70 + "\nLIVE PREFLIGHT GATES\n" + "=" * 70)
    lens = LogitLens(args.weights_dir)
    sampling = SamplingParams(temperature=0, max_tokens=args.max_gen_tokens, detokenize=True)

    results = {"gate2_no_think": [], "gate4_logit_lens": [], "layer_profile": {}}
    k, filler_type = CONDITIONS["dots_10"]
    profiles = []

    for i in range(args.n_gate_examples):
        rng = random.Random(i)
        text, ids = build_prompt_k25(tok, few_shot, problems[i], filler_type, k, rng=rng,
                                     dataset_type=dataset_type)

        ok_think, msg_think = check_thinking_suppressed(ids)
        ok_tail, msg_tail = check_prompt_tail(text)
        ok_scaf, msg_scaf = check_v3_scaffold(text, k)
        if not (ok_think and ok_tail and ok_scaf):
            raise SystemExit(f"GATE 1 FAIL on example {i}: {msg_think} | {msg_tail} | {msg_scaf}")

        qe, fs, fe = find_filler_boundaries(tok, torch.tensor([ids]), k)
        positions, boundaries = compute_all_positions(qe, len(ids), filler_start=fs,
                                                      filler_end=fe)
        states, gen_text, gen_ids = run_one(llm, ids, positions, sampling)

        # --- Gate 2: no <think> anywhere in the free-running output
        leaked = THINK_OPEN in gen_ids
        results["gate2_no_think"].append(not leaked)
        if leaked:
            raise SystemExit(f"GATE 2 FAIL: <think> ({THINK_OPEN}) in generation "
                             f"for example {i}: {gen_ids}")

        # --- shape / NaN checks
        ap = states["pos_%03d" % (len(positions) - 1)]
        arr = ap[N_LAYERS - 1]
        if arr.shape != (HIDDEN,):
            raise SystemExit(f"GATE FAIL: last-layer state shape {arr.shape} != ({HIDDEN},)")
        for name, layer_dict in states.items():
            for li, v in layer_dict.items():
                if not np.isfinite(v.astype(np.float32)).all():
                    raise SystemExit(f"GATE FAIL: non-finite state at {name} L{li}")

        # --- Gate 4: last-layer state at answer_prompt decodes to what vLLM emitted
        if not gen_ids:
            raise SystemExit(f"GATE FAIL: example {i} generated zero tokens "
                             f"(text={gen_text!r}) — nothing to compare the lens against")
        first_tok = gen_ids[0]
        pred = lens.top_token(arr)
        results["gate4_logit_lens"].append(int(pred == first_tok))
        profiles.append(lens.layer_profile(states, "pos_%03d" % (len(positions) - 1),
                                           first_tok))
        print(f"  ex{i}: gen={gen_text[:24]!r} first_tok={first_tok} "
              f"lens_argmax={pred} {'MATCH' if pred == first_tok else 'MISMATCH'} "
              f"| n_pos={len(positions)}")

    # --- layer profile: gradual emergence == residual stream
    agg = {li: float(np.mean([p[li] for p in profiles])) for li in profiles[0]}
    results["layer_profile"] = agg
    g4 = float(np.mean(results["gate4_logit_lens"]))
    print(f"\nGATE 4 (final-layer logit lens): {g4:.0%} match "
          f"({sum(results['gate4_logit_lens'])}/{len(results['gate4_logit_lens'])})")
    print("layer profile (fraction matching the emitted token):")
    for li in sorted(agg):
        if li >= N_LAYERS - 12 or li % 10 == 0:
            print(f"  L{li:2d}: {agg[li]:.0%}")

    late = [agg[li] for li in range(N_LAYERS - 10, N_LAYERS - 1)]
    print(f"\nmean over L{N_LAYERS - 10}..L{N_LAYERS - 2} = {np.mean(late):.1%}  "
          f"(residual stream => gradual emergence, well above 0%; "
          f"layer writes => ~0% until the final layer)")
    if g4 < 0.8:
        raise SystemExit(f"GATE 4 FAIL: only {g4:.0%} of final-layer states decode to "
                         "the emitted token — capture convention is wrong")
    print("LIVE GATES 2 + 4 PASSED")
    return results


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=Path, default=Path("/workspace/models/kimi-k2.5"))
    ap.add_argument("--dataset", type=Path, nargs="+",
                    default=[Path("data/chained_var_binding_easy_dataset.json")],
                    help="One or more dataset JSONs. Multiple = extracted sequentially "
                         "from a SINGLE vLLM load (the ~7.5 min load dominates otherwise).")
    ap.add_argument("--output-dir", type=Path, nargs="+",
                    default=[Path("data/extracted_states_varbind_allpos_kimi_k25")],
                    help="One output dir per --dataset (1:1).")
    ap.add_argument("--dataset-type", nargs="+", default=["varbind"],
                    choices=["1hop", "2fact", "letterpos", "varbind"],
                    help="One per --dataset, or a single value broadcast to all.")
    ap.add_argument("--weights-dir", type=Path, default=Path("data/model_weights/kimi_k25"))
    ap.add_argument("--results-json", type=Path,
                    default=Path("results/kimi_k25_varbind_accuracy.json"))
    ap.add_argument("--conditions", nargs="+", default=RUN_CONDITIONS)
    ap.add_argument("--max-problems", type=int, default=None)
    ap.add_argument("--tensor-parallel-size", type=int, default=8)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--max-gen-tokens", type=int, default=20)
    ap.add_argument("--quantization", default=None,
                    help="Leave unset to auto-detect from text_config.quantization_config")
    ap.add_argument("--n-gate-examples", type=int, default=5)
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--skip-gates", action="store_true",
                    help="Resume an interrupted run whose gates already passed")
    ap.add_argument("--no-skip-existing", action="store_true")
    args = ap.parse_args()

    for c in args.conditions:
        if c not in CONDITIONS:
            raise SystemExit(f"unknown condition {c!r}")

    # Pair up (dataset, output_dir, dataset_type) 1:1
    if len(args.output_dir) != len(args.dataset):
        ap.error(f"--output-dir count ({len(args.output_dir)}) must match "
                 f"--dataset count ({len(args.dataset)})")
    if len(args.dataset_type) == 1:
        dataset_types = list(args.dataset_type) * len(args.dataset)
    elif len(args.dataset_type) == len(args.dataset):
        dataset_types = list(args.dataset_type)
    else:
        ap.error(f"--dataset-type count ({len(args.dataset_type)}) must be 1 or "
                 f"equal --dataset count ({len(args.dataset)})")

    def load_dataset(path: Path):
        data = json.loads(path.read_text())
        problems = data["examples"] if isinstance(data, dict) else data
        few = ((data.get("few_shot_examples") or data.get("few_shot_facts") or [])
               if isinstance(data, dict) else [])[:5]
        if args.max_problems:
            problems = problems[:args.max_problems]
        return problems, few

    def results_path_for(output_dir: Path) -> Path:
        """One flat {condition: {...}} file per dataset. With a single dataset this
        is exactly --results-json, so the varbind run's filename is unchanged."""
        if len(args.dataset) == 1:
            return args.results_json
        return args.results_json.parent / f"kimi_k25_{output_dir.name}_accuracy.json"

    first_problems, first_few_shot = load_dataset(args.dataset[0])
    print(f"dataset {args.dataset[0]} ({dataset_types[0]}): "
          f"{len(first_problems)} problems, {len(first_few_shot)} few-shot")

    # Load the tokenizer through transformers directly rather than
    # llm.get_tokenizer(), so prompts are byte-identical to the ones the offline
    # preflight already validated (vLLM may hand back a wrapped tokenizer).
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)

    llm = build_llm(args)
    install_hooks(llm)

    gate_results = None
    if not args.skip_gates:
        gate_results = live_gates(llm, tok, first_problems, first_few_shot, args,
                                  dataset_type=dataset_types[0])
        args.results_json.parent.mkdir(parents=True, exist_ok=True)
        # Name the gates file after the run, not a fixed "kimi_k25_gates.json" —
        # that fixed name silently overwrote the varbind run's gate record when a
        # second run (1hop/2fact) reused the same results dir.
        gates_path = (args.results_json.parent /
                      f"kimi_k25_gates_{'_'.join(dict.fromkeys(dataset_types))}.json")
        gates_path.write_text(json.dumps(gate_results, indent=2))
        print(f"  gates written to {gates_path}")
    if args.preflight_only:
        print("\n--preflight-only: stopping before extraction")
        return

    from vllm import SamplingParams
    sampling = SamplingParams(temperature=0, max_tokens=args.max_gen_tokens, detokenize=True)

    for ds_idx, (dataset_path, output_dir, dataset_type) in enumerate(
            zip(args.dataset, args.output_dir, dataset_types)):
        print(f"\n{'#' * 70}\n# Dataset {ds_idx + 1}/{len(args.dataset)}: {dataset_path} "
              f"(type={dataset_type})\n# Output: {output_dir}\n{'#' * 70}")
        if ds_idx == 0:
            problems, few_shot = first_problems, first_few_shot
        else:
            problems, few_shot = load_dataset(dataset_path)
            print(f"  {len(problems)} problems, {len(few_shot)} few-shot")

        results_json = results_path_for(output_dir)
        run_dataset(llm, tok, problems, few_shot, dataset_type, output_dir,
                    results_json, args)

    print("\nDone.")


def _extra_metadata(p: dict, dataset_type: str) -> dict:
    """Task-specific fields that problem_metadata() does not already carry, matching
    what scripts/extract/extract_hidden_states.py writes into metadata.json."""
    if dataset_type == "2fact":
        return {"fact_phrase_1": p["fact_phrase_1"], "fact_phrase_2": p["fact_phrase_2"]}
    if dataset_type == "varbind":
        return {"queried_term": p["queried_term"], "definitions": p["definitions"],
                "question": p["question"]}
    if dataset_type == "letterpos":
        return {}
    return {"fact_phrase": p["fact_phrase"], "kind": p["kind"]}


def run_dataset(llm, tok, problems, few_shot, dataset_type, output_dir,
                results_json, args) -> None:
    """Extract every requested condition for one dataset."""
    from vllm import SamplingParams
    sampling = SamplingParams(temperature=0, max_tokens=args.max_gen_tokens, detokenize=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = [{"idx": i, **problem_metadata(p, dataset_type),
                 **_extra_metadata(p, dataset_type)}
                for i, p in enumerate(problems)]
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    accuracy = {}
    if results_json.exists():
        accuracy = json.loads(results_json.read_text())

    for cond_name in args.conditions:
        k, filler_type = CONDITIONS[cond_name]
        cond_dir = output_dir / cond_name
        cond_dir.mkdir(exist_ok=True)
        print(f"\n{'=' * 60}\nCondition {cond_name} (k={k}, {filler_type})\n{'=' * 60}")

        todo = [i for i in range(len(problems))
                if args.no_skip_existing or not (cond_dir / f"prob_{i:04d}.pkl").exists()]
        if not todo:
            print(f"  all {len(problems)} already extracted, skipping")
            continue
        print(f"  {len(todo)} to extract")

        t0, correct, total = time.time(), 0, 0
        for prob_idx in tqdm(todo, desc=cond_name):
            problem = problems[prob_idx]
            rng = random.Random(prob_idx)
            text, ids = build_prompt_k25(tok, few_shot, problem, filler_type, k, rng=rng,
                                         dataset_type=dataset_type)
            seq_len = len(ids)

            filler_start = filler_end = question_end = None
            boundaries = None
            if k > 0:
                question_end, filler_start, filler_end = find_filler_boundaries(
                    tok, torch.tensor([ids]), k)
                positions, boundaries = compute_all_positions(
                    question_end, seq_len, filler_start=filler_start,
                    filler_end=filler_end)
            else:
                positions = compute_baseline_positions(seq_len)
            positions = {p: i for p, i in positions.items() if 0 <= i < seq_len}

            states, gen_text, _ = run_one(llm, ids, positions, sampling)
            model_answer = parse_answer(gen_text)
            model_correct = (model_answer == problem["answer"]) if model_answer is not None else False
            total += 1
            correct += int(model_correct)

            result = {
                "problem_idx": prob_idx,
                "condition": cond_name,
                "k": k,
                **problem_metadata(problem, dataset_type),
                "positions": positions,
                "states": states,
                "model_response": gen_text,
                "model_answer": model_answer,
                "model_correct": model_correct,
                "seq_len": seq_len,
                "filler_start": filler_start,
                "filler_end": filler_end,
                "gen_prefix": None,
            }
            if boundaries is not None:
                result["boundaries"] = boundaries
            with open(cond_dir / f"prob_{prob_idx:04d}.pkl", "wb") as f:
                pickle.dump(result, f)

        elapsed = time.time() - t0
        acc = correct / total if total else float("nan")
        accuracy[cond_name] = {"correct": correct, "total": total, "accuracy": acc,
                               "seconds": round(elapsed, 1)}
        print(f"  {cond_name}: {correct}/{total} = {acc:.1%} in {elapsed:.0f}s "
              f"({elapsed / max(total, 1):.2f}s/example)")
        results_json.parent.mkdir(parents=True, exist_ok=True)
        results_json.write_text(json.dumps(accuracy, indent=2))

    print("\n" + "=" * 70 + "\nACCURACY\n" + "=" * 70)
    base = accuracy.get("baseline", {}).get("accuracy")
    for c in args.conditions:
        if c in accuracy:
            a = accuracy[c]["accuracy"]
            up = f"{(a - base) * 100:+.1f} pt" if base is not None else ""
            print(f"  {c:12s} {a:6.1%}  ({accuracy[c]['correct']}/{accuracy[c]['total']}) {up}")


if __name__ == "__main__":
    main()
