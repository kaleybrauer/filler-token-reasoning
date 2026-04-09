"""
probe_intervention.py

Probe-guided causal intervention: use the linear direction learned by a probe
to additively shift the model's internal representation of A during filler
tokens, then test whether the model's final answer shifts accordingly.

If the model reads A from the probe's direction to produce its answer,
adding delta * steering_vector to the residual stream should shift the
answer by ~delta.

Usage:
    source /workspace/config/probing_env.sh
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python scripts/probe_intervention.py \
        --model-path /workspace/models/deepseek-v3-awq \
        --output-dir intervention_results \
        --layer 58 \
        --position answer_prompt \
        --max-examples 50

    # Quick sanity test
    python scripts/probe_intervention.py \
        --model-path /workspace/models/deepseek-v3-awq \
        --output-dir intervention_results/test \
        --max-examples 5 --deltas -20,0,20
"""

import argparse
import json
import pickle
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

# Patch: autoawq imports PytorchGELUTanh which was renamed
import transformers.activations as _act
if not hasattr(_act, "PytorchGELUTanh"):
    _act.PytorchGELUTanh = _act.GELUTanh

from extract_hidden_states import (
    build_messages_for_condition,
    find_filler_boundaries,
    compute_extraction_positions,
    load_model,
)
from generate_1hop_dataset import build_system_message, build_user_turn


# ==============================================================================
# Probe fitting (to get steering vector)
# ==============================================================================

def load_condition_states(extraction_dir: Path, condition_name: str, layer: int, position: str):
    """Load hidden states for one (condition, layer, position) from extracted pickles."""
    cond_dir = extraction_dir / condition_name
    files = sorted(cond_dir.glob("prob_*.pkl"))
    if not files:
        raise FileNotFoundError(f"No files found in {cond_dir}")

    vectors = []
    metadata = []
    for f in files:
        with open(f, "rb") as fp:
            data = pickle.load(fp)
        if position not in data["states"]:
            raise KeyError(f"Position '{position}' not in {f.name}. "
                           f"Available: {list(data['states'].keys())}")
        if layer not in data["states"][position]:
            raise KeyError(f"Layer {layer} not in {f.name}. "
                           f"Available: {list(data['states'][position].keys())}")
        vec = data["states"][position][layer]
        vectors.append(vec)
        metadata.append({
            "problem_idx": data["problem_idx"],
            "fact_value": data["fact_value"],
            "x": data["x"],
            "answer": data["answer"],
            "model_correct": data["model_correct"],
            "model_answer": data.get("model_answer"),
        })

    X = np.stack(vectors, axis=0).astype(np.float32)
    print(f"  Loaded {len(X)} states from {condition_name}, "
          f"layer={layer}, position={position}, shape={X.shape}")
    return X, metadata


def fit_steering_probe(X, y, target_name="A", test_frac=0.2, seed=42):
    """
    Fit RidgeCV on a train split to get the probe direction, validate on
    a held-out test split.

    Using the full dataset gives R²=1.0 (memorization with 7168 features
    and 500 examples). Train/test split ensures the direction generalizes.

    Returns:
        steering_direction: np.ndarray (hidden_dim,) in raw hidden-state space.
            Adding delta * steering_direction to h shifts probe(h) by delta.
        probe_info: dict with diagnostics (r2_train, r2_test, alpha, etc.)
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import r2_score, mean_absolute_error
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_frac, random_state=seed
    )
    print(f"  Train/test split: {len(X_train)} train, {len(X_test)} test")

    # Normalize using TRAIN statistics only
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std < 1e-8] = 1.0
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std

    alphas = np.logspace(-2, 6, 20)
    probe = RidgeCV(alphas=alphas, cv=5)
    probe.fit(X_train_norm, y_train)

    y_pred_train = probe.predict(X_train_norm)
    y_pred_test = probe.predict(X_test_norm)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)

    # Un-normalize: probe predicts w_norm . ((h - mean) / std) + intercept
    # In raw space: w_raw = w_norm / std
    w_norm = probe.coef_  # (hidden_dim,)
    w_raw = w_norm / std  # direction in raw hidden-state space

    # steering_direction: adding delta * steering_direction shifts w_raw . h by delta
    # w_raw . (h + delta * v) = w_raw . h + delta * (w_raw . v) = ... + delta
    # So we need w_raw . v = 1, minimal norm: v = w_raw / ||w_raw||^2
    w_raw_norm_sq = np.dot(w_raw, w_raw)
    steering_direction = w_raw / w_raw_norm_sq

    # Verify: w_raw . steering_direction should be ~1.0
    dot_check = np.dot(w_raw, steering_direction)

    print(f"  Probe fit for {target_name}:")
    print(f"    Train: R²={r2_train:.4f}, MAE={mae_train:.2f}")
    print(f"    Test:  R²={r2_test:.4f}, MAE={mae_test:.2f}")
    print(f"    alpha={probe.alpha_:.1f}")
    print(f"    ||w_raw||={np.linalg.norm(w_raw):.4f}, "
          f"||steering||={np.linalg.norm(steering_direction):.6f}, "
          f"dot check={dot_check:.6f} (should be 1.0)")

    return steering_direction.astype(np.float32), {
        "target": target_name,
        "r2_train": float(r2_train),
        "r2_test": float(r2_test),
        "mae_train": float(mae_train),
        "mae_test": float(mae_test),
        "alpha": float(probe.alpha_),
        "w_raw_norm": float(np.linalg.norm(w_raw)),
        "steering_norm": float(np.linalg.norm(steering_direction)),
        "dot_check": float(dot_check),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "mean": mean,
        "std": std,
    }


# ==============================================================================
# Steering hook
# ==============================================================================

class SteeringHook:
    """
    Forward hook that additively perturbs the residual stream at specified
    token positions in a single layer.
    """

    def __init__(self, model, layer_idx: int, positions: List[int],
                 steering_vector: torch.Tensor):
        """
        Args:
            layer_idx: which layer to intervene at
            positions: absolute token positions to perturb
            steering_vector: tensor of shape (hidden_dim,) to ADD to the
                residual stream (already scaled by delta)
        """
        self.model = model
        self.layer_idx = layer_idx
        self.positions = positions
        self.steering_vector = steering_vector
        self._hook = None

    def _hook_fn(self, module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        # Only during prefill (full sequence), not autoregressive generation
        if hidden.shape[1] <= 1:
            return output

        hidden = hidden.clone()
        device, dtype = hidden.device, hidden.dtype
        sv = self.steering_vector.to(device=device, dtype=dtype)

        for pos in self.positions:
            if pos < hidden.shape[1]:
                hidden[0, pos] += sv

        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    def register(self):
        self.remove()
        module = self.model.model.layers[self.layer_idx]
        self._hook = module.register_forward_hook(self._hook_fn)

    def remove(self):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None


# ==============================================================================
# Generation helpers
# ==============================================================================

def parse_answer(raw: str) -> Optional[int]:
    """Parse numeric answer from model output."""
    m = re.search(r"Answer:\s*(-?\d+)", raw)
    if m:
        return int(m.group(1))
    m = re.search(r"(-?\d+)", raw)
    if m:
        return int(m.group(1))
    return None


def generate_answer(model, tokenizer, input_ids, attention_mask,
                    hook=None) -> Tuple[str, Optional[int]]:
    """Generate an answer, optionally with a steering hook active."""
    if hook is not None:
        hook.register()
    try:
        with torch.no_grad():
            gen_output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=20,
                do_sample=False,
                use_cache=True,
            )
    finally:
        if hook is not None:
            hook.remove()

    new_tokens = gen_output[0][input_ids.shape[1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    answer = parse_answer(raw)
    return raw, answer


# ==============================================================================
# Main experiment
# ==============================================================================

def get_intervention_positions(position_name: str, filler_start: int,
                               filler_end: int, seq_len: int) -> List[int]:
    """Convert a position name to a list of absolute token positions to steer at.

    Supports:
        answer_prompt — last token before generation
        question_end — last token before filler
        filler_0.50 — fraction-based (50% through filler)
        filler_k25 — absolute offset (25th filler token)
        all_filler — every filler token
        early_filler — first 25% of filler tokens
    """
    if position_name == "answer_prompt":
        return [seq_len - 1]
    elif position_name == "question_end":
        return [filler_start - 1]
    elif position_name.startswith("filler_k"):
        # Absolute offset: filler_k25 = 25th filler token (1-indexed)
        k = int(position_name.split("_k")[1])
        idx = filler_start + k - 1
        return [min(idx, filler_end)]
    elif position_name.startswith("filler_"):
        frac = float(position_name.split("_")[1])
        filler_len = filler_end - filler_start + 1
        idx = filler_start + int(frac * (filler_len - 1))
        return [min(idx, filler_end)]
    elif position_name == "all_filler":
        return list(range(filler_start, filler_end + 1))
    elif position_name == "early_filler":
        # First 25% of filler tokens
        filler_len = filler_end - filler_start + 1
        end = filler_start + max(1, filler_len // 4)
        return list(range(filler_start, end))
    else:
        raise ValueError(f"Unknown position: {position_name}")


def run_experiment(
    model,
    tokenizer,
    problems: list,
    few_shot: list,
    example_indices: List[int],
    steering_direction: np.ndarray,
    layer: int,
    position: str,
    deltas: List[float],
    output_dir: Path,
    filler_type: str = "dots",
    k: int = 250,
    control_type: Optional[str] = None,
    random_seed: int = 0,
):
    """
    Run the steering intervention experiment.

    For each example and delta, add delta * steering_direction to the residual
    stream at the specified layer/position, generate, and record the answer shift.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    first_param = next(model.parameters())
    input_device = first_param.device

    prob_by_idx = {p["idx"]: p for p in problems}

    # Prepare steering vector
    sv_np = steering_direction.copy()
    if control_type == "random":
        rng = np.random.RandomState(random_seed)
        sv_np = rng.randn(len(sv_np)).astype(np.float32)
        sv_np = sv_np / np.linalg.norm(sv_np) * np.linalg.norm(steering_direction)
        print(f"  Using RANDOM direction (seed={random_seed}), "
              f"same norm as steering vector")

    # Load checkpoint
    checkpoint_path = output_dir / "checkpoint.pkl"
    if checkpoint_path.exists():
        with open(checkpoint_path, "rb") as f:
            results = pickle.load(f)
        print(f"Loaded checkpoint with {len(results)} results")
    else:
        results = []

    completed_keys = set()
    for r in results:
        completed_keys.add((r["problem_idx"], r["delta"]))

    total = len(example_indices) * len(deltas)
    done = len(completed_keys)
    t0 = time.time()

    for ex_i, prob_idx in enumerate(example_indices):
        problem = prob_by_idx[prob_idx]

        # Build prompt
        example_rng = random.Random(prob_idx)
        messages = build_messages_for_condition(
            few_shot[:5], problem, filler_type, k, rng=example_rng
        )
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(full_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(input_device)
        attention_mask = inputs["attention_mask"].to(input_device)
        seq_len = input_ids.shape[1]

        # Find filler boundaries
        filler_start, filler_end = find_filler_boundaries(tokenizer, input_ids, k)

        # Get intervention positions
        steer_positions = get_intervention_positions(
            position, filler_start, filler_end, seq_len
        )

        for delta in deltas:
            key = (prob_idx, delta)
            if key in completed_keys:
                continue

            if delta == 0:
                # No hook needed for null
                raw, answer = generate_answer(
                    model, tokenizer, input_ids, attention_mask
                )
            else:
                scaled_sv = torch.from_numpy(sv_np * delta)
                hook = SteeringHook(model, layer, steer_positions, scaled_sv)
                raw, answer = generate_answer(
                    model, tokenizer, input_ids, attention_mask, hook=hook
                )

            result = {
                "problem_idx": prob_idx,
                "fact_value": problem["fact_value"],
                "x": problem["x"],
                "true_answer": problem["answer"],
                "delta": delta,
                "raw_response": raw,
                "steered_answer": answer,
                "layer": layer,
                "position": position,
                "n_steered_positions": len(steer_positions),
                "control_type": control_type,
            }
            results.append(result)
            completed_keys.add(key)
            done += 1

        # Checkpoint every 5 examples
        if (ex_i + 1) % 5 == 0 or ex_i == len(example_indices) - 1:
            with open(checkpoint_path, "wb") as f:
                pickle.dump(results, f)

            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            remaining = (total - done) / rate if rate > 0 else 0
            print(f"  [{done}/{total}] {elapsed:.0f}s elapsed, "
                  f"~{remaining:.0f}s remaining")

        del input_ids, attention_mask
        torch.cuda.empty_cache()

    # Save final results
    with open(output_dir / "intervention_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone: {len(results)} results saved to {output_dir}/")
    return results


# ==============================================================================
# Analysis and plotting
# ==============================================================================

def analyze_results(results: List[dict], output_dir: Path):
    """Compute per-delta aggregate metrics and overall slope."""

    # Group by delta
    by_delta = defaultdict(list)
    for r in results:
        if r["steered_answer"] is not None:
            by_delta[r["delta"]].append(r)

    # Get natural answers (delta=0)
    natural_answers = {}
    for r in by_delta.get(0, []):
        natural_answers[r["problem_idx"]] = r["steered_answer"]

    if not natural_answers:
        print("WARNING: No delta=0 results found — cannot compute shifts")
        return {}

    # Compute per-example shifts
    all_deltas = []
    all_shifts = []
    per_delta_summary = {}

    sorted_deltas = sorted(by_delta.keys())
    print(f"\n{'Delta':>8} {'N':>4} {'MeanShift':>10} {'MedianShift':>12} "
          f"{'Parse%':>7} {'GarbageN':>9}")
    print("-" * 60)

    for delta in sorted_deltas:
        items = by_delta[delta]
        shifts = []
        n_garbage = 0
        for r in items:
            nat = natural_answers.get(r["problem_idx"])
            if nat is None:
                continue
            shift = r["steered_answer"] - nat
            shifts.append(shift)
            all_deltas.append(delta)
            all_shifts.append(shift)
            # "Garbage" = answer shifted by more than 3x delta or wrong sign
            if delta != 0 and (abs(shift) > 3 * abs(delta) or
                               (shift != 0 and np.sign(shift) != np.sign(delta))):
                n_garbage += 1

        n_total = len([r for r in results if r["delta"] == delta])
        n_parsed = len(items)
        mean_shift = np.mean(shifts) if shifts else float("nan")
        median_shift = np.median(shifts) if shifts else float("nan")

        print(f"{delta:>8.0f} {n_parsed:>4} {mean_shift:>10.2f} "
              f"{median_shift:>12.2f} {n_parsed/max(n_total,1):>7.0%} "
              f"{n_garbage:>9}")

        per_delta_summary[str(delta)] = {
            "n": n_parsed,
            "mean_shift": float(mean_shift),
            "median_shift": float(median_shift),
            "n_garbage": n_garbage,
        }

    # Linear regression: shift vs delta
    all_deltas = np.array(all_deltas)
    all_shifts = np.array(all_shifts)

    if len(all_deltas) > 2 and np.std(all_deltas) > 0:
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            all_deltas, all_shifts
        )
        r2 = r_value ** 2
        print(f"\nLinear regression: shift = {slope:.4f} * delta + {intercept:.2f}")
        print(f"  R² = {r2:.4f}, p = {p_value:.2e}, slope SE = {std_err:.4f}")
        print(f"  Ideal slope = 1.0, actual = {slope:.4f}")
    else:
        slope, intercept, r2, p_value = None, None, None, None

    summary = {
        "per_delta": per_delta_summary,
        "slope": float(slope) if slope is not None else None,
        "intercept": float(intercept) if intercept is not None else None,
        "r2": float(r2) if r2 is not None else None,
        "p_value": float(p_value) if p_value is not None else None,
        "n_examples": len(natural_answers),
        "n_datapoints": len(all_deltas),
    }

    with open(output_dir / "analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def plot_results(results: List[dict], summary: dict, output_dir: Path):
    """Generate scatter and summary plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    # Get natural answers
    natural_answers = {}
    for r in results:
        if r["delta"] == 0 and r["steered_answer"] is not None:
            natural_answers[r["problem_idx"]] = r["steered_answer"]

    # Collect all (delta, shift) pairs
    deltas_arr = []
    shifts_arr = []
    for r in results:
        if r["steered_answer"] is None or r["delta"] == 0:
            continue
        nat = natural_answers.get(r["problem_idx"])
        if nat is None:
            continue
        deltas_arr.append(r["delta"])
        shifts_arr.append(r["steered_answer"] - nat)

    deltas_arr = np.array(deltas_arr)
    shifts_arr = np.array(shifts_arr)

    if len(deltas_arr) == 0:
        print("No data points for plotting")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Scatter of delta vs shift
    ax = axes[0]
    ax.scatter(deltas_arr, shifts_arr, alpha=0.3, s=20, color="#4CAF50",
               edgecolors="white", linewidth=0.3)

    # Add regression line
    if summary.get("slope") is not None:
        x_range = np.array([deltas_arr.min(), deltas_arr.max()])
        ax.plot(x_range, summary["slope"] * x_range + summary["intercept"],
                "r-", linewidth=2,
                label=f"slope={summary['slope']:.3f}, R²={summary['r2']:.3f}")

    # Ideal line
    lim = max(abs(deltas_arr).max(), abs(shifts_arr).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.3, label="ideal (slope=1)")
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.3)
    ax.axvline(x=0, color="gray", linestyle=":", alpha=0.3)

    ax.set_xlabel("Delta (steering magnitude)")
    ax.set_ylabel("Answer shift (steered - natural)")
    ax.set_title("Per-example: steering delta vs answer shift")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 2: Mean/median shift per delta
    ax = axes[1]
    per_delta = summary.get("per_delta", {})
    if per_delta:
        ds = sorted([float(d) for d in per_delta.keys()])
        # Keys may be stored as "0.0", "-20.0", etc. — look up by original key
        key_map = {float(k): k for k in per_delta.keys()}
        means = [per_delta[key_map[d]]["mean_shift"] for d in ds]
        medians = [per_delta[key_map[d]]["median_shift"] for d in ds]

        ax.plot(ds, means, "o-", color="#4CAF50", linewidth=2,
                markersize=8, label="Mean shift")
        ax.plot(ds, medians, "s--", color="#2196F3", linewidth=2,
                markersize=6, label="Median shift")
        ax.plot(ds, ds, "k--", alpha=0.3, label="ideal (shift=delta)")
        ax.axhline(y=0, color="gray", linestyle=":", alpha=0.3)
        ax.axvline(x=0, color="gray", linestyle=":", alpha=0.3)

    ax.set_xlabel("Delta")
    ax.set_ylabel("Answer shift")
    ax.set_title("Aggregate: mean/median shift per delta")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Add text with layer/position info
    sample = results[0]
    ctrl = sample.get("control_type") or "steering"
    fig.suptitle(
        f"Probe-guided intervention — Layer {sample['layer']}, "
        f"position={sample['position']}, type={ctrl}",
        fontsize=13, y=1.02,
    )

    plt.tight_layout()
    plt.savefig(output_dir / "intervention_results.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "intervention_results.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved plots to {output_dir}/")


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Probe-guided causal intervention")
    parser.add_argument("--model-path", type=str,
                        default="/workspace/models/deepseek-v3-awq")
    parser.add_argument("--dataset", type=Path,
                        default=Path("data/1hop_addition_dataset.json"))
    parser.add_argument("--categories", type=Path,
                        default=Path("probe_results/categories.json"))
    parser.add_argument("--extraction-dir", type=Path,
                        default=Path("data/extracted_states"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("intervention_results"))

    # Probe / steering config
    parser.add_argument("--condition", type=str, default="dots_250",
                        help="Condition to load extracted states from")
    parser.add_argument("--layer", type=int, default=58,
                        help="Layer to intervene at")
    parser.add_argument("--position", type=str, default="answer_prompt",
                        help="Position to steer at (answer_prompt, filler_0.50, "
                             "all_filler, early_filler, etc.)")
    parser.add_argument("--target", type=str, default="A",
                        choices=["A", "Y", "sum"],
                        help="Which value's probe direction to use for steering")

    # Experiment config
    parser.add_argument("--deltas", type=str,
                        default="-50,-30,-20,-10,-5,0,5,10,20,30,50",
                        help="Comma-separated delta values")
    parser.add_argument("--category", type=str, default="filler_helped",
                        help="Category of examples to use")
    parser.add_argument("--max-examples", type=int, default=50)

    # Filler config
    parser.add_argument("--filler-type", type=str, default="dots")
    parser.add_argument("--filler-k", type=int, default=250)

    # Controls
    parser.add_argument("--control", type=str, default=None,
                        choices=["random", None],
                        help="Control type: 'random' uses a random direction")
    parser.add_argument("--random-seed", type=int, default=0,
                        help="Seed for random direction control")

    # Probe position (for loading states — may differ from intervention position)
    parser.add_argument("--probe-position", type=str, default=None,
                        help="Position to load states for probe fitting "
                             "(default: same as --position). Use this when "
                             "intervening at all_filler but fitting probe on "
                             "answer_prompt.")

    args = parser.parse_args()

    # Parse deltas
    deltas = [float(d) for d in args.deltas.split(",")]
    print(f"Deltas: {deltas}")

    # Determine probe position (which extracted states to use for fitting)
    probe_position = args.probe_position or args.position
    # Map multi-token positions to a single position for probe fitting
    if probe_position in ("all_filler", "early_filler"):
        probe_position = "answer_prompt"
        print(f"  Probe fitted on position=answer_prompt "
              f"(intervention at {args.position})")

    # =========================================================================
    # Step 1: Fit steering probe from extracted states
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 1: Fitting steering probe")
    print("=" * 60)

    X, meta = load_condition_states(
        args.extraction_dir, args.condition, args.layer, probe_position
    )

    if args.target == "A":
        y = np.array([m["fact_value"] for m in meta], dtype=np.float32)
    elif args.target == "Y":
        y = np.array([m["x"] for m in meta], dtype=np.float32)
    elif args.target == "sum":
        y = np.array([m["answer"] for m in meta], dtype=np.float32)

    steering_direction, probe_info = fit_steering_probe(X, y, target_name=args.target)

    # Save probe info
    args.output_dir.mkdir(parents=True, exist_ok=True)
    probe_save = {k: v for k, v in probe_info.items()
                  if k not in ("mean", "std")}  # skip large arrays in JSON
    probe_save["layer"] = args.layer
    probe_save["position"] = probe_position
    probe_save["condition"] = args.condition
    with open(args.output_dir / "probe_info.json", "w") as f:
        json.dump(probe_save, f, indent=2)

    del X  # free memory

    # =========================================================================
    # Step 2: Select examples
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 2: Selecting examples")
    print("=" * 60)

    with open(args.dataset) as f:
        dataset = json.load(f)
    problems = dataset["examples"]
    few_shot = dataset["few_shot_facts"]

    # Add idx if missing
    for i, p in enumerate(problems):
        if "idx" not in p:
            p["idx"] = i

    with open(args.categories) as f:
        cat_data = json.load(f)

    # Filter by category — use per-condition field if available
    cat_field = f"category_{args.condition}"
    if cat_field not in cat_data["examples"][0]:
        cat_field = "category"
    print(f"  Using category field: {cat_field}")

    example_indices = [
        e["problem_idx"] for e in cat_data["examples"]
        if e.get(cat_field) == args.category
    ]
    if args.max_examples and len(example_indices) > args.max_examples:
        example_indices = example_indices[:args.max_examples]

    print(f"  Category '{args.category}': {len(example_indices)} examples")
    print(f"  Layer: {args.layer}, Position: {args.position}")

    # =========================================================================
    # Step 3: Load model and run
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 3: Loading model")
    print("=" * 60)

    model, tokenizer = load_model(args.model_path)

    # =========================================================================
    # Step 4: Null validation (delta=0 consistency check)
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 4: Null validation (delta=0 with hook)")
    print("=" * 60)

    # Run 5 examples with delta=0 WITH and WITHOUT hook to verify
    # that adding a zero vector doesn't change the answer
    null_sv = torch.zeros(len(steering_direction))
    first_param = next(model.parameters())
    input_device = first_param.device
    n_null = min(5, len(example_indices))
    all_pass = True

    for i in range(n_null):
        prob_idx = example_indices[i]
        problem = problems[prob_idx]
        example_rng = random.Random(prob_idx)
        messages = build_messages_for_condition(
            few_shot[:5], problem, args.filler_type, args.filler_k, rng=example_rng
        )
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(full_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(input_device)
        attention_mask = inputs["attention_mask"].to(input_device)
        seq_len = input_ids.shape[1]
        filler_start, filler_end = find_filler_boundaries(
            tokenizer, input_ids, args.filler_k
        )
        steer_positions = get_intervention_positions(
            args.position, filler_start, filler_end, seq_len
        )

        _, natural = generate_answer(model, tokenizer, input_ids, attention_mask)
        hook = SteeringHook(model, args.layer, steer_positions, null_sv)
        _, hooked = generate_answer(model, tokenizer, input_ids, attention_mask,
                                    hook=hook)

        passed = (natural == hooked)
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] prob_{prob_idx:04d}: natural={natural}, hooked={hooked}")
        if not passed:
            all_pass = False

        del input_ids, attention_mask
        torch.cuda.empty_cache()

    if not all_pass:
        print("\nWARNING: Null validation failed — hook may be buggy")
    else:
        print(f"\nNull validation PASSED ({n_null}/{n_null})")

    # =========================================================================
    # Step 5: Main experiment
    # =========================================================================
    print("\n" + "=" * 60)
    print(f"Step 5: Running intervention ({len(example_indices)} examples "
          f"x {len(deltas)} deltas = {len(example_indices) * len(deltas)} generations)")
    print("=" * 60)

    results = run_experiment(
        model, tokenizer, problems, few_shot,
        example_indices, steering_direction,
        layer=args.layer, position=args.position,
        deltas=deltas, output_dir=args.output_dir,
        filler_type=args.filler_type, k=args.filler_k,
        control_type=args.control, random_seed=args.random_seed,
    )

    # =========================================================================
    # Step 6: Analysis and plotting
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 6: Analysis")
    print("=" * 60)

    summary = analyze_results(results, args.output_dir)
    if summary:
        plot_results(results, summary, args.output_dir)


if __name__ == "__main__":
    main()
