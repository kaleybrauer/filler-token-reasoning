"""
probe_kv_latents.py

Probe the MLA compressed KV latents (512-dim) for intermediate values A, Y, A+Y.
Compares against probing the residual stream (7168-dim) at the same positions.

The KV transplant proved filler KV entries carry A-specific information.
This script tests whether that information is decodable via linear probes.

CPU-only — uses pre-extracted KV latents and hidden states.

Usage:
    uv run --project /workspace/filler-token-reasoning/probing \
        python probing/scripts/probe_kv_latents.py \
        --conditions dots_100 baseline \
        --output-dir probing/results/kv_latent_probes
"""

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_predict
from tqdm import tqdm


def rms_norm(x, weight, eps=1e-6):
    """Apply RMSNorm. x: (N, D), weight: (D,)."""
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight


def load_kv_latents(kv_dir, condition):
    """Load KV latents for one condition.

    Returns:
        positions: list of position names
        layers: list of layer indices
        data: {pos: {layer: ndarray (N, 576)}}
        metadata: list of dicts with fact_value, x, answer, problem_idx
    """
    cond_dir = Path(kv_dir) / condition
    files = sorted(cond_dir.glob("kv_*.pkl"))

    data = defaultdict(lambda: defaultdict(list))
    metadata = []

    for f in files:
        with open(f, "rb") as fp:
            d = pickle.load(fp)
        metadata.append({
            "problem_idx": d["problem_idx"],
            "fact_value": d["fact_value"],
            "x": d["x"],
            "answer": d["answer"],
        })
        for pos, layer_dict in d["kv_latents"].items():
            for layer, vec in layer_dict.items():
                data[pos][layer].append(vec.astype(np.float32))

    positions = list(d["kv_latents"].keys())
    layers = sorted(d["kv_latents"][positions[0]].keys())

    # Stack into arrays
    stacked = {}
    for pos in positions:
        stacked[pos] = {}
        for layer in layers:
            stacked[pos][layer] = np.stack(data[pos][layer], axis=0)

    return positions, layers, stacked, metadata


def load_hidden_states(hs_dir, condition):
    """Load residual stream hidden states for comparison."""
    cond_dir = Path(hs_dir) / condition
    files = sorted(cond_dir.glob("prob_*.pkl"))

    data = defaultdict(lambda: defaultdict(list))
    metadata = []

    for f in files:
        with open(f, "rb") as fp:
            d = pickle.load(fp)
        metadata.append({
            "problem_idx": d["problem_idx"],
            "fact_value": d["fact_value"],
            "x": d["x"],
            "answer": d["answer"],
        })
        for pos, layer_dict in d["states"].items():
            for layer, vec in layer_dict.items():
                data[pos][layer].append(vec.astype(np.float32))

    positions = list(d["states"].keys())
    layers = sorted(d["states"][positions[0]].keys())

    stacked = {}
    for pos in positions:
        stacked[pos] = {}
        for layer in layers:
            stacked[pos][layer] = np.stack(data[pos][layer], axis=0)

    return positions, layers, stacked, metadata


def build_targets(metadata):
    """Build probe targets from metadata."""
    A = np.array([m["fact_value"] for m in metadata], dtype=np.float32)
    Y = np.array([m["x"] for m in metadata], dtype=np.float32)
    AY = np.array([m["answer"] for m in metadata], dtype=np.float32)
    rng = np.random.RandomState(42)
    A_shuffled = rng.permutation(A)
    return {"A": A, "Y": Y, "A+Y": AY, "A_shuffled": A_shuffled}


def run_probe(X, y, n_folds=5):
    """Run cross-validated Ridge probe. Returns dict with r2, mae, corr."""
    alphas = np.logspace(-2, 6, 20)
    ridge = RidgeCV(alphas=alphas, cv=n_folds)
    y_pred = cross_val_predict(ridge, X, y, cv=n_folds)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    from scipy import stats
    corr = stats.pearsonr(y, y_pred)[0]
    return {"r2": float(r2), "mae": float(mae), "corr": float(corr)}


def _probe_one(args):
    """Worker for parallel probing."""
    pos, layer, target_name, X, y, norm_w = args
    if norm_w is not None:
        X_kv = X[:, :512]
        X_kv = rms_norm(X_kv, norm_w)
        if X.shape[1] > 512:
            X = np.concatenate([X_kv, X[:, 512:]], axis=1)
        else:
            X = X_kv
    result = run_probe(X, y)
    return pos, layer, target_name, result


def run_all_probes(data, layers, positions, targets, label, norm_weights=None, n_jobs=16):
    """Run probes for all (layer, position, target) combinations in parallel."""
    from joblib import Parallel, delayed

    jobs = []
    for pos in positions:
        for layer in layers:
            X = data[pos][layer]
            nw = norm_weights[layer] if norm_weights is not None else None
            for target_name, y in targets.items():
                jobs.append((pos, layer, target_name, X, y, nw))

    print(f"  {label}: {len(jobs)} probes across {n_jobs} workers...")
    outputs = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_probe_one)(job) for job in tqdm(jobs, desc=label)
    )

    results = {}
    for pos, layer, target_name, result in outputs:
        if pos not in results:
            results[pos] = {}
        if layer not in results[pos]:
            results[pos][layer] = {}
        results[pos][layer][target_name] = result

    return results


def print_summary(all_results, conditions, output_dir):
    """Print and save comparison tables."""
    outlines = []

    for cond in conditions:
        for rep_name, results in all_results[cond].items():
            positions = list(results.keys())
            layers = sorted(results[positions[0]].keys())

            outlines.append(f"\n{'='*70}")
            outlines.append(f"  {cond} — {rep_name}")
            outlines.append(f"{'='*70}")

            for target in ["A", "A+Y", "Y", "A_shuffled"]:
                outlines.append(f"\n  Target: {target}")
                outlines.append(f"  {'Position':<16} {'Best L':>7} {'R²':>8} {'MAE':>8}")
                outlines.append(f"  {'-'*42}")

                for pos in positions:
                    best_layer = max(layers, key=lambda l: results[pos][l][target]["r2"])
                    r = results[pos][best_layer][target]
                    outlines.append(f"  {pos:<16} L{best_layer:>5} {r['r2']:>8.3f} {r['mae']:>8.1f}")

    text = "\n".join(outlines)
    print(text)
    with open(output_dir / "summary.txt", "w") as f:
        f.write(text)


def print_head_to_head(all_results, condition, output_dir):
    """Print KV vs residual stream comparison for key positions/layers."""
    if "kv_raw" not in all_results[condition] or "residual" not in all_results[condition]:
        return

    kv = all_results[condition]["kv_raw"]
    hs = all_results[condition]["residual"]
    positions = [p for p in kv.keys() if p in hs]

    lines = [f"\n{'='*70}", f"  HEAD-TO-HEAD: KV latent (512d) vs Residual stream (7168d)", f"  {condition}", f"{'='*70}"]

    for target in ["A", "A+Y"]:
        lines.append(f"\n  Target: {target}")
        lines.append(f"  {'Position':<16} {'KV best L':>10} {'KV R²':>8} {'HS best L':>10} {'HS R²':>8} {'Δ R²':>8}")
        lines.append(f"  {'-'*62}")

        kv_layers = sorted(kv[positions[0]].keys())
        hs_layers = sorted(hs[positions[0]].keys())

        for pos in positions:
            kv_best = max(kv_layers, key=lambda l: kv[pos][l][target]["r2"])
            hs_best = max(hs_layers, key=lambda l: hs[pos][l][target]["r2"])
            kv_r2 = kv[pos][kv_best][target]["r2"]
            hs_r2 = hs[pos][hs_best][target]["r2"]
            delta = kv_r2 - hs_r2
            lines.append(f"  {pos:<16} L{kv_best:>8} {kv_r2:>8.3f} L{hs_best:>8} {hs_r2:>8.3f} {delta:>+8.3f}")

    text = "\n".join(lines)
    print(text)
    with open(output_dir / "head_to_head.txt", "w") as f:
        f.write(text)


def main():
    parser = argparse.ArgumentParser(description="Probe KV latents for A, Y, A+Y")
    parser.add_argument("--kv-dir", type=Path, default=Path("probing/extracted_kv_latents"))
    parser.add_argument("--hs-dir", type=Path, default=Path("probing/extracted_states"))
    parser.add_argument("--output-dir", type=Path, default=Path("probing/results/kv_latent_probes"))
    parser.add_argument("--norm-weights", type=Path, default=Path("probing/kv_a_layernorm_weights.npy"))
    parser.add_argument("--conditions", nargs="+", default=["dots_100", "baseline"])
    parser.add_argument("--compare-residual", action="store_true", default=True)
    parser.add_argument("--no-compare-residual", action="store_true")
    args = parser.parse_args()

    if args.no_compare_residual:
        args.compare_residual = False

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load norm weights
    norm_weights = None
    if args.norm_weights.exists():
        norm_weights = np.load(args.norm_weights)
        print(f"Loaded kv_a_layernorm weights: {norm_weights.shape}")

    all_results = {}

    for condition in args.conditions:
        print(f"\n{'='*70}")
        print(f"  Loading {condition}")
        print(f"{'='*70}")

        all_results[condition] = {}

        # Load KV latents
        positions, layers, kv_data, metadata = load_kv_latents(args.kv_dir, condition)
        targets = build_targets(metadata)
        print(f"  KV: {len(metadata)} examples, {len(positions)} positions, {len(layers)} layers")
        print(f"  Positions: {positions}")

        # Probe raw KV (first 512 dims)
        kv_raw_data = {pos: {l: kv_data[pos][l][:, :512] for l in layers} for pos in positions}
        print(f"\n  Probing KV raw (512d)...")
        all_results[condition]["kv_raw"] = run_all_probes(
            kv_raw_data, layers, positions, targets, f"{condition} kv_raw"
        )

        # Probe normed KV
        if norm_weights is not None:
            print(f"\n  Probing KV normed (512d)...")
            all_results[condition]["kv_norm"] = run_all_probes(
                kv_raw_data, layers, positions, targets, f"{condition} kv_norm",
                norm_weights=norm_weights,
            )

        # Probe full KV (576d including k_pe)
        print(f"\n  Probing KV full (576d)...")
        all_results[condition]["kv_full"] = run_all_probes(
            kv_data, layers, positions, targets, f"{condition} kv_full"
        )

        # Probe residual stream for comparison
        if args.compare_residual:
            hs_positions, hs_layers, hs_data, hs_metadata = load_hidden_states(
                args.hs_dir, condition
            )
            # Use only positions that exist in both
            common_positions = [p for p in positions if p in hs_positions]
            print(f"\n  Probing residual stream (7168d) at {common_positions}...")
            hs_filtered = {pos: {l: hs_data[pos][l] for l in hs_layers} for pos in common_positions}
            all_results[condition]["residual"] = run_all_probes(
                hs_filtered, hs_layers, common_positions, targets, f"{condition} residual"
            )

    # Print results
    print_summary(all_results, args.conditions, args.output_dir)
    for condition in args.conditions:
        print_head_to_head(all_results, condition, args.output_dir)

    # Save full results
    def make_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj

    with open(args.output_dir / "all_results.json", "w") as f:
        json.dump(make_serializable(all_results), f, indent=2)

    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
