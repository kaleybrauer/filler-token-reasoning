"""
pca_filler.py

PCA on filler hidden states at each layer. Correlate top PCs with A, Y, A+Y.
Fully unsupervised — PCA doesn't use labels. Correlation checked post-hoc.

Also builds a cross-validated unsupervised decoder:
  PCA (unsupervised) → ridge regression (supervised on train fold) → predict A on held-out fold.
Reports MAE and fraction-within-10 alongside a permutation null.

Runs on both individual filler positions and averaged-over-filler.

Usage:
    uv run --project /workspace/filler-token-reasoning/probing \
        python scripts/pca_filler.py \
        --condition dots_100 \
        --output-dir results/pca_filler
"""

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from tqdm import tqdm


def load_data(extraction_dir, condition, baseline_dir=None):
    """Load hidden states and metadata. Optionally build categories from baseline."""
    cond_dir = Path(extraction_dir) / condition
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
            "model_correct": d.get("model_correct", False),
        })
        for pos, layer_dict in d["states"].items():
            for layer, vec in layer_dict.items():
                data[pos][layer].append(vec.astype(np.float32))

    positions = list(d["states"].keys())
    layers = sorted(d["states"][positions[0]].keys())

    for pos in data:
        for layer in data[pos]:
            data[pos][layer] = np.stack(data[pos][layer])

    # Build categories if baseline provided
    categories = None
    if baseline_dir:
        bl_files = sorted(Path(baseline_dir).glob("prob_*.pkl"))
        categories = {}
        for bf, m in zip(bl_files, metadata):
            with open(bf, "rb") as f:
                bd = pickle.load(f)
            bc = bd.get("model_correct", False)
            fc = m["model_correct"]
            if bc and fc: categories[m["problem_idx"]] = "both_correct"
            elif not bc and fc: categories[m["problem_idx"]] = "filler_helped"
            elif bc and not fc: categories[m["problem_idx"]] = "filler_hurt"
            else: categories[m["problem_idx"]] = "both_wrong"

    return data, metadata, positions, layers, categories


def run_pca_analysis(X, A, Y, AY, n_components=20, label=""):
    """Run PCA and correlate top components with targets."""
    pca = PCA(n_components=min(n_components, X.shape[1], X.shape[0]))
    Z = pca.fit_transform(X)

    results = {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cumulative_variance": np.cumsum(pca.explained_variance_ratio_).tolist(),
        "correlations": {},
    }

    for target_name, target in [("A", A), ("Y", Y), ("A+Y", AY)]:
        corrs = []
        for pc_idx in range(Z.shape[1]):
            r, p = stats.pearsonr(Z[:, pc_idx], target)
            corrs.append({"pc": pc_idx, "r": float(r), "p": float(p), "abs_r": float(abs(r))})
        corrs.sort(key=lambda x: -x["abs_r"])
        results["correlations"][target_name] = corrs

    return results, pca, Z


def decode_cv(X, targets, n_components=20, n_folds=5, n_permutations=100, rng=None):
    """Cross-validated unsupervised decoder: PCA (on train) → ridge → predict on held-out.

    PCA is fit on the train fold only (unsupervised). Ridge maps PCs → target (supervised).
    Held-out fold is projected into train PCA space, then predicted via ridge.

    Returns dict with per-target MAE, frac_within_10, and permutation null stats.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = X.shape[0]
    n_comp = min(n_components, X.shape[1], n - 1)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    results = {}
    for target_name, y in targets.items():
        # Real decode
        preds = np.zeros(n)
        for train_idx, test_idx in kf.split(X):
            pca = PCA(n_components=n_comp)
            Z_train = pca.fit_transform(X[train_idx])
            Z_test = pca.transform(X[test_idx])
            ridge = Ridge(alpha=1.0)
            ridge.fit(Z_train, y[train_idx])
            preds[test_idx] = ridge.predict(Z_test)

        mae = float(np.mean(np.abs(preds - y)))
        frac10 = float(np.mean(np.abs(preds - y) <= 10))
        r = float(np.corrcoef(preds, y)[0, 1])

        # Permutation null
        null_maes = []
        null_rs = []
        for _ in range(n_permutations):
            y_shuf = rng.permutation(y)
            null_preds = np.zeros(n)
            for train_idx, test_idx in kf.split(X):
                pca = PCA(n_components=n_comp)
                Z_train = pca.fit_transform(X[train_idx])
                Z_test = pca.transform(X[test_idx])
                ridge = Ridge(alpha=1.0)
                ridge.fit(Z_train, y_shuf[train_idx])
                null_preds[test_idx] = ridge.predict(Z_test)
            null_maes.append(float(np.mean(np.abs(null_preds - y_shuf))))
            null_rs.append(float(np.corrcoef(null_preds, y_shuf)[0, 1]))

        results[target_name] = {
            "mae": mae,
            "frac_within_10": frac10,
            "r": r,
            "null_mae_mean": float(np.mean(null_maes)),
            "null_mae_std": float(np.std(null_maes)),
            "null_r_mean": float(np.mean(null_rs)),
            "null_r_std": float(np.std(null_rs)),
            "p_value": float(np.mean([nr >= abs(r) for nr in np.abs(null_rs)])),
        }

    return results


def logit_lens_decode(X, A, pca, lm_head, tokenizer, n_components=20):
    """Fully unsupervised decoder: project each PC through lm_head, read off number predictions.

    For each example:
    1. Project hidden state into PCA space (unsupervised)
    2. For each PC direction, compute logits = lm_head @ pc_direction
    3. Find which number tokens (0-300) each PC "points to"
    4. Weight PCs by their explained variance and combine predictions

    Also tries: reconstruct from top-k PCs, project reconstruction through lm_head,
    take argmax over number tokens as the prediction.
    """
    n_comp = min(n_components, pca.n_components_)
    Z = pca.transform(X)  # (n_examples, n_comp)
    components = pca.components_[:n_comp]  # (n_comp, 7168)
    mean = pca.mean_  # (7168,)

    lm_head_f32 = lm_head.astype(np.float32)

    # Build number token map: value -> token_id for integers 0-300
    num_token_ids = {}
    for v in range(301):
        ids = tokenizer.encode(str(v), add_special_tokens=False)
        if len(ids) == 1:
            num_token_ids[v] = ids[0]
    num_values = sorted(num_token_ids.keys())
    num_ids = [num_token_ids[v] for v in num_values]
    num_values_arr = np.array(num_values, dtype=np.float32)

    results = {}

    # Method 1: Per-PC logit lens
    # For each PC, project direction through lm_head, get logits over number tokens
    pc_logits = components @ lm_head_f32.T  # (n_comp, vocab)
    pc_num_logits = pc_logits[:, num_ids]  # (n_comp, n_numbers)

    # For each PC, the "preferred number" is the argmax
    pc_preferred = []
    for pc_idx in range(n_comp):
        best_num_idx = np.argmax(pc_num_logits[pc_idx])
        best_num = num_values[best_num_idx]
        # Also check negative direction
        best_num_idx_neg = np.argmax(-pc_num_logits[pc_idx])
        best_num_neg = num_values[best_num_idx_neg]
        pc_preferred.append({
            "pc": pc_idx,
            "pos_preferred": int(best_num),
            "neg_preferred": int(best_num_neg),
            "pos_logit": float(pc_num_logits[pc_idx, best_num_idx]),
            "neg_logit": float(-pc_num_logits[pc_idx, best_num_idx_neg]),
        })
    results["pc_preferred_numbers"] = pc_preferred

    # Method 2: Reconstruct from top-k PCs, add mean, project through lm_head
    # This is the real unsupervised decoder
    for k in [1, 3, 5, 10, 20]:
        k_use = min(k, n_comp)
        # Reconstruct: x_hat = mean + Z[:, :k] @ components[:k]
        X_hat = mean[None, :] + Z[:, :k_use] @ components[:k_use]  # (n, 7168)
        logits = X_hat @ lm_head_f32.T  # (n, vocab)
        num_logits = logits[:, num_ids]  # (n, n_numbers)

        # Prediction = number with highest logit
        pred_idx = np.argmax(num_logits, axis=1)
        preds = num_values_arr[pred_idx]

        mae = float(np.mean(np.abs(preds - A)))
        frac10 = float(np.mean(np.abs(preds - A) <= 10))
        r = float(np.corrcoef(preds, A)[0, 1]) if np.std(preds) > 0 else 0.0

        # Softmax-weighted prediction (expected value)
        num_logits_shifted = num_logits - num_logits.max(axis=1, keepdims=True)
        probs = np.exp(num_logits_shifted) / np.exp(num_logits_shifted).sum(axis=1, keepdims=True)
        preds_soft = (probs * num_values_arr[None, :]).sum(axis=1)

        mae_soft = float(np.mean(np.abs(preds_soft - A)))
        frac10_soft = float(np.mean(np.abs(preds_soft - A) <= 10))
        r_soft = float(np.corrcoef(preds_soft, A)[0, 1]) if np.std(preds_soft) > 0 else 0.0

        results[f"reconstruct_top{k}"] = {
            "argmax": {"mae": mae, "frac_within_10": frac10, "r": r},
            "softmax": {"mae": mae_soft, "frac_within_10": frac10_soft, "r": r_soft},
        }

    # Method 3: Full reconstruction (all PCs used by PCA, then lm_head)
    X_hat_full = pca.inverse_transform(Z)
    logits_full = X_hat_full @ lm_head_f32.T
    num_logits_full = logits_full[:, num_ids]
    pred_idx_full = np.argmax(num_logits_full, axis=1)
    preds_full = num_values_arr[pred_idx_full]
    mae_full = float(np.mean(np.abs(preds_full - A)))
    frac10_full = float(np.mean(np.abs(preds_full - A) <= 10))
    r_full = float(np.corrcoef(preds_full, A)[0, 1]) if np.std(preds_full) > 0 else 0.0
    results["reconstruct_full"] = {"mae": mae_full, "frac_within_10": frac10_full, "r": r_full}

    return results


def main():
    parser = argparse.ArgumentParser(description="PCA on filler hidden states")
    parser.add_argument("--extraction-dir", type=Path, default=Path("data/extracted_states"))
    parser.add_argument("--condition", type=str, default="dots_100")
    parser.add_argument("--output-dir", type=Path, default=Path("results/pca_filler"))
    parser.add_argument("--n-components", type=int, default=20)
    parser.add_argument("--lm-head", type=Path, default=Path("lm_head_weight.npy"))
    parser.add_argument("--tokenizer", type=str, default="/workspace/models/deepseek-v3-awq")
    parser.add_argument("--filter-categories", nargs="+",
                        default=["both_correct", "filler_helped"],
                        help="Only include these categories")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    baseline_dir = args.extraction_dir / "baseline"
    data, metadata, positions, layers, categories = load_data(
        args.extraction_dir, args.condition,
        baseline_dir=baseline_dir if baseline_dir.exists() else None,
    )

    # Filter to selected categories
    if categories and args.filter_categories:
        keep = [i for i, m in enumerate(metadata)
                if categories.get(m["problem_idx"]) in args.filter_categories]
        metadata = [metadata[i] for i in keep]
        for pos in data:
            for layer in data[pos]:
                data[pos][layer] = data[pos][layer][keep]
        print(f"Filtered to {len(metadata)} examples ({args.filter_categories})")
    else:
        print(f"Using all {len(metadata)} examples")

    A = np.array([m["fact_value"] for m in metadata], dtype=np.float32)
    Y = np.array([m["x"] for m in metadata], dtype=np.float32)
    AY = np.array([m["answer"] for m in metadata], dtype=np.float32)

    filler_positions = [p for p in positions if p.startswith("filler_k") or p == "pre_filler"]
    all_positions = filler_positions + ["answer_prompt"]

    print(f"Condition: {args.condition}")
    print(f"Filler positions: {filler_positions}")
    print(f"Layers: {len(layers)}")
    print(f"Examples: {len(metadata)}")

    all_results = {}

    # 1. PCA at individual filler positions
    for pos in tqdm(all_positions, desc="Individual positions"):
        all_results[pos] = {}
        for layer in layers:
            X = data[pos][layer]
            results, _, _ = run_pca_analysis(X, A, Y, AY, args.n_components)
            all_results[pos][layer] = results

    # 2. PCA on averaged filler
    print("\nComputing averaged filler...")
    avg_data = {}
    for layer in layers:
        vecs = [data[p][layer] for p in filler_positions if p in data and layer in data[p]]
        avg_data[layer] = np.mean(vecs, axis=0)

    all_results["avg_filler"] = {}
    for layer in tqdm(layers, desc="Avg filler PCA"):
        results, _, _ = run_pca_analysis(avg_data[layer], A, Y, AY, args.n_components)
        all_results["avg_filler"][layer] = results

    # 3. Cross-validated unsupervised decoder
    targets = {"A": A, "Y": Y, "A+Y": AY}
    decode_positions = ["avg_filler"] + filler_positions + ["answer_prompt"]

    # Pick best layer per position (highest |r| with A from PCA correlations)
    best_layers = {}
    for pos in decode_positions:
        best_r = 0
        best_l = layers[0]
        for layer in layers:
            if layer not in all_results.get(pos, {}):
                continue
            corrs = all_results[pos][layer]["correlations"]["A"]
            if corrs and corrs[0]["abs_r"] > best_r:
                best_r = corrs[0]["abs_r"]
                best_l = layer
        best_layers[pos] = best_l

    # Also decode at layer bands (max |r| layer within band)
    bands = {"early (0-15)": range(0, 16), "mid (16-30)": range(16, 31),
             "late_mid (31-45)": range(31, 46), "late (46-60)": range(46, 61)}

    decode_results = {}
    rng = np.random.default_rng(42)

    print(f"\n{'='*80}")
    print(f"  Cross-validated unsupervised decoder (PCA → ridge, 5-fold)")
    print(f"{'='*80}")
    print(f"  {'Position':<16} {'Layer':>7} {'MAE(A)':>8} {'±10(A)':>8} {'r(A)':>7}"
          f"  {'null r':>7} {'p':>7}")
    print(f"  {'-'*72}")

    for pos in decode_positions:
        layer = best_layers[pos]
        if pos == "avg_filler":
            X = avg_data[layer]
        else:
            X = data[pos][layer]

        dr = decode_cv(X, targets, n_components=args.n_components,
                       n_permutations=100, rng=rng)
        decode_results[f"{pos}_L{layer}"] = dr
        a = dr["A"]
        print(f"  {pos:<16} L{layer:>5} {a['mae']:>8.1f} {a['frac_within_10']:>7.1%}"
              f" {a['r']:>7.3f}  {a['null_r_mean']:>+.3f} {a['p_value']:>7.3f}")

    # Decode with layer bands on avg_filler
    print(f"\n  Avg filler — layer band decoder:")
    print(f"  {'Band':<16} {'Layer':>7} {'MAE(A)':>8} {'±10(A)':>8} {'r(A)':>7}"
          f"  {'null r':>7} {'p':>7}")
    print(f"  {'-'*72}")

    for band_name, band_range in bands.items():
        band_layers = [l for l in layers if l in band_range]
        if not band_layers:
            continue
        # Find best layer in band
        best_r = 0
        best_l = band_layers[0]
        for l in band_layers:
            if l not in all_results.get("avg_filler", {}):
                continue
            corrs = all_results["avg_filler"][l]["correlations"]["A"]
            if corrs and corrs[0]["abs_r"] > best_r:
                best_r = corrs[0]["abs_r"]
                best_l = l
        X = avg_data[best_l]
        dr = decode_cv(X, targets, n_components=args.n_components,
                       n_permutations=100, rng=rng)
        decode_results[f"avg_filler_band_{band_name}_L{best_l}"] = dr
        a = dr["A"]
        print(f"  {band_name:<16} L{best_l:>5} {a['mae']:>8.1f} {a['frac_within_10']:>7.1%}"
              f" {a['r']:>7.3f}  {a['null_r_mean']:>+.3f} {a['p_value']:>7.3f}")

    all_results["decoder"] = decode_results

    # 4. Logit lens on PCs — fully unsupervised decoding
    if args.lm_head.exists():
        from transformers import AutoTokenizer
        print(f"\n{'='*80}")
        print(f"  Logit lens decoder (fully unsupervised — no labels)")
        print(f"{'='*80}")

        lm_head = np.load(args.lm_head)
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

        # Run on avg_filler at best layers, plus answer_prompt
        lens_positions = [
            ("avg_filler", best_layers.get("avg_filler", 53)),
            ("answer_prompt", best_layers.get("answer_prompt", 59)),
        ]
        # Add best layer per band for avg_filler
        for band_name, band_range in bands.items():
            band_layers_list = [l for l in layers if l in band_range]
            if not band_layers_list:
                continue
            best_r = 0
            best_l = band_layers_list[0]
            for l in band_layers_list:
                if l not in all_results.get("avg_filler", {}):
                    continue
                corrs = all_results["avg_filler"][l]["correlations"]["A"]
                if corrs and corrs[0]["abs_r"] > best_r:
                    best_r = corrs[0]["abs_r"]
                    best_l = l
            lens_positions.append((f"avg_{band_name}", best_l))

        logit_lens_results = {}

        for pos_label, layer in lens_positions:
            if pos_label == "answer_prompt":
                X = data["answer_prompt"][layer]
            elif pos_label == "avg_filler":
                X = avg_data[layer]
            else:
                X = avg_data[layer]

            # Fit PCA on all data (unsupervised)
            pca = PCA(n_components=min(args.n_components, X.shape[1], X.shape[0] - 1))
            pca.fit(X)

            lr = logit_lens_decode(X, A, pca, lm_head, tokenizer, args.n_components)
            logit_lens_results[f"{pos_label}_L{layer}"] = lr

            # Print reconstruct results
            print(f"\n  {pos_label} L{layer}:")
            print(f"  {'Method':<22} {'MAE':>7} {'±10':>7} {'r':>7}")
            print(f"  {'-'*46}")
            for k in [1, 3, 5, 10, 20]:
                key = f"reconstruct_top{k}"
                if key in lr:
                    am = lr[key]["argmax"]
                    sm = lr[key]["softmax"]
                    print(f"  top-{k:<2} argmax        {am['mae']:>7.1f} {am['frac_within_10']:>6.1%} {am['r']:>7.3f}")
                    print(f"  top-{k:<2} softmax       {sm['mae']:>7.1f} {sm['frac_within_10']:>6.1%} {sm['r']:>7.3f}")
            if "reconstruct_full" in lr:
                rf = lr["reconstruct_full"]
                print(f"  full reconstruct    {rf['mae']:>7.1f} {rf['frac_within_10']:>6.1%} {rf['r']:>7.3f}")

        all_results["logit_lens"] = logit_lens_results

        # 4b. Full layer sweep: logit lens on avg_filler at every layer
        print(f"\n{'='*80}")
        print(f"  Logit lens layer sweep — avg_filler (fully unsupervised)")
        print(f"{'='*80}")
        print(f"  {'Layer':>7} {'top20_am MAE':>12} {'±10':>7} {'r':>7}"
              f"  {'top10_sm MAE':>12} {'±10':>7} {'r':>7}")
        print(f"  {'-'*72}")

        layer_sweep_results = {}
        for layer in tqdm(layers, desc="Logit lens sweep"):
            X = avg_data[layer]
            pca = PCA(n_components=min(args.n_components, X.shape[1], X.shape[0] - 1))
            pca.fit(X)
            lr = logit_lens_decode(X, A, pca, lm_head, tokenizer, args.n_components)
            layer_sweep_results[layer] = lr

            am20 = lr["reconstruct_top20"]["argmax"]
            sm10 = lr["reconstruct_top10"]["softmax"]
            print(f"  L{layer:>5} {am20['mae']:>12.1f} {am20['frac_within_10']:>6.1%} {am20['r']:>7.3f}"
                  f"  {sm10['mae']:>12.1f} {sm10['frac_within_10']:>6.1%} {sm10['r']:>7.3f}")

        all_results["logit_lens_layer_sweep"] = {
            str(l): v for l, v in layer_sweep_results.items()
        }

        # Print best layers (by lowest MAE)
        best_am = min(layer_sweep_results.items(),
                      key=lambda x: x[1]["reconstruct_top20"]["argmax"]["mae"])
        best_sm = min(layer_sweep_results.items(),
                      key=lambda x: x[1]["reconstruct_top10"]["softmax"]["mae"])
        print(f"\n  Best top-20 argmax:  L{best_am[0]} MAE={best_am[1]['reconstruct_top20']['argmax']['mae']:.1f}"
              f"  ±10={best_am[1]['reconstruct_top20']['argmax']['frac_within_10']:.1%}"
              f"  r={best_am[1]['reconstruct_top20']['argmax']['r']:.3f}")
        print(f"  Best top-10 softmax: L{best_sm[0]} MAE={best_sm[1]['reconstruct_top10']['softmax']['mae']:.1f}"
              f"  ±10={best_sm[1]['reconstruct_top10']['softmax']['frac_within_10']:.1%}"
              f"  r={best_sm[1]['reconstruct_top10']['softmax']['r']:.3f}")
    else:
        print(f"\n  Skipping logit lens (no lm_head at {args.lm_head})")

    # 5. Print summary: best PC correlation per (position, layer) for each target
    print(f"\n{'='*80}")
    print(f"  Best PC correlation with A, Y, A+Y (top |r| across PCs)")
    print(f"{'='*80}")

    summary_positions = ["pre_filler"] + [p for p in filler_positions if p != "pre_filler"] + ["avg_filler", "answer_prompt"]

    for target in ["A", "Y", "A+Y"]:
        print(f"\n  Target: {target}")
        print(f"  {'Position':<16} {'Best L':>7} {'PC':>4} {'|r|':>7} {'Var%':>6}")
        print(f"  {'-'*44}")

        for pos in summary_positions:
            if pos not in all_results:
                continue
            best_r = 0
            best_layer = 0
            best_pc = 0
            best_var = 0
            for layer in layers:
                if layer not in all_results[pos]:
                    continue
                corrs = all_results[pos][layer]["correlations"][target]
                if corrs and corrs[0]["abs_r"] > best_r:
                    best_r = corrs[0]["abs_r"]
                    best_layer = layer
                    best_pc = corrs[0]["pc"]
                    best_var = all_results[pos][layer]["explained_variance_ratio"][best_pc] * 100
            print(f"  {pos:<16} L{best_layer:>5} PC{best_pc:>2} {best_r:>7.3f} {best_var:>5.1f}%")

    # Save
    def make_serializable(obj):
        if isinstance(obj, (np.integer, np.int64)): return int(obj)
        if isinstance(obj, (np.floating, np.float64)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {str(k): make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list): return [make_serializable(v) for v in obj]
        return obj

    with open(args.output_dir / "pca_results.json", "w") as f:
        json.dump(make_serializable(all_results), f)

    # Heatmap: best |r| for A at each (layer, position)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for target in ["A", "A+Y"]:
            plot_positions = summary_positions
            matrix = np.zeros((len(layers), len(plot_positions)))

            for j, pos in enumerate(plot_positions):
                if pos not in all_results:
                    continue
                for i, layer in enumerate(layers):
                    if layer not in all_results[pos]:
                        continue
                    corrs = all_results[pos][layer]["correlations"][target]
                    matrix[i, j] = corrs[0]["abs_r"] if corrs else 0

            fig, ax = plt.subplots(figsize=(12, 10))
            im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=0.8)

            pos_labels = []
            for p in plot_positions:
                if p == "pre_filler": pos_labels.append("pre_f")
                elif p == "answer_prompt": pos_labels.append("ans")
                elif p == "avg_filler": pos_labels.append("avg")
                elif p.startswith("filler_k"): pos_labels.append(f"k={p.split('k')[1]}")
                else: pos_labels.append(p[:8])

            ax.set_xticks(range(len(plot_positions)))
            ax.set_xticklabels(pos_labels, rotation=45, ha="right", fontsize=10)
            layer_labels = [str(l) if l % 5 == 0 else "" for l in layers]
            ax.set_yticks(range(len(layers)))
            ax.set_yticklabels(layer_labels, fontsize=7)
            ax.set_ylabel("Layer")
            plt.colorbar(im, ax=ax, label="|r|", shrink=0.8)
            ax.set_title(f"PCA: best PC |correlation| with {target} ({args.condition})", fontsize=13)

            plt.tight_layout()
            plt.savefig(args.output_dir / f"pca_heatmap_{target.replace('+','plus')}.png",
                        dpi=150, bbox_inches="tight")
            plt.savefig(args.output_dir / f"pca_heatmap_{target.replace('+','plus')}.pdf",
                        bbox_inches="tight")
            plt.close()

        print(f"\nPlots saved to {args.output_dir}/")
    except ImportError:
        print("matplotlib not available, skipping plots")

    print(f"Results saved to {args.output_dir}/pca_results.json")


if __name__ == "__main__":
    main()
