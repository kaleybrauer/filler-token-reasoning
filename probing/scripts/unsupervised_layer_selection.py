"""Test unsupervised layer selection criteria for PCA logit lens decoder."""

import numpy as np
import sys
import json
from pathlib import Path
from sklearn.decomposition import PCA
from scipy.stats import entropy, spearmanr
from scipy.spatial.distance import pdist, squareform
sys.path.insert(0, str(Path(__file__).parent))
from pca_filler import load_data
from transformers import AutoTokenizer

# Load data for both conditions
extraction_dir = Path("probing/extracted_states")
baseline_dir = extraction_dir / "baseline"

conditions = {}
for cond in ["dots_100", "dots_10"]:
    data, metadata, positions, layers, categories = load_data(
        extraction_dir, cond, baseline_dir=baseline_dir)
    keep = [i for i, m in enumerate(metadata)
            if categories.get(m["problem_idx"]) in ["both_correct", "filler_helped"]]
    metadata_f = [metadata[i] for i in keep]
    for pos in data:
        for layer in data[pos]:
            data[pos][layer] = data[pos][layer][keep]
    A = np.array([m["fact_value"] for m in metadata_f], dtype=np.float32)
    filler_positions = [p for p in positions if p.startswith("filler_k") or p == "pre_filler"]
    avg_data = {}
    for layer in layers:
        vecs = [data[p][layer] for p in filler_positions if p in data and layer in data[p]]
        avg_data[layer] = np.mean(vecs, axis=0)
    conditions[cond] = {"avg_data": avg_data, "A": A, "metadata": metadata_f,
                         "problem_idxs": [m["problem_idx"] for m in metadata_f]}

# Match examples across conditions by problem_idx
idx_100 = {m["problem_idx"]: i for i, m in enumerate(conditions["dots_100"]["metadata"])}
idx_10 = {m["problem_idx"]: i for i, m in enumerate(conditions["dots_10"]["metadata"])}
shared = sorted(set(idx_100.keys()) & set(idx_10.keys()))
map_100 = [idx_100[p] for p in shared]
map_10 = [idx_10[p] for p in shared]
print(f"Shared examples across conditions: {len(shared)}")

lm_head = np.load("probing/lm_head_weight.npy").astype(np.float32)
tokenizer = AutoTokenizer.from_pretrained("/workspace/models/deepseek-v3-awq", trust_remote_code=True)

num_token_ids = {}
for v in range(301):
    ids = tokenizer.encode(str(v), add_special_tokens=False)
    if len(ids) == 1:
        num_token_ids[v] = ids[0]
num_ids = [num_token_ids[v] for v in sorted(num_token_ids.keys())]
num_values_arr = np.array(sorted(num_token_ids.keys()), dtype=np.float32)

def get_preds(X, n_components=20):
    """PCA reconstruct -> lm_head -> argmax and softmax over number tokens."""
    pca = PCA(n_components=n_components)
    X_hat = pca.inverse_transform(pca.fit_transform(X))
    logits = X_hat @ lm_head.T
    num_logits = logits[:, num_ids]
    preds_argmax = num_values_arr[np.argmax(num_logits, axis=1)]
    # softmax EV
    shifted = num_logits - num_logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    preds_softmax = (probs * num_values_arr[None, :]).sum(axis=1)
    mean_ent = np.mean([entropy(p) for p in probs])
    return preds_argmax, preds_softmax, pca, mean_ent

rng = np.random.default_rng(42)
n_bootstrap = 50

print(f"\n{'Layer':>5} {'MAE':>6} {'cross_r':>8} {'boot_std':>9} {'am_sm_d':>8} {'nn_rho':>7} {'entropy':>8} {'pred_std':>9} {'n_unique':>9}")
print("-" * 85)

results = {}
A_100 = conditions["dots_100"]["A"]

for layer in layers:
    X_100 = conditions["dots_100"]["avg_data"][layer]
    X_10 = conditions["dots_10"]["avg_data"][layer]

    preds_100_am, preds_100_sm, _, ent_100 = get_preds(X_100)
    preds_10_am, preds_10_sm, _, _ = get_preds(X_10)

    mae = float(np.mean(np.abs(preds_100_am - A_100)))

    # 1. Cross-condition agreement
    p100_shared = preds_100_am[map_100]
    p10_shared = preds_10_am[map_10]
    if np.std(p100_shared) > 0 and np.std(p10_shared) > 0:
        cross_r = float(np.corrcoef(p100_shared, p10_shared)[0, 1])
    else:
        cross_r = 0.0

    # 2. Bootstrap stability
    n = X_100.shape[0]
    boot_preds = np.zeros((n_bootstrap, n))
    for b in range(n_bootstrap):
        train = rng.choice(n, size=int(0.8 * n), replace=False)
        test = np.setdiff1d(np.arange(n), train)
        pca = PCA(n_components=20)
        pca.fit(X_100[train])
        X_hat = pca.inverse_transform(pca.transform(X_100))
        logits = X_hat @ lm_head.T
        num_logits = logits[:, num_ids]
        boot_preds[b] = num_values_arr[np.argmax(num_logits, axis=1)]
    boot_std = float(np.mean(np.std(boot_preds, axis=0)))

    # 3. Argmax-softmax agreement
    am_sm_diff = float(np.mean(np.abs(preds_100_am - preds_100_sm)))

    # 4. Nearest-neighbor consistency (subsample for speed)
    sub = rng.choice(n, size=min(100, n), replace=False)
    Z = PCA(n_components=20).fit_transform(X_100[sub])
    pca_dists = pdist(Z)
    pred_dists = pdist(preds_100_am[sub].reshape(-1, 1))
    if np.std(pca_dists) > 0 and np.std(pred_dists) > 0:
        nn_rho = float(spearmanr(pca_dists, pred_dists).correlation)
    else:
        nn_rho = 0.0

    pred_std = float(np.std(preds_100_am))
    n_unique = int(len(np.unique(preds_100_am)))

    results[layer] = {
        "mae": mae, "cross_r": cross_r, "boot_std": boot_std,
        "am_sm_diff": am_sm_diff, "nn_rho": nn_rho,
        "entropy": ent_100, "pred_std": pred_std, "n_unique": n_unique
    }

    print(f"L{layer:>3}  {mae:>6.1f} {cross_r:>8.3f} {boot_std:>9.1f} {am_sm_diff:>8.1f} {nn_rho:>7.3f} {ent_100:>8.2f} {pred_std:>9.1f} {n_unique:>9}")

# Save
with open("probing/results/pca_filler/unsupervised_criteria.json", "w") as f:
    json.dump({str(k): v for k, v in results.items()}, f)

# Plot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

layer_list = sorted(results.keys())
metrics = {
    "cross_r": ("Cross-condition r\n(dots_100 vs dots_10)", True),
    "boot_std": ("Bootstrap std\n(lower = more stable)", False),
    "am_sm_diff": ("Argmax−softmax |diff|\n(lower = more consistent)", False),
    "nn_rho": ("NN consistency ρ\n(PCA dist vs pred dist)", True),
}

fig, axes = plt.subplots(len(metrics) + 1, 1, figsize=(14, 16), sharex=True)

for ax, (metric, (ylabel, higher_better)) in zip(axes, metrics.items()):
    vals = [results[l][metric] for l in layer_list]
    color = "tab:blue"
    ax.plot(layer_list, vals, ".-", markersize=4, color=color)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(alpha=0.3)
    if higher_better:
        best_l = layer_list[np.argmax(vals)]
    else:
        best_l = layer_list[np.argmin(vals)]
    ax.axvline(x=best_l, color=color, linestyle="--", alpha=0.4)
    ax.annotate(f"L{best_l}", xy=(best_l, vals[best_l]), fontsize=10, color=color,
                xytext=(best_l + 2, vals[best_l]))

axes[0].set_title("Unsupervised layer selection criteria — avg filler, PCA→logit lens (dots_100)", fontsize=13)

# MAE ground truth
ax = axes[-1]
maes = [results[l]["mae"] for l in layer_list]
ax.plot(layer_list, maes, "k.-", markersize=4)
ax.axhline(y=25.7, color="gray", linestyle="--", alpha=0.5, label="predict mean")
ax.set_ylabel("MAE(A)\n(ground truth)", fontsize=9)
ax.set_xlabel("Layer")
ax.set_ylim(0, 220)
ax.grid(alpha=0.3)
ax.legend(fontsize=9)
best_mae = layer_list[np.argmin(maes)]
ax.axvline(x=best_mae, color="black", linestyle="--", alpha=0.4)
ax.annotate(f"L{best_mae}", xy=(best_mae, maes[best_mae]), fontsize=10, color="black",
            xytext=(best_mae + 2, maes[best_mae] + 8))

plt.tight_layout()
out = Path("probing/results/pca_filler")
plt.savefig(out / "unsupervised_layer_selection.png", dpi=150, bbox_inches="tight")
plt.savefig(out / "unsupervised_layer_selection.pdf", bbox_inches="tight")
print(f"\nPlot saved to {out}/unsupervised_layer_selection.png")
