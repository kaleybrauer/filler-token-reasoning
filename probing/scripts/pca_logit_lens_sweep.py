"""Quick standalone: logit lens layer sweep on avg_filler. No permutations."""

import argparse
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.decomposition import PCA
from tqdm import tqdm

# Import from main script
import sys
sys.path.insert(0, str(Path(__file__).parent))
from pca_filler import load_data, logit_lens_decode

parser = argparse.ArgumentParser()
parser.add_argument("--condition", type=str, default="dots_100")
parser.add_argument("--extraction-dir", type=Path, default=Path("probing/extracted_states"))
parser.add_argument("--output-dir", type=Path, default=None)
parser.add_argument("--lm-head", type=Path, default=Path("probing/lm_head_weight.npy"))
parser.add_argument("--tokenizer", type=str, default="/workspace/models/deepseek-v3-awq")
parser.add_argument("--n-components", type=int, default=20)
args = parser.parse_args()

extraction_dir = args.extraction_dir
condition = args.condition
n_components = args.n_components
lm_head_path = args.lm_head
tokenizer_path = args.tokenizer
output_dir = args.output_dir or Path(f"probing/results/pca_filler_{condition}")
output_dir = Path(output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

from transformers import AutoTokenizer

print("Loading data...")
baseline_dir = extraction_dir / "baseline"
data, metadata, positions, layers, categories = load_data(
    extraction_dir, condition,
    baseline_dir=baseline_dir if baseline_dir.exists() else None,
)

# Filter
filter_cats = ["both_correct", "filler_helped"]
keep = [i for i, m in enumerate(metadata)
        if categories.get(m["problem_idx"]) in filter_cats]
metadata = [metadata[i] for i in keep]
for pos in data:
    for layer in data[pos]:
        data[pos][layer] = data[pos][layer][keep]
print(f"Filtered to {len(metadata)} examples")

A = np.array([m["fact_value"] for m in metadata], dtype=np.float32)
filler_positions = [p for p in positions if p.startswith("filler_k") or p == "pre_filler"]

# Compute avg filler
print("Computing averaged filler...")
avg_data = {}
for layer in layers:
    vecs = [data[p][layer] for p in filler_positions if p in data and layer in data[p]]
    avg_data[layer] = np.mean(vecs, axis=0)

print("Loading lm_head and tokenizer...")
lm_head = np.load(lm_head_path)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

print(f"\n{'='*80}")
print(f"  Logit lens layer sweep — avg_filler (fully unsupervised)")
print(f"{'='*80}")
print(f"  {'Layer':>7} {'top20_am MAE':>12} {'±10':>7} {'r':>7}"
      f"  {'top10_sm MAE':>12} {'±10':>7} {'r':>7}"
      f"  {'top5_sm MAE':>12} {'±10':>7} {'r':>7}")
print(f"  {'-'*90}")

import json

results = {}
for layer in tqdm(layers, desc="Logit lens sweep"):
    X = avg_data[layer]
    pca = PCA(n_components=min(n_components, X.shape[1], X.shape[0] - 1))
    pca.fit(X)
    lr = logit_lens_decode(X, A, pca, lm_head, tokenizer, n_components)
    results[layer] = lr

    am20 = lr["reconstruct_top20"]["argmax"]
    sm10 = lr["reconstruct_top10"]["softmax"]
    sm5 = lr["reconstruct_top5"]["softmax"]
    print(f"  L{layer:>5} {am20['mae']:>12.1f} {am20['frac_within_10']:>6.1%} {am20['r']:>7.3f}"
          f"  {sm10['mae']:>12.1f} {sm10['frac_within_10']:>6.1%} {sm10['r']:>7.3f}"
          f"  {sm5['mae']:>12.1f} {sm5['frac_within_10']:>6.1%} {sm5['r']:>7.3f}")

# Best layers (by lowest MAE)
for method_label, method_key in [("top-20 argmax", ("reconstruct_top20", "argmax")),
                                   ("top-10 argmax", ("reconstruct_top10", "argmax")),
                                   ("top-10 softmax", ("reconstruct_top10", "softmax")),
                                   ("top-5 softmax", ("reconstruct_top5", "softmax")),
                                   ("top-3 softmax", ("reconstruct_top3", "softmax")),
                                   ("top-1 softmax", ("reconstruct_top1", "softmax"))]:
    best = min(results.items(), key=lambda x: x[1][method_key[0]][method_key[1]]["mae"])
    r = best[1][method_key[0]][method_key[1]]
    print(f"\n  Best {method_label}: L{best[0]}  MAE={r['mae']:.1f}  ±10={r['frac_within_10']:.1%}  r={r['r']:.3f}")

# Save
serializable = {str(k): v for k, v in results.items()}
with open(output_dir / "logit_lens_layer_sweep.json", "w") as f:
    json.dump(serializable, f)
print(f"\nSaved to {output_dir / 'logit_lens_layer_sweep.json'}")

# Plot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

layer_list = sorted(results.keys())
methods = [
    ("top-20 argmax", "reconstruct_top20", "argmax"),
    ("top-10 argmax", "reconstruct_top10", "argmax"),
    ("top-10 softmax", "reconstruct_top10", "softmax"),
    ("top-5 softmax", "reconstruct_top5", "softmax"),
]

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# MAE plot
ax = axes[0]
for label, rkey, skey in methods:
    maes = [results[l][rkey][skey]["mae"] for l in layer_list]
    ax.plot(layer_list, maes, marker=".", markersize=4, label=label)
ax.set_ylabel("MAE(A)")
ax.set_title(f"Logit lens unsupervised decoder — avg filler, by layer ({condition})")
ax.legend(fontsize=9)
ax.set_ylim(0, None)
ax.axhline(y=30, color="gray", linestyle="--", alpha=0.4, label="MAE=30 ref")
ax.grid(alpha=0.3)

# ±10 plot
ax = axes[1]
for label, rkey, skey in methods:
    frac = [results[l][rkey][skey]["frac_within_10"] * 100 for l in layer_list]
    ax.plot(layer_list, frac, marker=".", markersize=4, label=label)
ax.set_ylabel("% within ±10")
ax.set_xlabel("Layer")
ax.legend(fontsize=9)
ax.set_ylim(0, None)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "logit_lens_layer_sweep.png", dpi=150, bbox_inches="tight")
plt.savefig(output_dir / "logit_lens_layer_sweep.pdf", bbox_inches="tight")
plt.close()
print(f"Plot saved to {output_dir / 'logit_lens_layer_sweep.png'}")
