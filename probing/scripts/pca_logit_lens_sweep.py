"""Quick standalone: logit lens layer sweep on avg_filler. No permutations."""

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

extraction_dir = Path("probing/extracted_states")
condition = "dots_100"
n_components = 20
lm_head_path = Path("probing/lm_head_weight.npy")
tokenizer_path = "/workspace/models/deepseek-v3-awq"
output_dir = Path("probing/results/pca_filler")
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

# Best layers
for method_label, method_key in [("top-20 argmax", ("reconstruct_top20", "argmax")),
                                   ("top-10 softmax", ("reconstruct_top10", "softmax")),
                                   ("top-5 softmax", ("reconstruct_top5", "softmax")),
                                   ("top-3 softmax", ("reconstruct_top3", "softmax")),
                                   ("top-1 softmax", ("reconstruct_top1", "softmax"))]:
    best = max(results.items(), key=lambda x: x[1][method_key[0]][method_key[1]]["r"])
    r = best[1][method_key[0]][method_key[1]]
    print(f"\n  Best {method_label}: L{best[0]}  r={r['r']:.3f}  MAE={r['mae']:.1f}  ±10={r['frac_within_10']:.1%}")

# Save
serializable = {str(k): v for k, v in results.items()}
with open(output_dir / "logit_lens_layer_sweep.json", "w") as f:
    json.dump(serializable, f)
print(f"\nSaved to {output_dir / 'logit_lens_layer_sweep.json'}")
