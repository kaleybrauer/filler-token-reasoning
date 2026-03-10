"""
Analyze filler token embeddings in Qwen's embedding space.

Computes:
1. Pairwise cosine distances within each filler type
2. Distance from filler tokens to "useful" number tokens (0-200)
3. Effective dimensionality (PCA) of each filler type's embedding subspace
4. Visualization of embeddings via PCA projection

Usage:
    python analyze_embeddings.py --model Qwen/Qwen2.5-72B-Instruct [--load-in-4bit]
"""

import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from sklearn.decomposition import PCA
import json
import os

def get_token_ids(tokenizer, texts):
    """Get single-token IDs for a list of texts. Returns list of (text, token_id) pairs."""
    results = []
    for t in texts:
        ids = tokenizer.encode(t, add_special_tokens=False)
        for tid in ids:
            results.append((t, tid))
    return results


def get_embeddings(model, token_ids, device="cpu"):
    """Extract embedding vectors for a list of token IDs."""
    # Get the embedding layer
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        embed = model.model.embed_tokens
    elif hasattr(model, 'get_input_embeddings'):
        embed = model.get_input_embeddings()
    else:
        raise ValueError("Cannot find embedding layer")
    
    with torch.no_grad():
        ids_tensor = torch.tensor(token_ids, dtype=torch.long, device=embed.weight.device)
        vecs = embed(ids_tensor).float().cpu().numpy()
    return vecs


def analyze_filler_type(name, token_texts, embeddings, number_embeddings, number_labels):
    """Analyze a single filler type's embeddings."""
    n = len(embeddings)
    print(f"\n{'='*60}")
    print(f"  {name}: {n} tokens")
    print(f"{'='*60}")
    
    if n == 0:
        print("  No tokens found!")
        return {}
    
    # Unique embeddings
    unique_vecs = np.unique(embeddings, axis=0)
    n_unique = len(unique_vecs)
    print(f"  Unique embedding vectors: {n_unique} / {n}")
    
    # Pairwise cosine distances within filler type
    if n_unique > 1:
        dists = cosine_distances(unique_vecs)
        # Get upper triangle (exclude diagonal)
        upper = dists[np.triu_indices_from(dists, k=1)]
        print(f"  Pairwise cosine distance (unique vecs):")
        print(f"    Mean: {upper.mean():.4f}")
        print(f"    Std:  {upper.std():.4f}")
        print(f"    Min:  {upper.min():.4f}")
        print(f"    Max:  {upper.max():.4f}")
    else:
        print(f"  All tokens have identical embeddings (1 unique vector)")
    
    # Effective dimensionality via PCA
    if n_unique > 1:
        n_components = min(n_unique, 50)
        pca = PCA(n_components=n_components)
        pca.fit(unique_vecs)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        dim_90 = np.searchsorted(cumvar, 0.90) + 1
        dim_95 = np.searchsorted(cumvar, 0.95) + 1
        dim_99 = np.searchsorted(cumvar, 0.99) + 1
        print(f"  Effective dimensionality (PCA):")
        print(f"    Components for 90% variance: {dim_90}")
        print(f"    Components for 95% variance: {dim_95}")
        print(f"    Components for 99% variance: {dim_99}")
        print(f"    Top 5 explained variance: {pca.explained_variance_ratio_[:5].tolist()}")
    else:
        print(f"  Effective dimensionality: 0 (single point)")
    
    # Distance to number embeddings
    if len(number_embeddings) > 0:
        # Mean cosine similarity to all number tokens
        sim_to_numbers = cosine_similarity(embeddings, number_embeddings)
        mean_sim = sim_to_numbers.mean()
        max_sim_per_filler = sim_to_numbers.max(axis=1)  # closest number for each filler token
        print(f"  Cosine similarity to number tokens (0-200):")
        print(f"    Mean similarity: {mean_sim:.4f}")
        print(f"    Mean of max similarity (closest number): {max_sim_per_filler.mean():.4f}")
        
        # Which number tokens are closest to these filler tokens?
        closest_indices = sim_to_numbers.mean(axis=0).argsort()[-5:][::-1]
        print(f"    Top 5 closest number tokens (by mean sim):")
        for idx in closest_indices:
            print(f"      '{number_labels[idx]}': {sim_to_numbers.mean(axis=0)[idx]:.4f}")
    
    return {
        "name": name,
        "n_tokens": n,
        "n_unique": n_unique,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-72B-Instruct")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--outdir", type=str, default="results/embedding_analysis")
    args = parser.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    
    # Load tokenizer
    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
    
    # Load model (only need embeddings, but need to load full model for 4-bit)
    print(f"Loading model: {args.model}")
    model_kwargs = {"torch_dtype": torch.bfloat16}
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = "auto"
    
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.eval()
    
    # ── Define filler token sets ──
    
    # 1. Dot filler: just "."
    dot_texts = ["."]
    dot_pairs = get_token_ids(tokenizer, dot_texts)
    
    # 2. Counting filler: tokens that appear in "1 2 3 ... 256"
    counting_str = " ".join(str(i) for i in range(1, 257))
    counting_ids = tokenizer.encode(counting_str, add_special_tokens=False)
    counting_unique_ids = sorted(set(counting_ids))
    counting_pairs = [(tokenizer.decode([tid]), tid) for tid in counting_unique_ids]
    
    # 3. Alphabet filler: a-z, A-Z
    alpha_texts = [f" {c}" for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"]
    alpha_pairs = get_token_ids(tokenizer, alpha_texts)
    # Deduplicate
    seen = set()
    alpha_pairs_dedup = []
    for text, tid in alpha_pairs:
        if tid not in seen:
            alpha_pairs_dedup.append((text, tid))
            seen.add(tid)
    alpha_pairs = alpha_pairs_dedup
    
    # 4. Digit-letter replacements: the letter tokens that replace digits
    digit_letter_map = {
        "0": " a", "1": " b", "2": " c", "3": " d", "4": " e",
        "5": " f", "6": " g", "7": " h", "8": " i", "9": " j",
    }
    dl_texts = list(digit_letter_map.values())
    dl_pairs = get_token_ids(tokenizer, dl_texts)
    seen = set()
    dl_pairs_dedup = []
    for text, tid in dl_pairs:
        if tid not in seen:
            dl_pairs_dedup.append((text, tid))
            seen.add(tid)
    dl_pairs = dl_pairs_dedup
    
    # 5. "Useful" number tokens: 0-200 as standalone tokens
    number_texts = [f" {i}" for i in range(201)]
    number_pairs = get_token_ids(tokenizer, number_texts)
    
    # Print token info
    print(f"\n{'='*60}")
    print("TOKEN INVENTORY")
    print(f"{'='*60}")
    print(f"Dot tokens: {len(dot_pairs)} — {dot_pairs}")
    print(f"Counting tokens (unique in '1 2 3...256'): {len(counting_pairs)}")
    print(f"  IDs: {[tid for _, tid in counting_pairs]}")
    print(f"  Decoded: {[text for text, _ in counting_pairs]}")
    print(f"Alphabet tokens: {len(alpha_pairs)}")
    print(f"  Decoded: {[text for text, _ in alpha_pairs]}")
    print(f"Digit-letter tokens: {len(dl_pairs)}")
    print(f"  Decoded: {[text for text, _ in dl_pairs]}")
    print(f"Number tokens (0-200): {len(number_pairs)}")
    
    # ── Extract embeddings ──
    print("\nExtracting embeddings...")
    
    dot_embeds = get_embeddings(model, [tid for _, tid in dot_pairs])
    counting_embeds = get_embeddings(model, [tid for _, tid in counting_pairs])
    alpha_embeds = get_embeddings(model, [tid for _, tid in alpha_pairs])
    dl_embeds = get_embeddings(model, [tid for _, tid in dl_pairs])
    number_embeds = get_embeddings(model, [tid for _, tid in number_pairs])
    number_labels = [text for text, _ in number_pairs]
    
    # ── Analyze each filler type ──
    results = []
    results.append(analyze_filler_type("Dots", [t for t, _ in dot_pairs], dot_embeds, number_embeds, number_labels))
    results.append(analyze_filler_type("Counting (unique tokens in 1..256)", [t for t, _ in counting_pairs], counting_embeds, number_embeds, number_labels))
    results.append(analyze_filler_type("Alphabet (a-z, A-Z)", [t for t, _ in alpha_pairs], alpha_embeds, number_embeds, number_labels))
    results.append(analyze_filler_type("Digit-letter (a-j replacing 0-9)", [t for t, _ in dl_pairs], dl_embeds, number_embeds, number_labels))
    
    # ── Cross-type comparison ──
    print(f"\n{'='*60}")
    print("CROSS-TYPE COMPARISON")
    print(f"{'='*60}")
    
    # Combine all unique filler embeddings and project to shared PCA space
    all_embeds = np.vstack([counting_embeds, alpha_embeds, dl_embeds, number_embeds])
    all_labels = (
        [f"count:{t}" for t, _ in counting_pairs] +
        [f"alpha:{t}" for t, _ in alpha_pairs] +
        [f"dl:{t}" for t, _ in dl_pairs] +
        [f"num:{t}" for t, _ in number_pairs]
    )
    
    pca = PCA(n_components=2)
    projected = pca.fit_transform(all_embeds)
    print(f"PCA on all filler + number tokens:")
    print(f"  Variance explained: {pca.explained_variance_ratio_}")
    
    # Mean cosine similarity between filler types and number tokens
    types = {
        "Counting": counting_embeds,
        "Alphabet": alpha_embeds, 
        "Digit-letter": dl_embeds,
    }
    for name, embeds in types.items():
        sim = cosine_similarity(embeds, number_embeds).mean()
        print(f"  Mean cosine sim ({name} <-> Numbers): {sim:.4f}")
    
    # Cosine similarity between digit-letter tokens and their digit counterparts
    print(f"\n  Digit vs Digit-letter token similarity:")
    for digit_char, letter_text in digit_letter_map.items():
        digit_id = tokenizer.encode(f" {digit_char}", add_special_tokens=False)
        letter_id = tokenizer.encode(letter_text, add_special_tokens=False)
        if digit_id and letter_id:
            d_emb = get_embeddings(model, digit_id[:1])
            l_emb = get_embeddings(model, letter_id[:1])
            sim = cosine_similarity(d_emb, l_emb)[0, 0]
            print(f"    '{digit_char}' vs '{letter_text.strip()}': {sim:.4f}")
    
    # Save PCA coordinates for plotting
    pca_data = {
        "labels": all_labels,
        "x": projected[:, 0].tolist(),
        "y": projected[:, 1].tolist(),
        "variance_explained": pca.explained_variance_ratio_.tolist(),
    }
    outfile = os.path.join(args.outdir, "pca_embeddings.json")
    with open(outfile, "w") as f:
        json.dump(pca_data, f, indent=2)
    print(f"\nSaved PCA data to {outfile}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()