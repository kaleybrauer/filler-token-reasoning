---
license: mit
base_model: deepseek-ai/DeepSeek-V3-0324
tags:
- interpretability
- jacobian-lens
- deepseek-v3
---

# Jacobian lens for DeepSeek-V3-0324

A [Jacobian lens](https://transformer-circuits.pub/2026/workspace/index.html) (J-lens) for DeepSeek-V3-0324, fit with
Anthropic's reference estimator ([`jacobian-lens`](https://github.com/anthropics/jacobian-lens)) on the 4-bit AWQ
checkpoint [`cognitivecomputations/DeepSeek-V3-0324-AWQ`](https://huggingface.co/cognitivecomputations/DeepSeek-V3-0324-AWQ).
Write-up: [LINK].

## Files

| File | Target layer | Prompts | Source layers | Size | What it's for |
|---|---|---|---|---|---|
| `lens.pt` | 60 (final) | 100 | 0–59 | 6.2 GB | The main lens |
| `halves/half_a.pt`, `halves/half_b.pt` | 60 (final) | 50 each, disjoint | 0–59 | 6.2 GB each | Error bars: their prompt-weighted mean is `lens.pt` |
| `penultimate/target59_n10.pt` | 59 (penultimate) | 10 | 24 layers* | 2.5 GB | Same estimator, last block left out |
| `penultimate/target60_n10.pt` | 60 (final) | the same 10 | 24 layers* | 2.5 GB | The paired comparison for the file above |
| `meta/` | | | | small | Build records, per-prompt Jacobian norms, the fit prompts |

\* 0, 2, 5, 8, 10, 12, 15, 18, 20, 22, 25, 28, 30, 32, 35, 38, 40, 42, 45, 48, 50, 52, 55, 58 (the 25-layer grid of the
paper's Figure 28, below the target).

All files are in the library's own format (`JacobianLens.save`, fp16).

## Loading

```python
import jlens
lens = jlens.JacobianLens.from_pretrained("kbrauer/deepseek-v3-jacobian-lens")                      # lens.pt
pen = jlens.JacobianLens.from_pretrained("kbrauer/deepseek-v3-jacobian-lens", filename="penultimate/target59_n10.pt")
```

The library upcasts to float32 on load, so `lens.pt` needs about 12 GB of RAM. To read one layer at a time instead,
`torch.load(path, mmap=True, weights_only=True)["J"][layer]`.

## Read this before using `lens.pt`

`lens.pt` uses the library's default target, the final layer. On DeepSeek-V3 that matters a lot. The model's last
block amplifies one direction about twelve times more than a typical one, and that direction separates
Chinese-character tokens from Latin-alphabet tokens. (The next direction is amplified almost as much, 10.0 against
12.1, and separates the scripts almost as well, so this is a two-dimensional script subspace.) A lens that
differentiates through the last block inherits it as the top singular direction of every layer's Jacobian, so every
readout carries a script shift that the residual stream at that layer doesn't contain and that doesn't depend on
the input:

- at half depth, 46% of readouts on English text have mostly Chinese tokens in their top 20 (logit lens: 4%);
- the three script-class means explain 19–53% of a readout's variance at every layer (logit lens: 0.2–0.3%);
- final-target lenses also lift the whole block of number tokens in the first two-thirds of the network, which
  inflates any "best rank over layers" metric on numeric intermediates.

The paired lenses in `penultimate/` show the effect directly: the same ten prompts with layer 59 as the target cut the
script share about twenty-fold. They were fit on ten prompts and hold 24 layers, so treat them as a comparison, not a
replacement. The Anthropic paper's own Sonnet lens used the penultimate target.

## How it was fit

- Model: `cognitivecomputations/DeepSeek-V3-0324-AWQ` (61 layers, d_model 7168), on 3× H200.
- Estimator: `jlens.fitting` as released, with the model's MoE and AWQ layers patched for autograd. Settings:
  target layer 60, source layers 0–59, 128-token prompts, the first 16 positions skipped, dim_batch 64, cotangent
  scale 1/64, gradient checkpointing. About 40 minutes per prompt.
- Prompts: one middle paragraph per article from WikiText-103 (train split), at least 600 characters. Indices and
  text are in `meta/fit_prompts.json`.
- Pre-filter: a plain average over the first 100 prompts was dominated by four of them (84% of the summed Jacobian
  norm). Prompts whose largest ‖J_ℓ‖_F/√d over layers exceeded 40 (five times the median) were removed, twelve in
  all, and replaced by the next twelve. Their norms are in `meta/per_prompt_norms.json`; the four largest drop to
  5.5–21 when refit with the penultimate target.
- Checks: a stored Jacobian matches finite differences of the 4-bit forward pass to 0.5% at layer 59 and 2% at layer
  50 (one prompt). The two halves pick the same top-1 token 13% of the time at a quarter of the depth, 58% at half,
  90% at 83%: trust the first third of the network less.
- MoE routing depends on the batch, so single-prompt Jacobians do too; averaged readouts agree about 95% either way.

## License

The lens files: MIT, matching DeepSeek-V3-0324. `meta/fit_prompts.json` contains excerpts of WikiText-103
(Merity et al., 2016), CC BY-SA 3.0.

## Citation

If you use these lenses, please cite the write-up [LINK] and the paper:
Gurnee, Sofroniew, et al. (2026), *Verbalizable Representations Form a Global Workspace in Language Models*.
