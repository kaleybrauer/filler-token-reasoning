"""Training loop for paired-view SAE.

Usage:
    uv run --project /workspace/filler-token-reasoning/probing \
        python scripts/sae/train.py \
        --extraction-dir data/extracted_states \
        --position answer_prompt \
        --layer 58 \
        --output-dir sae_results/answer_prompt_L58

    # Layer sweep:
    uv run --project /workspace/filler-token-reasoning/probing \
        python scripts/sae/train.py \
        --layer-sweep 55,56,57,58,59,60 \
        --output-dir sae_results/layer_sweep
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from configs import SAEConfig
from data_loader import load_paired_views, load_delta_views, PairedViewDataset
from model import PairedViewSAE
from losses import compute_total_loss


def compute_l0(z: torch.Tensor, threshold: float = 0.01) -> float:
    """Mean number of active latents (|z| > threshold) per example."""
    return (z.abs() > threshold).float().sum(dim=-1).mean().item()


def compute_cross_view_cosine(z_a: torch.Tensor, z_b: torch.Tensor) -> float:
    """Mean cosine similarity between z_shared for paired views."""
    cos = torch.nn.functional.cosine_similarity(z_a, z_b, dim=-1)
    return cos.mean().item()


def explained_variance(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    """1 - MSE / Var(x)."""
    mse = ((x - x_hat) ** 2).mean().item()
    var = x.var().item()
    return 1 - mse / max(var, 1e-8)


def train_one_config(config: SAEConfig):
    """Train a single SAE configuration. Returns results dict."""
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    extraction_dir = Path(config.extraction_dir)
    categories_file = Path(config.categories_file)

    if config.delta:
        print(f"Loading delta views: {config.conditions} at {config.delta_positions} L{config.layer}")
        view_a, view_b, metadata = load_delta_views(
            extraction_dir, config.conditions, config.delta_positions, config.layer,
            categories_file=categories_file,
        )
        print(f"  {len(metadata)} examples, dim={view_a.shape[1]} (delta mode)")
    else:
        print(f"Loading views: {config.conditions} at {config.position} L{config.layer}")
        view_a, view_b, metadata = load_paired_views(
            extraction_dir, config.conditions, config.position, config.layer,
            categories_file=categories_file,
        )
        print(f"  {len(metadata)} examples, dim={view_a.shape[1]}")

    # Build dataset (filtered to included categories)
    include_cats = config.include_categories if config.include_categories else None
    dataset = PairedViewDataset(
        view_a, view_b, metadata, config.conditions, normalize=True,
        include_categories=include_cats,
    )
    print(f"  After filtering: {len(dataset)} examples "
          f"(categories: {include_cats or 'all'})")
    loader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True,
        drop_last=False, num_workers=0,
    )

    # Ground truth for logging
    gt = dataset.get_ground_truth()
    n_both_correct = gt["correct_mask"].sum()
    print(f"  Both-correct: {n_both_correct}/{len(metadata)}")

    # Model
    model = PairedViewSAE(
        input_dim=config.input_dim,
        encoder_hidden=config.encoder_hidden,
        dim_shared=config.dim_shared,
        dim_view=config.dim_view,
        topk_shared=config.topk_shared,
        topk_view=config.topk_view,
    )
    print(f"  Model: {model}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    # Training loop
    history = []
    t0 = time.time()

    for epoch in range(config.n_epochs):
        model.train()
        epoch_losses = {
            "total": 0, "recon": 0, "sparse_shared": 0, "sparse_view": 0,
            "consist": 0, "separate": 0,
        }
        epoch_l0_shared = 0
        epoch_l0_view = 0
        epoch_cosine = 0
        epoch_ev = 0
        n_batches = 0

        for x_a, x_b, correct_mask, idx in loader:
            # Forward both views
            x_hat_a, z_shared_a, z_view_a = model(x_a)
            x_hat_b, z_shared_b, z_view_b = model(x_b)

            # Compute losses
            losses = compute_total_loss(
                x_a, x_b, x_hat_a, x_hat_b,
                z_shared_a, z_shared_b, z_view_a, z_view_b,
                correct_mask,
                lambda_shared=config.lambda_shared,
                lambda_view=config.lambda_view,
                lambda_consist=config.lambda_consist,
                lambda_separate=config.lambda_separate,
            )

            # Backward
            optimizer.zero_grad()
            losses["total"].backward()
            optimizer.step()

            # Accumulate metrics
            for k, v in losses.items():
                epoch_losses[k] += v.item()
            epoch_l0_shared += (compute_l0(z_shared_a) + compute_l0(z_shared_b)) / 2
            epoch_l0_view += (compute_l0(z_view_a) + compute_l0(z_view_b)) / 2
            epoch_cosine += compute_cross_view_cosine(z_shared_a, z_shared_b)
            epoch_ev += (explained_variance(x_a, x_hat_a) + explained_variance(x_b, x_hat_b)) / 2
            n_batches += 1

        # Average over batches
        for k in epoch_losses:
            epoch_losses[k] /= n_batches
        epoch_l0_shared /= n_batches
        epoch_l0_view /= n_batches
        epoch_cosine /= n_batches
        epoch_ev /= n_batches

        record = {
            "epoch": epoch,
            **epoch_losses,
            "l0_shared": epoch_l0_shared,
            "l0_view": epoch_l0_view,
            "cosine_shared": epoch_cosine,
            "explained_var": epoch_ev,
        }
        history.append(record)

        if epoch % 10 == 0 or epoch == config.n_epochs - 1:
            print(f"  Epoch {epoch:>4}: "
                  f"total={epoch_losses['total']:.4f} "
                  f"recon={epoch_losses['recon']:.4f} "
                  f"consist={epoch_losses['consist']:.4f} "
                  f"L0_sh={epoch_l0_shared:.1f} "
                  f"L0_v={epoch_l0_view:.1f} "
                  f"cos={epoch_cosine:.3f} "
                  f"EV={epoch_ev:.3f}")

        # Checkpoint
        if (epoch + 1) % 100 == 0 or epoch == config.n_epochs - 1:
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config.__dict__,
                "epoch": epoch,
                "history": history,
                "stats": dataset.get_stats(),
            }, output_dir / "checkpoint.pt")

    elapsed = time.time() - t0
    print(f"  Training complete: {elapsed:.1f}s")

    # Save final model and training history
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config.__dict__,
        "epoch": config.n_epochs - 1,
        "history": history,
        "stats": dataset.get_stats(),
    }, output_dir / "model.pt")

    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f)

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config.__dict__, f, indent=2)

    return model, dataset, history


def main():
    parser = argparse.ArgumentParser(description="Train paired-view SAE")
    parser.add_argument("--extraction-dir", type=str, default="data/extracted_states")
    parser.add_argument("--categories", type=str,
                        default="results/probe_results/categories.json")
    parser.add_argument("--conditions", nargs=2, default=["baseline", "dots_50"])
    parser.add_argument("--position", type=str, default="answer_prompt")
    parser.add_argument("--delta", action="store_true",
                        help="Train on h(cond_b) - h(cond_a) at two positions")
    parser.add_argument("--delta-positions", nargs=2,
                        default=["answer_prompt", "pre_answer"],
                        help="Two positions for delta views (default: answer_prompt pre_answer)")
    parser.add_argument("--layer", type=int, default=58)
    parser.add_argument("--layer-sweep", type=str, default=None,
                        help="Comma-separated layers to sweep (trains one SAE per layer)")
    parser.add_argument("--output-dir", type=str, default="sae_results")

    # Model
    parser.add_argument("--encoder-hidden", type=int, default=512)
    parser.add_argument("--dim-shared", type=int, default=64)
    parser.add_argument("--dim-view", type=int, default=32)
    parser.add_argument("--topk-shared", type=int, default=0,
                        help="TopK sparsity for z_shared (0=L1 only, >0=keep top K)")
    parser.add_argument("--topk-view", type=int, default=0,
                        help="TopK sparsity for z_view (0=L1 only)")

    # Training
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)

    # Loss weights
    parser.add_argument("--lambda-shared", type=float, default=0.01)
    parser.add_argument("--lambda-view", type=float, default=0.01)
    parser.add_argument("--lambda-consist", type=float, default=1.0)
    parser.add_argument("--lambda-separate", type=float, default=0.1)

    # Filtering
    parser.add_argument("--include-categories", nargs="+", default=None,
                        help="Only train on these categories (default: all)")
    parser.add_argument("--all-categories", action="store_true",
                        help="Train on all categories (same as default, kept for clarity)")

    # Evaluation
    parser.add_argument("--evaluate", action="store_true",
                        help="Run evaluation after training")

    args = parser.parse_args()

    include_cats = tuple(args.include_categories) if args.include_categories else None

    if args.layer_sweep:
        layers = [int(l) for l in args.layer_sweep.split(",")]
        for layer in layers:
            print(f"\n{'='*60}")
            print(f"  Layer {layer}")
            print(f"{'='*60}")

            config = SAEConfig(
                extraction_dir=args.extraction_dir,
                categories_file=args.categories,
                conditions=tuple(args.conditions),
                position=args.position,
                layer=layer,
                encoder_hidden=args.encoder_hidden,
                dim_shared=args.dim_shared,
                dim_view=args.dim_view,
                delta=args.delta,
                delta_positions=tuple(args.delta_positions),
                topk_shared=args.topk_shared,
                topk_view=args.topk_view,
                lr=args.lr,
                batch_size=args.batch_size,
                n_epochs=args.n_epochs,
                seed=args.seed,
                lambda_shared=args.lambda_shared,
                lambda_view=args.lambda_view,
                lambda_consist=args.lambda_consist,
                lambda_separate=args.lambda_separate,
                include_categories=include_cats,
                output_dir=f"{args.output_dir}/L{layer}",
            )
            model, dataset, history = train_one_config(config)

            if args.evaluate:
                from evaluate import run_evaluation
                run_evaluation(model, dataset, Path(config.output_dir))
    else:
        config = SAEConfig(
            extraction_dir=args.extraction_dir,
            categories_file=args.categories,
            conditions=tuple(args.conditions),
            position=args.position,
            layer=args.layer,
            encoder_hidden=args.encoder_hidden,
            dim_shared=args.dim_shared,
            dim_view=args.dim_view,
            topk_shared=args.topk_shared,
            topk_view=args.topk_view,
            lr=args.lr,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            seed=args.seed,
            lambda_shared=args.lambda_shared,
            lambda_view=args.lambda_view,
            lambda_consist=args.lambda_consist,
            lambda_separate=args.lambda_separate,
            include_categories=include_cats,
            output_dir=args.output_dir,
        )
        model, dataset, history = train_one_config(config)

        if args.evaluate:
            from evaluate import run_evaluation
            run_evaluation(model, dataset, Path(config.output_dir))

    print("\nDone.")


if __name__ == "__main__":
    main()
