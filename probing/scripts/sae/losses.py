"""Loss functions for paired-view SAE training."""

import torch
from typing import Dict


def reconstruction_loss(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """MSE reconstruction loss, normalized per dimension."""
    return ((x - x_hat) ** 2).mean()


def sparsity_loss(z: torch.Tensor) -> torch.Tensor:
    """L1 sparsity penalty, mean over batch and dimensions."""
    return z.abs().mean()


def consistency_loss(
    z_shared_a: torch.Tensor,
    z_shared_b: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Same-problem consistency: z_shared should match across views.

    Only applied to examples where the model is correct in both conditions.

    Args:
        z_shared_a: (B, dim_shared) — z_shared for view A
        z_shared_b: (B, dim_shared) — z_shared for view B
        mask: (B,) bool — True for examples to include
    """
    if mask.sum() == 0:
        return torch.tensor(0.0, device=z_shared_a.device)

    diff = z_shared_a[mask] - z_shared_b[mask]
    return (diff ** 2).mean()


def separation_loss(z_shared: torch.Tensor, target_std: float = 1.0) -> torch.Tensor:
    """Different-problem separation: prevent z_shared from collapsing.

    Penalizes deviation from a target standard deviation. If std is too low
    (collapse), this pushes it up. If std is too high (explosion), this
    constrains it. Bounded loss — doesn't go to -infinity.

    Args:
        z_shared: (B, dim_shared) — concatenated z_shared from both views
        target_std: target per-dimension standard deviation
    """
    std_per_dim = z_shared.std(dim=0)  # (dim_shared,)
    return ((std_per_dim - target_std) ** 2).mean()


def compute_total_loss(
    x_a: torch.Tensor,
    x_b: torch.Tensor,
    x_hat_a: torch.Tensor,
    x_hat_b: torch.Tensor,
    z_shared_a: torch.Tensor,
    z_shared_b: torch.Tensor,
    z_view_a: torch.Tensor,
    z_view_b: torch.Tensor,
    correct_mask: torch.Tensor,
    lambda_shared: float = 0.01,
    lambda_view: float = 0.01,
    lambda_consist: float = 1.0,
    lambda_separate: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """Compute all losses and return a dict.

    Args:
        x_a, x_b: (B, D) input activations for views A and B
        x_hat_a, x_hat_b: (B, D) reconstructions
        z_shared_a, z_shared_b: (B, dim_shared) shared latents
        z_view_a, z_view_b: (B, dim_view) view-specific latents
        correct_mask: (B,) bool — True for examples correct in both conditions

    Returns:
        Dict with "total" and all component losses.
    """
    l_recon_a = reconstruction_loss(x_a, x_hat_a)
    l_recon_b = reconstruction_loss(x_b, x_hat_b)
    l_recon = (l_recon_a + l_recon_b) / 2

    l_sparse_shared = (sparsity_loss(z_shared_a) + sparsity_loss(z_shared_b)) / 2
    l_sparse_view = (sparsity_loss(z_view_a) + sparsity_loss(z_view_b)) / 2

    l_consist = consistency_loss(z_shared_a, z_shared_b, correct_mask)

    z_shared_all = torch.cat([z_shared_a, z_shared_b], dim=0)
    l_separate = separation_loss(z_shared_all)

    total = (
        l_recon
        + lambda_shared * l_sparse_shared
        + lambda_view * l_sparse_view
        + lambda_consist * l_consist
        + lambda_separate * l_separate
    )

    return {
        "total": total,
        "recon": l_recon,
        "sparse_shared": l_sparse_shared,
        "sparse_view": l_sparse_view,
        "consist": l_consist,
        "separate": l_separate,
    }
