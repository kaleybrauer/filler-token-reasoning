"""Paired-View Sparse Autoencoder model."""

import torch
import torch.nn as nn
from typing import Tuple


def _topk_mask(z: torch.Tensor, k: int) -> torch.Tensor:
    """Keep only top-k absolute values per row, zero the rest."""
    if k >= z.shape[-1]:
        return z
    topk_vals, topk_idx = torch.topk(z.abs(), k, dim=-1)
    mask = torch.zeros_like(z)
    mask.scatter_(-1, topk_idx, 1.0)
    return z * mask


class PairedViewSAE(nn.Module):
    """Sparse autoencoder with shared and view-specific latent spaces.

    Architecture:
        h (input_dim) → encoder trunk (ReLU) → z_shared + z_view → decoder → h_hat

    z_shared captures problem-invariant information (stable across views).
    z_view captures condition-specific nuisance (filler type, format, position).
    Sparsity via L1 penalty and/or TopK activation.
    """

    def __init__(
        self,
        input_dim: int = 7168,
        encoder_hidden: int = 512,
        dim_shared: int = 64,
        dim_view: int = 32,
        topk_shared: int = 0,
        topk_view: int = 0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.encoder_hidden = encoder_hidden
        self.dim_shared = dim_shared
        self.dim_view = dim_view
        self.topk_shared = topk_shared  # 0 = no topk, use L1 only
        self.topk_view = topk_view

        # Encoder trunk: shared across both latent spaces
        self.encoder_trunk = nn.Sequential(
            nn.Linear(input_dim, encoder_hidden),
            nn.ReLU(),
        )

        # Latent heads
        self.shared_head = nn.Linear(encoder_hidden, dim_shared)
        self.view_head = nn.Linear(encoder_hidden, dim_view)

        # Decoder: reconstructs from concatenated latents
        self.decoder = nn.Linear(dim_shared + dim_view, input_dim)

        self._init_weights()

    def _init_weights(self):
        # Kaiming for ReLU layers
        nn.init.kaiming_uniform_(self.encoder_trunk[0].weight, nonlinearity="relu")
        nn.init.zeros_(self.encoder_trunk[0].bias)

        # Xavier for linear heads
        nn.init.xavier_uniform_(self.shared_head.weight)
        nn.init.zeros_(self.shared_head.bias)
        nn.init.xavier_uniform_(self.view_head.weight)
        nn.init.zeros_(self.view_head.bias)

        # Decoder: unit-norm columns (helps with sparse feature scaling)
        nn.init.xavier_uniform_(self.decoder.weight)
        with torch.no_grad():
            self.decoder.weight.div_(
                self.decoder.weight.norm(dim=0, keepdim=True).clamp(min=1e-8)
            )
        nn.init.zeros_(self.decoder.bias)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to (z_shared, z_view)."""
        h = self.encoder_trunk(x)
        z_shared = self.shared_head(h)
        z_view = self.view_head(h)

        if self.topk_shared > 0:
            z_shared = _topk_mask(z_shared, self.topk_shared)
        if self.topk_view > 0:
            z_view = _topk_mask(z_view, self.topk_view)

        return z_shared, z_view

    def decode(self, z_shared: torch.Tensor, z_view: torch.Tensor) -> torch.Tensor:
        """Decode from latent spaces to reconstruction."""
        z_cat = torch.cat([z_shared, z_view], dim=-1)
        return self.decoder(z_cat)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass.

        Args:
            x: (batch, input_dim) input activation

        Returns:
            x_hat: (batch, input_dim) reconstruction
            z_shared: (batch, dim_shared) shared latents
            z_view: (batch, dim_view) view-specific latents
        """
        z_shared, z_view = self.encode(x)
        x_hat = self.decode(z_shared, z_view)
        return x_hat, z_shared, z_view

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def __repr__(self):
        return (
            f"PairedViewSAE(input={self.input_dim}, hidden={self.encoder_hidden}, "
            f"shared={self.dim_shared}, view={self.dim_view}, "
            f"params={self.n_params():,})"
        )
