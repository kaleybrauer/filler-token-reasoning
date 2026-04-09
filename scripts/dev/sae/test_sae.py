"""
Unit tests for paired-view SAE.

Run:
    uv run --project /workspace/filler-token-reasoning/probing \
        python -m pytest probing/scripts/sae/test_sae.py -v
"""

import numpy as np
import pytest
import torch

from configs import SAEConfig
from data_loader import ExampleMetadata, PairedViewDataset
from model import PairedViewSAE
from losses import (
    reconstruction_loss,
    sparsity_loss,
    consistency_loss,
    separation_loss,
    compute_total_loss,
)


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def config():
    return SAEConfig(
        input_dim=128,
        encoder_hidden=32,
        dim_shared=8,
        dim_view=4,
        batch_size=16,
        n_epochs=5,
    )


@pytest.fixture
def model(config):
    return PairedViewSAE(
        input_dim=config.input_dim,
        encoder_hidden=config.encoder_hidden,
        dim_shared=config.dim_shared,
        dim_view=config.dim_view,
    )


@pytest.fixture
def dummy_data():
    """Create dummy paired-view data."""
    rng = np.random.RandomState(42)
    n = 50
    dim = 128

    # Create data where view_a and view_b share a common signal
    signal = rng.randn(n, 10).astype(np.float32)
    noise_a = rng.randn(n, dim).astype(np.float32) * 0.1
    noise_b = rng.randn(n, dim).astype(np.float32) * 0.1

    # Embed signal into both views via a shared projection
    proj = rng.randn(10, dim).astype(np.float32)
    view_a = signal @ proj + noise_a
    view_b = signal @ proj + noise_b

    metadata = []
    for i in range(n):
        correct_a = rng.random() > 0.3
        correct_b = rng.random() > 0.3
        if correct_a and correct_b:
            cat = "both_correct"
        elif not correct_a and correct_b:
            cat = "filler_helped"
        elif correct_a and not correct_b:
            cat = "filler_hurt"
        else:
            cat = "both_wrong"

        metadata.append(ExampleMetadata(
            problem_idx=i,
            fact_value=rng.randint(1, 100),
            x=rng.randint(1, 100),
            answer=rng.randint(2, 200),
            model_correct={"baseline": correct_a, "dots_50": correct_b},
            category=cat,
        ))

    return view_a, view_b, metadata


# ==============================================================================
# Model tests
# ==============================================================================

class TestModel:
    def test_forward_shapes(self, model, config):
        x = torch.randn(16, config.input_dim)
        x_hat, z_shared, z_view = model(x)

        assert x_hat.shape == (16, config.input_dim)
        assert z_shared.shape == (16, config.dim_shared)
        assert z_view.shape == (16, config.dim_view)

    def test_encode_shapes(self, model, config):
        x = torch.randn(8, config.input_dim)
        z_shared, z_view = model.encode(x)

        assert z_shared.shape == (8, config.dim_shared)
        assert z_view.shape == (8, config.dim_view)

    def test_decode_shapes(self, model, config):
        z_shared = torch.randn(8, config.dim_shared)
        z_view = torch.randn(8, config.dim_view)
        x_hat = model.decode(z_shared, z_view)

        assert x_hat.shape == (8, config.input_dim)

    def test_roundtrip(self, model, config):
        """Forward pass should be equivalent to encode + decode."""
        x = torch.randn(8, config.input_dim)
        x_hat_1, z_shared_1, z_view_1 = model(x)

        z_shared_2, z_view_2 = model.encode(x)
        x_hat_2 = model.decode(z_shared_2, z_view_2)

        torch.testing.assert_close(x_hat_1, x_hat_2)
        torch.testing.assert_close(z_shared_1, z_shared_2)

    def test_n_params(self, model):
        assert model.n_params() > 0
        # Verify count is consistent
        manual_count = sum(p.numel() for p in model.parameters())
        assert model.n_params() == manual_count

    def test_gradients_flow(self, model, config):
        x = torch.randn(8, config.input_dim, requires_grad=False)
        x_hat, z_shared, z_view = model(x)
        loss = x_hat.sum()
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.all(param.grad == 0), f"Zero gradient for {name}"


# ==============================================================================
# Loss tests
# ==============================================================================

class TestLosses:
    def test_reconstruction_loss_zero(self):
        x = torch.randn(8, 64)
        assert reconstruction_loss(x, x).item() == pytest.approx(0.0, abs=1e-6)

    def test_reconstruction_loss_positive(self):
        x = torch.randn(8, 64)
        x_hat = x + torch.randn_like(x)
        assert reconstruction_loss(x, x_hat).item() > 0

    def test_sparsity_loss_zero(self):
        z = torch.zeros(8, 16)
        assert sparsity_loss(z).item() == pytest.approx(0.0, abs=1e-6)

    def test_sparsity_loss_positive(self):
        z = torch.randn(8, 16)
        assert sparsity_loss(z).item() > 0

    def test_sparsity_increases_with_magnitude(self):
        z_small = torch.randn(8, 16) * 0.1
        z_large = torch.randn(8, 16) * 10.0
        assert sparsity_loss(z_small) < sparsity_loss(z_large)

    def test_consistency_loss_zero_when_identical(self):
        z = torch.randn(8, 16)
        mask = torch.ones(8, dtype=torch.bool)
        assert consistency_loss(z, z, mask).item() == pytest.approx(0.0, abs=1e-6)

    def test_consistency_loss_positive_when_different(self):
        z_a = torch.randn(8, 16)
        z_b = torch.randn(8, 16)
        mask = torch.ones(8, dtype=torch.bool)
        assert consistency_loss(z_a, z_b, mask).item() > 0

    def test_consistency_loss_respects_mask(self):
        z_a = torch.randn(8, 16)
        z_b = torch.randn(8, 16)

        # All masked: should be 0
        mask_none = torch.zeros(8, dtype=torch.bool)
        assert consistency_loss(z_a, z_b, mask_none).item() == pytest.approx(0.0, abs=1e-6)

        # Partial mask should give different result than full mask
        mask_all = torch.ones(8, dtype=torch.bool)
        mask_half = torch.zeros(8, dtype=torch.bool)
        mask_half[:4] = True

        loss_all = consistency_loss(z_a, z_b, mask_all)
        loss_half = consistency_loss(z_a, z_b, mask_half)
        assert loss_all.item() != pytest.approx(loss_half.item(), abs=1e-4)

    def test_separation_loss_at_target(self):
        """Loss should be ~0 when std matches target."""
        z = torch.randn(100, 16)  # std ≈ 1.0
        loss = separation_loss(z, target_std=1.0)
        assert loss.item() < 0.1  # should be close to 0

    def test_separation_loss_penalizes_collapse(self):
        """Collapsed z_shared (all same) should have high loss."""
        z_collapsed = torch.ones(100, 16) * 0.5
        z_spread = torch.randn(100, 16)
        assert separation_loss(z_collapsed, target_std=1.0) > separation_loss(z_spread, target_std=1.0)

    def test_compute_total_loss_keys(self):
        B, D = 8, 64
        x_a = torch.randn(B, D)
        x_b = torch.randn(B, D)
        x_hat_a = torch.randn(B, D)
        x_hat_b = torch.randn(B, D)
        z_shared_a = torch.randn(B, 16)
        z_shared_b = torch.randn(B, 16)
        z_view_a = torch.randn(B, 8)
        z_view_b = torch.randn(B, 8)
        mask = torch.ones(B, dtype=torch.bool)

        losses = compute_total_loss(
            x_a, x_b, x_hat_a, x_hat_b,
            z_shared_a, z_shared_b, z_view_a, z_view_b,
            mask,
        )

        expected_keys = {"total", "recon", "sparse_shared", "sparse_view", "consist", "separate"}
        assert set(losses.keys()) == expected_keys
        assert all(torch.is_tensor(v) for v in losses.values())

    def test_total_loss_gradients(self):
        """Total loss should be differentiable."""
        B, D = 8, 64
        x_a = torch.randn(B, D)
        x_b = torch.randn(B, D)
        z_shared_a = torch.randn(B, 16, requires_grad=True)
        z_shared_b = torch.randn(B, 16, requires_grad=True)
        z_view_a = torch.randn(B, 8, requires_grad=True)
        z_view_b = torch.randn(B, 8, requires_grad=True)

        # Reconstruct via simple linear (to make x_hat differentiable)
        W = torch.randn(24, D, requires_grad=True)
        x_hat_a = torch.cat([z_shared_a, z_view_a], dim=-1) @ W
        x_hat_b = torch.cat([z_shared_b, z_view_b], dim=-1) @ W

        mask = torch.ones(B, dtype=torch.bool)

        losses = compute_total_loss(
            x_a, x_b, x_hat_a, x_hat_b,
            z_shared_a, z_shared_b, z_view_a, z_view_b,
            mask,
        )
        losses["total"].backward()

        assert z_shared_a.grad is not None
        assert z_shared_b.grad is not None
        assert z_view_a.grad is not None


# ==============================================================================
# Dataset tests
# ==============================================================================

class TestDataset:
    def test_dataset_length(self, dummy_data):
        view_a, view_b, metadata = dummy_data
        dataset = PairedViewDataset(
            view_a, view_b, metadata,
            conditions=("baseline", "dots_50"),
            normalize=True,
            include_categories=None,
        )
        assert len(dataset) == len(metadata)

    def test_dataset_item_shapes(self, dummy_data):
        view_a, view_b, metadata = dummy_data
        dataset = PairedViewDataset(
            view_a, view_b, metadata,
            conditions=("baseline", "dots_50"),
            include_categories=None,
        )
        x_a, x_b, correct, idx = dataset[0]
        assert x_a.shape == (128,)
        assert x_b.shape == (128,)
        assert isinstance(correct.item(), bool)
        assert isinstance(idx, int)

    def test_dataset_filtering(self, dummy_data):
        view_a, view_b, metadata = dummy_data
        n_all = len(metadata)

        dataset_all = PairedViewDataset(
            view_a, view_b, metadata,
            conditions=("baseline", "dots_50"),
            include_categories=None,
        )
        assert len(dataset_all) == n_all

        dataset_filtered = PairedViewDataset(
            view_a, view_b, metadata,
            conditions=("baseline", "dots_50"),
            include_categories=("both_correct", "filler_helped"),
        )
        n_expected = sum(1 for m in metadata
                         if m.category in ("both_correct", "filler_helped"))
        assert len(dataset_filtered) == n_expected
        assert len(dataset_filtered) < n_all

    def test_dataset_normalization(self, dummy_data):
        view_a, view_b, metadata = dummy_data
        dataset = PairedViewDataset(
            view_a, view_b, metadata,
            conditions=("baseline", "dots_50"),
            normalize=True,
            include_categories=None,
        )
        # Normalized data should have mean ≈ 0, std ≈ 1
        mean_a = dataset.view_a.mean(dim=0)
        std_a = dataset.view_a.std(dim=0)
        assert mean_a.abs().mean().item() < 0.15
        assert (std_a - 1.0).abs().mean().item() < 0.15

    def test_dataset_no_normalization(self, dummy_data):
        view_a, view_b, metadata = dummy_data
        dataset = PairedViewDataset(
            view_a, view_b, metadata,
            conditions=("baseline", "dots_50"),
            normalize=False,
            include_categories=None,
        )
        # Should match raw data
        np.testing.assert_allclose(
            dataset.view_a.numpy(), view_a, atol=1e-5
        )

    def test_correct_mask(self, dummy_data):
        view_a, view_b, metadata = dummy_data
        dataset = PairedViewDataset(
            view_a, view_b, metadata,
            conditions=("baseline", "dots_50"),
            include_categories=None,
        )
        for i, m in enumerate(metadata):
            expected = (m.model_correct["baseline"] and m.model_correct["dots_50"])
            assert dataset.correct_mask[i] == expected

    def test_ground_truth(self, dummy_data):
        view_a, view_b, metadata = dummy_data
        dataset = PairedViewDataset(
            view_a, view_b, metadata,
            conditions=("baseline", "dots_50"),
            include_categories=None,
        )
        gt = dataset.get_ground_truth()
        assert gt["A"].shape == (len(metadata),)
        assert gt["Y"].shape == (len(metadata),)
        assert gt["A+Y"].shape == (len(metadata),)
        assert gt["category"].shape == (len(metadata),)
        assert gt["A"][0] == metadata[0].fact_value
        assert gt["Y"][0] == metadata[0].x

    def test_stats_roundtrip(self, dummy_data):
        view_a, view_b, metadata = dummy_data
        dataset1 = PairedViewDataset(
            view_a, view_b, metadata,
            conditions=("baseline", "dots_50"),
            normalize=True,
            include_categories=None,
        )
        stats = dataset1.get_stats()

        dataset2 = PairedViewDataset(
            view_a, view_b, metadata,
            conditions=("baseline", "dots_50"),
            normalize=True,
            stats=stats,
            include_categories=None,
        )
        torch.testing.assert_close(dataset1.view_a, dataset2.view_a)
        torch.testing.assert_close(dataset1.view_b, dataset2.view_b)


# ==============================================================================
# Integration test
# ==============================================================================

class TestIntegration:
    def test_train_step(self, model, config, dummy_data):
        """One training step should reduce loss."""
        view_a, view_b, metadata = dummy_data

        dataset = PairedViewDataset(
            view_a[:32, :config.input_dim],
            view_b[:32, :config.input_dim],
            metadata[:32],
            conditions=("baseline", "dots_50"),
            include_categories=None,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Get initial loss
        x_a, x_b = dataset.view_a, dataset.view_b
        mask = dataset.correct_mask_t

        x_hat_a, z_shared_a, z_view_a = model(x_a)
        x_hat_b, z_shared_b, z_view_b = model(x_b)
        losses_0 = compute_total_loss(
            x_a, x_b, x_hat_a, x_hat_b,
            z_shared_a, z_shared_b, z_view_a, z_view_b,
            mask,
        )
        loss_0 = losses_0["total"].item()

        # Train for a few steps
        for _ in range(20):
            x_hat_a, z_shared_a, z_view_a = model(x_a)
            x_hat_b, z_shared_b, z_view_b = model(x_b)
            losses = compute_total_loss(
                x_a, x_b, x_hat_a, x_hat_b,
                z_shared_a, z_shared_b, z_view_a, z_view_b,
                mask,
            )
            optimizer.zero_grad()
            losses["total"].backward()
            optimizer.step()

        loss_final = losses["total"].item()
        assert loss_final < loss_0, f"Loss didn't decrease: {loss_0:.4f} -> {loss_final:.4f}"

    def test_consistency_improves(self, config):
        """With consistency loss, z_shared should converge across views."""
        rng = np.random.RandomState(123)
        n, dim = 30, config.input_dim

        # Shared signal
        signal = rng.randn(n, 5).astype(np.float32)
        proj = rng.randn(5, dim).astype(np.float32)
        view_a = signal @ proj + rng.randn(n, dim).astype(np.float32) * 0.1
        view_b = signal @ proj + rng.randn(n, dim).astype(np.float32) * 0.1

        metadata = [
            ExampleMetadata(i, 50, 30, 80,
                            {"baseline": True, "dots_50": True}, "both_correct")
            for i in range(n)
        ]

        dataset = PairedViewDataset(
            view_a, view_b, metadata,
            conditions=("baseline", "dots_50"),
            include_categories=None,
        )

        model = PairedViewSAE(
            input_dim=dim, encoder_hidden=config.encoder_hidden,
            dim_shared=config.dim_shared, dim_view=config.dim_view,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Measure initial consistency
        with torch.no_grad():
            z_a, _ = model.encode(dataset.view_a)
            z_b, _ = model.encode(dataset.view_b)
            cos_0 = torch.nn.functional.cosine_similarity(z_a, z_b, dim=-1).mean().item()

        # Train
        for _ in range(100):
            x_hat_a, z_shared_a, z_view_a = model(dataset.view_a)
            x_hat_b, z_shared_b, z_view_b = model(dataset.view_b)
            losses = compute_total_loss(
                dataset.view_a, dataset.view_b, x_hat_a, x_hat_b,
                z_shared_a, z_shared_b, z_view_a, z_view_b,
                dataset.correct_mask_t,
                lambda_consist=5.0,
            )
            optimizer.zero_grad()
            losses["total"].backward()
            optimizer.step()

        with torch.no_grad():
            z_a, _ = model.encode(dataset.view_a)
            z_b, _ = model.encode(dataset.view_b)
            cos_final = torch.nn.functional.cosine_similarity(z_a, z_b, dim=-1).mean().item()

        assert cos_final > cos_0, (
            f"Consistency didn't improve: {cos_0:.3f} -> {cos_final:.3f}"
        )
