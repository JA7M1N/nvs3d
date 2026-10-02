"""Unit tests for nerf/sampler.py: stratified sampler.

(verify): t_i = t_near + (i + U_i)/N · (t_far − t_near), U_i ~ U[0,1);
          deterministic midpoints at eval — blueprint line 142.

Stratified sampler interface:
    sample_stratified(rays_o, rays_d, near, far, N_samples, perturb, generator)
    → t_vals [B, N_samples]

near/far can be scalar or [B] / [B,1] tensors (per-ray bounds).
"""

import torch
import pytest

from nvs3d.nerf.sampler import sample_stratified

SEED = 42


# ── Samples sorted and in bounds ─────────────────────────────────────────


def test_stratified_samples_sorted():
    """Output t_vals are sorted along dim=-1."""
    B, N = 16, 32
    rays_o = torch.randn(B, 3)
    rays_d = torch.randn(B, 3)
    gen = torch.Generator().manual_seed(SEED)
    t = sample_stratified(rays_o, rays_d, near=2.0, far=6.0,
                          N_samples=N, perturb=True, generator=gen)
    assert t.shape == (B, N)
    # Check sorted: t[..., i+1] >= t[..., i]
    diffs = t[:, 1:] - t[:, :-1]
    assert (diffs >= 0).all(), "Samples are not sorted"


def test_stratified_samples_in_bounds():
    """All t ∈ [near, far]."""
    B, N = 16, 32
    rays_o = torch.randn(B, 3)
    rays_d = torch.randn(B, 3)
    near, far = 2.0, 6.0
    gen = torch.Generator().manual_seed(SEED)
    t = sample_stratified(rays_o, rays_d, near=near, far=far,
                          N_samples=N, perturb=True, generator=gen)
    assert (t >= near).all(), f"Min t = {t.min().item()}, expected >= {near}"
    assert (t <= far).all(), f"Max t = {t.max().item()}, expected <= {far}"


# ── Sample i in bin i ────────────────────────────────────────────────────


def test_sample_i_in_bin_i():
    """Sample i lies within bin i = [near + i·Δ, near + (i+1)·Δ] where Δ = (far-near)/N.

    This verifies the stratified property: each sample is constrained
    to its own bin, preventing clumping.
    """
    B, N = 8, 16
    near, far = 2.0, 6.0
    delta = (far - near) / N
    rays_o = torch.randn(B, 3)
    rays_d = torch.randn(B, 3)
    gen = torch.Generator().manual_seed(SEED)
    t = sample_stratified(rays_o, rays_d, near=near, far=far,
                          N_samples=N, perturb=True, generator=gen)

    for i in range(N):
        bin_lo = near + i * delta
        bin_hi = near + (i + 1) * delta
        col = t[:, i]
        assert (col >= bin_lo - 1e-6).all(), (
            f"Sample {i}: min={col.min().item():.6f}, bin_lo={bin_lo:.6f}"
        )
        assert (col <= bin_hi + 1e-6).all(), (
            f"Sample {i}: max={col.max().item():.6f}, bin_hi={bin_hi:.6f}"
        )


# ── Deterministic midpoints ─────────────────────────────────────────────


def test_stratified_deterministic_midpoints():
    """With perturb=False, samples are at bin midpoints.

    t_i = near + (i + 0.5) / N · (far - near).
    """
    B, N = 4, 8
    near, far = 2.0, 6.0
    rays_o = torch.randn(B, 3)
    rays_d = torch.randn(B, 3)
    t = sample_stratified(rays_o, rays_d, near=near, far=far,
                          N_samples=N, perturb=False)

    expected = near + (torch.arange(N, dtype=torch.float32) + 0.5) / N * (far - near)
    expected = expected.unsqueeze(0).expand(B, N)
    torch.testing.assert_close(t, expected, atol=1e-6, rtol=0)


# ── No gradient through samples ─────────────────────────────────────────


def test_stratified_no_gradient():
    """t_vals.requires_grad is False even when inputs require grad.

    Samples must be detached: gradients should NOT flow through the
    sampled t values (blueprint line 143: "detach the samples").
    """
    B, N = 4, 8
    rays_o = torch.randn(B, 3, requires_grad=True)
    rays_d = torch.randn(B, 3, requires_grad=True)
    gen = torch.Generator().manual_seed(SEED)
    t = sample_stratified(rays_o, rays_d, near=2.0, far=6.0,
                          N_samples=N, perturb=True, generator=gen)
    assert not t.requires_grad, "t_vals should not require grad"


# ── Generator reproducibility ───────────────────────────────────────────


def test_stratified_generator_reproducibility():
    """Same generator seed → same t_vals."""
    B, N = 8, 16
    rays_o = torch.randn(B, 3)
    rays_d = torch.randn(B, 3)

    gen1 = torch.Generator().manual_seed(SEED)
    t1 = sample_stratified(rays_o, rays_d, near=2.0, far=6.0,
                           N_samples=N, perturb=True, generator=gen1)

    gen2 = torch.Generator().manual_seed(SEED)
    t2 = sample_stratified(rays_o, rays_d, near=2.0, far=6.0,
                           N_samples=N, perturb=True, generator=gen2)

    torch.testing.assert_close(t1, t2, atol=0, rtol=0)


# ── Per-ray near/far ────────────────────────────────────────────────────


def test_stratified_per_ray_near_far():
    """near and far can be [B] tensors (one per ray).

    Ray 0: near=1, far=3 → samples ∈ [1, 3]
    Ray 1: near=5, far=9 → samples ∈ [5, 9]
    """
    B, N = 2, 16
    rays_o = torch.randn(B, 3)
    rays_d = torch.randn(B, 3)
    near = torch.tensor([1.0, 5.0])
    far = torch.tensor([3.0, 9.0])
    gen = torch.Generator().manual_seed(SEED)
    t = sample_stratified(rays_o, rays_d, near=near, far=far,
                          N_samples=N, perturb=True, generator=gen)

    assert t.shape == (B, N)
    assert (t[0] >= 1.0 - 1e-6).all() and (t[0] <= 3.0 + 1e-6).all(), (
        f"Ray 0 bounds violated: [{t[0].min():.4f}, {t[0].max():.4f}]"
    )
    assert (t[1] >= 5.0 - 1e-6).all() and (t[1] <= 9.0 + 1e-6).all(), (
        f"Ray 1 bounds violated: [{t[1].min():.4f}, {t[1].max():.4f}]"
    )
