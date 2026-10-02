"""Unit tests for nerf/field.py: NeRFField MLP.

(verify): 8 layers × 256, ReLU, skip concat of γ(x) at layer 5 — blueprint line 155.
(verify): σ head (1) + 256-d feature; concat γ(d) → 128-unit layer → RGB (3, sigmoid) — line 155.
(verify): σ = softplus(raw + b) — blueprint line 156.
(verify): Density bias init: "bias so initial σ is small (avoid opaque start)" — line 157.
          No exact value given; this is a (verify) item.

skip_layer=5 definition: The input to layer 5 (0-indexed) is concatenated with
γ(x). So layers 0–4 are standard FC+ReLU (5 layers), then layer 5 gets
[output_of_layer4, γ(x)] as input.
"""

import torch
import pytest

from nvs3d.nerf.encoding import PositionalEncoding
from nvs3d.nerf.field import NeRFField

SEED = 42


# ── Output shapes ────────────────────────────────────────────────────────


def test_field_output_shapes():
    """NeRFField maps (pos [B,3], dir [B,3]) → sigma [B,1], rgb [B,3]."""
    field = NeRFField()
    torch.manual_seed(SEED)
    pos = torch.randn(8, 3)
    dirs = torch.randn(8, 3)
    sigma, rgb = field(pos, dirs)
    assert sigma.shape == (8, 1), f"sigma shape: expected (8,1), got {sigma.shape}"
    assert rgb.shape == (8, 3), f"rgb shape: expected (8,3), got {rgb.shape}"


def test_field_layer_shapes():
    """Verify exact internal layer input/output dimensions.

    - PE position (L=10): dim = 63
    - Backbone layers 0-4: 63→256, 256→256 ×4
    - Skip layer 5 input: 256 + 63 = 319 → 256
    - Backbone layers 6-7: 256→256 ×2
    - σ head: 256 → 1
    - Color input: 256 + 27 (PE dir L=4) = 283 → 128 → 3
    """
    field = NeRFField(pos_L=10, dir_L=4, hidden=256, skip_layer=5)

    # Check backbone layer shapes
    # Layer 0: in=63, out=256
    assert field.backbone[0].in_features == 63
    assert field.backbone[0].out_features == 256

    # Layers 1-4: in=256, out=256
    for i in range(1, 5):
        assert field.backbone[i].in_features == 256, f"Layer {i} in_features"
        assert field.backbone[i].out_features == 256, f"Layer {i} out_features"

    # Layer 5 (skip): in=256+63=319, out=256
    assert field.backbone[5].in_features == 319, f"Skip layer in_features"
    assert field.backbone[5].out_features == 256

    # Layers 6-7: in=256, out=256
    for i in range(6, 8):
        assert field.backbone[i].in_features == 256, f"Layer {i} in_features"
        assert field.backbone[i].out_features == 256, f"Layer {i} out_features"

    # σ head: 256 → 1
    assert field.sigma_head.in_features == 256
    assert field.sigma_head.out_features == 1

    # Color layers: 256+27=283 → 128, 128 → 3
    assert field.color_hidden.in_features == 283
    assert field.color_hidden.out_features == 128
    assert field.color_out.in_features == 128
    assert field.color_out.out_features == 3


def test_field_skip_layer_pin():
    """Pin: the skip concat is at trunk layer index 5 (0-based) of 8 total.

    Blueprint line 155: "8 layers × 256, ReLU, skip concat of γ(x) at layer 5".
    backbone[5].in_features == 256 + 63 = 319 is the ONLY layer with
    in_features > 256.  Total trunk layers = 8.
    """
    field = NeRFField(pos_L=10, dir_L=4, hidden=256, skip_layer=5)
    assert len(field.backbone) == 8, f"Expected 8 trunk layers, got {len(field.backbone)}"

    skip_indices = [i for i, l in enumerate(field.backbone) if l.in_features > 256]
    assert skip_indices == [5], (
        f"Expected skip at index [5], got {skip_indices}"
    )
    assert field.backbone[5].in_features == 256 + 63


# ── Density properties ───────────────────────────────────────────────────


def test_field_sigma_nonneg():
    """σ ≥ 0 for random inputs (softplus is always positive)."""
    field = NeRFField()
    torch.manual_seed(SEED)
    pos = torch.randn(32, 3)
    dirs = torch.randn(32, 3)
    sigma, _ = field(pos, dirs)
    assert (sigma >= 0).all(), f"Found negative σ: min={sigma.min().item()}"


def test_field_sigma_small_at_init():
    """Initial σ should be small (< 1.0 mean) to avoid opaque start.

    (verify): Blueprint says "bias so initial σ is small (avoid opaque start)"
    but does not specify the exact bias value.
    """
    field = NeRFField()
    torch.manual_seed(SEED)
    pos = torch.randn(64, 3)
    dirs = torch.randn(64, 3)
    sigma, _ = field(pos, dirs)
    mean_sigma = sigma.mean().item()
    assert mean_sigma < 1.0, (
        f"Initial mean σ = {mean_sigma:.4f}, expected < 1.0 for a good init"
    )


def test_field_sigma_invariant_to_direction():
    """Same position, different direction → identical σ (exact equality).

    Density path: pos → PE(pos) → backbone → sigma_head → softplus.
    Direction is never used in this path, so the output is bit-identical
    regardless of direction.
    """
    field = NeRFField()
    torch.manual_seed(SEED)
    pos = torch.randn(8, 3)
    dir1 = torch.randn(8, 3)
    dir2 = torch.randn(8, 3)
    sigma1, _ = field(pos, dir1)
    sigma2, _ = field(pos, dir2)
    torch.testing.assert_close(sigma1, sigma2, atol=0, rtol=0)


def test_field_sigma_zero_gradient_wrt_direction():
    """σ has zero gradient with respect to the direction input.

    Since σ depends only on position, ∂σ/∂d = 0.
    Autograd returns None when an input isn't in the computation graph,
    which is equivalent to zero gradient.
    """
    field = NeRFField()
    torch.manual_seed(SEED)
    pos = torch.randn(4, 3)
    dirs = torch.randn(4, 3, requires_grad=True)
    sigma, _ = field(pos, dirs)
    grad = torch.autograd.grad(sigma.sum(), dirs, allow_unused=True)[0]
    # None means dirs wasn't part of the sigma computation → zero gradient
    if grad is not None:
        torch.testing.assert_close(
            grad, torch.zeros_like(grad), atol=0, rtol=0
        )


# ── Color properties ────────────────────────────────────────────────────


def test_field_rgb_in_01():
    """RGB ∈ (0, 1) for random inputs (sigmoid output)."""
    field = NeRFField()
    torch.manual_seed(SEED)
    pos = torch.randn(32, 3)
    dirs = torch.randn(32, 3)
    _, rgb = field(pos, dirs)
    assert (rgb > 0).all(), f"Found rgb ≤ 0: min={rgb.min().item()}"
    assert (rgb < 1).all(), f"Found rgb ≥ 1: max={rgb.max().item()}"


def test_field_rgb_changes_with_direction():
    """RGB changes when only the direction changes (view-dependent color).

    Color depends on direction through the color head.
    """
    field = NeRFField()
    torch.manual_seed(SEED)
    pos = torch.randn(8, 3)
    dir1 = torch.tensor([[1.0, 0.0, 0.0]] * 8)
    dir2 = torch.tensor([[0.0, 0.0, 1.0]] * 8)
    _, rgb1 = field(pos, dir1)
    _, rgb2 = field(pos, dir2)
    # At least some RGB values should differ
    assert not torch.allclose(rgb1, rgb2, atol=1e-4), (
        "RGB did not change with direction — view dependence is broken"
    )


# ── Determinism ──────────────────────────────────────────────────────────


def test_field_forward_deterministic():
    """Same input → identical output (no stochasticity in eval mode)."""
    field = NeRFField()
    field.eval()
    pos = torch.randn(4, 3)
    dirs = torch.randn(4, 3)
    sigma1, rgb1 = field(pos, dirs)
    sigma2, rgb2 = field(pos, dirs)
    torch.testing.assert_close(sigma1, sigma2, atol=0, rtol=0)
    torch.testing.assert_close(rgb1, rgb2, atol=0, rtol=0)
