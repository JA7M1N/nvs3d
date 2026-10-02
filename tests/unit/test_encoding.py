"""Unit tests for nerf/encoding.py: PositionalEncoding.

(verify): γ(p) = [p, sin(2⁰πp), cos(2⁰πp), …, sin(2^{L-1}πp), cos(2^{L-1}πp)]
          — blueprint line 149.
(verify): L=10 position → dim 63; L=4 direction → dim 27 — blueprint line 149.
(verify): Frequencies use π (not 2π): freq_l = 2^l · π — blueprint line 149.

Layout (pinned): For D-dimensional input, output is
  [p₁..pD, sin(f₀·p₁)..sin(f₀·pD), cos(f₀·p₁)..cos(f₀·pD), …,
   sin(f_{L-1}·p₁)..sin(f_{L-1}·pD), cos(f_{L-1}·p₁)..cos(f_{L-1}·pD)]
where f_l = 2^l · π.  Total dim = D·(1 + 2L).
"""

import math
import torch
import pytest

from nvs3d.nerf.encoding import PositionalEncoding

SEED = 42


# ── Shape tests ──────────────────────────────────────────────────────────


def test_pe_output_shape_position():
    """PE(L=10) on [B,3] → [B, 63].  dim = 3·(1 + 2·10) = 63."""
    pe = PositionalEncoding(L=10, include_input=True)
    x = torch.randn(8, 3)
    out = pe(x)
    assert out.shape == (8, 63), f"Expected (8,63), got {out.shape}"


def test_pe_output_shape_direction():
    """PE(L=4) on [B,3] → [B, 27].  dim = 3·(1 + 2·4) = 27."""
    pe = PositionalEncoding(L=4, include_input=True)
    x = torch.randn(8, 3)
    out = pe(x)
    assert out.shape == (8, 27), f"Expected (8,27), got {out.shape}"


def test_pe_no_include_input():
    """include_input=False → dim = 3·2L.  L=10 → 60."""
    pe = PositionalEncoding(L=10, include_input=False)
    x = torch.randn(4, 3)
    out = pe(x)
    assert out.shape == (4, 60), f"Expected (4,60), got {out.shape}"


# ── Exact value tests (float64) ──────────────────────────────────────────


def test_pe_known_values_float64():
    """Exact PE values at a non-degenerate point in float64.

    p = [0.3, -0.7, 1.2].  Check the first few entries manually.
    freq_0 = 2^0 · π = π.
    Layout: [p, sin(πp), cos(πp), sin(2πp), cos(2πp), ...]
    """
    pe = PositionalEncoding(L=3, include_input=True)
    p = torch.tensor([[0.3, -0.7, 1.2]], dtype=torch.float64)
    out = pe(p)
    assert out.dtype == torch.float64

    # dim = 3·(1 + 2·3) = 21
    assert out.shape == (1, 21)

    pi = math.pi
    expected_entries = [
        # [0:3] = p
        0.3, -0.7, 1.2,
        # [3:6] = sin(π·p)
        math.sin(pi * 0.3), math.sin(pi * (-0.7)), math.sin(pi * 1.2),
        # [6:9] = cos(π·p)
        math.cos(pi * 0.3), math.cos(pi * (-0.7)), math.cos(pi * 1.2),
        # [9:12] = sin(2π·p)
        math.sin(2 * pi * 0.3), math.sin(2 * pi * (-0.7)), math.sin(2 * pi * 1.2),
        # [12:15] = cos(2π·p)
        math.cos(2 * pi * 0.3), math.cos(2 * pi * (-0.7)), math.cos(2 * pi * 1.2),
        # [15:18] = sin(4π·p)
        math.sin(4 * pi * 0.3), math.sin(4 * pi * (-0.7)), math.sin(4 * pi * 1.2),
        # [18:21] = cos(4π·p)
        math.cos(4 * pi * 0.3), math.cos(4 * pi * (-0.7)), math.cos(4 * pi * 1.2),
    ]
    expected = torch.tensor([expected_entries], dtype=torch.float64)
    torch.testing.assert_close(out, expected, atol=1e-14, rtol=0)


def test_pe_layout_pin():
    """Pin the output layout: sin block precedes cos block for each band.

    For L=2, D=2, output is: [p1,p2, sin(πp1),sin(πp2), cos(πp1),cos(πp2),
                               sin(2πp1),sin(2πp2), cos(2πp1),cos(2πp2)]
    """
    pe = PositionalEncoding(L=2, include_input=True)
    p = torch.tensor([[0.25, 0.75]], dtype=torch.float64)
    out = pe(p)

    pi = math.pi
    assert out.shape == (1, 2 + 2 * 2 * 2)  # 2·(1+4) = 10

    # Verify sin comes before cos for each band
    # Band 0: indices 2,3 = sin(πp), indices 4,5 = cos(πp)
    torch.testing.assert_close(out[0, 2], torch.tensor(math.sin(pi * 0.25), dtype=torch.float64), atol=1e-14, rtol=0)
    torch.testing.assert_close(out[0, 4], torch.tensor(math.cos(pi * 0.25), dtype=torch.float64), atol=1e-14, rtol=0)


# ── Float32 vs float64 consistency ───────────────────────────────────────


def test_pe_float32_vs_float64_consistency():
    """Float32 PE matches float64 PE within a tolerance justified by
    the top-band argument size.

    For L=10, p=0.3: top-band arg = 2^9 · π · 0.3 ≈ 482.7.
    Float32 relative error ≈ 1.2e-7, so arg error ≈ 5.8e-5.
    sin/cos error ≤ arg error ≈ 6e-5.
    Tolerance: 5e-4 (roughly 8× worst case, gives margin for
    accumulated arithmetic errors across all multiplications).
    Measured max error: 1.67e-4 (3× margin).
    """
    pe = PositionalEncoding(L=10, include_input=True)
    p_f64 = torch.tensor([[0.3, -0.7, 1.2]], dtype=torch.float64)
    p_f32 = p_f64.float()

    out_f64 = pe(p_f64)
    out_f32 = pe(p_f32)

    torch.testing.assert_close(out_f32.double(), out_f64, atol=5e-4, rtol=0)


# ── Gradient test ────────────────────────────────────────────────────────


def test_pe_gradient_finite():
    """Gradients through PE are finite and non-zero."""
    pe = PositionalEncoding(L=10, include_input=True)
    x = torch.tensor([[0.3, -0.7, 1.2]], requires_grad=True)
    out = pe(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all(), "PE gradient has non-finite values"
    assert (x.grad.abs() > 0).any(), "PE gradient is all zeros"
