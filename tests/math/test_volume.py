"""Math tests for nerf/volume.py: volume rendering.

All analytic tests use float64.  Gradcheck uses float64 with σ ∈ [0.5, 2].

volume_render(sigma, rgb, t_vals, bg_color, dirs=None, last_delta=1e10):
    sigma: [B, N], rgb: [B, N, 3], t_vals: [B, N], bg_color: [3], dirs: [B, 3] or None.
    If dirs is provided, δ is scaled by ‖dirs‖ to convert parametric Δt to
    Euclidean distance.  bg_color is an explicit [3] tensor (no white_bkgd flag).
    last_delta: configurable last δ (default 1e10).

(verify): δ_i = t_{i+1} − t_i, last δ = 1e10 — blueprint line 164.
(verify): α_i = 1 − exp(−σ_i δ_i) — blueprint line 164.
(verify): T_i = ∏_{j<i}(1 − α_j)  (exclusive cumprod) — blueprint line 164.
(verify): Background: Ĉ + (1−acc)·bg — blueprint line 164.
"""

import math
import torch
import pytest

from nvs3d.nerf.volume import volume_render

SEED = 42


# ── Helper ───────────────────────────────────────────────────────────────

def _vr(sigma, rgb, t_vals, bg_color, **kwargs):
    """Convenience wrapper: ensures inputs are 2D/3D batched."""
    if sigma.dim() == 1:
        sigma = sigma.unsqueeze(0)
    if rgb.dim() == 2:
        rgb = rgb.unsqueeze(0)
    if t_vals.dim() == 1:
        t_vals = t_vals.unsqueeze(0)
    return volume_render(sigma, rgb, t_vals, bg_color, **kwargs)


# ── Hand-computed 2-sample case ──────────────────────────────────────────


def test_two_sample_hand_computed():
    """Hand-computed 2-sample case with explicit numbers in float64.

    t_vals = [2.0, 4.0]
    σ      = [0.5, 0.0]    (σ=0 on last → last δ=1e10 is irrelevant)
    c      = [[1,0,0], [0,1,0]]
    bg     = [0,0,1]

    δ_0 = 2.0,  δ_1 = 1e10
    σδ = [1.0, 0]
    α  = [1−e⁻¹, 0] ≈ [0.6321, 0]
    T  = [1, e⁻¹]   ≈ [1, 0.3679]
    w  = [1−e⁻¹, 0]
    acc = 1−e⁻¹ ≈ 0.6321
    rgb = (1−e⁻¹)·[1,0,0] + 0·[0,1,0] + e⁻¹·[0,0,1]
        = [1−e⁻¹, 0, e⁻¹]
    depth = (1−e⁻¹)·2.0 = 2(1−e⁻¹)
    """
    e1 = math.exp(-1.0)
    sigma = torch.tensor([0.5, 0.0], dtype=torch.float64)
    rgb = torch.tensor([[1., 0., 0.], [0., 1., 0.]], dtype=torch.float64)
    t_vals = torch.tensor([2.0, 4.0], dtype=torch.float64)
    bg = torch.tensor([0., 0., 1.], dtype=torch.float64)

    out = _vr(sigma, rgb, t_vals, bg)

    expected_acc = 1.0 - e1
    expected_rgb = torch.tensor([[1.0 - e1, 0.0, e1]], dtype=torch.float64)
    expected_depth = 2.0 * (1.0 - e1)

    torch.testing.assert_close(out["acc"], torch.tensor([expected_acc], dtype=torch.float64), atol=1e-12, rtol=0)
    torch.testing.assert_close(out["rgb"], expected_rgb, atol=1e-12, rtol=0)
    torch.testing.assert_close(out["depth"], torch.tensor([expected_depth], dtype=torch.float64), atol=1e-12, rtol=0)


# ── Piecewise σ, non-uniform t_vals ──────────────────────────────────────


def test_piecewise_sigma_nonuniform_t():
    """Non-uniform t_vals with piecewise σ.  σ=0 on last sample.

    t_vals = [0.5, 1.0, 2.0, 5.0]
    σ      = [1.0, 2.0, 0.5, 0.0]
    δ      = [0.5, 1.0, 3.0, 1e10]
    σδ     = [0.5, 2.0, 1.5, 0]
    Σσδ    = 4.0
    acc    = 1 − exp(−4.0)
    """
    sigma = torch.tensor([1.0, 2.0, 0.5, 0.0], dtype=torch.float64)
    rgb = torch.rand(4, 3, dtype=torch.float64)
    t_vals = torch.tensor([0.5, 1.0, 2.0, 5.0], dtype=torch.float64)
    bg = torch.zeros(3, dtype=torch.float64)

    out = _vr(sigma, rgb, t_vals, bg)

    expected_acc = 1.0 - math.exp(-4.0)
    torch.testing.assert_close(
        out["acc"], torch.tensor([expected_acc], dtype=torch.float64),
        atol=1e-10, rtol=0,
    )


# ── Constant σ slab (analytic) ───────────────────────────────────────────


def test_constant_sigma_slab():
    """Constant σ over a slab: acc = 1 − exp(−σ·L).

    σ = 0.3 on 100 uniform samples over [2, 6] (L=4).  σ=0 on last sample
    so δ_last = 1e10 doesn't pollute.
    Expected: acc = 1 − exp(−0.3 · 4.0) = 1 − exp(−1.2).
    Tolerance: 1e-10 (float64; discretization error is O(δ²)≈1.6e-5, but
    since σ is constant, the discrete sum equals the integral exactly:
    Σσ_i δ_i = σ·Σδ_i = σ·L, independent of N).
    """
    N = 100
    near, far = 2.0, 6.0
    sigma_val = 0.3
    t_vals = torch.linspace(near, far, N, dtype=torch.float64).unsqueeze(0)  # [1, N]
    sigma = torch.full((1, N), sigma_val, dtype=torch.float64)
    # Set σ=0 on last sample so last δ=1e10 is irrelevant
    sigma[0, -1] = 0.0
    rgb = torch.ones(1, N, 3, dtype=torch.float64)
    bg = torch.zeros(3, dtype=torch.float64)

    out = volume_render(sigma, rgb, t_vals, bg)
    L = far - near
    expected_acc = 1.0 - math.exp(-sigma_val * L)

    torch.testing.assert_close(
        out["acc"], torch.tensor([expected_acc], dtype=torch.float64),
        atol=1e-10, rtol=0,
    )


# ── σ=0 gives background ────────────────────────────────────────────────


def test_sigma_zero_gives_bg():
    """σ=0 everywhere → acc=0, rgb=bg_color.  Float64, atol=1e-12."""
    sigma = torch.zeros(1, 8, dtype=torch.float64)
    rgb = torch.randn(1, 8, 3, dtype=torch.float64)
    t_vals = torch.linspace(2, 6, 8, dtype=torch.float64).unsqueeze(0)
    bg = torch.tensor([0.2, 0.4, 0.8], dtype=torch.float64)

    out = volume_render(sigma, rgb, t_vals, bg)
    torch.testing.assert_close(out["rgb"], bg.unsqueeze(0), atol=1e-12, rtol=0)
    torch.testing.assert_close(out["acc"], torch.tensor([0.0], dtype=torch.float64), atol=1e-12, rtol=0)


# ── One opaque sample ───────────────────────────────────────────────────


def test_one_opaque_sample():
    """Single sample with very large σ → color ≈ that sample, acc ≈ 1.

    σ = [100] (only 1 sample), δ = 1e10, α = 1−exp(−100·1e10) ≈ 1.
    Weight ≈ 1, acc ≈ 1, rgb ≈ c_0.
    Tolerance: 1e-6 (float64, residual exp(−1e12) is 0 to machine precision).
    """
    sigma = torch.tensor([[100.0]], dtype=torch.float64)
    c0 = torch.tensor([[[0.8, 0.3, 0.1]]], dtype=torch.float64)
    t_vals = torch.tensor([[3.0]], dtype=torch.float64)
    bg = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)

    out = volume_render(sigma, c0, t_vals, bg)
    torch.testing.assert_close(out["rgb"], c0.squeeze(0), atol=1e-6, rtol=0)
    torch.testing.assert_close(out["acc"], torch.tensor([1.0], dtype=torch.float64), atol=1e-6, rtol=0)


# ── Occlusion test ──────────────────────────────────────────────────────


def test_opaque_middle_sample_occludes():
    """An opaque middle sample occludes everything behind it.  Analytic values.

    3 samples: σ = [0.1, 100.0, 0.1], t = [1.0, 2.0, 3.0], bg = [0.5, 0.5, 0.5].
    c = [[1,0,0], [0,1,0], [0,0,1]].

    δ = [1.0, 1.0, 1e10]
    σδ = [0.1, 100, 0.1·1e10]
    α  = [1-e^{-0.1}, 1-e^{-100}≈1, 1-e^{-1e9}≈1]
    T  = [1, e^{-0.1}, e^{-0.1}·(1-α₁)≈0]

    w₀ = 1·(1-e^{-0.1}) = 1-e^{-0.1} ≈ 0.09516
    w₁ = e^{-0.1}·1 = e^{-0.1} ≈ 0.90484
    w₂ ≈ 0  (occluded)
    acc = w₀+w₁ ≈ 1 (no bg contribution)

    rgb = w₀·[1,0,0] + w₁·[0,1,0] + 0·bg = [1-e^{-0.1}, e^{-0.1}, 0]
    depth = w₀·1 + w₁·2 = (1-e^{-0.1})·1 + e^{-0.1}·2

    Tolerance: 1e-6 (float64; the "≈1" terms have residual e^{-100} ≈ 3.7e-44).
    """
    import math
    e01 = math.exp(-0.1)

    sigma = torch.tensor([[0.1, 100.0, 0.1]], dtype=torch.float64)
    rgb = torch.tensor([[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]], dtype=torch.float64)
    t_vals = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
    bg = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float64)

    out = volume_render(sigma, rgb, t_vals, bg)

    exp_w0 = 1.0 - e01
    exp_w1 = e01   # × (1−e^{-100}) ≈ e01
    # w2 ≈ 0

    exp_acc = exp_w0 + exp_w1  # ≈ 1
    exp_rgb = torch.tensor([[exp_w0, exp_w1, 0.0]], dtype=torch.float64)
    # + (1-acc)·bg ≈ 0
    exp_depth = exp_w0 * 1.0 + exp_w1 * 2.0

    torch.testing.assert_close(out["acc"], torch.tensor([exp_acc], dtype=torch.float64), atol=1e-6, rtol=0)
    torch.testing.assert_close(out["rgb"], exp_rgb, atol=1e-6, rtol=0)
    torch.testing.assert_close(out["depth"], torch.tensor([exp_depth], dtype=torch.float64), atol=1e-6, rtol=0)
    # Back sample weight ≈ 0
    assert out["weights"][0, 2].item() < 1e-40, f"w₂ = {out['weights'][0,2].item()}, expected ≈ 0"


# ── Weights sum properties ──────────────────────────────────────────────


def test_weights_sum_leq_one():
    """Σw ≤ 1 always.  Random sigma in float64.  Exact check: Σw ≤ 1+ε."""
    torch.manual_seed(SEED)
    B, N = 16, 32
    sigma = torch.rand(B, N, dtype=torch.float64) * 5.0
    rgb = torch.rand(B, N, 3, dtype=torch.float64)
    t_vals = torch.linspace(2, 6, N, dtype=torch.float64).unsqueeze(0).expand(B, N)
    bg = torch.zeros(3, dtype=torch.float64)

    out = volume_render(sigma, rgb, t_vals, bg)
    w_sum = out["weights"].sum(dim=-1)

    # In float64, machine epsilon ≈ 2.2e-16.  Allow 1e-12 for accumulation.
    assert (w_sum <= 1.0 + 1e-12).all(), f"max Σw = {w_sum.max().item()}"


def test_weights_sum_equals_one_minus_Tfinal():
    """Σw_i = 1 − T_final where T_final = ∏(1 − α_i).

    This is the fundamental identity of volume rendering: the weights plus
    the residual transmittance sum to exactly 1.
    Float64, atol=1e-10.
    """
    torch.manual_seed(SEED)
    B, N = 8, 16
    sigma = torch.rand(B, N, dtype=torch.float64) * 3.0
    rgb = torch.rand(B, N, 3, dtype=torch.float64)
    t_vals = torch.linspace(2, 6, N, dtype=torch.float64).unsqueeze(0).expand(B, N)
    bg = torch.zeros(3, dtype=torch.float64)

    out = volume_render(sigma, rgb, t_vals, bg)
    w_sum = out["weights"].sum(dim=-1)  # [B]
    acc = out["acc"]  # [B]

    # w_sum should equal acc
    torch.testing.assert_close(w_sum, acc, atol=1e-10, rtol=0)

    # Also verify: acc = 1 - T_final
    deltas = t_vals[:, 1:] - t_vals[:, :-1]
    last_d = torch.full_like(deltas[:, :1], 1e10)
    deltas = torch.cat([deltas, last_d], dim=-1)
    alpha = 1.0 - torch.exp(-sigma * deltas)
    T_final = torch.prod(1.0 - alpha, dim=-1)
    torch.testing.assert_close(acc, 1.0 - T_final, atol=1e-10, rtol=0)


# ── Exclusive cumprod off-by-one ─────────────────────────────────────────


def test_exclusive_cumprod_off_by_one():
    """T[0] = 1 always; T[i] = ∏_{j<i}(1 − α_j), verified vs manual loop.

    This catches the classic inclusive-vs-exclusive cumprod bug.
    Float64, atol=1e-12.
    """
    torch.manual_seed(SEED)
    sigma = torch.rand(1, 5, dtype=torch.float64) * 2.0
    rgb = torch.rand(1, 5, 3, dtype=torch.float64)
    t_vals = torch.tensor([[1.0, 2.0, 3.5, 5.0, 6.0]], dtype=torch.float64)
    bg = torch.zeros(3, dtype=torch.float64)

    out = volume_render(sigma, rgb, t_vals, bg)
    weights = out["weights"]  # [1, 5]

    # Recompute manually
    deltas = t_vals[0, 1:] - t_vals[0, :-1]
    deltas = torch.cat([deltas, torch.tensor([1e10], dtype=torch.float64)])
    alpha = 1.0 - torch.exp(-sigma[0] * deltas)

    T_manual = torch.ones(5, dtype=torch.float64)
    for i in range(1, 5):
        T_manual[i] = T_manual[i - 1] * (1.0 - alpha[i - 1])

    w_manual = T_manual * alpha

    # T[0] must be 1
    # (We verify indirectly: w[0] = T[0]*α[0] = α[0])
    torch.testing.assert_close(
        weights[0, 0], alpha[0], atol=1e-12, rtol=0,
    )

    # Full weights match
    torch.testing.assert_close(weights[0], w_manual, atol=1e-12, rtol=0)


# ── Non-unit direction test ──────────────────────────────────────────────


def test_nonunit_direction_scales_delta():
    """Rendering with scaled direction + adjusted t_vals gives same result.

    If dirs = 2·d̂ (norm=2), and t_vals = t_orig/2, then:
    δ_parametric = Δt/2, δ_euclidean = Δt/2 · 2 = Δt (unchanged).
    So σ·δ_euclidean is the same, and the rendering should be identical.

    dirs is used to convert parametric Δt to Euclidean distance.
    """
    torch.manual_seed(SEED)
    B, N = 4, 8
    sigma = torch.rand(B, N, dtype=torch.float64)
    rgb = torch.rand(B, N, 3, dtype=torch.float64)
    t_orig = torch.linspace(2, 6, N, dtype=torch.float64).unsqueeze(0).expand(B, N)
    bg = torch.zeros(3, dtype=torch.float64)

    # Unit directions
    d_unit = torch.randn(B, 3, dtype=torch.float64)
    d_unit = d_unit / d_unit.norm(dim=-1, keepdim=True)

    # Render with unit dirs + original t
    out1 = volume_render(sigma, rgb, t_orig, bg, dirs=d_unit)

    # Render with 2x dirs + halved t
    d_2x = d_unit * 2.0
    t_half = t_orig / 2.0
    out2 = volume_render(sigma, rgb, t_half, bg, dirs=d_2x)

    torch.testing.assert_close(out1["rgb"], out2["rgb"], atol=1e-10, rtol=0)
    torch.testing.assert_close(out1["acc"], out2["acc"], atol=1e-10, rtol=0)


# ── Float64 gradcheck ───────────────────────────────────────────────────


def _make_vr_inputs(B=4, N=8):
    """Create volume render inputs for gradcheck: σ ∈ [0.5, 2]."""
    torch.manual_seed(SEED)
    sigma = 0.5 + 1.5 * torch.rand(B, N, dtype=torch.float64, requires_grad=True)
    rgb = torch.rand(B, N, 3, dtype=torch.float64, requires_grad=True)
    t_vals = torch.linspace(2, 6, N, dtype=torch.float64).unsqueeze(0).expand(B, N).clone()
    return sigma, rgb, t_vals


def test_volume_render_gradcheck_black_bg():
    """Float64 gradcheck with bg_color = [0,0,0]."""
    sigma, rgb, t_vals = _make_vr_inputs()
    bg = torch.zeros(3, dtype=torch.float64)

    def fn(s, c):
        out = volume_render(s, c, t_vals, bg)
        return out["rgb"], out["depth"], out["acc"]

    torch.autograd.gradcheck(fn, (sigma, rgb), raise_exception=True)


def test_volume_render_gradcheck_white_bg():
    """Float64 gradcheck with bg_color = [1,1,1]."""
    sigma, rgb, t_vals = _make_vr_inputs()
    bg = torch.ones(3, dtype=torch.float64)

    def fn(s, c):
        out = volume_render(s, c, t_vals, bg)
        return out["rgb"], out["depth"], out["acc"]

    torch.autograd.gradcheck(fn, (sigma, rgb), raise_exception=True)
