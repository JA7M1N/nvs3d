"""Math tests for SE3, Sim3, Umeyama, and slerp in core/transforms.py."""

import math
import torch
import pytest

from nvs3d.core.transforms import (
    se3_inverse,
    se3_compose,
    Sim3,
    sim3_inverse,
    sim3_compose,
    sim3_transform_points,
    umeyama,
    slerp,
)

SEED = 42


# ── SE3 ──────────────────────────────────────────────────────────────────


def _random_se3(rng: torch.Generator, dtype=torch.float64) -> torch.Tensor:
    """Create a random valid SE3 matrix [4,4]."""
    # Random rotation via QR decomposition
    A = torch.randn(3, 3, dtype=dtype, generator=rng)
    Q, R_ = torch.linalg.qr(A)
    # Ensure det(Q) = +1
    Q = Q * torch.det(Q).sign()
    t = torch.randn(3, dtype=dtype, generator=rng)
    T = torch.eye(4, dtype=dtype)
    T[:3, :3] = Q
    T[:3, 3] = t
    return T


def test_se3_inverse():
    """T @ inv(T) = I."""
    rng = torch.Generator().manual_seed(SEED)
    T = _random_se3(rng)
    T_inv = se3_inverse(T)
    I_approx = T @ T_inv
    torch.testing.assert_close(I_approx, torch.eye(4, dtype=torch.float64), atol=1e-12, rtol=0)


def test_se3_compose_inverse():
    """(A @ B) @ inv(B) = A."""
    rng = torch.Generator().manual_seed(SEED)
    A = _random_se3(rng)
    B = _random_se3(rng)
    AB = se3_compose(A, B)
    result = se3_compose(AB, se3_inverse(B))
    torch.testing.assert_close(result, A, atol=1e-12, rtol=0)


# ── Sim3 ─────────────────────────────────────────────────────────────────

def _random_sim3(rng: torch.Generator, dtype=torch.float64) -> Sim3:
    """Create a random Sim3 with positive scale."""
    A = torch.randn(3, 3, dtype=dtype, generator=rng)
    Q, _ = torch.linalg.qr(A)
    Q = Q * torch.det(Q).sign()
    s = 0.5 + 2.0 * torch.rand(1, dtype=dtype, generator=rng).item()
    t = torch.randn(3, dtype=dtype, generator=rng)
    return Sim3(s=s, R=Q, t=t)


def test_sim3_inverse_with_points():
    """Sim3 inverse recovers original points: inv(S)(S(x)) = x."""
    rng = torch.Generator().manual_seed(SEED)
    S = _random_sim3(rng)
    pts = torch.randn(20, 3, dtype=torch.float64, generator=rng)

    pts_transformed = sim3_transform_points(S, pts)
    S_inv = sim3_inverse(S)
    pts_recovered = sim3_transform_points(S_inv, pts_transformed)

    torch.testing.assert_close(pts_recovered, pts, atol=1e-12, rtol=0)


def test_sim3_compose_with_points():
    """compose(S1, S2)(x) = S1(S2(x))."""
    rng = torch.Generator().manual_seed(SEED)
    S1 = _random_sim3(rng)
    S2 = _random_sim3(rng)
    pts = torch.randn(20, 3, dtype=torch.float64, generator=rng)

    # Apply S2 then S1 sequentially
    pts_via_seq = sim3_transform_points(S1, sim3_transform_points(S2, pts))

    # Apply composed
    S12 = sim3_compose(S1, S2)
    pts_via_composed = sim3_transform_points(S12, pts)

    torch.testing.assert_close(pts_via_composed, pts_via_seq, atol=1e-12, rtol=0)


def test_sim3_compose_is_associative():
    """compose(S1, compose(S2, S3)) ≡ compose(compose(S1, S2), S3) on points."""
    rng = torch.Generator().manual_seed(SEED)
    S1 = _random_sim3(rng)
    S2 = _random_sim3(rng)
    S3 = _random_sim3(rng)
    pts = torch.randn(15, 3, dtype=torch.float64, generator=rng)

    left = sim3_transform_points(sim3_compose(S1, sim3_compose(S2, S3)), pts)
    right = sim3_transform_points(sim3_compose(sim3_compose(S1, S2), S3), pts)

    torch.testing.assert_close(left, right, atol=1e-12, rtol=0)


# ── Umeyama ──────────────────────────────────────────────────────────────


def test_umeyama_noisefree():
    """(a) Noise-free float64 recovery of a planted Sim3, tolerance ~1e-9."""
    rng = torch.Generator().manual_seed(SEED)
    # Planted Sim3
    planted_s = 2.5
    A = torch.randn(3, 3, dtype=torch.float64, generator=rng)
    Q, _ = torch.linalg.qr(A)
    Q = Q * torch.det(Q).sign()
    planted_R = Q
    planted_t = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float64)

    src = torch.randn(20, 3, dtype=torch.float64, generator=rng)
    dst = planted_s * (src @ planted_R.T) + planted_t  # x' = s*R*x + t

    recovered = umeyama(src, dst)

    assert abs(recovered.s - planted_s) < 1e-9, f"Scale: expected {planted_s}, got {recovered.s}"
    torch.testing.assert_close(recovered.R, planted_R, atol=1e-9, rtol=0)
    torch.testing.assert_close(recovered.t, planted_t, atol=1e-9, rtol=0)


def test_umeyama_noisy():
    """(b) Noisy recovery: tolerance at 3x the measured error for this seed.

    Measured errors (seed=42, n=50, sigma=0.01, planted_s=1.8):
        Scale error:    0.00069
        Rotation error: 0.00114  (Frobenius norm)
        Translation error: 0.00099

    Expected error scale ~ sigma * s / sqrt(n) = 0.01 * 1.8 / 7.07 ≈ 0.0025.
    Tolerances are set to 3x measured to handle minor floating-point
    variation while being much tighter than the old 0.1.
    """
    rng = torch.Generator().manual_seed(SEED)
    planted_s = 1.8
    A = torch.randn(3, 3, dtype=torch.float64, generator=rng)
    Q, _ = torch.linalg.qr(A)
    Q = Q * torch.det(Q).sign()
    planted_R = Q
    planted_t = torch.tensor([0.5, 1.0, -0.5], dtype=torch.float64)

    n_points = 50
    noise_std = 0.01
    src = torch.randn(n_points, 3, dtype=torch.float64, generator=rng)
    dst_clean = planted_s * (src @ planted_R.T) + planted_t
    noise = noise_std * torch.randn_like(dst_clean, generator=rng)
    dst_noisy = dst_clean + noise

    recovered = umeyama(src, dst_noisy)

    # 3x measured errors
    scale_tol = 0.0021   # 3 × 0.00069
    rot_tol = 0.0035     # 3 × 0.00114
    trans_tol = 0.003    # 3 × 0.00099

    s_err = abs(recovered.s - planted_s)
    assert s_err < scale_tol, (
        f"Scale error {s_err:.6f} exceeds tolerance {scale_tol} "
        f"(3x measured 0.00069)"
    )

    rot_err = torch.norm(recovered.R - planted_R).item()
    assert rot_err < rot_tol, (
        f"Rotation Frobenius error {rot_err:.6f} exceeds tolerance {rot_tol} "
        f"(3x measured 0.00114)"
    )

    t_err = torch.norm(recovered.t - planted_t).item()
    assert t_err < trans_tol, (
        f"Translation error {t_err:.6f} exceeds tolerance {trans_tol} "
        f"(3x measured 0.00099)"
    )


def test_umeyama_det_positive():
    """(c) det(R) = +1 even when data invites a reflection."""
    rng = torch.Generator().manual_seed(SEED)
    src = torch.randn(20, 3, dtype=torch.float64, generator=rng)
    # Reflect across x-axis to invite a reflection
    dst = src.clone()
    dst[:, 0] *= -1.0

    recovered = umeyama(src, dst)
    det_R = torch.det(recovered.R).item()
    assert abs(det_R - 1.0) < 1e-10, f"det(R) = {det_R}, expected +1 (reflection guard failed)"


def test_umeyama_too_few_points():
    """(d) Clear error for fewer than 3 points."""
    src = torch.randn(2, 3, dtype=torch.float64)
    dst = torch.randn(2, 3, dtype=torch.float64)
    with pytest.raises(ValueError, match="at least 3"):
        umeyama(src, dst)


def test_umeyama_collinear():
    """(d) Clear error for collinear (degenerate) input."""
    # All points on a line
    t_vals = torch.linspace(0, 1, 10, dtype=torch.float64).unsqueeze(1)
    src = t_vals * torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
    dst = src * 2.0 + 1.0
    with pytest.raises(ValueError, match="[Dd]egenerate|[Cc]ollinear|rank"):
        umeyama(src, dst)


# ── Slerp ────────────────────────────────────────────────────────────────


def test_slerp_endpoints():
    """slerp(q1, q2, 0) = q1, slerp(q1, q2, 1) = q2."""
    q1 = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)  # identity
    q2 = torch.tensor(
        [math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0], dtype=torch.float64
    )  # 90° around x

    r0 = slerp(q1, q2, 0.0)
    r1 = slerp(q1, q2, 1.0)

    torch.testing.assert_close(r0, q1, atol=1e-12, rtol=0)
    torch.testing.assert_close(r1, q2, atol=1e-12, rtol=0)


def test_slerp_midpoint():
    """slerp midpoint is at half the angle."""
    q1 = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)  # identity
    # 90° rotation around z → quaternion (cos45, 0, 0, sin45)
    angle = math.pi / 2
    q2 = torch.tensor(
        [math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)], dtype=torch.float64
    )

    mid = slerp(q1, q2, 0.5)
    # Expected: 45° around z → (cos22.5, 0, 0, sin22.5)
    half_angle = angle / 4
    expected = torch.tensor(
        [math.cos(half_angle), 0.0, 0.0, math.sin(half_angle)], dtype=torch.float64
    )
    torch.testing.assert_close(mid, expected, atol=1e-12, rtol=0)


def test_slerp_unit_norm():
    """slerp output is always unit norm."""
    rng = torch.Generator().manual_seed(SEED)
    q1 = torch.randn(4, dtype=torch.float64, generator=rng)
    q1 = q1 / q1.norm()
    q2 = torch.randn(4, dtype=torch.float64, generator=rng)
    q2 = q2 / q2.norm()

    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        r = slerp(q1, q2, t)
        assert abs(r.norm().item() - 1.0) < 1e-12, f"slerp(t={t}) norm = {r.norm()}"
