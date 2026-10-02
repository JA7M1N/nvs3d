"""SE3 and Sim3 transforms, Umeyama alignment, and quaternion slerp.

Convention:
    - SE3: 4x4 homogeneous matrix, stored as torch.Tensor.
    - Sim3: x' = s * R @ x + t, stored as Sim3 dataclass.
    - Quaternions: (w, x, y, z), unit-normalized before use.

Dtype policy:
    - float32 for production use.
    - float64 supported and recommended for tests / Umeyama alignment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


# ── Sim3 dataclass ───────────────────────────────────────────────────────


@dataclass
class Sim3:
    """Similarity transform: x' = s * R @ x + t.

    Attributes:
        s: Positive scale factor (float).
        R: Rotation matrix, shape [3, 3].
        t: Translation vector, shape [3].
    """

    s: float
    R: Tensor  # [3, 3]
    t: Tensor  # [3]


# ── SE3 operations ───────────────────────────────────────────────────────


def se3_inverse(T: Tensor) -> Tensor:
    """Compute the inverse of an SE3 matrix efficiently.

    Uses R^T and -R^T @ t instead of a full matrix inverse.

    Args:
        T: SE3 matrix, shape [4, 4].

    Returns:
        Inverse SE3 matrix, shape [4, 4].
    """
    R = T[:3, :3]
    t = T[:3, 3]
    R_inv = R.T
    t_inv = -(R_inv @ t)
    T_inv = torch.eye(4, dtype=T.dtype, device=T.device)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv
    return T_inv


def se3_compose(T1: Tensor, T2: Tensor) -> Tensor:
    """Compose two SE3 transforms: result = T1 @ T2.

    Args:
        T1: First SE3 matrix, shape [4, 4].
        T2: Second SE3 matrix, shape [4, 4].

    Returns:
        Composed SE3 matrix, shape [4, 4].
    """
    return T1 @ T2


# ── Sim3 operations ──────────────────────────────────────────────────────


def sim3_transform_points(sim3: Sim3, pts: Tensor) -> Tensor:
    """Apply Sim3 transform to points: x' = s * R @ x + t.

    Args:
        sim3: Sim3 transform.
        pts: Points, shape [N, 3].

    Returns:
        Transformed points, shape [N, 3].
    """
    return sim3.s * (pts @ sim3.R.T) + sim3.t


def sim3_inverse(sim3: Sim3) -> Sim3:
    """Compute the inverse Sim3 transform.

    If x' = s * R @ x + t, then x = (1/s) * R^T @ (x' - t).
    So: s_inv = 1/s, R_inv = R^T, t_inv = -(1/s) * R^T @ t.

    Args:
        sim3: Sim3 transform.

    Returns:
        Inverse Sim3 transform.
    """
    s_inv = 1.0 / sim3.s
    R_inv = sim3.R.T
    t_inv = -s_inv * (R_inv @ sim3.t)
    return Sim3(s=s_inv, R=R_inv, t=t_inv)


def sim3_compose(s1: Sim3, s2: Sim3) -> Sim3:
    """Compose two Sim3 transforms: result applies s2 first, then s1.

    x'' = s1 * R1 @ (s2 * R2 @ x + t2) + t1
         = (s1*s2) * (R1@R2) @ x + (s1 * R1 @ t2 + t1)

    Args:
        s1: Outer Sim3 transform.
        s2: Inner Sim3 transform (applied first).

    Returns:
        Composed Sim3 transform.
    """
    s = s1.s * s2.s
    R = s1.R @ s2.R
    t = s1.s * (s1.R @ s2.t) + s1.t
    return Sim3(s=s, R=R, t=t)


# ── Umeyama alignment ───────────────────────────────────────────────────


def umeyama(src: Tensor, dst: Tensor) -> Sim3:
    """Least-squares Sim3 alignment via the Umeyama method.

    Finds (s, R, t) minimizing || dst - (s * R @ src + t) ||^2.
    Convention: x' = s * R @ x + t.

    Uses SVD with a reflection guard to ensure det(R) = +1.

    Args:
        src: Source points, shape [N, 3], dtype float64 recommended.
        dst: Destination points, shape [N, 3], same dtype as src.

    Returns:
        Sim3 transform.

    Raises:
        ValueError: If fewer than 3 points, or points are degenerate (collinear).
    """
    if src.shape[0] < 3:
        raise ValueError(f"Umeyama requires at least 3 points, got {src.shape[0]}")
    if src.shape != dst.shape:
        raise ValueError(f"Shape mismatch: src {src.shape} vs dst {dst.shape}")

    n = src.shape[0]
    dim = src.shape[1]

    # Centroids
    mu_src = src.mean(dim=0)
    mu_dst = dst.mean(dim=0)

    # Center the points
    src_c = src - mu_src
    dst_c = dst - mu_dst

    # Check for degeneracy: rank of centered source must be >= 2
    # (i.e., not collinear)
    _, S_check, _ = torch.linalg.svd(src_c)
    # For 3D points, if the second singular value is near-zero, points are collinear
    if S_check[1] < 1e-10 * S_check[0]:
        raise ValueError(
            "Degenerate input: source points are collinear (rank < 2). "
            "Umeyama requires non-degenerate point configurations."
        )

    # Variances
    var_src = (src_c * src_c).sum() / n

    # Cross-covariance matrix
    Sigma = (dst_c.T @ src_c) / n  # [3, 3]

    # SVD
    U, D, Vt = torch.linalg.svd(Sigma)

    # Reflection guard: ensure det(R) = +1
    S = torch.ones(dim, dtype=src.dtype, device=src.device)
    if torch.det(U) * torch.det(Vt) < 0:
        S[-1] = -1

    # Rotation
    R = U @ torch.diag(S) @ Vt

    # Scale
    s = (torch.sum(D * S) / var_src).item()

    # Translation
    t = mu_dst - s * (R @ mu_src)

    return Sim3(s=s, R=R, t=t)


# ── Quaternion slerp ─────────────────────────────────────────────────────
# Part 14 step 2: "core/transforms.py (SE3, Sim3, Umeyama, slerp) + tests."


def slerp(q1: Tensor, q2: Tensor, t: float) -> Tensor:
    """Spherical linear interpolation between two unit quaternions.

    Quaternion convention: (w, x, y, z).

    Args:
        q1: Start quaternion, shape [4], unit norm.
        q2: End quaternion, shape [4], unit norm.
        t: Interpolation parameter in [0, 1].

    Returns:
        Interpolated unit quaternion, shape [4].
    """
    # Normalize inputs
    q1 = q1 / q1.norm()
    q2 = q2 / q2.norm()

    # Dot product
    dot = torch.dot(q1, q2).clamp(-1.0, 1.0)

    # If dot < 0, negate one to take the short path
    if dot.item() < 0:
        q2 = -q2
        dot = -dot

    # If nearly identical, use linear interpolation
    if dot.item() > 0.9995:
        result = q1 + t * (q2 - q1)
        return result / result.norm()

    # Slerp formula
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)

    w1 = torch.sin((1.0 - t) * theta) / sin_theta
    w2 = torch.sin(t * theta) / sin_theta

    result = w1 * q1 + w2 * q2
    return result / result.norm()
