"""Volume rendering for NeRF.

(verify): δ_i = t_{i+1} − t_i, last δ = 1e10 — blueprint line 164.
(verify): α_i = 1 − exp(−σ_i δ_i) — blueprint line 164.
(verify): T_i = ∏_{j<i}(1 − α_j)  (exclusive cumprod) — blueprint line 164.
(verify): w_i = T_i α_i, Ĉ = Σ w_i c_i, depth = Σ w_i t_i — blueprint line 164.
(verify): Background: Ĉ + (1−acc)·bg — blueprint line 164.

Interface:
    volume_render(sigma, rgb, t_vals, bg_color, dirs=None, last_delta=1e10)
    bg_color is an explicit [3] tensor.  No white_bkgd flag.
    If dirs is provided, δ is scaled by ‖dirs‖ (parametric → Euclidean).
"""

from __future__ import annotations

import torch
from torch import Tensor


def volume_render(
    sigma: Tensor,
    rgb: Tensor,
    t_vals: Tensor,
    bg_color: Tensor,
    dirs: Tensor | None = None,
    last_delta: float = 1e10,
) -> dict[str, Tensor]:
    """Discrete volume rendering along rays.

    Args:
        sigma: Density values, shape [B, N].
        rgb: Color values, shape [B, N, 3].
        t_vals: Sample distances along rays, shape [B, N].
        bg_color: Background color, shape [3].
        dirs: Optional ray directions, shape [B, 3].
            If provided, deltas are scaled by ‖dirs‖ to convert from
            parametric Δt to Euclidean distance.
        last_delta: Value for the last interval (default 1e10).

    Returns:
        Dict with:
            rgb: Rendered color, shape [B, 3].
            depth: Expected depth, shape [B].
            acc: Accumulated opacity, shape [B].
            weights: Per-sample weights, shape [B, N].
    """
    # δ_i = t_{i+1} − t_i, last δ = last_delta
    deltas = t_vals[..., 1:] - t_vals[..., :-1]  # [B, N-1]
    B = sigma.shape[0]
    last_d = torch.full((B, 1), last_delta, dtype=t_vals.dtype, device=t_vals.device)
    deltas = torch.cat([deltas, last_d], dim=-1)  # [B, N]

    # If dirs provided, scale by ‖dirs‖ (parametric → Euclidean distance)
    if dirs is not None:
        dir_norm = dirs.norm(dim=-1, keepdim=True)  # [B, 1]
        deltas = deltas * dir_norm

    # α_i = 1 − exp(−σ_i · δ_i)
    alpha = 1.0 - torch.exp(-sigma * deltas)  # [B, N]

    # T_i = ∏_{j<i}(1 − α_j)  — exclusive cumprod
    # T[0] = 1, T[i] = (1-α_0)·(1-α_1)·…·(1-α_{i-1})
    one_minus_alpha = 1.0 - alpha  # [B, N]
    # cumprod gives [x0, x0·x1, x0·x1·x2, ...]  (inclusive)
    # exclusive: shift right by 1, prepend 1
    cumprod_inclusive = torch.cumprod(one_minus_alpha, dim=-1)  # [B, N]
    T = torch.cat([
        torch.ones_like(cumprod_inclusive[..., :1]),  # T[0] = 1
        cumprod_inclusive[..., :-1],  # T[1..N-1]
    ], dim=-1)  # [B, N]

    # w_i = T_i · α_i
    weights = T * alpha  # [B, N]

    # Ĉ = Σ w_i · c_i + (1 − acc) · bg
    acc = weights.sum(dim=-1)  # [B]
    rgb_rendered = (weights.unsqueeze(-1) * rgb).sum(dim=-2)  # [B, 3]
    rgb_rendered = rgb_rendered + (1.0 - acc).unsqueeze(-1) * bg_color  # [B, 3]

    # depth = Σ w_i · t_i
    depth = (weights * t_vals).sum(dim=-1)  # [B]

    return {
        "rgb": rgb_rendered,
        "depth": depth,
        "acc": acc,
        "weights": weights,
    }
