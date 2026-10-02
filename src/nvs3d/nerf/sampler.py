"""Stratified sampler for NeRF.

(verify): t_i = t_near + (i + U_i)/N · (t_far − t_near), U_i ~ U[0,1);
          deterministic midpoints at eval — blueprint line 142.

Interface:
    sample_stratified(rays_o, rays_d, near, far, N_samples, perturb, generator)
    → t_vals [B, N_samples]

near/far: scalar or [B] / [B,1] tensor (per-ray bounds).
Samples are detached (no gradient flows through t_vals).
"""

from __future__ import annotations

import torch
from torch import Tensor


def sample_stratified(
    rays_o: Tensor,
    rays_d: Tensor,
    near: float | Tensor,
    far: float | Tensor,
    N_samples: int,
    perturb: bool = True,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Stratified sampling along rays.

    Each sample i is drawn from bin [near + i·Δ, near + (i+1)·Δ]
    where Δ = (far - near) / N_samples.

    Args:
        rays_o: Ray origins, shape [B, 3]. Not used directly but defines B.
        rays_d: Ray directions, shape [B, 3]. Not used directly.
        near: Near bound. Scalar or [B] / [B, 1] tensor.
        far: Far bound. Scalar or [B] / [B, 1] tensor.
        N_samples: Number of samples per ray.
        perturb: If True, jitter within bins. If False, sample at midpoints.
        generator: Optional torch.Generator for reproducibility.

    Returns:
        t_vals: Sample distances, shape [B, N_samples]. Detached.
    """
    B = rays_o.shape[0]
    device = rays_o.device
    dtype = rays_o.dtype

    # Convert near/far to [B, 1] tensors
    if isinstance(near, (int, float)):
        near_t = torch.full((B, 1), near, dtype=dtype, device=device)
    else:
        near_t = near.reshape(B, 1).to(dtype=dtype, device=device)

    if isinstance(far, (int, float)):
        far_t = torch.full((B, 1), far, dtype=dtype, device=device)
    else:
        far_t = far.reshape(B, 1).to(dtype=dtype, device=device)

    # Bin edges: i / N for i = 0, 1, ..., N-1
    # t_i = near + (i + offset) / N · (far - near)
    i_vals = torch.arange(N_samples, dtype=dtype, device=device)  # [N]

    if perturb:
        # U_i ~ U[0, 1)
        u = torch.rand(B, N_samples, dtype=dtype, device=device, generator=generator)
        t_vals = near_t + (i_vals + u) / N_samples * (far_t - near_t)
    else:
        # Deterministic midpoints: offset = 0.5
        t_vals = near_t + (i_vals + 0.5) / N_samples * (far_t - near_t)

    # Samples must be detached (no gradient through t_vals)
    return t_vals.detach()
