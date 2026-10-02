"""Integration test: random NeRF field + volume rendering is finite.

Uses P1 Camera + ray generation with a small synthetic camera (no dataset).
Creates a NeRFField, samples points along rays, evaluates the field,
volume renders, and checks all outputs are finite and in valid ranges.
"""

import torch
import pytest

from nvs3d.core.cameras import Camera
from nvs3d.core.rays import generate_rays
from nvs3d.nerf.field import NeRFField
from nvs3d.nerf.sampler import sample_stratified
from nvs3d.nerf.volume import volume_render

SEED = 42


def test_random_field_render_finite():
    """End-to-end: camera → rays → sample → field → volume render → finite output.

    Small synthetic camera W=16, H=16, identity pose.
    Verifies the entire P2 pipeline produces valid numbers.
    """
    torch.manual_seed(SEED)

    # Small synthetic camera (identity pose, looking down +z)
    fx, fy = 32.0, 32.0
    cx, cy = 8.0, 8.0
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32)
    c2w = torch.eye(4, dtype=torch.float32)
    cam = Camera(K=K, c2w=c2w, W=16, H=16)

    # Generate rays
    origins, directions = generate_rays(cam)  # [H, W, 3]
    rays_o = origins.reshape(-1, 3)  # [256, 3]
    rays_d = directions.reshape(-1, 3)  # [256, 3]

    B = rays_o.shape[0]
    N_samples = 16
    near, far = 2.0, 6.0

    # Sample t_vals
    gen = torch.Generator().manual_seed(SEED)
    t_vals = sample_stratified(rays_o, rays_d, near=near, far=far,
                               N_samples=N_samples, perturb=True, generator=gen)
    assert t_vals.shape == (B, N_samples)

    # Compute sample positions: pts = o + t·d  → [B, N, 3]
    pts = rays_o.unsqueeze(1) + t_vals.unsqueeze(-1) * rays_d.unsqueeze(1)
    assert pts.shape == (B, N_samples, 3)

    # Evaluate field
    field = NeRFField()
    field.eval()
    pts_flat = pts.reshape(-1, 3)  # [B*N, 3]
    dirs_flat = rays_d.unsqueeze(1).expand(-1, N_samples, -1).reshape(-1, 3)  # [B*N, 3]

    with torch.no_grad():
        sigma_flat, rgb_flat = field(pts_flat, dirs_flat)

    sigma = sigma_flat.reshape(B, N_samples)  # [B, N]
    rgb = rgb_flat.reshape(B, N_samples, 3)  # [B, N, 3]

    # Volume render
    bg = torch.ones(3, dtype=torch.float32)  # white background
    out = volume_render(sigma, rgb, t_vals, bg)

    # All outputs must be finite
    assert torch.isfinite(out["rgb"]).all(), "RGB has non-finite values"
    assert torch.isfinite(out["depth"]).all(), "Depth has non-finite values"
    assert torch.isfinite(out["acc"]).all(), "Acc has non-finite values"
    assert torch.isfinite(out["weights"]).all(), "Weights have non-finite values"

    # RGB in [0, 1] (due to sigmoid + background compositing)
    assert (out["rgb"] >= 0).all(), f"RGB min = {out['rgb'].min().item()}"
    assert (out["rgb"] <= 1).all(), f"RGB max = {out['rgb'].max().item()}"

    # Depth ≥ 0
    assert (out["depth"] >= 0).all(), f"Depth min = {out['depth'].min().item()}"

    # Acc in [0, 1]
    assert (out["acc"] >= -1e-6).all(), f"Acc min = {out['acc'].min().item()}"
    assert (out["acc"] <= 1 + 1e-6).all(), f"Acc max = {out['acc'].max().item()}"

    # Image shape matches
    rendered = out["rgb"].reshape(16, 16, 3)
    assert rendered.shape == (16, 16, 3)
