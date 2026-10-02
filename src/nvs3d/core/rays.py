"""Ray generation from cameras.

Convention:
    - Rays originate at the camera center (c2w[:3, 3]).
    - Directions are unit-normalized world-space vectors.
    - No per-ray Python loops; fully vectorized.

Dtype policy:
    - float32 for production.
"""

from __future__ import annotations

import torch
from torch import Tensor

from nvs3d.core.cameras import Camera, pixel_grid


def generate_rays(cam: Camera) -> tuple[Tensor, Tensor]:
    """Generate rays for all pixels in an image.

    Args:
        cam: Camera instance.

    Returns:
        origins: Ray origins in world space, shape [H, W, 3].
        directions: Unit-normalized ray directions in world space, shape [H, W, 3].
    """
    grid = pixel_grid(cam.W, cam.H, dtype=cam.K.dtype, device=cam.K.device)  # [H, W, 2]
    pixels_flat = grid.reshape(-1, 2)  # [H*W, 2]

    origins_flat, dirs_flat = generate_rays_for_pixels(cam, pixels_flat)

    origins = origins_flat.reshape(cam.H, cam.W, 3)
    directions = dirs_flat.reshape(cam.H, cam.W, 3)
    return origins, directions


def generate_rays_for_pixels(cam: Camera, pixels: Tensor) -> tuple[Tensor, Tensor]:
    """Generate rays for specific pixel coordinates.

    Args:
        cam: Camera instance.
        pixels: Pixel coordinates (u, v), shape [N, 2].

    Returns:
        origins: Ray origins in world space, shape [N, 3].
        directions: Unit-normalized ray directions in world space, shape [N, 3].
    """
    fx, fy = cam.K[0, 0], cam.K[1, 1]
    cx, cy = cam.K[0, 2], cam.K[1, 2]

    u, v = pixels[:, 0], pixels[:, 1]

    # Camera-space direction (OpenCV: x right, y down, z forward)
    d_cam = torch.stack([
        (u - cx) / fx,
        (v - cy) / fy,
        torch.ones_like(u),
    ], dim=-1)  # [N, 3]

    # Rotate to world space
    R_c2w = cam.c2w[:3, :3]  # [3, 3]
    d_world = d_cam @ R_c2w.T  # [N, 3]

    # Normalize to unit length
    d_world = d_world / d_world.norm(dim=-1, keepdim=True)

    # Origin is the camera center (same for all rays)
    origin = cam.c2w[:3, 3]  # [3]
    origins = origin.unsqueeze(0).expand(len(pixels), 3)  # [N, 3]

    return origins, d_world
