"""Camera model: pinhole projection, unprojection, pixel grid, intrinsics scaling.

Convention:
    - OpenCV: x right, y down, z forward.
    - Poses are camera-to-world (c2w), 4x4. w2c is derived, never stored.
    - Pixel centers at (i + 0.5, j + 0.5).
    - Camera holds torch tensors (float32 by default).

Dtype policy:
    - K and c2w are float32 in production.
    - float64 supported for tests via explicit dtype.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class Camera:
    """Pinhole camera with intrinsics and extrinsics.

    Attributes:
        K: Intrinsic matrix, shape [3, 3]. K = [[fx,0,cx],[0,fy,cy],[0,0,1]].
        c2w: Camera-to-world pose, shape [4, 4].
        W: Image width in pixels.
        H: Image height in pixels.
    """

    K: Tensor  # [3, 3], float32
    c2w: Tensor  # [4, 4], float32
    W: int
    H: int

    @property
    def w2c(self) -> Tensor:
        """World-to-camera transform (derived, never stored).

        Returns:
            [4, 4] tensor, inverse of c2w.
        """
        R = self.c2w[:3, :3]
        t = self.c2w[:3, 3]
        w2c = torch.eye(4, dtype=self.c2w.dtype, device=self.c2w.device)
        w2c[:3, :3] = R.T
        w2c[:3, 3] = -(R.T @ t)
        return w2c


def project(pts_world: Tensor, cam: Camera) -> Tensor:
    """Project 3D world points to 2D pixel coordinates.

    Args:
        pts_world: World-space points, shape [N, 3].
        cam: Camera instance.

    Returns:
        Pixel coordinates (u, v), shape [N, 2].
    """
    w2c = cam.w2c
    R, t = w2c[:3, :3], w2c[:3, 3]
    pts_cam = (pts_world @ R.T) + t  # [N, 3]
    fx, fy = cam.K[0, 0], cam.K[1, 1]
    cx, cy = cam.K[0, 2], cam.K[1, 2]
    u = fx * pts_cam[:, 0] / pts_cam[:, 2] + cx
    v = fy * pts_cam[:, 1] / pts_cam[:, 2] + cy
    return torch.stack([u, v], dim=-1)  # [N, 2]


def unproject(pixels: Tensor, depths: Tensor, cam: Camera) -> Tensor:
    """Unproject 2D pixels + depths to 3D world points.

    The depth is along the camera z-axis (not ray distance).

    Args:
        pixels: Pixel coordinates (u, v), shape [N, 2].
        depths: Depth values along z-axis, shape [N].
        cam: Camera instance.

    Returns:
        World-space points, shape [N, 3].
    """
    fx, fy = cam.K[0, 0], cam.K[1, 1]
    cx, cy = cam.K[0, 2], cam.K[1, 2]

    u, v = pixels[:, 0], pixels[:, 1]
    x_cam = (u - cx) / fx * depths
    y_cam = (v - cy) / fy * depths
    z_cam = depths

    pts_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)  # [N, 3]

    # Transform to world space
    R_c2w = cam.c2w[:3, :3]
    t_c2w = cam.c2w[:3, 3]
    pts_world = (pts_cam @ R_c2w.T) + t_c2w  # [N, 3]
    return pts_world


def pixel_grid(W: int, H: int, dtype: torch.dtype = torch.float32,
               device: torch.device | str = "cpu") -> Tensor:
    """Generate a grid of pixel center coordinates with +0.5 offset.

    Pixel centers are at (i + 0.5, j + 0.5) where i is the column index
    and j is the row index.

    Args:
        W: Image width.
        H: Image height.
        dtype: Output dtype.
        device: Output device.

    Returns:
        Pixel coordinates (u, v), shape [H, W, 2].
    """
    u = torch.arange(W, dtype=dtype, device=device) + 0.5  # [W]
    v = torch.arange(H, dtype=dtype, device=device) + 0.5  # [H]
    grid_v, grid_u = torch.meshgrid(v, u, indexing="ij")  # [H, W] each
    return torch.stack([grid_u, grid_v], dim=-1)  # [H, W, 2]


def resize_intrinsics(K: Tensor, scale: float) -> Tensor:
    """Scale intrinsics for a resized image.

    With pixel centers at i + 0.5, the mapping from original to resized
    continuous pixel coordinates is u' = u * scale. This gives:
        fx' = fx * scale
        cx' = cx * scale
    No shift is needed because the half-pixel convention makes the
    coordinate origin coincide with the sensor edge.

    Args:
        K: Intrinsic matrix, shape [3, 3].
        scale: Scale factor (e.g., 0.5 for half resolution).

    Returns:
        Scaled intrinsic matrix, shape [3, 3].
    """
    K_new = K.clone()
    K_new[0, 0] *= scale  # fx
    K_new[1, 1] *= scale  # fy
    K_new[0, 2] *= scale  # cx
    K_new[1, 2] *= scale  # cy
    return K_new
