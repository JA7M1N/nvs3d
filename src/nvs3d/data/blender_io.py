"""Blender/NeRF synthetic data loader.

Reads transforms_{split}.json and the corresponding RGBA PNG images.
Converts from Blender/OpenGL convention to OpenCV convention.

Convention flip (ONLY here, nowhere else):
    OpenGL: x right, y up, -z forward
    OpenCV: x right, y down, z forward
    Applied as: c2w[:3, 1:3] *= -1  (negate y and z columns of rotation)

(verify): The flip formula c2w[:3, 1:3] *= -1 is from blueprint line 122.
(verify): camera_angle_x -> focal: fx = W / (2 * tan(camera_angle_x / 2))

Images are stored as uint8 [H, W, 3] after RGBA compositing to white background.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch import Tensor
from PIL import Image
import numpy as np

from nvs3d.core.cameras import Camera, resize_intrinsics


def load_blender_split(
    root: Path | str,
    split: str,
    scale: float = 1.0,
    white_bkgd: bool = True,
) -> tuple[list[Tensor], list[Camera]]:
    """Load a Blender/NeRF synthetic dataset split.

    Args:
        root: Path to the scene directory (e.g., datasets/nerf_synthetic/lego).
        split: One of 'train', 'val', 'test'.
        scale: Image downscale factor (e.g., 0.5 for half resolution).
        white_bkgd: If True, composite RGBA to RGB over white background.

    Returns:
        images: List of [H, W, 3] uint8 tensors (RGB).
        cameras: List of Camera instances in OpenCV convention.
    """
    root = Path(root)
    transforms_path = root / f"transforms_{split}.json"
    if not transforms_path.exists():
        raise FileNotFoundError(f"Transforms file not found: {transforms_path}")

    with open(transforms_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    camera_angle_x: float = meta["camera_angle_x"]
    frames: list[dict] = meta["frames"]

    images: list[Tensor] = []
    cameras: list[Camera] = []

    for frame in frames:
        # ── Load image ───────────────────────────────────────────────
        file_path = frame["file_path"]
        # Add .png extension if missing
        if not file_path.endswith(".png"):
            file_path = file_path + ".png"
        img_path = root / file_path
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        img_pil = Image.open(img_path)
        img_np = np.array(img_pil)  # [H, W, 4] uint8 for RGBA

        # Get original dimensions
        H_orig, W_orig = img_np.shape[:2]

        # Downscale if needed
        if scale != 1.0:
            W_new = int(W_orig * scale)
            H_new = int(H_orig * scale)
            img_pil_resized = img_pil.resize((W_new, H_new), Image.LANCZOS)
            img_np = np.array(img_pil_resized)
        else:
            W_new, H_new = W_orig, H_orig

        # RGBA compositing to RGB
        if img_np.shape[-1] == 4:
            img_f = img_np.astype(np.float32) / 255.0
            alpha = img_f[..., 3:4]
            rgb = img_f[..., :3]
            if white_bkgd:
                rgb = rgb * alpha + (1.0 - alpha) * 1.0
            else:
                rgb = rgb * alpha
            img_uint8 = (rgb * 255.0).clip(0, 255).astype(np.uint8)
        else:
            img_uint8 = img_np[..., :3]

        images.append(torch.from_numpy(img_uint8))  # [H, W, 3] uint8

        # ── Build camera ─────────────────────────────────────────────
        # Focal length from camera_angle_x
        fx = W_orig / (2.0 * math.tan(camera_angle_x / 2.0))
        fy = fx  # Square pixels assumed for Blender synthetic
        cx = W_orig / 2.0
        cy = H_orig / 2.0

        K = torch.tensor([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ], dtype=torch.float32)

        # Apply scale to intrinsics (plain scale, no shift, per +0.5 pixel center convention)
        if scale != 1.0:
            K = resize_intrinsics(K, scale)

        # Parse c2w from transforms (OpenGL convention)
        c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32)  # [4, 4]

        # ── Convention flip: OpenGL → OpenCV ─────────────────────────
        # Negate y and z columns of the rotation part.
        # This converts (y-up, -z-forward) → (y-down, z-forward).
        # (verify): blueprint line 122: c2w[:3, 1:3] *= -1
        c2w[:3, 1:3] *= -1

        cameras.append(Camera(K=K, c2w=c2w, W=W_new, H=H_new))

    return images, cameras
