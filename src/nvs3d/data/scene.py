"""Scene data representation.

SceneData is the universal interface between data loading (L2) and training (L5).
Trainers never read COLMAP or Blender files directly; they receive SceneData.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor

from nvs3d.core.cameras import Camera
from nvs3d.core.transforms import Sim3
from nvs3d.data.blender_io import load_blender_split


@dataclass
class SceneData:
    """Universal scene representation consumed by trainers.

    Attributes:
        images: List of [H, W, 3] uint8 tensors, or a stacked tensor.
        cameras: List of Camera instances (OpenCV convention, c2w).
        points_xyz: Optional SfM point cloud, shape [P, 3].
        points_rgb: Optional point colors, shape [P, 3].
        split: Dict mapping split names to lists of image indices.
                e.g., {"train": [0,1,...], "test": [100,101,...]}.
        extent: Scene extent (1.1 * max camera distance from centroid).
        norm: Sim3 normalization record (COLMAP-world → training-world).
    """

    images: list[Tensor] | Tensor
    cameras: list[Camera]
    points_xyz: Tensor | None = None
    points_rgb: Tensor | None = None
    split: dict[str, list[int]] = field(default_factory=dict)
    extent: float = 1.0
    norm: Sim3 = field(default_factory=lambda: Sim3(
        s=1.0,
        R=torch.eye(3, dtype=torch.float32),
        t=torch.zeros(3, dtype=torch.float32),
    ))


def load_blender_scene_data(
    root: Path | str,
    scale: float = 1.0,
    white_bkgd: bool = True,
) -> SceneData:
    """Load a full Blender/NeRF synthetic scene into SceneData.

    Loads train + val + test splits and computes scene extent from camera
    positions.

    Args:
        root: Path to the scene directory.
        scale: Image downscale factor.
        white_bkgd: Composite RGBA to white background.

    Returns:
        SceneData instance.
    """
    root = Path(root)

    all_images: list[Tensor] = []
    all_cameras: list[Camera] = []
    split_indices: dict[str, list[int]] = {}

    idx = 0
    for split_name in ["train", "val", "test"]:
        transforms_path = root / f"transforms_{split_name}.json"
        if not transforms_path.exists():
            continue
        imgs, cams = load_blender_split(root, split_name, scale=scale, white_bkgd=white_bkgd)
        split_indices[split_name] = list(range(idx, idx + len(imgs)))
        all_images.extend(imgs)
        all_cameras.extend(cams)
        idx += len(imgs)

    # Compute scene extent: 1.1 * max camera distance from centroid
    centers = torch.stack([cam.c2w[:3, 3] for cam in all_cameras])  # [N, 3]
    centroid = centers.mean(dim=0)
    dists = (centers - centroid).norm(dim=1)
    extent = 1.1 * dists.max().item()

    # Identity normalization for Blender scenes (already centered)
    norm = Sim3(
        s=1.0,
        R=torch.eye(3, dtype=torch.float32),
        t=torch.zeros(3, dtype=torch.float32),
    )

    return SceneData(
        images=all_images,
        cameras=all_cameras,
        points_xyz=None,
        points_rgb=None,
        split=split_indices,
        extent=extent,
        norm=norm,
    )
