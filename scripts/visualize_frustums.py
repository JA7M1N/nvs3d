"""Frustum visualization script for camera poses.

Plots camera positions and optical axes in 3D using matplotlib.
Uses Agg backend (no GUI) and saves to a PNG file.

Usage:
    python scripts/visualize_frustums.py --data_dir datasets/nerf_synthetic/lego --output frustums.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize camera frustums as a 3D plot")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to Blender scene directory")
    parser.add_argument("--output", type=str, default="outputs/frustums.png", help="Output PNG path")
    parser.add_argument("--split", type=str, default="train", help="Split to visualize")
    parser.add_argument("--scale", type=float, default=1.0, help="Image downscale factor")
    args = parser.parse_args()

    # Import matplotlib with Agg backend (no GUI dependency)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError:
        print("matplotlib is required for this script. Install with: pip install matplotlib")
        print("Or install the viz extras: pip install -e '.[viz]'")
        sys.exit(1)

    import torch
    import numpy as np
    from nvs3d.data.blender_io import load_blender_split

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        sys.exit(1)

    print(f"Loading {args.split} split from {data_dir}...")
    _, cameras = load_blender_split(data_dir, args.split, scale=args.scale)
    print(f"Loaded {len(cameras)} cameras")

    # Extract camera centers and optical axes
    centers = torch.stack([cam.c2w[:3, 3] for cam in cameras]).numpy()
    # Optical axis = z-column of c2w rotation (OpenCV: z forward)
    axes = torch.stack([cam.c2w[:3, 2] for cam in cameras]).numpy()

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Plot camera centers
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
               c="blue", s=20, alpha=0.7, label="Camera centers")

    # Plot optical axes as arrows
    arrow_len = 0.5
    ax.quiver(centers[:, 0], centers[:, 1], centers[:, 2],
              axes[:, 0] * arrow_len, axes[:, 1] * arrow_len, axes[:, 2] * arrow_len,
              color="red", alpha=0.5, arrow_length_ratio=0.2)

    # Plot origin
    ax.scatter([0], [0], [0], c="green", s=100, marker="*", label="Origin")

    # Labels and formatting
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Camera Frustums: {data_dir.name} ({args.split})")
    ax.legend()

    # Equal aspect ratio
    max_range = max(
        np.ptp(centers[:, 0]), np.ptp(centers[:, 1]), np.ptp(centers[:, 2])
    ) / 2.0
    mid = centers.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    output_path = Path(args.output)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved frustum plot to {output_path}")

    # Print summary statistics
    radii = np.linalg.norm(centers, axis=1)
    print(f"Camera radius: mean={radii.mean():.3f}, std={radii.std():.3f}, "
          f"CV={radii.std()/radii.mean():.4f}")

    # Check optical axes point toward origin
    dirs_to_origin = -centers / np.linalg.norm(centers, axis=1, keepdims=True)
    cos_angles = np.sum(axes * dirs_to_origin, axis=1).clip(-1, 1)
    angles_deg = np.degrees(np.arccos(cos_angles))
    print(f"Angle to origin: mean={angles_deg.mean():.2f}°, max={angles_deg.max():.2f}°")


if __name__ == "__main__":
    main()
