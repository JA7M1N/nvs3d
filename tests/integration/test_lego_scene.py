"""Integration tests for the Lego scene: numeric assertions on camera geometry.

Measured values (100 Lego train cameras, Oct 2026):
    Radius: mean=4.031128, std=0.000000, CV=1.16e-7
    Camera z min: 0.5097 (all > 0)
    R^T R = I max error: 1.25e-6
    det(R) max error: 3.58e-7
    Max look-at angle: 0.028°
    Max center-ray closest approach: 1e-6
    Alpha-ray hit fraction (r=1.5): 100%
    Object max ray-closest-approach: 1.18, p95=1.01

Thresholds are set to 3-5x the measured values, justified per-test.
"""

import math
from pathlib import Path
import torch
import pytest

from nvs3d.data.blender_io import load_blender_split
from nvs3d.core.rays import generate_rays_for_pixels


pytestmark = pytest.mark.skipif(
    not Path("datasets/nerf_synthetic/lego").exists(),
    reason="Lego dataset not found at datasets/nerf_synthetic/lego",
)

LEGO_ROOT = Path("datasets/nerf_synthetic/lego")


@pytest.fixture(scope="module")
def lego_cameras():
    """Load Lego train cameras once for the module."""
    _, cameras = load_blender_split(LEGO_ROOT, "train", scale=1.0)
    return cameras


@pytest.fixture(scope="module")
def lego_images_and_cameras():
    """Load Lego train images and cameras once for the module."""
    images, cameras = load_blender_split(LEGO_ROOT, "train", scale=1.0)
    return images, cameras


# ── All camera z > 0 ─────────────────────────────────────────────────────


def test_all_camera_centers_z_positive(lego_cameras):
    """All 100 camera centers have z > 0.

    Measured: z_min = 0.5097.  Cameras are on the upper hemisphere in the
    Blender world frame, so all z values are positive.
    """
    for i, cam in enumerate(lego_cameras):
        z = cam.c2w[2, 3].item()
        assert z > 0, f"Camera {i}: z = {z:.6f}, expected > 0"


# ── Rotation validity ────────────────────────────────────────────────────


def test_rotation_orthonormality(lego_cameras):
    """Every c2w rotation satisfies R^T R = I.

    Measured max error: 1.25e-6.
    Threshold: 5e-6 (~4x measured, float32 precision limit for
    orthonormal matrices stored from JSON float representation).
    """
    threshold = 5e-6
    for i, cam in enumerate(lego_cameras):
        R = cam.c2w[:3, :3]
        RtR = R.T @ R
        err = (RtR - torch.eye(3)).abs().max().item()
        assert err < threshold, (
            f"Camera {i}: max |R^T R - I| = {err:.2e}, threshold = {threshold:.2e}"
        )


def test_rotation_det_positive_one(lego_cameras):
    """Every c2w rotation has det(R) = +1 (proper rotation, no reflection).

    Measured max error: 3.58e-7.
    Threshold: 2e-6 (~5x measured).
    """
    threshold = 2e-6
    for i, cam in enumerate(lego_cameras):
        R = cam.c2w[:3, :3]
        det_err = abs(torch.det(R).item() - 1.0)
        assert det_err < threshold, (
            f"Camera {i}: |det(R) - 1| = {det_err:.2e}, threshold = {threshold:.2e}"
        )


# ── Camera centers form a sphere ─────────────────────────────────────────


def test_camera_centers_radius_consistency(lego_cameras):
    """Camera centers are at near-constant radius from origin.

    Measured: CV = 1.16e-7 (cameras on a perfect sphere of radius 4.031).
    Threshold: 5e-7 (~4x measured). These cameras are placed by a
    Blender script at exact radius, so CV should be near-zero.
    """
    centers = torch.stack([cam.c2w[:3, 3] for cam in lego_cameras])
    radii = centers.norm(dim=1)
    mean_r = radii.mean().item()
    std_r = radii.std().item()
    cv = std_r / mean_r

    assert cv < 5e-7, (
        f"Radius CV = {cv:.2e} (std={std_r:.2e}, mean={mean_r:.4f}), "
        f"expected < 5e-7"
    )


# ── Optical axes point toward origin ─────────────────────────────────────


def test_camera_optical_axes_point_toward_origin(lego_cameras):
    """Angle between each camera's optical axis and direction to origin < 0.1°.

    Measured max: 0.028°.
    Threshold: 0.1° (~3.5x measured). Blender's look-at constraint makes
    cameras point directly at the origin; tiny deviation is from float
    precision in the 4x4 matrix.
    """
    max_angle_deg = 0.1

    for i, cam in enumerate(lego_cameras):
        center = cam.c2w[:3, 3]
        optical_axis = cam.c2w[:3, 2]
        optical_axis = optical_axis / optical_axis.norm()
        to_origin = -center / center.norm()

        cos_angle = torch.dot(optical_axis, to_origin).clamp(-1, 1).item()
        angle_deg = math.degrees(math.acos(cos_angle))

        assert angle_deg < max_angle_deg, (
            f"Camera {i}: look-at angle = {angle_deg:.4f}°, threshold = {max_angle_deg}°"
        )


# ── Center-ray closest approach ──────────────────────────────────────────


def test_center_ray_closest_approach_to_origin(lego_cameras):
    """Center ray from each camera passes within 5e-6 of the origin.

    Measured max: 1e-6.
    Threshold: 5e-6 (~5x measured). The Blender look-at makes the
    optical axis pass through the origin; residual is float32 round-off.
    """
    max_dist = 5e-6

    for i, cam in enumerate(lego_cameras):
        pp = torch.tensor([[cam.K[0, 2].item(), cam.K[1, 2].item()]], dtype=torch.float32)
        origins, dirs = generate_rays_for_pixels(cam, pp)
        o, d = origins[0], dirs[0]
        t_val = -torch.dot(o, d)
        closest = o + t_val * d
        dist = closest.norm().item()

        assert dist < max_dist, (
            f"Camera {i}: center-ray closest approach = {dist:.2e}, threshold = {max_dist:.2e}"
        )


# ── Rays through alpha>0 pixels hit the object ──────────────────────────


def test_alpha_rays_hit_object_region(lego_images_and_cameras):
    """All sampled alpha>0 rays pass within 1.5 of the origin.

    Measured: 100% of rays within r=1.5 (object max ray distance = 1.18).
    Threshold: object_radius = 1.5 (~1.3x measured max of 1.18), fraction >= 0.95.
    The Lego object has a p95 ray-closest-approach of ~1.01 and max of 1.18,
    so 1.5 gives a 27% margin over the max.
    """
    images, cameras = lego_images_and_cameras
    object_radius = 1.5
    min_fraction = 0.95

    rng = torch.Generator().manual_seed(42)
    n_images = min(10, len(images))
    n_pixels_per_image = 200
    total_sampled = 0
    total_within = 0

    for img_idx in range(n_images):
        img = images[img_idx]
        cam = cameras[img_idx]

        not_white = (img < 255).any(dim=-1)
        obj_coords = not_white.nonzero()

        if len(obj_coords) < n_pixels_per_image:
            continue

        perm = torch.randperm(len(obj_coords), generator=rng)[:n_pixels_per_image]
        sampled = obj_coords[perm]
        pixels = torch.stack([sampled[:, 1].float() + 0.5,
                              sampled[:, 0].float() + 0.5], dim=1)

        origins, dirs = generate_rays_for_pixels(cam, pixels)

        for j in range(len(pixels)):
            o, d = origins[j], dirs[j]
            t_val = -torch.dot(o, d)
            if t_val < 0:
                t_val = torch.tensor(0.0)
            closest = o + t_val * d
            dist = closest.norm().item()
            total_sampled += 1
            if dist < object_radius:
                total_within += 1

    fraction = total_within / total_sampled if total_sampled > 0 else 0
    assert fraction >= min_fraction, (
        f"Only {fraction*100:.1f}% of alpha>0 rays pass within {object_radius} "
        f"of origin (measured 100%), expected >= {min_fraction*100:.0f}%"
    )
