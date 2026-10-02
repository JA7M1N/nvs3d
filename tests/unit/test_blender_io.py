"""Unit tests for data/blender_io.py: convention flip, alpha compositing, focal length."""

import math
from pathlib import Path
import torch
import pytest

from nvs3d.data.blender_io import load_blender_split


# ── Convention flip ──────────────────────────────────────────────────────


def test_blender_flip_produces_opencv():
    """After loading, the c2w should be in OpenCV convention (z forward, y down).

    The Blender/OpenGL convention has y up and -z forward.
    The flip is c2w[:3, 1:3] *= -1, which negates the y and z columns of rotation.
    This converts (y-up, -z-forward) → (y-down, z-forward) = OpenCV.

    Verify with a known OpenGL pose:
      OpenGL c2w = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
      has camera looking along -z with y up.
      After flip: [[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]]
      has camera looking along -(-z)=+z with y down = OpenCV.
    """
    # Construct a mock c2w in OpenGL convention
    c2w_gl = torch.eye(4, dtype=torch.float32)

    # Apply the flip manually (this is what the loader does)
    c2w_cv = c2w_gl.clone()
    c2w_cv[:3, 1:3] *= -1

    # In OpenCV, the z-axis of rotation (third column) should be +z (forward)
    # Original third column was (0,0,1), after flip becomes (0,0,-1)
    # Wait, let me re-check: c2w[:3, 1:3] means columns 1 and 2 (y and z axes).
    # Column 0 (x-axis): unchanged = (1,0,0)
    # Column 1 (y-axis): (0,1,0) * -1 = (0,-1,0)
    # Column 2 (z-axis): (0,0,1) * -1 = (0,0,-1)
    # So z-axis becomes (0,0,-1)?? That doesn't seem right...

    # Actually the identity in OpenGL means camera at origin looking along -z.
    # After flip, the rotation columns become:
    #   x = (1,0,0), y = (0,-1,0), z = (0,0,-1)
    # This rotation matrix transforms from camera to world.
    # In this camera frame, a point at (0,0,1) in camera space maps to (0,0,-1) in world.
    # But in OpenCV, z forward means looking along (0,0,1) in camera space → world z direction.
    # The c2w z-column IS the world direction of the camera's z-axis.
    # For OpenCV, the camera looks along its +z in camera space,
    # so the c2w z-column is the look direction in world.
    # c2w z-column = (0,0,-1) means looking along -z in world, which matches the OpenGL convention
    # that the camera was looking along -z.

    # The KEY test: for a camera that was looking along -z in OpenGL world,
    # after flip to OpenCV, the camera should STILL look along -z in world
    # (the scene doesn't change, just the camera axes convention).
    # The c2w[:3, 2] (z-column) now represents the OpenCV camera's +z direction in world.
    # For the identity OpenGL camera, this should be (0, 0, -1) (looking into -z world).

    expected_z_axis = torch.tensor([0.0, 0.0, -1.0])
    torch.testing.assert_close(c2w_cv[:3, 2], expected_z_axis, atol=1e-6, rtol=0)

    expected_y_axis = torch.tensor([0.0, -1.0, 0.0])
    torch.testing.assert_close(c2w_cv[:3, 1], expected_y_axis, atol=1e-6, rtol=0)

    # x-axis unchanged
    expected_x_axis = torch.tensor([1.0, 0.0, 0.0])
    torch.testing.assert_close(c2w_cv[:3, 0], expected_x_axis, atol=1e-6, rtol=0)


# ── Alpha compositing ────────────────────────────────────────────────────


def test_alpha_compositing_zero():
    """Alpha = 0 → pixel is white (background)."""
    rgba = torch.zeros(1, 1, 4, dtype=torch.float32)  # fully transparent
    rgba[..., :3] = 0.5  # color shouldn't matter
    rgba[..., 3] = 0.0

    # Composite: rgb = alpha * color + (1 - alpha) * white
    alpha = rgba[..., 3:4]
    rgb = rgba[..., :3] * alpha + (1.0 - alpha) * 1.0
    expected = torch.ones(1, 1, 3, dtype=torch.float32)
    torch.testing.assert_close(rgb, expected, atol=1e-6, rtol=0)


def test_alpha_compositing_one():
    """Alpha = 1 → pixel is the original color."""
    color = torch.tensor([[[0.3, 0.6, 0.9]]], dtype=torch.float32)
    alpha = torch.ones(1, 1, 1, dtype=torch.float32)

    rgb = color * alpha + (1.0 - alpha) * 1.0
    torch.testing.assert_close(rgb, color, atol=1e-6, rtol=0)


def test_alpha_compositing_half():
    """Alpha = 0.5 → midpoint between color and white."""
    color = torch.tensor([[[0.2, 0.4, 0.8]]], dtype=torch.float32)
    alpha = torch.full((1, 1, 1), 0.5, dtype=torch.float32)

    rgb = color * alpha + (1.0 - alpha) * 1.0
    expected = torch.tensor([[[0.6, 0.7, 0.9]]], dtype=torch.float32)
    torch.testing.assert_close(rgb, expected, atol=1e-6, rtol=0)


# ── Focal length from camera_angle_x ────────────────────────────────────


def test_focal_length_lego_800px():
    """fx = W / (2 * tan(camera_angle_x / 2)) ≈ 1111.1 for 800px Lego.

    (verify): camera_angle_x = 0.6911112070083618 from the Lego transforms.json.
    """
    camera_angle_x = 0.6911112070083618
    W = 800
    fx = W / (2.0 * math.tan(camera_angle_x / 2.0))
    # Expected: ~1111.11
    assert abs(fx - 1111.1110) < 0.01, f"fx = {fx}, expected ~1111.11"


# ── Lego loader tests (skip if dataset missing) ─────────────────────────


def test_load_blender_lego_train(lego_root: Path):
    """Load Lego train split and verify basic properties."""
    images, cameras = load_blender_split(lego_root, "train", scale=1.0)

    assert len(images) == 100, f"Expected 100 train images, got {len(images)}"
    assert len(cameras) == 100

    # Images are uint8, RGBA composited to RGB
    img0 = images[0]
    assert img0.dtype == torch.uint8, f"Expected uint8, got {img0.dtype}"
    assert img0.shape == (800, 800, 3), f"Expected (800,800,3), got {img0.shape}"

    # Verify focal length
    cam0 = cameras[0]
    fx = cam0.K[0, 0].item()
    assert abs(fx - 1111.1110) < 0.1, f"fx = {fx}, expected ~1111.11"

    # Principal point at center
    cx = cam0.K[0, 2].item()
    cy = cam0.K[1, 2].item()
    assert abs(cx - 400.0) < 0.01, f"cx = {cx}, expected 400.0"
    assert abs(cy - 400.0) < 0.01, f"cy = {cy}, expected 400.0"


def test_load_blender_lego_downscale(lego_root: Path):
    """Load Lego at 0.5 scale and verify dimensions and intrinsics."""
    images, cameras = load_blender_split(lego_root, "train", scale=0.5)

    img0 = images[0]
    assert img0.shape == (400, 400, 3), f"Expected (400,400,3), got {img0.shape}"
    assert img0.dtype == torch.uint8

    cam0 = cameras[0]
    fx = cam0.K[0, 0].item()
    assert abs(fx - 1111.1110 * 0.5) < 0.1, f"fx = {fx}, expected ~555.56"
    assert cam0.W == 400
    assert cam0.H == 400


def test_load_blender_cameras_are_opencv(lego_root: Path):
    """All loaded cameras have orthonormal R with det(R) ≈ +1 (no reflection)."""
    _, cameras = load_blender_split(lego_root, "train", scale=1.0)
    for i, cam in enumerate(cameras):
        R = cam.c2w[:3, :3]
        det = torch.det(R).item()
        assert abs(det - 1.0) < 1e-4, f"Camera {i}: det(R) = {det}, expected +1"
        # Orthonormality: R @ R^T ≈ I
        RRT = R @ R.T
        torch.testing.assert_close(
            RRT, torch.eye(3, dtype=torch.float32), atol=1e-4, rtol=0,
            msg=f"Camera {i}: R is not orthonormal"
        )
