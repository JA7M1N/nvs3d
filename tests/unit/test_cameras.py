"""Unit tests for core/cameras.py: Camera, project, unproject, pixel_grid, resize_intrinsics."""

import math
import torch
import pytest

from nvs3d.core.cameras import Camera, project, unproject, pixel_grid, resize_intrinsics

SEED = 42


def _make_identity_camera(W: int = 640, H: int = 480, fx: float = 500.0, fy: float = 500.0,
                           cx: float | None = None, cy: float | None = None) -> Camera:
    """Create a camera at the origin looking along +z (OpenCV)."""
    if cx is None:
        cx = W / 2.0
    if cy is None:
        cy = H / 2.0
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32)
    c2w = torch.eye(4, dtype=torch.float32)
    return Camera(K=K, c2w=c2w, W=W, H=H)


def _make_offcenter_nonsquare_camera() -> Camera:
    """Non-square (W=640, H=480) camera with off-center principal point."""
    W, H = 640, 480
    fx, fy = 550.0, 510.0
    cx, cy = 330.0, 250.0  # off-center
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32)
    # Camera rotated 30° around y-axis, translated
    angle = math.radians(30)
    c, s = math.cos(angle), math.sin(angle)
    c2w = torch.tensor([
        [c, 0, s, 1.0],
        [0, 1, 0, 2.0],
        [-s, 0, c, 3.0],
        [0, 0, 0, 1.0],
    ], dtype=torch.float32)
    return Camera(K=K, c2w=c2w, W=W, H=H)


# ── w2c is inverse of c2w ────────────────────────────────────────────────


def test_w2c_is_inverse_of_c2w():
    """Camera.w2c @ Camera.c2w ≈ I."""
    cam = _make_offcenter_nonsquare_camera()
    I_approx = cam.w2c @ cam.c2w
    torch.testing.assert_close(I_approx, torch.eye(4, dtype=torch.float32), atol=1e-6, rtol=0)


# ── project / unproject round trip ───────────────────────────────────────


def test_project_unproject_roundtrip_identity():
    """project(unproject(p, d)) = p for identity camera."""
    cam = _make_identity_camera()
    pixels = torch.tensor([[320.5, 240.5], [0.5, 0.5], [639.5, 479.5]], dtype=torch.float32)
    depths = torch.tensor([5.0, 3.0, 10.0], dtype=torch.float32)

    pts_world = unproject(pixels, depths, cam)
    pixels_back = project(pts_world, cam)
    torch.testing.assert_close(pixels_back, pixels, atol=1e-4, rtol=0)


def test_project_unproject_roundtrip_nonsquare_offcenter():
    """project(unproject(p, d)) = p for non-square, off-center camera."""
    cam = _make_offcenter_nonsquare_camera()
    pixels = torch.tensor([
        [330.0, 250.0],   # principal point
        [0.5, 0.5],       # corner
        [639.5, 479.5],   # opposite corner
        [100.0, 400.0],   # arbitrary
    ], dtype=torch.float32)
    depths = torch.tensor([5.0, 3.0, 10.0, 7.0], dtype=torch.float32)

    pts_world = unproject(pixels, depths, cam)
    pixels_back = project(pts_world, cam)
    torch.testing.assert_close(pixels_back, pixels, atol=1e-4, rtol=0)


# ── Known cube corners ──────────────────────────────────────────────────


def test_project_known_point():
    """A known 3D point projects to the expected pixel."""
    cam = _make_identity_camera(W=640, H=480, fx=500.0, fy=500.0)
    # Point at (1, 2, 10) in world = camera space (identity c2w)
    # u = fx * X/Z + cx = 500 * 1/10 + 320 = 370
    # v = fy * Y/Z + cy = 500 * 2/10 + 240 = 340
    pt = torch.tensor([[1.0, 2.0, 10.0]], dtype=torch.float32)
    pix = project(pt, cam)
    expected = torch.tensor([[370.0, 340.0]], dtype=torch.float32)
    torch.testing.assert_close(pix, expected, atol=1e-4, rtol=0)


# ── Pixel grid ───────────────────────────────────────────────────────────


def test_pixel_grid_half_pixel_offset():
    """Pixel grid has +0.5 center: corner pixel is (0.5, 0.5)."""
    grid = pixel_grid(4, 3)  # W=4, H=3
    assert grid.shape == (3, 4, 2)
    # Top-left pixel center
    torch.testing.assert_close(grid[0, 0], torch.tensor([0.5, 0.5], dtype=torch.float32), atol=1e-6, rtol=0)
    # Bottom-right pixel center
    torch.testing.assert_close(grid[2, 3], torch.tensor([3.5, 2.5], dtype=torch.float32), atol=1e-6, rtol=0)


# ── resize_intrinsics ────────────────────────────────────────────────────


def test_resize_intrinsics_plain_scale():
    """With +0.5 pixel centers, resize is a plain scale with no shift."""
    K = torch.tensor([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1]], dtype=torch.float32)
    K_half = resize_intrinsics(K, 0.5)
    expected = torch.tensor([[250.0, 0, 160.0], [0, 250.0, 120.0], [0, 0, 1]], dtype=torch.float32)
    torch.testing.assert_close(K_half, expected, atol=1e-6, rtol=0)


def test_resize_intrinsics_projects_same_point():
    """Projecting a 3D point at full and half resolution gives consistent pixel coords.

    With pixel centers at i+0.5, scaling the intrinsics by k means:
    u_full = fx * X/Z + cx, and u_half = (fx*k) * X/Z + (cx*k) = k * u_full.
    So the half-res pixel should be exactly k * full-res pixel.
    """
    cam_full = _make_identity_camera(W=640, H=480, fx=500.0, fy=500.0)
    K_half = resize_intrinsics(cam_full.K, 0.5)
    cam_half = Camera(K=K_half, c2w=cam_full.c2w, W=320, H=240)

    pt = torch.tensor([[2.0, -1.0, 8.0]], dtype=torch.float32)
    pix_full = project(pt, cam_full)
    pix_half = project(pt, cam_half)

    torch.testing.assert_close(pix_half, pix_full * 0.5, atol=1e-4, rtol=0)


# ── OpenCV axis signs ────────────────────────────────────────────────────


def test_opencv_axes_center_pixel():
    """Center pixel direction in camera space is (0, 0, 1)."""
    cam = _make_identity_camera(W=640, H=480, fx=500.0, fy=500.0)
    # Center pixel: (cx, cy) = (320, 240)
    # Camera-space ray: ((cx-cx)/fx, (cy-cy)/fy, 1) = (0, 0, 1)
    center_px = torch.tensor([[320.0, 240.0]], dtype=torch.float32)
    depth = torch.tensor([1.0], dtype=torch.float32)
    pt_world = unproject(center_px, depth, cam)
    # With identity c2w, world = camera space, so pt should be (0, 0, 1)
    torch.testing.assert_close(pt_world, torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32), atol=1e-6, rtol=0)


def test_opencv_axes_right_is_plus_x():
    """One pixel to the right of center → +x in camera space."""
    cam = _make_identity_camera(W=640, H=480, fx=500.0, fy=500.0)
    right_px = torch.tensor([[321.0, 240.0]], dtype=torch.float32)  # one pixel right
    depth = torch.tensor([1.0], dtype=torch.float32)
    pt = unproject(right_px, depth, cam)
    # x should be positive, y ≈ 0, z ≈ 1
    assert pt[0, 0].item() > 0, f"Expected +x, got {pt[0, 0].item()}"
    assert abs(pt[0, 1].item()) < 1e-5, f"Expected y≈0, got {pt[0, 1].item()}"


def test_opencv_axes_down_is_plus_y():
    """One pixel down from center → +y in camera space."""
    cam = _make_identity_camera(W=640, H=480, fx=500.0, fy=500.0)
    down_px = torch.tensor([[320.0, 241.0]], dtype=torch.float32)  # one pixel down
    depth = torch.tensor([1.0], dtype=torch.float32)
    pt = unproject(down_px, depth, cam)
    # y should be positive, x ≈ 0, z ≈ 1
    assert pt[0, 1].item() > 0, f"Expected +y, got {pt[0, 1].item()}"
    assert abs(pt[0, 0].item()) < 1e-5, f"Expected x≈0, got {pt[0, 0].item()}"
