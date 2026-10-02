"""Unit tests for core/rays.py: ray generation, unit norm, optical axis, known-point hit."""

import math
import torch
import pytest

from nvs3d.core.cameras import Camera, pixel_grid
from nvs3d.core.rays import generate_rays, generate_rays_for_pixels

SEED = 42


def _make_camera(W=640, H=480, fx=500.0, fy=500.0, cx=None, cy=None, c2w=None) -> Camera:
    if cx is None:
        cx = W / 2.0
    if cy is None:
        cy = H / 2.0
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32)
    if c2w is None:
        c2w = torch.eye(4, dtype=torch.float32)
    return Camera(K=K, c2w=c2w, W=W, H=H)


def _make_offcenter_nonsquare_camera() -> Camera:
    """Non-square (W=640, H=480) camera with off-center principal point and non-trivial pose."""
    W, H = 640, 480
    fx, fy = 550.0, 510.0
    cx, cy = 330.0, 250.0  # off-center
    angle = math.radians(30)
    c, s = math.cos(angle), math.sin(angle)
    c2w = torch.tensor([
        [c, 0, s, 1.0],
        [0, 1, 0, 2.0],
        [-s, 0, c, 3.0],
        [0, 0, 0, 1.0],
    ], dtype=torch.float32)
    return _make_camera(W, H, fx, fy, cx, cy, c2w)


# ── Center pixel = optical axis ──────────────────────────────────────────


def test_center_pixel_is_optical_axis():
    """Ray through the principal point = the camera's optical axis (z direction in world).

    For the identity camera with cx=320, cy=240, the principal point pixel
    is (320, 240). The camera-space direction at this pixel is (0, 0, 1).
    """
    cam = _make_camera()
    # Use the principal point pixel, not H//2, W//2 index
    pp = torch.tensor([[cam.K[0, 2].item(), cam.K[1, 2].item()]], dtype=torch.float32)
    _, dirs = generate_rays_for_pixels(cam, pp)
    d_center = dirs[0]

    # For identity c2w, optical axis is (0, 0, 1)
    expected = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
    torch.testing.assert_close(d_center, expected, atol=1e-5, rtol=0)


def test_center_pixel_optical_axis_nonsquare():
    """Ray through center pixel = optical axis for non-square off-center camera."""
    cam = _make_offcenter_nonsquare_camera()
    # The principal point pixel (cx, cy) should give direction = c2w[:3, 2]
    pp = torch.tensor([[cam.K[0, 2].item(), cam.K[1, 2].item()]], dtype=torch.float32)
    origins, dirs = generate_rays_for_pixels(cam, pp)

    optical_axis = cam.c2w[:3, 2]  # z-column of rotation
    optical_axis = optical_axis / optical_axis.norm()

    torch.testing.assert_close(dirs[0], optical_axis, atol=1e-5, rtol=0)


# ── All rays unit norm ───────────────────────────────────────────────────


def test_all_rays_unit_norm():
    """All generated rays have unit norm."""
    cam = _make_camera(W=64, H=48)  # small for speed
    _, dirs = generate_rays(cam)
    norms = dirs.norm(dim=-1)
    torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=0)


def test_all_rays_unit_norm_nonsquare():
    """All generated rays unit norm for non-square off-center camera."""
    cam = _make_offcenter_nonsquare_camera()
    pixels = torch.tensor([
        [0.5, 0.5], [639.5, 0.5], [0.5, 479.5], [639.5, 479.5],
        [330.0, 250.0], [100.0, 400.0],
    ], dtype=torch.float32)
    _, dirs = generate_rays_for_pixels(cam, pixels)
    norms = dirs.norm(dim=-1)
    torch.testing.assert_close(norms, torch.ones(len(pixels)), atol=1e-5, rtol=0)


# ── Ray hits known 3D point ──────────────────────────────────────────────


def test_ray_hits_known_point():
    """Ray through a known 3D point's pixel has closest-approach distance < 1e-5."""
    cam = _make_camera()
    target = torch.tensor([2.0, -1.0, 8.0], dtype=torch.float32)

    # Project target to get its pixel
    w2c = cam.w2c
    pt_cam = (w2c[:3, :3] @ target) + w2c[:3, 3]
    u = cam.K[0, 0] * pt_cam[0] / pt_cam[2] + cam.K[0, 2]
    v = cam.K[1, 1] * pt_cam[1] / pt_cam[2] + cam.K[1, 2]
    pixel = torch.tensor([[u.item(), v.item()]], dtype=torch.float32)

    origins, dirs = generate_rays_for_pixels(cam, pixel)
    o = origins[0]
    d = dirs[0]

    # Closest point on ray to target: t = dot(target - o, d)
    t_val = torch.dot(target - o, d)
    closest = o + t_val * d
    dist = (closest - target).norm().item()

    assert dist < 1e-5, f"Ray closest approach to target = {dist}, expected < 1e-5"


def test_ray_hits_known_point_nonsquare():
    """Ray through a known 3D point's pixel hits it for non-square camera."""
    cam = _make_offcenter_nonsquare_camera()
    target = torch.tensor([3.0, 4.0, 12.0], dtype=torch.float32)

    w2c = cam.w2c
    pt_cam = (w2c[:3, :3] @ target) + w2c[:3, 3]
    u = cam.K[0, 0] * pt_cam[0] / pt_cam[2] + cam.K[0, 2]
    v = cam.K[1, 1] * pt_cam[1] / pt_cam[2] + cam.K[1, 2]
    pixel = torch.tensor([[u.item(), v.item()]], dtype=torch.float32)

    origins, dirs = generate_rays_for_pixels(cam, pixel)
    o, d = origins[0], dirs[0]
    t_val = torch.dot(target - o, d)
    closest = o + t_val * d
    dist = (closest - target).norm().item()

    assert dist < 1e-4, f"Ray closest approach to target = {dist}, expected < 1e-4"


# ── OpenCV axis directions in rays ───────────────────────────────────────


def test_ray_right_is_plus_x_in_camera():
    """Moving one pixel right increases the x-component of the camera-space direction."""
    cam = _make_camera()
    cx, cy = cam.K[0, 2].item(), cam.K[1, 2].item()
    center_px = torch.tensor([[cx, cy]], dtype=torch.float32)
    right_px = torch.tensor([[cx + 1.0, cy]], dtype=torch.float32)

    _, d_center = generate_rays_for_pixels(cam, center_px)
    _, d_right = generate_rays_for_pixels(cam, right_px)

    # For identity c2w, world = camera space. Right pixel should have larger x.
    assert d_right[0, 0].item() > d_center[0, 0].item(), "Right pixel should increase x"


def test_ray_down_is_plus_y_in_camera():
    """Moving one pixel down increases the y-component of the camera-space direction."""
    cam = _make_camera()
    cx, cy = cam.K[0, 2].item(), cam.K[1, 2].item()
    center_px = torch.tensor([[cx, cy]], dtype=torch.float32)
    down_px = torch.tensor([[cx, cy + 1.0]], dtype=torch.float32)

    _, d_center = generate_rays_for_pixels(cam, center_px)
    _, d_down = generate_rays_for_pixels(cam, down_px)

    assert d_down[0, 1].item() > d_center[0, 1].item(), "Down pixel should increase y"
