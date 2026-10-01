"""Tests for the Lego dataset verify logic in scripts/download_lego.py.

Uses tmp_path to build tiny fake NeRF-synthetic trees — these are ONLY for
testing the verifier function, never used as real training data.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# Import the verify function from the download script
import importlib.util
import sys


def _import_download_lego():
    """Import download_lego.py as a module (it lives in scripts/, not a package)."""
    scripts_dir = Path(__file__).resolve().parent.parent.parent / "scripts"
    spec = importlib.util.spec_from_file_location(
        "download_lego", scripts_dir / "download_lego.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


download_lego = _import_download_lego()
verify_lego_dataset = download_lego.verify_lego_dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rgba_png(path: Path, width: int = 800, height: int = 800) -> None:
    """Create a tiny valid RGBA PNG at *path*."""
    from PIL import Image

    img = Image.new("RGBA", (width, height), (255, 0, 0, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def _make_transforms_json(
    dest: Path,
    name: str,
    n_frames: int,
    split: str,
    *,
    camera_angle_x: float | None = 0.6911,
    create_images: bool = True,
    image_size: tuple[int, int] = (800, 800),
) -> None:
    """Write a transforms JSON file and optionally create matching PNGs."""
    frames = []
    for i in range(n_frames):
        file_path = f"./{split}/r_{i}"
        frames.append({
            "file_path": file_path,
            "rotation": 0.0,
            "transform_matrix": [[1, 0, 0, 0]] * 4,
        })
        if create_images:
            _make_rgba_png(dest / split / f"r_{i}.png", *image_size)

    data: dict = {"frames": frames}
    if camera_angle_x is not None:
        data["camera_angle_x"] = camera_angle_x

    dest.mkdir(parents=True, exist_ok=True)
    (dest / name).write_text(json.dumps(data), encoding="utf-8")


def _make_valid_lego(dest: Path, image_size: tuple[int, int] = (800, 800)) -> None:
    """Create a minimal but fully valid Lego-like dataset tree."""
    _make_transforms_json(dest, "transforms_train.json", 100, "train", image_size=image_size)
    _make_transforms_json(dest, "transforms_val.json", 100, "val", image_size=image_size)
    _make_transforms_json(dest, "transforms_test.json", 200, "test", image_size=image_size)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestVerifyAllPass:
    """A valid tree should produce all-PASS results (uses 8x8 images for speed)."""

    def test_all_pass(self, tmp_path: Path) -> None:
        lego = tmp_path / "lego"
        _make_valid_lego(lego, image_size=(8, 8))

        results = verify_lego_dataset(lego, expected_image_size=(8, 8))
        statuses = {r["check"]: r["status"] for r in results}
        for check, status in statuses.items():
            assert status == "PASS", f"{check} should PASS but got {status}"


class TestVerifyMissingJson:
    """Missing JSON files should produce FAIL entries."""

    def test_missing_train_json(self, tmp_path: Path) -> None:
        lego = tmp_path / "lego"
        _make_valid_lego(lego, image_size=(8, 8))
        (lego / "transforms_train.json").unlink()

        results = verify_lego_dataset(lego, expected_image_size=(8, 8))
        by_check = {r["check"]: r["status"] for r in results}
        assert by_check["transforms_train.json exists"] == "FAIL"

    def test_missing_all_jsons(self, tmp_path: Path) -> None:
        lego = tmp_path / "lego"
        lego.mkdir(parents=True)

        results = verify_lego_dataset(lego)
        fails = [r for r in results if r["status"] == "FAIL"]
        assert len(fails) >= 3  # at least the 3 JSON existence checks


class TestVerifyWrongCounts:
    """Wrong frame counts should FAIL."""

    def test_train_wrong_count(self, tmp_path: Path) -> None:
        lego = tmp_path / "lego"
        # Only 50 train frames instead of 100
        _make_transforms_json(lego, "transforms_train.json", 50, "train", image_size=(8, 8))
        _make_transforms_json(lego, "transforms_val.json", 100, "val", image_size=(8, 8))
        _make_transforms_json(lego, "transforms_test.json", 200, "test", image_size=(8, 8))

        results = verify_lego_dataset(lego, expected_image_size=(8, 8))
        by_check = {r["check"]: r["status"] for r in results}
        assert by_check["transforms_train.json frame count"] == "FAIL"
        assert by_check["transforms_val.json frame count"] == "PASS"
        assert by_check["transforms_test.json frame count"] == "PASS"


class TestVerifyMissingImages:
    """Missing image files should cause the PNG resolution check to FAIL."""

    def test_missing_images(self, tmp_path: Path) -> None:
        lego = tmp_path / "lego"
        # Create JSONs but don't create the images
        _make_transforms_json(lego, "transforms_train.json", 100, "train",
                              create_images=False)
        _make_transforms_json(lego, "transforms_val.json", 100, "val",
                              create_images=False)
        _make_transforms_json(lego, "transforms_test.json", 200, "test",
                              create_images=False)

        results = verify_lego_dataset(lego)
        by_check = {r["check"]: r["status"] for r in results}
        assert by_check["All file_path PNGs resolve"] == "FAIL"


class TestVerifyMissingCameraAngleX:
    """Missing camera_angle_x should FAIL."""

    def test_no_camera_angle_x(self, tmp_path: Path) -> None:
        lego = tmp_path / "lego"
        _make_transforms_json(lego, "transforms_train.json", 100, "train",
                              camera_angle_x=None, image_size=(8, 8))
        _make_transforms_json(lego, "transforms_val.json", 100, "val", image_size=(8, 8))
        _make_transforms_json(lego, "transforms_test.json", 200, "test", image_size=(8, 8))

        results = verify_lego_dataset(lego, expected_image_size=(8, 8))
        by_check = {r["check"]: r["status"] for r in results}
        assert by_check["camera_angle_x in all JSONs"] == "FAIL"


class TestVerifyWrongImageSize:
    """Images at wrong size should FAIL the sample image check."""

    def test_wrong_size(self, tmp_path: Path) -> None:
        lego = tmp_path / "lego"
        # Create 8x8 images but verify against the default 800x800
        _make_transforms_json(lego, "transforms_train.json", 100, "train",
                              image_size=(8, 8))
        _make_transforms_json(lego, "transforms_val.json", 100, "val", image_size=(8, 8))
        _make_transforms_json(lego, "transforms_test.json", 200, "test", image_size=(8, 8))

        results = verify_lego_dataset(lego)  # default expected_image_size=(800,800)
        by_check = {r["check"]: r["status"] for r in results}
        assert by_check["Sample image RGBA 800x800"] == "FAIL"


class TestVerifyConfigCLIOverrides:
    """Test that data_dir and output_root work as CLI overrides on BaseConfig."""

    def test_data_dir_override(self) -> None:
        from nvs3d.utils.config import BaseConfig, parse_cli_overrides

        cfg = BaseConfig()
        assert cfg.data_dir == ""
        updated = parse_cli_overrides(cfg, ["--data_dir", "/my/data"])
        assert updated.data_dir == "/my/data"

    def test_output_root_override(self) -> None:
        from nvs3d.utils.config import BaseConfig, parse_cli_overrides

        cfg = BaseConfig()
        assert cfg.output_root == ""
        updated = parse_cli_overrides(cfg, ["--output_root", "/my/output"])
        assert updated.output_root == "/my/output"

    def test_data_dir_from_yaml(self, tmp_path: Path) -> None:
        """YAML files with data_dir should load it into the field, not extra."""
        from nvs3d.utils.config import BaseConfig, save_config_yaml, load_config_yaml

        cfg = BaseConfig(data_dir="datasets/nerf_synthetic/lego")
        yaml_path = tmp_path / "test.yaml"
        save_config_yaml(cfg, yaml_path)
        loaded = load_config_yaml(yaml_path, config_cls=BaseConfig)
        assert loaded.data_dir == "datasets/nerf_synthetic/lego"

    def test_old_yaml_without_new_fields_loads(self, tmp_path: Path) -> None:
        """Old YAML files that don't have data_dir or output_root still load fine."""
        from nvs3d.utils.config import BaseConfig, load_config_yaml

        # Write a YAML like the original P0 base.yaml (no data_dir, no output_root)
        yaml_path = tmp_path / "old.yaml"
        yaml_path.write_text(
            "exp_name: old_exp\nseed: 42\ndevice: cpu\noutput_dir: runs\n",
            encoding="utf-8",
        )
        loaded = load_config_yaml(yaml_path, config_cls=BaseConfig)
        assert loaded.exp_name == "old_exp"
        assert loaded.data_dir == ""  # default
        assert loaded.output_root == ""  # default

    def test_yaml_with_extra_fields_still_works(self, tmp_path: Path) -> None:
        """YAML files with unknown fields still load into extra dict."""
        from nvs3d.utils.config import BaseConfig, load_config_yaml

        yaml_path = tmp_path / "with_extra.yaml"
        yaml_path.write_text(
            "exp_name: test\ndata_dir: datasets/lego\n"
            "white_bkgd: true\nnear: 2.0\n",
            encoding="utf-8",
        )
        loaded = load_config_yaml(yaml_path, config_cls=BaseConfig)
        assert loaded.data_dir == "datasets/lego"
        assert loaded.extra["white_bkgd"] is True
        assert loaded.extra["near"] == 2.0
