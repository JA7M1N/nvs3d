"""Pytest configuration and shared fixtures for nvs3d tests."""

from pathlib import Path
import pytest


@pytest.fixture
def tmp_run_dir(tmp_path: Path) -> Path:
    """Provides a clean temporary runs directory."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    return runs_dir


def _repo_root() -> Path:
    """Return the repository root (directory containing pyproject.toml)."""
    d = Path(__file__).resolve().parent
    while d != d.parent:
        if (d / "pyproject.toml").exists():
            return d
        d = d.parent
    raise RuntimeError("Could not locate repo root (no pyproject.toml found)")


@pytest.fixture
def lego_root() -> Path:
    """Return the path to the NeRF-synthetic Lego dataset.

    Skips the test with a clear message if the data has not been
    downloaded (i.e. ``datasets/nerf_synthetic/lego/transforms_train.json``
    does not exist).  Never synthesises substitute data.
    """
    root = _repo_root() / "datasets" / "nerf_synthetic" / "lego"
    if not (root / "transforms_train.json").exists():
        pytest.skip(
            f"Lego dataset not found at {root}. "
            "Run `python scripts/download_lego.py` to download it."
        )
    return root
