"""Pytest configuration and shared fixtures for nvs3d tests."""

from pathlib import Path
import pytest


@pytest.fixture
def tmp_run_dir(tmp_path: Path) -> Path:
    """Provides a clean temporary runs directory."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    return runs_dir


@pytest.fixture
def lego_root() -> Path:
    """Path to the Lego NeRF synthetic dataset. Skips if missing."""
    root = Path("datasets/nerf_synthetic/lego")
    if not root.exists():
        pytest.skip("Lego dataset not found at datasets/nerf_synthetic/lego")
    return root
