"""Pytest configuration and shared fixtures for nvs3d tests."""

from pathlib import Path
import pytest


@pytest.fixture
def tmp_run_dir(tmp_path: Path) -> Path:
    """Provides a clean temporary runs directory."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    return runs_dir
