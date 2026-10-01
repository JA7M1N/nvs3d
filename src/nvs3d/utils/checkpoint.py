"""Checkpoint serialization, loading, and state management conforming to blueprint Part 6."""

from pathlib import Path
import random
import subprocess
from typing import Any
import numpy as np
import torch


def get_git_hash() -> str:
    """Retrieve current git commit hash, falling back to 'unknown' if git fails or is unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def capture_rng_state() -> dict[str, Any]:
    """Capture current RNG states across Python, NumPy, and PyTorch."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(rng_state: dict[str, Any]) -> None:
    """Restore RNG states across Python, NumPy, and PyTorch."""
    if "python" in rng_state and rng_state["python"] is not None:
        random.setstate(rng_state["python"])
    if "numpy" in rng_state and rng_state["numpy"] is not None:
        np.random.set_state(rng_state["numpy"])
    if "torch_cpu" in rng_state and rng_state["torch_cpu"] is not None:
        torch.set_rng_state(rng_state["torch_cpu"])
    if "torch_cuda" in rng_state and rng_state["torch_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng_state["torch_cuda"])


def save_checkpoint(
    path: Path | str,
    kind: str,
    config_yaml: str,
    step: int,
    state_dict: dict[str, Any],
    optim_state: dict[str, Any],
    scene_meta: dict[str, Any] | None = None,
    version: int = 1,
    git_hash: str | None = None,
) -> None:
    """Save training checkpoint adhering to blueprint Part 6 contract."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if git_hash is None:
        git_hash = get_git_hash()

    checkpoint_data = {
        "version": version,
        "kind": kind,
        "config_yaml": config_yaml,
        "step": step,
        "state_dict": state_dict,
        "optim_state": optim_state,
        "rng_state": capture_rng_state(),
        "scene_meta": scene_meta or {},
        "git_hash": git_hash,
    }

    torch.save(checkpoint_data, path)


def load_checkpoint(
    path: Path | str,
    expected_kind: str | None = None,
    expected_version: int | None = 1,
    restore_rng: bool = False,
    map_location: str | torch.device = "cpu",
    weights_only: bool = False,
) -> dict[str, Any]:
    """Load and validate training checkpoint."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {path}")

    ckpt = torch.load(path, map_location=map_location, weights_only=weights_only)

    actual_version = ckpt.get("version")
    if expected_version is not None and actual_version != expected_version:
        raise ValueError(
            f"Unsupported checkpoint version {actual_version}, expected {expected_version}"
        )

    actual_kind = ckpt.get("kind")
    if expected_kind is not None and actual_kind != expected_kind:
        raise ValueError(
            f"Expected checkpoint kind '{expected_kind}', got '{actual_kind}'"
        )

    if restore_rng and "rng_state" in ckpt:
        restore_rng_state(ckpt["rng_state"])

    return ckpt
