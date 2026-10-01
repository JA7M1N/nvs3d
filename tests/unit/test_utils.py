"""Unit tests for nvs3d.utils (seed, config, logging, checkpoint, profiling)."""

import json
from pathlib import Path
import random
import numpy as np
import pytest
import torch

from nvs3d.utils.seed import seed_everything
from nvs3d.utils.config import BaseConfig, load_config_yaml, save_config_yaml, parse_cli_overrides
from nvs3d.utils.logging import RunLogger
from nvs3d.utils.checkpoint import save_checkpoint, load_checkpoint, get_git_hash
from nvs3d.utils.profiling import Timer, get_vram_usage_mb


def test_seed_everything():
    """Verify seed_everything provides deterministic CPU sequences across random, numpy, and torch."""
    seed_everything(42)
    py_val1 = random.random()
    np_val1 = np.random.rand(3)
    torch_val1 = torch.rand(3)

    # Re-seed with identical seed
    seed_everything(42)
    py_val2 = random.random()
    np_val2 = np.random.rand(3)
    torch_val2 = torch.rand(3)

    assert py_val1 == py_val2
    np.testing.assert_array_equal(np_val1, np_val2)
    assert torch.equal(torch_val1, torch_val2)

    # Seed with different seed
    seed_everything(43)
    py_val3 = random.random()
    np_val3 = np.random.rand(3)
    torch_val3 = torch.rand(3)

    assert py_val1 != py_val3
    assert not np.array_equal(np_val1, np_val3)
    assert not torch.equal(torch_val1, torch_val3)


def test_config_roundtrip(tmp_path: Path):
    """Verify dataclass config serialization and deserialization to/from YAML."""
    cfg = BaseConfig(exp_name="test_exp", seed=123, iters=500)
    cfg_file = tmp_path / "config.yaml"
    save_config_yaml(cfg, cfg_file)

    assert cfg_file.exists()
    loaded_cfg = load_config_yaml(cfg_file, config_cls=BaseConfig)
    assert loaded_cfg.exp_name == "test_exp"
    assert loaded_cfg.seed == 123
    assert loaded_cfg.iters == 500


def test_config_cli_overrides():
    """Verify command-line key=value or flag overrides update config fields."""
    cfg = BaseConfig(exp_name="default", seed=42, iters=100)
    overrides = ["--exp_name", "overridden_exp", "--seed", "999", "--iters", "250"]
    updated_cfg = parse_cli_overrides(cfg, overrides)

    assert updated_cfg.exp_name == "overridden_exp"
    assert updated_cfg.seed == 999
    assert updated_cfg.iters == 250


def test_logging_and_run_dir(tmp_path: Path):
    """Verify run directory creation, Windows-safe naming, and metrics.jsonl logging."""
    runs_dir = tmp_path / "runs"
    cfg = BaseConfig(exp_name="smoke_test", seed=42, iters=10)
    logger = RunLogger(base_dir=runs_dir, config=cfg)

    run_dir = logger.run_dir
    assert run_dir.exists()
    # Check Windows safety: no colons in path name
    assert ":" not in run_dir.name

    # Check required blueprint Part 6 subdirectories / files
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "tb").is_dir()
    assert (run_dir / "renders").is_dir()
    assert (run_dir / "ckpt").is_dir()
    assert (run_dir / "report.md").exists()
    assert (run_dir / "metrics.jsonl").exists()

    # Log metrics
    logger.log_metrics({"step": 1, "loss": 0.5, "psnr": 20.0})
    logger.log_metrics({"step": 2, "loss": 0.3, "psnr": 22.5})
    logger.close()

    # Read back metrics.jsonl
    metrics_file = run_dir / "metrics.jsonl"
    lines = [json.loads(line) for line in metrics_file.read_text(encoding="utf-8").strip().splitlines()]
    assert len(lines) == 2
    assert lines[0]["step"] == 1
    assert lines[0]["loss"] == 0.5
    assert lines[1]["step"] == 2
    assert lines[1]["psnr"] == 22.5


def test_checkpoint_roundtrip_and_rng_state(tmp_path: Path):
    """Verify saving and loading checkpoints preserves tensors, rng state, and validates version/kind."""
    seed_everything(100)
    # Burn a few random numbers
    random.random()
    np.random.rand()
    torch.rand(1)

    ckpt_path = tmp_path / "checkpoint.pt"
    dummy_model_state = {"weight": torch.tensor([1.0, 2.0, 3.0])}
    dummy_optim_state = {"param_groups": []}
    scene_meta = {"extent": 1.5, "norm": None, "sh_degree": 0, "K": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}

    save_checkpoint(
        path=ckpt_path,
        kind="nerf",
        config_yaml="exp_name: test",
        step=50,
        state_dict=dummy_model_state,
        optim_state=dummy_optim_state,
        scene_meta=scene_meta,
        version=1,
    )

    # Generate reference values without restoring state
    expected_py = random.random()
    expected_np = np.random.rand()
    expected_torch = torch.rand(1)

    # Mutate RNG state
    seed_everything(999)
    assert random.random() != expected_py

    # Load checkpoint and restore state
    loaded = load_checkpoint(ckpt_path, expected_kind="nerf", expected_version=1, restore_rng=True)
    assert loaded["step"] == 50
    assert torch.equal(loaded["state_dict"]["weight"], dummy_model_state["weight"])
    assert loaded["scene_meta"]["extent"] == 1.5

    # Check restored RNG matches expected subsequent values
    restored_py = random.random()
    restored_np = np.random.rand()
    restored_torch = torch.rand(1)

    assert restored_py == expected_py
    assert restored_np == expected_np
    assert torch.equal(restored_torch, expected_torch)


def test_checkpoint_validation_errors(tmp_path: Path):
    """Verify loading fails on version or kind mismatch."""
    ckpt_path = tmp_path / "bad_ckpt.pt"
    save_checkpoint(
        path=ckpt_path,
        kind="nerf",
        config_yaml="exp_name: test",
        step=1,
        state_dict={},
        optim_state={},
        scene_meta={},
        version=1,
    )

    with pytest.raises(ValueError, match="Expected checkpoint kind 'gs'"):
        load_checkpoint(ckpt_path, expected_kind="gs")

    with pytest.raises(ValueError, match="Unsupported checkpoint version"):
        load_checkpoint(ckpt_path, expected_kind="nerf", expected_version=2)


def test_git_hash_fallback():
    """Verify get_git_hash returns a string and handles fallback gracefully."""
    gh = get_git_hash()
    assert isinstance(gh, str)
    assert len(gh) > 0


def test_profiling_cpu():
    """Verify Timer and VRAM profiler run on CPU without throwing errors."""
    with Timer("cpu_op") as t:
        _ = torch.zeros(100, 100).sum()
    assert t.elapsed_seconds >= 0.0

    vram = get_vram_usage_mb()
    assert isinstance(vram, float)
    assert vram >= 0.0
