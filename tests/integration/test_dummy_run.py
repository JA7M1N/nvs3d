"""Integration test for dummy training run."""

import json
from pathlib import Path
import pytest

from nvs3d.utils.config import BaseConfig
from nvs3d.train.dummy_runner import run_dummy


def test_dummy_training_run(tmp_path: Path):
    """Verify that a dummy run creates a complete run directory with config.yaml and metrics.jsonl."""
    runs_dir = tmp_path / "runs"
    cfg = BaseConfig(exp_name="dummy_test", seed=42, iters=5)

    run_dir = run_dummy(config=cfg, base_dir=runs_dir)

    assert run_dir.exists()
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "metrics.jsonl").exists()
    assert (run_dir / "ckpt").is_dir()
    assert (run_dir / "report.md").exists()

    metrics_file = run_dir / "metrics.jsonl"
    lines = [json.loads(line) for line in metrics_file.read_text(encoding="utf-8").strip().splitlines()]
    assert len(lines) == 5
    for i, entry in enumerate(lines):
        assert entry["step"] == i + 1
        assert "loss" in entry
        assert "psnr" in entry

    # Verify a dummy checkpoint was written
    ckpts = list((run_dir / "ckpt").glob("*.pt"))
    assert len(ckpts) > 0
