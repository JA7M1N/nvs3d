"""Structured experiment run logging utilities."""

from datetime import datetime
import json
from pathlib import Path
from typing import Any

from nvs3d.utils.config import BaseConfig, save_config_yaml


class RunLogger:
    """Manages the run directory and structured metrics logging according to blueprint Part 6.

    Layout:
        runs/<exp_name>/<timestamp>/
            ├── config.yaml
            ├── metrics.jsonl
            ├── tb/
            ├── renders/
            ├── ckpt/
            └── report.md
    """

    def __init__(self, base_dir: Path | str, config: BaseConfig) -> None:
        self.base_dir = Path(base_dir)
        self.config = config

        # Windows-safe timestamp: YYYYMMDD-HHMMSS (no colons)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_dir = self.base_dir / config.exp_name / timestamp
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.run_dir / "tb").mkdir(exist_ok=True)
        (self.run_dir / "renders").mkdir(exist_ok=True)
        (self.run_dir / "ckpt").mkdir(exist_ok=True)

        # Write resolved config.yaml
        save_config_yaml(self.config, self.run_dir / "config.yaml")

        # Initialize metrics.jsonl and report.md
        self.metrics_path = self.run_dir / "metrics.jsonl"
        if not self.metrics_path.exists():
            self.metrics_path.touch()

        self.report_path = self.run_dir / "report.md"
        if not self.report_path.exists():
            self.report_path.write_text(
                f"# Run Report: {config.exp_name}\n\nStarted: {timestamp}\n",
                encoding="utf-8",
            )

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        """Append a single line of scalar metrics to metrics.jsonl."""
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + "\n")

    def update_report(self, text: str) -> None:
        """Append summary information to report.md."""
        with open(self.report_path, "a", encoding="utf-8") as f:
            f.write(text + "\n")

    def close(self) -> None:
        """Clean up any active file handlers or writers."""
        pass
