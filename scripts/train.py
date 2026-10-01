"""Entry point for training (supports --dummy for Phase P0 scaffolding verification)."""

import argparse
from pathlib import Path
import sys

from nvs3d.utils.config import BaseConfig, load_config_yaml, parse_cli_overrides
from nvs3d.train.dummy_runner import run_dummy


def main() -> None:
    parser = argparse.ArgumentParser(description="nvs3d training entry point")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--dummy", action="store_true", help="Execute dummy run for Phase P0 verification")
    parser.add_argument("--exp_name", type=str, default="dummy_run", help="Experiment name")
    parser.add_argument("--iters", type=int, default=5, help="Number of iterations")
    parser.add_argument("--output_dir", type=str, default="runs", help="Base directory for runs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args, unknown = parser.parse_known_args()

    if args.config:
        cfg = load_config_yaml(args.config, config_cls=BaseConfig)
    else:
        cfg = BaseConfig(
            exp_name=args.exp_name,
            seed=args.seed,
            iters=args.iters,
            output_dir=args.output_dir,
        )

    if unknown:
        cfg = parse_cli_overrides(cfg, unknown)

    if args.dummy:
        print(f"[P0 Scaffolding] Running dummy training (seed={cfg.seed}, iters={cfg.iters})...")
        run_dir = run_dummy(config=cfg, base_dir=cfg.output_dir)
        print(f"[P0 Scaffolding] Dummy run complete. Output written to: {run_dir}")
        print(f"  - Config: {run_dir / 'config.yaml'}")
        print(f"  - Metrics: {run_dir / 'metrics.jsonl'}")
        print(f"  - Checkpoint: {run_dir / 'ckpt' / 'ckpt_dummy.pt'}")
    else:
        print("Training pipeline placeholder. Use --dummy for Phase P0 verification.")
        print("NeRF training begins in P3; 3DGS training begins in P10.")


if __name__ == "__main__":
    main()
