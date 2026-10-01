"""Dummy training runner for Phase P0 scaffolding verification."""

from pathlib import Path
import torch

from nvs3d.utils.config import BaseConfig
from nvs3d.utils.logging import RunLogger
from nvs3d.utils.seed import seed_everything
from nvs3d.utils.checkpoint import save_checkpoint
from nvs3d.utils.profiling import get_vram_usage_mb


def run_dummy(config: BaseConfig | None = None, base_dir: Path | str = "runs") -> Path:
    """Execute a dummy training run to verify logging, directories, and checkpoint IO.

    Args:
        config: Optional configuration instance.
        base_dir: Base directory where runs are placed.

    Returns:
        The created run directory Path.
    """
    if config is None:
        config = BaseConfig(exp_name="dummy_run", seed=42, iters=5)

    seed_everything(config.seed)
    logger = RunLogger(base_dir=base_dir, config=config)
    run_dir = logger.run_dir

    # Dummy training loop
    for step in range(1, config.iters + 1):
        dummy_loss = 1.0 / step
        dummy_psnr = 20.0 + step * 0.5
        vram = get_vram_usage_mb()
        logger.log_metrics({
            "step": step,
            "loss": round(dummy_loss, 4),
            "psnr": round(dummy_psnr, 2),
            "vram_mb": vram,
        })

    # Save a dummy checkpoint
    ckpt_path = run_dir / "ckpt" / "ckpt_dummy.pt"
    save_checkpoint(
        path=ckpt_path,
        kind="dummy",
        config_yaml="dummy: true",
        step=config.iters,
        state_dict={"dummy_param": torch.tensor([1.0, 2.0])},
        optim_state={"step": config.iters},
        scene_meta={"extent": 1.0},
    )

    logger.update_report(f"Completed {config.iters} dummy steps successfully.")
    logger.close()
    return run_dir
