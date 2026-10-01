"""Random seed utilities for deterministic execution."""

import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic: bool = True) -> int:
    """Seed all random number generators across Python, NumPy, and PyTorch.

    Args:
        seed: The integer seed to set.
        deterministic: Whether to configure PyTorch/cuDNN for deterministic execution.

    Returns:
        The seed value that was set.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    return seed
