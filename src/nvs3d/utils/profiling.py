"""Minimal profiling and timing utilities."""

from pathlib import Path
import time
import torch


class Timer:
    """Simple context manager to measure wall-clock elapsed time."""

    def __init__(self, name: str = "operation") -> None:
        self.name = name
        self.start_time: float = 0.0
        self.elapsed_seconds: float = 0.0

    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.elapsed_seconds = time.perf_counter() - self.start_time


def get_vram_usage_mb() -> float:
    """Return current peak allocated GPU memory in MB, or 0.0 if CUDA is not available."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    return 0.0
