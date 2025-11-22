# graphtune/efficiency/profiler.py
import time
from typing import Dict

import torch


class StageProfiler:
    """
    Context manager to measure wall time and GPU peak memory for a stage.
    (legacy와 동일)
    """
    def __init__(self, device: str = "cuda"):
        self.device = device
        self._t0 = None

    def __enter__(self):
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def result(self) -> Dict[str, float]:
        if self._t0 is None:
            return {}
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
        else:
            peak_mem = 0.0
        dt = time.perf_counter() - self._t0
        return {"train_time_sec": float(dt), "peak_mem_mb": float(peak_mem)}
