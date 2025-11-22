import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .utils import count_trainable_params, param_size_mb


def profile_model_static(model: nn.Module) -> Dict[str, Any]:
    """Static resource metrics from model parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = count_trainable_params(model)
    size_mb = param_size_mb(model)

    return {
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "param_size_mb": float(size_mb),
        "trainable_ratio": float(trainable_params / max(total_params, 1)),
    }


class _FirstOutputWrapper(nn.Module):
    """Wrap a model that may return tuple so FLOP profiler sees a tensor."""
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, (tuple, list)):
            return out[0]
        return out


def estimate_flops(
    model: nn.Module,
    sample_x: torch.Tensor,
    device: str = "cuda"
) -> Optional[Dict[str, float]]:
    """
    Estimate FLOPs/MACs using thop if available.
    Returns dict with 'macs' and 'flops' (per forward), or None.
    """
    try:
        from thop import profile  # pip install thop
    except Exception:
        return None

    wrapped = _FirstOutputWrapper(model).to(device)
    wrapped.eval()
    with torch.no_grad():
        macs, params = profile(wrapped, inputs=(sample_x.to(device),), verbose=False)

    # thop returns MACs; FLOPs ≈ 2 * MACs
    return {
        "macs": float(macs),
        "flops": float(macs * 2.0),
        "thop_params": float(params),
    }


class StageProfiler:
    """
    Context manager to measure wall time and GPU peak memory for a stage.
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
