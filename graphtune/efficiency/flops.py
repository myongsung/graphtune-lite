# graphtune/efficiency/flops.py
from typing import Dict, Optional

import torch
import torch.nn as nn


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
    (legacy와 동일)
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
