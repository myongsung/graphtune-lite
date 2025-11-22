# graphtune/efficiency/static.py
from typing import Any, Dict

import torch.nn as nn

from ..utils import count_trainable_params, param_size_mb


def profile_model_static(model: nn.Module) -> Dict[str, Any]:
    """Static resource metrics from model parameters. (legacy와 동일)"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = count_trainable_params(model)
    size_mb = param_size_mb(model)

    return {
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "param_size_mb": float(size_mb),
        "trainable_ratio": float(trainable_params / max(total_params, 1)),
    }
