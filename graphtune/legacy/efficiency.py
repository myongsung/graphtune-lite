# graphtune/legacy/efficiency.py
from graphtune.efficiency.static import profile_model_static  # noqa
from graphtune.efficiency.flops import estimate_flops  # noqa
from graphtune.efficiency.profiler import StageProfiler  # noqa

__all__ = ["profile_model_static", "estimate_flops", "StageProfiler"]
