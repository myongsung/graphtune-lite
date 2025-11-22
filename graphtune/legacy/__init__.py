"""
Legacy namespace (v2 migration).

Legacy is now only for old code that hasn't been moved yet.
Do NOT import legacy.data here anymore.
"""

from .train import train_one_stage  # noqa: F401
from .utils import load_partial_state  # noqa: F401

__all__ = [
    "train_one_stage",
    "load_partial_state",
]
