"""
Legacy namespace (v2 migration).

Do NOT import models here anymore.
Models are provided from graphtune.models (v2).
"""

from .data import prepare_dataset  # noqa: F401
from .train import train_one_stage  # noqa: F401
from .utils import load_partial_state  # noqa: F401

__all__ = [
    "prepare_dataset",
    "train_one_stage",
    "load_partial_state",
]
