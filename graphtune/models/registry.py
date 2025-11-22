# graphtune/models/registry.py
from typing import Callable, Dict, Any

_MODEL_REGISTRY: Dict[str, Callable[..., Any]] = {}

def register_model(name: str):
    name = name.lower()
    def deco(factory: Callable[..., Any]):
        _MODEL_REGISTRY[name] = factory
        return factory
    return deco

def build_model(model_name: str, dataset_bundle: dict, **kwargs):
    key = model_name.lower()
    if key not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model_name: {model_name}. Available={list(_MODEL_REGISTRY)}")
    return _MODEL_REGISTRY[key](dataset_bundle, **kwargs)
