"""FoodCrisisEnv package with typed exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "FoodCrisisEnv",
    "FoodCrisisAction",
    "FoodCrisisObservation",
    "FoodCrisisState",
    "FoodCrisisEnvClient",
    "HTTPEnvClient",
]


def __getattr__(name: str) -> Any:
    if name == "FoodCrisisEnv":
        return import_module(".environment", __name__).FoodCrisisEnv

    if name in {"FoodCrisisAction", "FoodCrisisObservation", "FoodCrisisState"}:
        module = import_module(".models", __name__)
        return getattr(module, name)

    if name in {"HTTPEnvClient", "FoodCrisisEnvClient"}:
        module = import_module(".client", __name__)
        return getattr(module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
