"""FoodCrisisEnv compatibility package with lazy exports to keep runtime imports lightweight."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "HTTPEnvClient",
    "IRCEAction",
    "IRCEEnv",
    "IRCEEnvClient",
    "IRCEObservation",
    "IRCEState",
]


def __getattr__(name: str) -> Any:
    if name == "IRCEEnv":
        return import_module(".environment", __name__).IRCEEnv

    if name in {"IRCEAction", "IRCEObservation", "IRCEState"}:
        module = import_module(".models", __name__)
        return getattr(module, name)

    if name in {"HTTPEnvClient", "IRCEEnvClient"}:
        module = import_module(".client", __name__)
        return getattr(module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
