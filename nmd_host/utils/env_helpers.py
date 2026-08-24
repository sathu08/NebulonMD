"""Shared, lenient environment-variable readers.

Previously re-declared as private ``_env_int`` / ``_env_float`` / ``_env_bool``
in ``core.config``, ``lifecycle.ranking`` / ``lifecycle.retrieval`` /
``lifecycle.context``, ``intelligence.providers`` and ``agent.config``.
Those modules now import and alias them (``env_int as _env_int``) so call
sites stay unchanged.

A missing or malformed variable falls back to the provided default — a
misbehaving ``.env`` never crashes config load.
"""

from __future__ import annotations

import os

_TRUE_VALUES = ("1", "true", "yes", "on")


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in _TRUE_VALUES


__all__ = ["env_bool", "env_float", "env_int"]
