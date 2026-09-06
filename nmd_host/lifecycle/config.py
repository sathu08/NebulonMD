"""Step 3 configuration, loaded from ``.env`` (never breaks existing keys).

Suggested variables (all optional — sensible defaults apply):

    NMD_RETRIEVAL_TOP_K=5
    NMD_RETRIEVAL_CANDIDATES=15
    NMD_RANKING_SEMANTIC_WEIGHT=0.50
    NMD_RANKING_IMPORTANCE_WEIGHT=0.20
    NMD_RANKING_CONFIDENCE_WEIGHT=0.15
    NMD_RANKING_RECENCY_WEIGHT=0.15
    NMD_CONTEXT_MAX_ITEMS=10
    NMD_CONTEXT_MAX_CHARACTERS=6000
    NMD_LIFECYCLE_AUTO_CLEANUP=false
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


__all__ = [
    "env_bool",
    "env_float",
    "env_int",
]
