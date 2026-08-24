"""Time/epoch helpers shared across stores and lifecycle code.

Stores mix UTC datetimes and float epochs depending on the backend, so every
consumer needs the same lenient conversion. ``to_epoch`` is the canonical
converter (previously duplicated as ``_to_epoch`` in ``agents.memory_agent``
and ``agents.task_agent``); ``fmt_timestamp`` / ``created_ts`` previously
lived privately in ``lifecycle.context``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from nmd_host.core.models import Memory


def to_epoch(value: Any) -> float:
    """Accept float or datetime timestamps (the store mixes both).

    ``None`` → 0.0; naive datetimes are assumed UTC.
    """
    if value is None:
        return 0.0
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.timestamp()
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def created_ts(memory: Memory) -> float:
    """Epoch of a memory's creation (falling back to its update time)."""
    created: Optional[datetime] = memory.lifecycle.created_at
    if created is None:
        created = memory.lifecycle.updated_at
    if created is None:
        return 0.0
    return created.timestamp()


def fmt_timestamp(value: Any) -> str:
    """datetime or float-epoch → UTC ISO string; ``None`` → ``"unknown"``."""
    if value is None:
        return "unknown"
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
    else:
        try:
            value = datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (TypeError, ValueError, OverflowError):
            return "unknown"
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


__all__ = ["created_ts", "fmt_timestamp", "to_epoch"]
