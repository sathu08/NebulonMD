"""Common lifecycle representation (Step 3).

Answers *when* a memory is alive: expired or active, how old it is, how long
it has been idle, and which of the four lifecycle states it is in.

Step 3 works with already-created ``Memory`` objects — deciding *what* becomes
memory belongs to Step 2. All evaluation is a pure function of the Memory's
own lifecycle fields; a wall clock can be injected for deterministic tests.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Optional

from nmd_host.core.models import Memory, MemoryStatus, RetentionPolicy


class MemoryState(str, Enum):
    ACTIVE = "active"
    EXPIRED = "expired"
    SESSION = "session"
    ARCHIVED = "archived"


Clock = Callable[[], datetime]


def now_utc() -> datetime:
    """Timezone-aware "now" used everywhere in Step 3."""
    return datetime.now(timezone.utc)


def _aware(value: Optional[datetime]) -> Optional[datetime]:
    """Naive datetimes are interpreted as UTC; ``None`` stays ``None``."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def is_expired(memory: Memory, now: Optional[datetime] = None) -> bool:
    """Whether a memory is past its usable life.

    * ``PERMANENT`` — never expires by the clock.
    * ``TEMPORARY`` — expired when ``expires_at`` is in the past.
    * ``SESSION`` — lifetime is governed by the session lifecycle, not the
      clock; a session memory is never "expired" here.
    * Missing ``expires_at`` on a TEMPORARY memory — cannot prove expiry,
      so it is treated as alive (see ``retention.is_valid_configuration``
      for explicit validation of that misconfiguration).

    Archived memories are frozen, not expired.
    """
    if memory.status is MemoryStatus.ARCHIVED:
        return False
    policy = memory.lifecycle.retention_policy
    if policy in (RetentionPolicy.PERMANENT, RetentionPolicy.SESSION):
        return False
    expires_at = _aware(memory.lifecycle.expires_at)
    if expires_at is None:
        return False
    return expires_at < _aware(now if now is not None else now_utc())


def get_state(memory: Memory, now: Optional[datetime] = None) -> MemoryState:
    """Classify a memory into one of the four lifecycle states.

    ARCHIVED wins (frozen), then SESSION (session-scoped), then EXPIRED,
    then ACTIVE.
    """
    if memory.status is MemoryStatus.ARCHIVED:
        return MemoryState.ARCHIVED
    if memory.lifecycle.retention_policy is RetentionPolicy.SESSION:
        return MemoryState.SESSION
    if is_expired(memory, now):
        return MemoryState.EXPIRED
    return MemoryState.ACTIVE


def memory_age(memory: Memory, now: Optional[datetime] = None) -> float:
    """Age in seconds since creation (falls back to last update)."""
    reference = _aware(memory.lifecycle.created_at) or _aware(
        memory.lifecycle.updated_at
    )
    if reference is None:
        return 0.0
    current = _aware(now if now is not None else now_utc())
    return max(0.0, (current - reference).total_seconds())


def time_since_access(memory: Memory, now: Optional[datetime] = None) -> float:
    """Idle time in seconds (last_accessed; falls back to last update)."""
    reference = _aware(memory.lifecycle.last_accessed)
    if reference is None:
        reference = _aware(memory.lifecycle.updated_at) or _aware(
            memory.lifecycle.created_at
        )
    if reference is None:
        return 0.0
    current = _aware(now if now is not None else now_utc())
    return max(0.0, (current - reference).total_seconds())


__all__ = [
    "Clock",
    "MemoryState",
    "get_state",
    "is_expired",
    "memory_age",
    "now_utc",
    "time_since_access",
]
