"""Retention system (Step 3).

Decides whether a memory should remain available, using the Step 1
``RetentionPolicy``:

* ``PERMANENT`` — never automatically expires.
* ``TEMPORARY`` — expires at ``expires_at``; without ``expires_at`` the
  configuration is invalid. The evaluator retains it (safe default) and
  ``is_valid_configuration`` flags it explicitly — the system never invents
  an arbitrary expiration time.
* ``SESSION`` — retained until the session lifecycle says otherwise
  (``session_ended=True``).

Nothing is ever deleted merely because it is old.
"""

from __future__ import annotations

from typing import Optional

from nmd_host.core.models import Memory, MemoryStatus, RetentionPolicy

from .lifecycle import Clock, is_expired, now_utc


class RetentionError(Exception):
    """Raised on an explicitly invalid retention configuration."""


class RetentionEvaluator:
    """Policy evaluation for one memory. ``clock`` may be injected for tests."""

    def __init__(self, clock: Optional[Clock] = None) -> None:
        self._clock = clock or now_utc

    # ------------------------------------------------------------------ #
    # Evaluation                                                         #
    # ------------------------------------------------------------------ #

    def is_expired(self, memory: Memory) -> bool:
        """Clock-based expiry, per the memory's retention policy."""
        if memory.status is MemoryStatus.ARCHIVED:
            return False
        return is_expired(memory, now=self._clock())

    def is_valid_configuration(self, memory: Memory) -> bool:
        """True when the retention configuration is self-consistent.

        TEMPORARY without ``expires_at`` is an invalid configuration: the
        evaluator still retains it (it cannot prove expiry), but this flag
        lets callers surface the misconfiguration instead of silently
        inventing an expiry time.
        """
        if memory.lifecycle.retention_policy is RetentionPolicy.TEMPORARY:
            return memory.lifecycle.expires_at is not None
        return True

    def should_retain(self, memory: Memory, session_ended: bool = False) -> bool:
        """Whether the memory should stay available right now."""
        if memory.status is MemoryStatus.ARCHIVED:
            return True  # archived is frozen, not deleted
        policy = memory.lifecycle.retention_policy
        if policy is RetentionPolicy.SESSION:
            return not session_ended
        if policy is RetentionPolicy.PERMANENT:
            return True
        if memory.lifecycle.expires_at is None:
            return True  # invalid config -> retain (safe), never invent expiry
        return not self.is_expired(memory)

    def should_delete(self, memory: Memory, session_ended: bool = False) -> bool:
        """Whether the memory is eligible for deletion *by policy*.

        PERMANENT is never eligible; TEMPORARY is eligible once expired;
        SESSION is eligible once the session has ended.
        """
        return not self.should_retain(memory, session_ended=session_ended)


__all__ = ["RetentionError", "RetentionEvaluator"]
