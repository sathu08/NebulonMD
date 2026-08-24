"""Forgetting (Step 3).

Automatic cleanup of memories that lifecycle policy says may go:

* ``PERMANENT`` — never automatically deleted.
* ``TEMPORARY`` + expired — eligible for deletion.
* ``SESSION`` + session ended — eligible for deletion.

Nothing is ever deleted because it is old, or because its importance is low.
Deletion goes through the Step 1 architecture (repository → NebulonMind →
graph → vector → truth); Step 3 never touches the stores directly.
"""

from __future__ import annotations

from typing import List, Optional

from .retention import RetentionEvaluator
from nmd_host.core.repository import MemoryRepository
from nmd_host.core.models import Memory, MemoryStatus, RetentionPolicy


class ForgettingManager:
    def __init__(
        self,
        repository: MemoryRepository,
        retention: Optional[RetentionEvaluator] = None,
    ) -> None:
        self._repository = repository
        self._retention = retention or RetentionEvaluator()

    def find_expired(
        self,
        memories: Optional[List[Memory]] = None,
        user_id: Optional[str] = None,
        session_ended: bool = False,
    ) -> List[Memory]:
        """Memories eligible for forgetting under lifecycle policy."""
        memories = memories if memories is not None else self._repository.list_all(user_id)
        eligible: List[Memory] = []
        for memory in memories:
            if self._retention.is_expired(memory):
                eligible.append(memory)
            elif session_ended and memory.lifecycle.retention_policy is RetentionPolicy.SESSION:
                eligible.append(memory)
        return eligible

    def forget(self, memory_id: str) -> bool:
        """Delete one memory through the repository (graph → vector → truth)."""
        return self._repository.delete(memory_id)

    def cleanup(self, user_id: Optional[str] = None, session_ended: bool = False) -> int:
        """Delete every eligible memory; returns the number deleted."""
        deleted = 0
        for memory in self.find_expired(user_id=user_id, session_ended=session_ended):
            if memory.lifecycle.retention_policy is RetentionPolicy.PERMANENT:
                continue  # belt-and-braces: permanent is never forgotten
            if memory.status is MemoryStatus.ARCHIVED:
                continue  # archived is frozen, never forgotten
            if self.forget(memory.memory_id or ""):
                deleted += 1
        return deleted


__all__ = ["ForgettingManager"]