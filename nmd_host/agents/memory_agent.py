"""14.1 Memory Agent — keep the memory system healthy.

Runs the *existing* ``MemoryConsolidator`` over a user's memories (the same
pairwise-similarity pipeline Step 5 uses) and reports the KEEP / UPDATE /
MERGE / IGNORE decisions. Crucially it does **not** create a second
consolidation system, and it **never deletes**: MERGE decisions are reported
so an operator/UI can apply them, and in ``consolidate`` mode it creates one
merged memory from the newest source rather than discarding anything.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

from ..core.models import (
    Classification,
    Importance,
    Lifecycle,
    Memory,
    MemoryContent,
    MemoryStatus,
    MemoryType,
    Provenance,
    RetentionPolicy,
)
from ..utils.time_helpers import to_epoch as _to_epoch

logger = logging.getLogger("nmd_host.agents.memory_agent")


def _source_timestamp(memory: Memory) -> float:
    """Newest available lifecycle timestamp (stores vary; assume fresh)."""
    stamp = memory.lifecycle.updated_at or memory.lifecycle.created_at
    if stamp is None:
        return 0.0
    return _to_epoch(stamp)


class MemoryAgent:
    """Background consolidation agent for one user.

    ``mode="review"`` (default) only reports what the existing consolidator
    recommends. ``mode="consolidate"`` additionally creates one merged memory
    per MERGE decision — built from the *newest* source's content (nothing is
    fabricated or merged by the agent itself) with a summary that records its
    provenance. Source memories are always left untouched; ``None`` deletes.
    """

    def __init__(
        self,
        repository: Any,
        manager: Any,
        user_id: str = "user_001",
        mode: str = "review",
    ) -> None:
        self._repository = repository
        self._manager = manager
        self._user_id = user_id
        self._mode = mode

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def run(self, mode: Optional[str] = None) -> Dict[str, Any]:
        """Run one consolidation pass and return a structured report."""
        active_mode = mode or self._mode
        decisions = self._manager.consolidate(user_id=self._user_id)
        by_action: Dict[str, List[Any]] = defaultdict(list)
        for decision in decisions:
            by_action[decision.action].append(decision)

        created: List[str] = []
        if active_mode == "consolidate":
            created = [
                mid for mid in (self._consolidate(d) for d in by_action["MERGE"]) if mid
            ]

        return {
            "agent": "memory",
            "user_id": self._user_id,
            "mode": active_mode,
            "run_at": time.time(),
            "memories_reviewed": len(decisions),
            "by_action": {k: len(v) for k, v in by_action.items()},
            "created_memory_ids": created,
            "deleted_memory_ids": [],
            "decisions": [d.model_dump() for d in decisions],
        }

    # ------------------------------------------------------------------ #
    # Consolidation (conservative)                                       #
    # ------------------------------------------------------------------ #

    def _consolidate(self, decision: Any) -> Optional[str]:
        """Create one merged memory for a MERGE decision (never deletes)."""
        sources = [
            self._repository.get(mid)
            for mid in decision.source_memory_ids
            if mid
        ]
        sources = [s for s in sources if s is not None]
        if not sources:
            logger.warning(
                "memory agent: MERGE decision has no reachable sources (%s)",
                decision,
            )
            return None
        newest = max(sources, key=_source_timestamp)
        source_ids = ", ".join(s.memory_id or "?" for s in sources)
        merged = Memory(
            user_id=self._user_id,
            content=MemoryContent(
                text=newest.content.text,
                summary=(
                    f"Consolidated by Memory Agent from {len(sources)} source "
                    f"memories ({source_ids})."
                ),
            ),
            classification=Classification(
                memory_type=newest.classification.memory_type,
                category=newest.classification.category,
                source="memory_agent",
            ),
            provenance=Provenance(
                source_type="agent",
                created_by="memory_agent",
            ),
            importance=Importance(
                score=max((s.importance.score for s in sources), default=0.5),
                confidence=max(
                    (s.importance.confidence for s in sources), default=0.5
                ),
                priority=newest.importance.priority,
            ),
            lifecycle=Lifecycle(
                retention_policy=newest.lifecycle.retention_policy,
                created_at=time.time(),
                updated_at=time.time(),
            ),
            entities=list({e for s in sources for e in s.entities}),
            status=MemoryStatus.ACTIVE,
        )
        stored = self._repository.create(merged)
        logger.info(
            "memory agent consolidated %s -> %s",
            newest.memory_id,
            stored.memory_id,
        )
        return stored.memory_id


__all__ = ["MemoryAgent"]