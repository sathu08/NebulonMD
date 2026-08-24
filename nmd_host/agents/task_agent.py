"""14.2 Task Agent — scheduled work (weekly summary).

The canonical example from changes.txt 14.2: every Sunday, recall the
week's work memories, group them, generate a weekly summary, and store it
as a memory — so "what did I work on last week?" already has an answer.

This implementation groups recent memories by ``category`` and persists a
summary memory through the existing Step 3 ingest gate (same gate the Chat
Agent's ``remember`` uses), so the summary is retrievable like any other
memory. No second summarisation/consolidation system is introduced.
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

logger = logging.getLogger("nmd_host.agents.task_agent")


def _memory_timestamp(memory: Memory) -> float:
    """Newest available lifecycle timestamp; none → assume just-created.

    Stores differ: ``NebulonMind.store`` stamps UTC datetimes, the
    in-memory repository does not. A memory with no recorded timestamp is
    assumed fresh so the weekly summary never silently drops it.
    """
    stamp = memory.lifecycle.updated_at or memory.lifecycle.created_at
    if stamp is None:
        return time.time()
    return _to_epoch(stamp)


class TaskAgent:
    """Weekly-summary task agent for one user.

    ``days`` is the look-back window. The summary is deterministic and
    template-based so the agent never requires an LLM to do its job; an
    optional ``llm`` provider (with a ``complete`` method) is used to
    rewrite the summary when provided — never required.
    """

    def __init__(
        self,
        repository: Any,
        manager: Any,
        user_id: str = "user_001",
        days: int = 7,
        llm: Optional[Any] = None,
    ) -> None:
        self._repository = repository
        self._manager = manager
        self._user_id = user_id
        self._days = days
        self._llm = llm

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def run(self, days: Optional[int] = None) -> Dict[str, Any]:
        """Scan the window, group by category, persist a summary memory."""
        window = days or self._days
        cutoff = time.time() - window * 86400
        memories = self._repository.list_all(self._user_id)
        recent = [m for m in memories if _memory_timestamp(m) >= cutoff]

        groups: Dict[str, List[Memory]] = defaultdict(list)
        for memory in recent:
            groups[memory.classification.category].append(memory)

        summary = self._render_summary(recent, groups)
        stored_id = self._store_summary(summary) if recent else None

        return {
            "agent": "task",
            "user_id": self._user_id,
            "run_at": time.time(),
            "period_days": window,
            "memories_scanned": len(memories),
            "memories_covered": len(recent),
            "groups": {k: len(v) for k, v in groups.items()},
            "summary": summary,
            "stored_memory_id": stored_id,
        }

    # ------------------------------------------------------------------ #
    # Summary rendering                                                  #
    # ------------------------------------------------------------------ #

    def _render_summary(self, recent: List[Memory], groups: Dict) -> str:
        if not recent:
            return "No new memories in the last window."
        lines = [
            f"Weekly summary — {len(recent)} memory(ies) over the last "
            f"{self._days} day(s).",
        ]
        for category in sorted(groups):
            members = sorted(groups[category], key=_memory_timestamp, reverse=True)
            lines.append(f"  {category} ({len(members)}):")
            for memory in members[:5]:
                text = memory.content.text.strip().replace("\n", " ")
                lines.append(f"    - {text[:160]}")
        base = "\n".join(lines)
        if self._llm is None:
            return base
        try:
            reply = self._llm.complete(base, system=None, temperature=0.0)
            rendered = (reply.text or "").strip()
            return rendered or base
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("task agent LLM rewrite failed; using template: %s", exc)
            return base

    # ------------------------------------------------------------------ #
    # Persistence (through the Step 3 gate)                              #
    # ------------------------------------------------------------------ #

    def _store_summary(self, summary: str) -> Optional[str]:
        memory = Memory(
            user_id=self._user_id,
            content=MemoryContent(
                text=summary,
                summary="Weekly summary generated by Task Agent.",
            ),
            classification=Classification(
                memory_type=MemoryType.KNOWLEDGE,
                category="summary",
                source="task_agent",
            ),
            provenance=Provenance(
                source_type="agent",
                created_by="task_agent",
            ),
            importance=Importance(score=0.5, confidence=0.7, priority="medium"),
            lifecycle=Lifecycle(
                retention_policy=RetentionPolicy.PERMANENT,
                created_at=time.time(),
                updated_at=time.time(),
            ),
            status=MemoryStatus.ACTIVE,
        )
        verdict = self._manager.ingest(memory, user_id=self._user_id)
        if verdict.action != "STORE":
            logger.info("task agent summary not stored: %s", verdict.action)
            return None
        stored = self._repository.create(memory)
        logger.info("task agent stored summary %s", stored.memory_id)
        return stored.memory_id


__all__ = ["TaskAgent"]