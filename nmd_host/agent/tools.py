"""Step 6 — Agent tools: the memory toolkit.

Two tools ship, both backed entirely by the existing Steps 2/3 pipeline and
persisted through the user's repository (NebulonDB the only persistent
store):

* ``RememberTool`` — conversation → Step 2 decisions → Step 3
  ``ingest()`` gate → ``STORE``-approved memories persisted.
* ``RecallTool`` — Step 3 ``MemoryRetriever`` → bounded context rendering.

The runtime is stateless; anything worth keeping lives in NebulonDB.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Protocol

from ..core.models import Provenance
from ..intelligence.bridge import MemoryIntelligence
from ..intelligence.converter import candidate_to_memory
from ..intelligence.extractor import MemoryExtractor
from ..intelligence.schemas import Conversation
from ..lifecycle.context import MemoryContextBuilder
from .trace import RecallSpan

logger = logging.getLogger("nmd_host.agent.tools")


class AgentTool(Protocol):
    """Contract every tool implements."""

    name: str
    description: str

    def run(self, arguments: dict) -> str:
        ...


class RememberTool:
    """Persist what should be remembered (Step 2 → Step 3 gate → store)."""

    name = "remember"
    description = (
        'Persist a fact worth remembering: {"text": "<what to remember>", '
        '"category": "<optional category>"}.'
    )

    def __init__(
        self,
        repository: Any,
        manager: Any,
        user_id: str,
        session_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        extractor: Optional[MemoryExtractor] = None,
    ) -> None:
        self._repository = repository
        self._manager = manager
        self._user_id = user_id
        self._session_id = session_id
        self._conversation_id = conversation_id
        self._extractor = extractor

    def run(self, arguments: dict) -> str:
        text = str(arguments.get("text", "")).strip()
        if not text:
            return "error: remember requires a non-empty text argument"
        intelligence = MemoryIntelligence(
            self._repository, user_id=self._user_id, extractor=self._extractor
        )
        conversation = Conversation.from_user_message(text)
        conversation.session_id = self._session_id
        conversation.conversation_id = self._conversation_id
        try:
            decisions = intelligence.decide(conversation)
        except Exception as exc:  # LLM extraction failure → keep the tool alive
            if self._extractor is None:
                return f"error: remember failed ({type(exc).__name__})"
            logger.warning(
                "agent remember LLM extraction failed (%s); falling back to rules",
                exc,
            )
            intelligence = MemoryIntelligence(self._repository, user_id=self._user_id)
            decisions = intelligence.decide(conversation)
        if self._extractor is not None and not decisions:
            logger.warning(
                "agent remember LLM extraction returned no decisions; "
                "falling back to rules"
            )
            intelligence = MemoryIntelligence(self._repository, user_id=self._user_id)
            decisions = intelligence.decide(conversation)
        stored = skipped = 0
        for decision in decisions:
            memory = candidate_to_memory(decision.candidate, self._user_id, source="agent")
            memory.provenance.source_type = "agent"
            memory.provenance.created_by = "agent"
            memory.provenance.session_id = self._session_id
            memory.provenance.conversation_id = self._conversation_id
            verdict = self._manager.ingest(memory, user_id=self._user_id)
            if verdict.action in ("STORE", "SUPERSEDE"):
                created = self._repository.create(memory)
                if verdict.action == "SUPERSEDE":
                    self._manager.archive_superseded(
                        created.memory_id or "", verdict.superseded_memory_ids
                    )
                stored += 1
            else:
                skipped += 1
                logger.info(
                    "agent remember skipped: action=%s reason=%s",
                    verdict.action, verdict.reason,
                )
        summary = (
            f"stored {stored} new memory/memories"
            if stored else "nothing new stored"
        )
        if skipped:
            summary += f" ({skipped} duplicate/invalid skipped)"
        return summary


class RecallTool:
    """Search the user's memories through the Step 3 retrieval pipeline."""

    name = "recall"
    description = (
        'Search the user\'s stored memories: {"query": "<search text>"}.'
    )

    def __init__(
        self, manager: Any, user_id: str, max_characters: int = 2000
    ) -> None:
        self._manager = manager
        self._user_id = user_id
        self._max_characters = max_characters
        self._trace = None

    def attach_trace(self, trace: Any) -> None:
        """Step 12 observability hook: record the retrieval outcome."""
        self._trace = trace

    def run(self, arguments: dict) -> str:
        query = str(arguments.get("query", "")).strip()
        if not query:
            return "error: recall requires a non-empty query argument"
        try:
            memories = self._manager.retrieve(
                query, top_k=5, user_id=self._user_id
            )
        except Exception as exc:  # defensive: retrieval must not crash the loop
            logger.warning("agent recall failed: %s", exc)
            return f"error: recall failed ({type(exc).__name__})"
        if self._trace is not None:
            if self._trace.recall is None:
                self._trace.recall = RecallSpan(query=query)
            self._trace.recall.memories = len(memories)
        if not memories:
            return "no memories found for that query"
        context = MemoryContextBuilder().build(
            memories, max_items=5, max_characters=self._max_characters
        )
        return f"relevant memories:\n{context}"


def build_memory_toolkit(
    repository: Any,
    manager: Any,
    user_id: str,
    session_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
    extractor: Optional[MemoryExtractor] = None,
) -> list:
    """Construct the two default memory tools for a user's bundle.

    ``session_id`` / ``conversation_id`` (optional) are stamped onto every
    memory the agent persists via ``RememberTool``. ``extractor`` (optional)
    overrides the decision engine's classifier — e.g. an ``LLMExtractor``
    when ``NMD_LLM_EXTRACTOR=true`` — and falls back to rules on failure.
    """
    return [
        RememberTool(
            repository,
            manager,
            user_id,
            session_id=session_id,
            conversation_id=conversation_id,
            extractor=extractor,
        ),
        RecallTool(manager, user_id),
    ]


__all__ = [
    "AgentTool",
    "RecallTool",
    "RememberTool",
    "build_memory_toolkit",
]