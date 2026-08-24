"""Step 3 orchestration — ``MemoryLifecycleManager``.

The manager wires the lifecycle modules together and exposes high-level
operations; each responsibility stays in its own module. It is also the
integration point for the Step 2 pipeline:

    MemoryDecisionEngine → MemoryDecision → candidate_to_memory()
      → MemoryLifecycleManager.ingest()  (duplicate / lifecycle checks)
      → NebulonMind.store()
"""

from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import BaseModel, Field

from nmd_host.core.models import Memory, MemoryStatus
from nmd_host.core.repository import MemoryRepository


from .consolidation import ConsolidationDecision, MemoryConsolidator, _memory_slots
from .context import ContextConfig, MemoryContextBuilder
from .deduplication import DuplicateDetector, DuplicateResult
from .forgetting import ForgettingManager
from .ranking import MemoryRanker, RankingConfig
from .retention import RetentionEvaluator
from .retrieval import MemoryRetriever, RetrievalConfig
from .lifecycle import now_utc


class IngestionResult(BaseModel):
    """What the Step 3 gate decided about a memory about to be stored."""

    action: Literal["STORE", "DUPLICATE", "EXPIRED", "INVALID", "SUPERSEDE"] = "STORE"
    memory: Memory
    existing_memory: Optional[Memory] = None
    superseded_memory_ids: List[str] = Field(default_factory=list)
    reason: str = ""


class MemoryLifecycleManager:
    def __init__(
        self,
        repository: MemoryRepository,
        retriever: MemoryRetriever,
        retention: RetentionEvaluator,
        deduplicator: DuplicateDetector,
        ranker: MemoryRanker,
        consolidator: MemoryConsolidator,
        forgetting: ForgettingManager,
        context_builder: MemoryContextBuilder,
    ) -> None:
        self.repository = repository
        self.retriever = retriever
        self.retention = retention
        self.deduplicator = deduplicator
        self.ranker = ranker
        self.consolidator = consolidator
        self.forgetting = forgetting
        self.context_builder = context_builder

    # ------------------------------------------------------------------ #
    # High-level operations                                              #
    # ------------------------------------------------------------------ #

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        user_id: Optional[str] = None,
    ) -> List[Memory]:
        return self.retriever.retrieve(query, top_k=top_k, user_id=user_id)

    def check_duplicate(
        self,
        memory: Memory,
        candidates: Optional[List[Memory]] = None,
        user_id: Optional[str] = None,
    ) -> DuplicateResult:
        if candidates is None:
            candidates = self.repository.list_all(user_id)
        return self.deduplicator.check(memory, candidates)

    def evaluate_retention(self, memory: Memory, session_ended: bool = False) -> dict:
        """Full retention verdict for one memory."""
        from .lifecycle import get_state, is_expired

        return {
            "state": get_state(memory).value,
            "expired": is_expired(memory),
            "should_retain": self.retention.should_retain(memory, session_ended=session_ended),
            "should_delete": self.retention.should_delete(memory, session_ended=session_ended),
            "valid_configuration": self.retention.is_valid_configuration(memory),
        }

    def cleanup(self, user_id: Optional[str] = None, session_ended: bool = False) -> int:
        return self.forgetting.cleanup(user_id=user_id, session_ended=session_ended)

    def consolidate(
        self,
        memories: Optional[List[Memory]] = None,
        user_id: Optional[str] = None,
    ) -> List[ConsolidationDecision]:
        if memories is None:
            memories = self.repository.list_all(user_id)
        return self.consolidator.analyze(memories)

    def build_context(
        self,
        memories: List[Memory],
        max_items: int = 10,
        max_characters: int = 6000,
    ) -> str:
        return self.context_builder.build(
            memories, max_items=max_items, max_characters=max_characters
        )

    # ------------------------------------------------------------------ #
    # Step 2 integration                                                 #
    # ------------------------------------------------------------------ #

    def ingest(
        self,
        memory: Memory,
        candidates: Optional[List[Memory]] = None,
        user_id: Optional[str] = None,
    ) -> IngestionResult:
        """Gate a new memory before it reaches ``NebulonMind.store()``.

        1. invalid retention configuration → INVALID (explicit, no invention)
        2. expired → EXPIRED
        3. duplicate of an existing memory → DUPLICATE (with the existing one)
        4. otherwise → STORE
        """
        if not self.retention.is_valid_configuration(memory):
            return IngestionResult(
                action="INVALID",
                memory=memory,
                reason="TEMPORARY memory has no expires_at; retention config invalid",
            )
        if self.retention.is_expired(memory):
            return IngestionResult(
                action="EXPIRED",
                memory=memory,
                reason="memory is already expired; refusing to store",
            )
        duplicate = self.check_duplicate(memory, candidates=candidates, user_id=user_id)
        if duplicate.is_duplicate:
            existing = self._find_existing(duplicate.existing_memory_id, candidates)
            return IngestionResult(
                action="DUPLICATE",
                memory=memory,
                existing_memory=existing,
                reason=(
                    f"duplicate of {duplicate.existing_memory_id} "
                    f"(similarity {duplicate.similarity:.2f})"
                ),
            )
        superseded = self._find_superseded(memory, candidates=candidates, user_id=user_id)
        if superseded:
            return IngestionResult(
                action="SUPERSEDE",
                memory=memory,
                superseded_memory_ids=[m.memory_id for m in superseded if m.memory_id],
                reason=(
                    f"conflicts with {len(superseded)} existing memory/memories "
                    "filling the same fact slot; the new value supersedes the old"
                ),
            )
        return IngestionResult(action="STORE", memory=memory)

    # ------------------------------------------------------------------ #
    # Supersede (Tier 3)                                                  #
    # ------------------------------------------------------------------ #

    def archive_superseded(self, new_memory_id: str, superseded_ids: List[str]) -> int:
        """Archive memories superseded by a newer one; returns the count archived.

        Never deletes: each old memory is frozen (``ARCHIVED``, out of
        retrieval) and stamped with ``structured_data.superseded_by`` so its
        historical value and provenance stay answerable.
        """
        archived = 0
        for old_id in superseded_ids:
            existing = self.repository.get(old_id)
            if existing is None or existing.status is MemoryStatus.ARCHIVED:
                continue
            data = existing.model_dump()
            content = dict(data.get("content") or {})
            structured = dict(content.get("structured_data") or {})
            structured["superseded_by"] = new_memory_id
            content["structured_data"] = structured
            lifecycle = dict(data.get("lifecycle") or {})
            lifecycle["updated_at"] = now_utc()
            self.repository.update(
                old_id,
                {
                    "status": MemoryStatus.ARCHIVED.value,
                    "content": content,
                    "lifecycle": lifecycle,
                },
            )
            archived += 1
        return archived

    def _find_superseded(
        self,
        memory: Memory,
        candidates: Optional[List[Memory]] = None,
        user_id: Optional[str] = None,
    ) -> List[Memory]:
        """Existing non-archived memories that fill the same slot differently."""
        new_slots = _memory_slots(memory)
        if not new_slots:
            return []
        if candidates is None:
            candidates = self.repository.list_all(user_id)
        superseded: List[Memory] = []
        for existing in candidates:
            if existing.user_id != memory.user_id:
                continue
            if existing.memory_id == memory.memory_id:
                continue
            if existing.status is MemoryStatus.ARCHIVED:
                continue
            old_slots = _memory_slots(existing)
            if any(
                key in old_slots and old_slots[key] != value
                for key, value in new_slots.items()
            ):
                superseded.append(existing)
        return superseded

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #

    def _find_existing(
        self, memory_id: Optional[str], candidates: Optional[List[Memory]]
    ) -> Optional[Memory]:
        if not memory_id:
            return None
        if candidates:
            for memory in candidates:
                if memory.memory_id == memory_id:
                    return memory
        return self.repository.get(memory_id)


def build_lifecycle_manager(
    repository: MemoryRepository,
    searcher=None,
    ranking_config: Optional[RankingConfig] = None,
    retrieval_config: Optional[RetrievalConfig] = None,
    context_config: Optional[ContextConfig] = None,
) -> MemoryLifecycleManager:
    """Wired-up Step 3 manager for a repository.

    ``searcher`` (optional) feeds the semantic duplicate layer — pass a
    ``NebulonMind``/``VectorStore`` for the live pipeline, or an
    ``InMemoryRepository`` for offline semantic-detection tests.
    """
    retention = RetentionEvaluator()
    deduplicator = DuplicateDetector(searcher=searcher)
    ranker = MemoryRanker(ranking_config or RankingConfig())
    retriever = MemoryRetriever(
        repository,
        ranker=ranker,
        retention=retention,
        deduplicator=deduplicator,
        config=retrieval_config or RetrievalConfig(),
    )
    consolidator = MemoryConsolidator()
    forgetting = ForgettingManager(repository, retention)
    context_builder = MemoryContextBuilder(context_config or ContextConfig())
    return MemoryLifecycleManager(
        repository=repository,
        retriever=retriever,
        retention=retention,
        deduplicator=deduplicator,
        ranker=ranker,
        consolidator=consolidator,
        forgetting=forgetting,
        context_builder=context_builder,
    )


__all__ = [
    "IngestionResult",
    "MemoryLifecycleManager",
    "build_lifecycle_manager",
]
