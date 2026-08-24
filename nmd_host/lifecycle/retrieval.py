"""Retrieval (Step 3).

The retrieval pipeline:

    query
      ▼
    VectorStore / NebulonMind.recall()      ← candidate provider (never bypassed)
      ▼
    candidate memories  (top_k × candidate_multiplier — never top_k raw)
      ▼
    retention filtering
      ▼
    ranking (full candidate set)
      ▼
    deduplication (keeps the best-ranked memory of each duplicate group)
      ▼
    top-k memories

Ranking runs before deduplication so a duplicate group never survives as
two entries and the *best-ranked* representative is the one kept. This
order is the single retrieval pipeline: the API server's ``/search`` and
``/memory/context`` both delegate here (no second ranking implementation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from nmd_host.core.models import Memory
from nmd_host.core.repository import MemoryRepository
from nmd_host.utils.env_helpers import env_int as _env_int

from .deduplication import DuplicateDetector, jaccard_similarity, normalize_text

from .ranking import MemoryRanker
from .retention import RetentionEvaluator


@dataclass
class RetrievalConfig:
    top_k: int = 5
    candidate_multiplier: int = 3
    include_expired: bool = False

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")
        if self.candidate_multiplier < 1:
            raise ValueError("candidate_multiplier must be >= 1")

    @classmethod
    def from_env(cls) -> "RetrievalConfig":
        top_k = _env_int("NMD_RETRIEVAL_TOP_K", 5)
        candidates = _env_int("NMD_RETRIEVAL_CANDIDATES", top_k * 3)
        multiplier = max(1, candidates // max(1, top_k))
        return cls(top_k=top_k, candidate_multiplier=multiplier)


class MemoryRetriever:
    """Pipeline orchestrator: candidates -> filter -> rank -> dedupe -> top-k."""

    def __init__(
        self,
        repository: MemoryRepository,
        ranker: Optional[MemoryRanker] = None,
        retention: Optional[RetentionEvaluator] = None,
        deduplicator: Optional[DuplicateDetector] = None,
        config: Optional[RetrievalConfig] = None,
    ) -> None:
        self._repository = repository
        self.ranker = ranker or MemoryRanker()
        self.retention = retention or RetentionEvaluator()
        self.deduplicator = deduplicator or DuplicateDetector()
        self.config = config or RetrievalConfig()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        user_id: Optional[str] = None,
    ) -> List[Memory]:
        """Ranked top-k memories for ``query``, lifecycle-filtered.

        Candidates are fetched *over-expanded* (configurable multiplier) so
        expired / duplicate candidates cannot crowd out valid memories after
        filtering. User isolation is preserved by construction: the
        candidate provider (a NebulonMind segment or a per-user repository)
        is already user-scoped, and a ``user_id`` guard is applied again.
        """
        effective_k = top_k if top_k else self.config.top_k
        candidate_count = max(
            effective_k, effective_k * self.config.candidate_multiplier
        )
        candidates = self._repository.search(query, top_k=candidate_count)
        return self.filter_and_rank(query, candidates, top_k=effective_k, user_id=user_id)

    def filter_and_rank(
        self,
        query: str,
        candidates: List[Memory],
        top_k: int = 5,
        user_id: Optional[str] = None,
    ) -> List[Memory]:
        """Run the post-candidate pipeline over a given candidate list.

        Lets callers keep their own candidate provisioning (e.g. graph
        expansion) while still using the single Step 3 ranking pipeline.
        Ranking runs over the full candidate set *before* deduplication so
        the best-ranked memory of each duplicate group is the one kept.
        """
        if user_id is not None:
            candidates = [m for m in candidates if m.user_id == user_id]
        if not self.config.include_expired:
            candidates = [
                m for m in candidates if self.retention.should_retain(m)
            ]
        ranked = self.ranker.rank(query, candidates)
        deduplicated = self._deduplicate(ranked)
        return deduplicated[:top_k]

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #

    def _deduplicate(self, candidates: List[Memory]) -> List[Memory]:
        """Collapse near-duplicates, keeping the first (highest-ranked) hit.

        Normalized equality always collapses. Similarity-based merging uses
        the *Jaccard* index (conservative) above a strict threshold, so
        genuinely different facts (e.g. "works with Python" vs "works with
        Python and Rust") survive.
        """
        kept: List[Memory] = []
        seen_ids: set = set()
        seen_normalized: set = set()
        for memory in candidates:
            if memory.memory_id and memory.memory_id in seen_ids:
                continue
            normalized = normalize_text(memory.content.text)
            if normalized in seen_normalized:
                continue
            if any(
                jaccard_similarity(memory.content.text, kept_memory.content.text) >= 0.95
                for kept_memory in kept
            ):
                continue
            kept.append(memory)
            if memory.memory_id:
                seen_ids.add(memory.memory_id)
            seen_normalized.add(normalized)
        return kept


__all__ = ["MemoryRetriever", "RetrievalConfig"]