"""Deduplication (Step 3).

Prevents "My name is Sathya.", "My name is Sathya" and "Sathya is my name."
from becoming three independent permanent memories.

The detection is a layered strategy:

1. exact ``memory_id`` match
2. normalized text equality
3. same category + highly similar content (token-level)
4. semantic similarity via the existing vector retrieval infrastructure
   (injected searcher — never a new embedding model)

``text_similarity``/``normalize_text`` live here so the ranking layer can
reuse the same lexical similarity as a semantic signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from nmd_host.core.models import Memory

from nmd_host.utils.constants import SIZE_RATIO_FLOOR
from nmd_host.utils.text_helpers import (
    jaccard_similarity,
    normalize_text,
    text_similarity,
    token_set,
)


def _size_ratio_ok(text_a: str, text_b: str) -> bool:
    """False when one text is a largely expanded version of the other.

    Guards the overlap-coefficient similarity against subset traps like
    "I like Python" vs "I don't like Python anymore" (overlap 1.0 but the
    negated rewrite must not count as a duplicate).
    """
    size_a, size_b = len(token_set(text_a)), len(token_set(text_b))
    if not size_a or not size_b:
        return size_a == size_b
    return min(size_a, size_b) / max(size_a, size_b) >= SIZE_RATIO_FLOOR


def content_similarity(memory_a: Memory, memory_b: Memory) -> float:
    """Lexical similarity of two memories (text + summary)."""
    text_a = " ".join(
        filter(None, [memory_a.content.text, memory_a.content.summary])
    )
    text_b = " ".join(
        filter(None, [memory_b.content.text, memory_b.content.summary])
    )
    return text_similarity(text_a, text_b)


@dataclass(frozen=True)
class DuplicateResult:
    """Explicit outcome of a duplicate check — callers never guess."""

    is_duplicate: bool
    existing_memory_id: Optional[str] = None
    similarity: float = 0.0


def is_duplicate(memory_a: Memory, memory_b: Memory) -> bool:
    """Convenience: is ``memory_b`` a duplicate of ``memory_a``?"""
    return DuplicateDetector().check(memory_a, [memory_b]).is_duplicate


class DuplicateDetector:
    """Layered duplicate detection. ``searcher`` is optional.

    ``searcher`` must expose ``search(text, top_k)`` (VectorStore-style,
    returning dicts with ``memory_id``/``score``, or Memory objects) or
    ``recall(text, top_k)`` (NebulonMind-style). When present, layer 4 uses
    it: if searching for one memory's text surfaces the other, they are
    treated as semantically duplicate. When absent, layer 4 yields no
    evidence and lexical layers carry the decision.
    """

    def __init__(
        self,
        searcher=None,
        content_threshold: float = 0.90,
        semantic_threshold: float = 0.92,
        semantic_top_k: int = 5,
    ) -> None:
        self._searcher = searcher
        self._content_threshold = content_threshold
        self._semantic_threshold = semantic_threshold
        self._semantic_top_k = semantic_top_k

    def find_duplicate(
        self, memory: Memory, candidates: List[Memory]
    ) -> Optional[Memory]:
        """Return the first candidate that duplicates ``memory``, else None."""
        result = self.check(memory, candidates)
        if result.existing_memory_id is None:
            return None
        for candidate in candidates:
            if candidate.memory_id == result.existing_memory_id:
                return candidate
        return None

    def check(
        self,
        memory: Memory,
        candidates: List[Memory],
        threshold: Optional[float] = None,
    ) -> DuplicateResult:
        """Run the layered strategy against every candidate."""
        for candidate in candidates:
            if candidate is memory:
                continue  # the same object is never a duplicate
            result = self._compare(memory, candidate, threshold)
            if result.is_duplicate:
                return result
        return DuplicateResult(is_duplicate=False, similarity=0.0)

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #

    def _compare(
        self, memory: Memory, candidate: Memory, threshold: Optional[float]
    ) -> DuplicateResult:
        if memory.memory_id and memory.memory_id == candidate.memory_id:
            return DuplicateResult(True, candidate.memory_id, 1.0)

        if normalize_text(memory.content.text) == normalize_text(
            candidate.content.text
        ):
            return DuplicateResult(True, candidate.memory_id, 1.0)

        if (
            memory.classification.category == candidate.classification.category
            and _size_ratio_ok(memory.content.text, candidate.content.text)
        ):
            similarity = content_similarity(memory, candidate)
            if similarity >= (threshold or self._content_threshold):
                return DuplicateResult(True, candidate.memory_id, similarity)

        similarity = self._semantic_similarity(memory, candidate)
        if similarity >= (threshold or self._semantic_threshold):
            return DuplicateResult(True, candidate.memory_id, similarity)
        return DuplicateResult(False, similarity=similarity)

    def _semantic_similarity(self, memory: Memory, candidate: Memory) -> float:
        """Similarity via the injected vector retrieval infrastructure.

        Searches for the candidate's text and asks whether ``memory`` shows
        up among the nearest neighbours.
        """
        if self._searcher is None:
            return 0.0
        hits = self._search(self._searcher, candidate.content.text)
        for memory_id, score in hits:
            if memory_id == memory.memory_id:
                return 1.0 if score is None else max(0.0, min(1.0, score))
        return 0.0

    @staticmethod
    def _search(searcher, text: str) -> List[tuple]:
        raw = []
        if hasattr(searcher, "search"):
            raw = searcher.search(text, top_k=10)
        elif hasattr(searcher, "recall"):
            raw = searcher.recall(text, top_k=10)
        normalized: List[tuple] = []
        for item in raw:
            if isinstance(item, dict):
                memory_id = (item.get("metadata") or {}).get("memory_id")
                if not memory_id:
                    memory_id = item.get("memory_id")
                normalized.append((memory_id, item.get("score")))
            else:
                normalized.append((getattr(item, "memory_id", None), None))
        return [(memory_id, score) for memory_id, score in normalized if memory_id]


__all__ = [
    "DuplicateDetector",
    "DuplicateResult",
    "content_similarity",
    "is_duplicate",
    "jaccard_similarity",
    "normalize_text",
    "text_similarity",
]
