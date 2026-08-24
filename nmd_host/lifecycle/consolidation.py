"""Consolidation (Step 3).

Consolidation asks whether several memories represent the same *evolving*
fact — and then deliberately does very little. It is conservative:

* similar memories are never automatically merged or deleted;
* conflicting facts ("I like Python" vs "I don't like Python anymore") are
  flagged IGNORE — similarity never implies truth equivalence;
* version drift ("Python 3.10" → "Python 3.13") is flagged UPDATE as a
  recommendation, preserving the historical memory's provenance;
* every decision names its ``source_memory_ids`` so callers can trace the
  reasoning back to the exact memories involved.

The output is a recommendation only; nothing here writes to storage.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pydantic import BaseModel
from typing import Any, Dict, List, Literal, Tuple

from pydantic import Field

from nmd_host.core.models import Memory

from .deduplication import content_similarity


_NEGATION_TERMS = (
    "don't",
    "dont",
    "doesn't",
    "doesnt",
    "didn't",
    "didnt",
    "no longer",
    "never",
    "not anymore",
    "dislike",
    "hate",
    "stopped",
    "quit",
    "no",
)
_VERSION_PATTERN = re.compile(r"\d+(?:\.\d+)+")


class ConsolidationDecision(BaseModel):
    action: Literal["KEEP", "UPDATE", "MERGE", "IGNORE"] = "KEEP"
    source_memory_ids: List[str] = Field(default_factory=list)
    reason: str = ""


class MemoryConsolidator:
    """Groups similar memories and recommends a conservative outcome."""

    def __init__(self, similarity_threshold: float = 0.80) -> None:
        self._threshold = similarity_threshold

    def analyze(self, memories: List[Memory]) -> List[ConsolidationDecision]:
        """One decision per group of related memories (singletons → KEEP)."""
        if not memories:
            return []
        components = self._similar_components(memories)
        decisions: List[ConsolidationDecision] = []
        for component in components:
            if len(component) == 1:
                decisions.append(
                    ConsolidationDecision(
                        action="KEEP",
                        source_memory_ids=[component[0].memory_id],
                        reason="unique memory; nothing to consolidate",
                    )
                )
                continue
            reasons = self._component_reason(component)
            for action, ids, reason in reasons:
                decisions.append(
                    ConsolidationDecision(
                        action=action,
                        source_memory_ids=ids,
                        reason=reason,
                    )
                )
        return decisions

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #

    def _similar_components(
        self, memories: List[Memory]
    ) -> List[List[Memory]]:
        """Connected components of the pairwise similarity graph.

        Two memories are linked only when they share a user and a category
        and their content is similar enough — the same gate used by the
        deduplication layer.
        """
        parent = {id(m): id(m) for m in memories}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            parent[find(a)] = find(b)

        for i, memory_a in enumerate(memories):
            for memory_b in memories[i + 1 :]:
                if self._linkable(memory_a, memory_b):
                    union(id(memory_a), id(memory_b))

        groups: dict = {}
        for memory in memories:
            groups.setdefault(find(id(memory)), []).append(memory)
        return [sorted(group, key=lambda m: m.memory_id or "") for group in groups.values()]

    def _linkable(self, memory_a: Memory, memory_b: Memory) -> bool:
        if memory_a.user_id != memory_b.user_id:
            return False
        if memory_a.classification.category != memory_b.classification.category:
            return False
        if content_similarity(memory_a, memory_b) >= self._threshold:
            return True
        if self._version_related(memory_a.content.text, memory_b.content.text):
            return True
        if self._slot_conflict(memory_a, memory_b):
            return True
        return False

    @staticmethod
    def _version_related(text_a: str, text_b: str) -> bool:
        """Two texts that differ mainly by their version numbers.

        "I use Python 3.10 for my projects" vs "I upgraded to Python 3.13
        for my projects": the version numbers are stripped to a placeholder
        and the remaining textual frame must still overlap strongly enough.
        """
        numbers_a = _VERSION_PATTERN.findall(text_a)
        numbers_b = _VERSION_PATTERN.findall(text_b)
        if not numbers_a or not numbers_b:
            return False
        if set(numbers_a) == set(numbers_b):
            return False
        frame_a = _VERSION_PATTERN.sub("VERSION", text_a.lower())
        frame_b = _VERSION_PATTERN.sub("VERSION", text_b.lower())
        from .deduplication import text_similarity

        return text_similarity(frame_a, frame_b) >= 0.6

    def _component_reason(
        self, component: List[Memory]
    ) -> List[Tuple[str, List[str], str]]:
        """Decide the outcome for one similar group (conservative by design)."""
        ids = [m.memory_id for m in component]
        if self._has_conflict(component):
            return [
                (
                    "IGNORE",
                    ids,
                    "conflicting facts detected; similarity does not imply "
                    "truth equivalence — do not merge",
                )
            ]
        if self._has_version_drift(component):
            return [
                (
                    "UPDATE",
                    ids,
                    "evolving fact (version drift); recommend updating the "
                    "current value while preserving historical provenance",
                )
            ]
        return [
            (
                "MERGE",
                ids,
                "near-identical content and category; merge candidate — no "
                "automatic merge performed",
            )
        ]

    @staticmethod
    def _has_conflict(component: List[Memory]) -> bool:
        negated = [m for m in component if _is_negated(m.content.text)]
        affirmed = [m for m in component if not _is_negated(m.content.text)]
        return bool(negated) and bool(affirmed) or MemoryConsolidator._has_slot_conflict(component)

    @staticmethod
    def _has_slot_conflict(component: List[Memory]) -> bool:
        """Same fact slot with two different values (e.g. employer: Apple vs Samsung).

        Two affirmative statements can still contradict: "I work at Apple" and
        "I now work at Samsung" carry different values for the same normalized
        slot, so they must be flagged even though neither is negated.
        """
        seen: Dict[str, set] = defaultdict(set)
        for memory in component:
            for key, value in _memory_slots(memory).items():
                seen[key].add(str(value))
        return any(len(values) > 1 for values in seen.values())

    @staticmethod
    def _slot_conflict(memory_a: Memory, memory_b: Memory) -> bool:
        """Do two memories fill the same slot with different values?"""
        slots_a = _memory_slots(memory_a)
        slots_b = _memory_slots(memory_b)
        if not slots_a or not slots_b:
            return False
        return any(
            key in slots_b and slots_b[key] != value_a
            for key, value_a in slots_a.items()
        )

    @staticmethod
    def _has_version_drift(component: List[Memory]) -> bool:
        versions: set = set()
        for memory in component:
            versions.update(_VERSION_PATTERN.findall(memory.content.text))
        return len(versions) >= 2


def _is_negated(text: str) -> bool:
    lowered = (text or "").lower()
    return any(term in lowered for term in _NEGATION_TERMS)


def _memory_slots(memory: Memory) -> Dict[str, Any]:
    """Flatten ``content.structured_data["slots"]`` into ``{slug: value}``.

    ``slots`` looks like ``{"identity": {"employer": "Apple"}}``; the group
    key and the slot key are joined so distinct groups never collide.
    """
    data = (memory.content.structured_data or {}).get("slots")
    if not isinstance(data, dict):
        return {}
    flat: Dict[str, Any] = {}
    for group, slots in data.items():
        if isinstance(slots, dict):
            for key, value in slots.items():
                flat[f"{group}.{key}"] = value
        elif slots is not None:
            flat[str(group)] = slots
    return flat


__all__ = ["ConsolidationDecision", "MemoryConsolidator"]