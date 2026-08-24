"""Context builder (Step 3).

Turns retrieved memories into a bounded context for an LLM. Ordering:

1. highest relevance (the memory ranked first by the retriever stays first),
2. highest importance,
3. newer information.

Each entry carries its provenance (memory_id, type, category, importance,
confidence, source) so the AI layer can see *where* the fact came from, and
the output is capped by item count and character budget.

Contract: this module is strictly formatting/selection — it never calls an
LLM and never mutates the ``Memory`` objects it is given.
"""

from __future__ import annotations

from typing import List, Optional

from dataclasses import dataclass

from nmd_host.core.models import Memory, MemoryStatus
from nmd_host.utils.constants import (
    CONTEXT_MAX_CHARACTERS_DEFAULT,
    CONTEXT_MAX_ITEMS_DEFAULT,
)
from nmd_host.utils.env_helpers import env_int as _env_int
from nmd_host.utils.time_helpers import created_ts as _created_ts
from nmd_host.utils.time_helpers import fmt_timestamp as _fmt_timestamp


@dataclass
class ContextConfig:
    max_items: int = CONTEXT_MAX_ITEMS_DEFAULT
    max_characters: int = CONTEXT_MAX_CHARACTERS_DEFAULT

    def __post_init__(self) -> None:
        if self.max_items < 1:
            raise ValueError("max_items must be >= 1")
        if self.max_characters < 1:
            raise ValueError("max_characters must be >= 1")

    @classmethod
    def from_env(cls) -> "ContextConfig":
        return cls(
            max_items=_env_int("NMD_CONTEXT_MAX_ITEMS", CONTEXT_MAX_ITEMS_DEFAULT),
            max_characters=_env_int(
                "NMD_CONTEXT_MAX_CHARACTERS", CONTEXT_MAX_CHARACTERS_DEFAULT
            ),
        )


class MemoryContextBuilder:
    def __init__(self, config: Optional[ContextConfig] = None) -> None:
        self.config = config or ContextConfig()

    def build(
        self,
        memories: List[Memory],
        max_items: int = CONTEXT_MAX_ITEMS_DEFAULT,
        max_characters: int = CONTEXT_MAX_CHARACTERS_DEFAULT,
    ) -> str:
        if not memories:
            return ""
        ordered = self._order(memories)[:max_items]
        blocks: List[str] = []
        used = 0
        for index, memory in enumerate(ordered):
            block = self._format(index + 1, memory)
            if used + len(block) > max_characters:
                if not blocks:
                    block = block[: max(0, max_characters - 3)] + "..."
                else:
                    break
            blocks.append(block)
            used += len(block)
        return "\n\n".join(blocks)

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _order(memories: List[Memory]) -> List[Memory]:
        """Top relevance first, then importance desc, then newest first.

        ``memories`` arrive relevance-ranked from the retriever; the first
        entry leads the context, the remainder are sorted by importance and
        recency so important recent facts are not cut off by the budget.
        """
        if len(memories) < 2:
            return list(memories)
        first = memories[0]
        rest = sorted(
            memories[1:],
            key=lambda m: (
                int(m.status is MemoryStatus.ARCHIVED),
                -float(m.importance.score),
                -_created_ts(m),
            ),
        )
        return [first, *rest]

    @staticmethod
    def _format(index: int, memory: Memory) -> str:
        provenance = " | ".join(
            [
                f"memory_id: {memory.memory_id or 'unknown'}",
                f"type: {memory.classification.memory_type.value}",
                f"category: {memory.classification.category}",
                f"importance: {memory.importance.score:.2f}",
                f"confidence: {memory.importance.confidence:.2f}",
                f"source: {memory.classification.source}",
                f"created: {_fmt_timestamp(memory.lifecycle.created_at)}",
                f"updated: {_fmt_timestamp(memory.lifecycle.updated_at)}",
            ]
        )
        superseded_by = (memory.content.structured_data or {}).get("superseded_by")
        if superseded_by:
            provenance += f" | superseded: yes, by {superseded_by}"
        return f"[{index}] {memory.content.text}\n    {provenance}"


__all__ = ["ContextConfig", "MemoryContextBuilder"]