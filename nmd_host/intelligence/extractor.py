"""The extractor contract: any extractor (rule-based or LLM) returns the
same structured shape, so the engine never knows where a decision came from.
"""

from __future__ import annotations

from typing import List, Protocol

from .schemas import Conversation, MemoryDecision


class MemoryExtractor(Protocol):
    """Common contract for rule-based and LLM-based extraction.

    Implementations must be pure (no I/O): the engine decides, the bridge
    persists.
    """

    def extract(self, conversation: Conversation) -> List[MemoryDecision]:
        ...


__all__ = ["MemoryExtractor"]
