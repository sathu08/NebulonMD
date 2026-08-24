"""Bridge: ties the intelligence layer to Step 1 persistence.

``MemoryIntelligence`` runs the decision engine over a conversation and
persists accepted candidates through any ``MemoryRepository``-compatible
store (``NebulonMind`` / ``NebulonMindRepository`` for the real pipeline,
``InMemoryRepository`` for tests).
"""

from __future__ import annotations

from typing import List, Optional, Union

from nmd_host.core.models import Memory
from nmd_host.core.mind import NebulonMind
from nmd_host.core.repository import InMemoryRepository, NebulonMindRepository

from .converter import candidate_to_memory

from .engine import MemoryDecisionEngine
from .extractor import MemoryExtractor
from .schemas import Conversation, MemoryDecision



StoreLike = Union["NebulonMind", InMemoryRepository]

class MemoryIntelligence:
    """Facade: conversation -> decisions -> ``Memory`` objects -> store."""

    def __init__(
        self,
        store: StoreLike,
        user_id: Optional[str] = None,
        extractor: Optional[MemoryExtractor] = None,
    ) -> None:
        self._store = self._normalize(store)
        self._user_id = user_id or getattr(store, "user_id", "user_001")
        self.engine = MemoryDecisionEngine(extractor)

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def decide(self, conversation: Conversation) -> List[MemoryDecision]:
        """Return the engine's verdicts without persisting anything."""
        return self.engine.extract(conversation)

    def process(
        self,
        conversation: Conversation,
        persist: bool = True,
    ) -> List[Memory]:
        """Decide over a conversation and store what should be remembered."""
        decisions = self.engine.extract(conversation)
        memories = [candidate_to_memory(d.candidate, self._user_id) for d in decisions]
        if persist:
            for memory in memories:
                self._store.create(memory)
        return memories

    def process_text(self, text: str, persist: bool = True) -> List[Memory]:
        return self.process(Conversation.from_user_message(text), persist=persist)

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalize(store: StoreLike) -> object:
        if hasattr(store, "create"):
            return store
        if hasattr(store, "store"):
            return NebulonMindRepository(store)
        raise TypeError(
            "store must be a MemoryRepository (has create) or a NebulonMind (has store)"
        )


__all__ = ["MemoryIntelligence"]
