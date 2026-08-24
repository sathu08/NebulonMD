"""Memory decision engine: conversation -> memory decisions.

The engine is pure — no storage, no network. It runs any ``MemoryExtractor``
(rule-based or LLM), drops rejected candidates, and validates what survives.
Persistence is the bridge's job.
"""

from __future__ import annotations

from typing import List, Optional

from .extractor import MemoryExtractor
from .rules import RuleBasedExtractor
from .schemas import Conversation, MemoryDecision

from .validation import validate_decisions


class MemoryDecisionEngine:
    """Converts a conversation into a list of validated memory decisions."""

    def __init__(self, extractor: Optional[MemoryExtractor] = None) -> None:
        self.extractor = extractor if extractor is not None else RuleBasedExtractor()

    def extract(self, conversation: Conversation) -> List[MemoryDecision]:
        decisions = self.extractor.extract(conversation)
        accepted = [d for d in decisions if d.should_remember]
        return validate_decisions(accepted)

    def decide(self, conversation: Conversation) -> List[MemoryDecision]:
        """Full verdict list including rejected candidates (for logging)."""
        return self.extractor.extract(conversation)


__all__ = ["MemoryDecisionEngine"]
