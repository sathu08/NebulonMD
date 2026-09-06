"""LLM-based extractor: a ``MemoryExtractor`` implementation backed by any
``LLMProvider`` (OpenAI / Claude / Gemini / Qwen / Ollama — swappable).

The LLM only fills the same decision schema the rule-based extractor produces;
it never touches storage. Output passes through the validation layer before
the engine will accept it.
"""

from __future__ import annotations

import logging
from typing import List

from .providers import LLMProvider
from .schemas import (
    Conversation,
    MemoryCategory,
    MemoryDecision,
    MemoryDecisionList,
)

from .validation import sanitize_decision

logger = logging.getLogger("nmd_host.intelligence.llm_extractor")


_CATEGORIES = ", ".join(c.value for c in MemoryCategory)

_SYSTEM = (
    "You are a memory extraction system for an AI assistant. "
    "Convert the conversation into structured memory candidates. "
    "Only return JSON matching the requested schema, nothing else."
)

_PROMPT_TEMPLATE = """Extract what should be remembered from this conversation.

Rules:
- Memory = durable information about the user: identity, preferences, skills, goals, projects, facts, events, tasks, opinions.
- Skip chit-chat, greetings, one-off questions, and anything transient.
- "text" is the exact sentence(s) from the conversation to remember.
- "summary" is a short neutral paraphrase (optional).
- "category" must be one of: {categories}.
- "importance" and "confidence" are floats in [0, 1].
- "entities" are proper nouns and technical terms (e.g. Python, Sathya).
- "relationships" are triples of source -> relation -> target, e.g.
  {{"source": "Sathya", "target": "Python", "relation": "HAS_SKILL"}}.
- "should_remember": false means this should NOT be stored.
- "reason": one short phrase explaining the decision.

Return JSON of shape:
{{"decisions": [{{"candidate": {{"text": "...", "summary": "...", "category": "...",
"importance": 0.8, "confidence": 0.9, "entities": [], "relationships": [],
"subject": "..."}}, "should_remember": true, "reason": "..."}}]}}

Conversation:
{turns}
"""


class LLMExtractor:
    """Structured extraction via an LLM provider."""

    def __init__(
        self,
        provider: LLMProvider,
        min_confidence: float = 0.0,
        max_decisions: int = 20,
    ) -> None:
        self.provider = provider
        self.min_confidence = min_confidence
        self.max_decisions = max_decisions

    def extract(self, conversation: Conversation) -> List[MemoryDecision]:
        prompt = _PROMPT_TEMPLATE.format(
            categories=_CATEGORIES,
            turns=_render_turns(conversation),
        )
        raw = self.provider.structured(prompt, MemoryDecisionList)
        items = raw.get("decisions", []) if isinstance(raw, dict) else []
        decisions: List[MemoryDecision] = []
        for item in items:
            sanitized = sanitize_decision(item)
            if sanitized is not None:
                decisions.append(sanitized)
        if self.min_confidence > 0.0:
            decisions = [
                d for d in decisions if d.candidate.confidence >= self.min_confidence
            ]
        _stamp_provenance(decisions, conversation)
        result = decisions[: self.max_decisions]
        if not result:
            logger.warning(
                "LLM extraction returned no valid decisions. Raw response: %s. Prompt tokens: ~%d",
                str(raw)[:500],
                len(prompt) // 4,
            )
        return result


def _stamp_provenance(
    decisions: List[MemoryDecision], conversation: Conversation
) -> None:
    """Tie accepted candidates back to the producing session/conversation."""
    session_id = conversation.session_id
    conversation_id = conversation.conversation_id
    if not session_id and not conversation_id:
        return
    from ..core.models import Provenance

    for decision in decisions:
        candidate = decision.candidate
        if candidate.provenance is None:
            candidate.provenance = Provenance(source_type="conversation")
        if session_id:
            candidate.provenance.session_id = session_id
        if conversation_id:
            candidate.provenance.conversation_id = conversation_id


def _render_turns(conversation: Conversation) -> str:
    lines = []
    for i, turn in enumerate(conversation.turns, start=1):
        lines.append(f"{turn.role}: {turn.content}")
    return "\n".join(lines)


__all__ = ["LLMExtractor"]
