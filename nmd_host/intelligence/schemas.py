"""Step 2 decision schemas: structured output describing what should be remembered.

These models are pure decisions — nothing is persisted through them. The
``converter`` maps accepted ``MemoryCandidate``s into the Step 1 ``Memory``
model that NebulonMind stores.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from nmd_host.core.models import Provenance, Relationship


Role = Literal["user", "assistant", "system"]

class MemoryCategory(str, Enum):
    IDENTITY = "identity"
    PREFERENCE = "preference"
    SKILL = "skill"
    GOAL = "goal"
    PROJECT = "project"
    FACT = "fact"
    EVENT = "event"
    TASK = "task"
    OPINION = "opinion"
    KNOWLEDGE = "knowledge"


class Turn(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    role: Role = "user"
    content: str = Field(min_length=1)


class Conversation(BaseModel):
    """A (possibly multi-turn) conversation the engine can decide over.

    ``session_id`` / ``conversation_id`` (optional) let callers tie any
    extracted candidates back to the session/conversation that produced
    them — stamped onto each accepted memory's ``Provenance``.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    turns: List[Turn] = Field(default_factory=list)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    conversation_id: Optional[str] = None

    @classmethod
    def from_user_message(cls, text: str) -> "Conversation":
        return cls(turns=[Turn(role="user", content=text)])

    @classmethod
    def from_text(cls, text: str, role: Role = "user") -> "Conversation":
        """Split raw text into sentence-sized turns (one turn per line)."""
        turns = [
            Turn(role=role, content=line)
            for line in text.splitlines()
            if line.strip()
        ]
        return cls(turns=turns)


class MemoryCandidate(BaseModel):
    """A single thing that could be remembered, fully classified and scored."""

    model_config = ConfigDict(str_strip_whitespace=True)

    text: str = Field(min_length=1)
    summary: Optional[str] = None
    category: MemoryCategory = MemoryCategory.FACT
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    entities: List[str] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)
    subject: Optional[str] = None
    reasoning: Optional[str] = None
    provenance: Optional[Provenance] = None
    slots: Optional[Dict[str, Any]] = None


class MemoryDecision(BaseModel):
    """The engine's verdict for one candidate: remember or not, and why."""

    candidate: MemoryCandidate
    should_remember: bool = True
    reason: Optional[str] = None


class MemoryDecisionList(BaseModel):
    """Batch output shape used by LLM providers (JSON mode)."""

    decisions: List[MemoryDecision] = Field(default_factory=list)


__all__ = [
    "Conversation",
    "MemoryCandidate",
    "MemoryCategory",
    "MemoryDecision",
    "MemoryDecisionList",
    "Role",
    "Turn",
]
