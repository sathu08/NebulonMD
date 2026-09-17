"""Memory domain models (pydantic)."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class MemoryType(str, Enum):
    """WHAT the memory is — the nature of the content.

    Orthogonal to ``RetentionPolicy`` (how long to keep it): a long-lived
    piece of knowledge is ``SEMANTIC``, while a one-off event is
    ``EPISODIC`` regardless of how long it is retained.
    """

    WORKING = "working"
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    KNOWLEDGE = "knowledge"
    DOC = "doc"


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RetentionPolicy(str, Enum):
    """HOW LONG the memory is kept.

    A lifecycle concern, independent of ``MemoryType``: "My name is Sathya"
    can be ``semantic`` + ``permanent``, while "I am debugging HNSW today"
    can be ``working`` + ``session``.
    """

    TEMPORARY = "temporary"
    SESSION = "session"
    PERMANENT = "permanent"


class MemoryStatus(str, Enum):
    """Health/handling of a memory (Step 3 lifecycle concern).

    ``ACTIVE`` means the memory participates in retrieval; ``ARCHIVED``
    means it is deliberately frozen out of retrieval while retaining its
    historical value. Both are orthogonal to ``RetentionPolicy``.
    """

    ACTIVE = "active"
    ARCHIVED = "archived"


class MemoryContent(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    text: str = Field(min_length=1)
    summary: Optional[str] = None
    structured_data: Optional[Dict[str, Any]] = None


class Classification(BaseModel):
    memory_type: MemoryType = MemoryType.SEMANTIC
    category: str = "general"
    source: str = "conversation"
    lang: str = "en"


class Provenance(BaseModel):
    """Where a memory came from (Step 6: agents need origin + session context).

    ``source_type`` is the producing path (``conversation`` / ``agent`` /
    ``api`` / ``import``); ``created_by`` names the creator (e.g. an agent
    runtime); ``session_id`` / ``conversation_id`` tie the memory back to
    the session/conversation that produced it. All optional — nothing here
    is required to store a memory.
    """

    source_type: str = "conversation"
    created_by: Optional[str] = None
    session_id: Optional[str] = None
    conversation_id: Optional[str] = None


class Importance(BaseModel):
    score: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    priority: Priority = Priority.MEDIUM


class Lifecycle(BaseModel):
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    last_accessed: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    retention_policy: RetentionPolicy = RetentionPolicy.TEMPORARY


class Relationship(BaseModel):
    """Directed relationship between two entity labels, e.g. Sathya -HAS_SKILL-> Python.

    ``source_memory_id`` is provenance: the memory whose content supports
    this claim (e.g. "I am learning Python" backs ``Sathya →HAS_SKILL→ Python``).
    It is stamped automatically by the store layer and persisted in the
    truth document, so the graph (a derived representation) can always be
    re-grounded or questioned via its supporting memory.
    """

    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    relation: str = "related"
    source_memory_id: Optional[str] = None


class Memory(BaseModel):
    memory_id: Optional[str] = None
    user_id: str = Field(min_length=1)
    content: MemoryContent
    classification: Classification = Field(default_factory=Classification)
    provenance: Provenance = Field(default_factory=Provenance)
    importance: Importance = Field(default_factory=Importance)
    lifecycle: Lifecycle = Field(default_factory=Lifecycle)
    status: MemoryStatus = MemoryStatus.ACTIVE
    entities: List[str] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)

    def to_truth_doc(self) -> Dict[str, Any]:
        """Full Memory JSON minus embedding, ready for the COSMOS truth store.

        The truth store JSON-encodes the whole doc into the COSMOS ``text``
        column (the engine only persists ``{text, lang, type, created_at}``).
        """
        doc = self.model_dump(mode="json")
        # Add top-level lang/type for NebulonDB dashboard visibility
        # Use NebulonDB document types (chat/doc/other) for 'type' field
        classification = doc.get("classification", {})
        if classification.get("lang"):
            doc["lang"] = classification["lang"]
        memory_type = classification.get("memory_type", "semantic")
        # Display type mirrors the doc_type sent to NebulonDB verbatim.
        doc["type"] = "doc" if memory_type == "doc" else "chat_memory"
        return doc

    @classmethod
    def from_truth_doc(cls, doc: Dict[str, Any]) -> "Memory":
        return cls.model_validate(dict(doc))

    def stamp_provenance(self) -> "Memory":
        """Attach this memory's id to every relationship it supports.

        Called by the store layer after ``memory_id`` is resolved, so the
        truth document keeps the provenance record (the graph is derived
        from it and can be rebuilt).
        """
        if self.memory_id:
            for relationship in self.relationships:
                relationship.source_memory_id = self.memory_id
        return self

class FeedbackRequest(BaseModel):
    """Feedback submission payload."""
    trace_id: str
    score: float = 0.0
    tag: str = ""
    comment: str = ""