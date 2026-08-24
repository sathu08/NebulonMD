"""Phase 4 — request/response schemas for the NebulonMind API.

Pure presentation models. Nothing here redefines the ``Memory`` domain
model: create/update accept the same fields the core model already
validates, and Step 2 / Step 3 objects (``Conversation``, ``MemoryDecision``)
are reused as-is, so the API never invents its own decision, retrieval or
context semantics (Phases 1–3 stay the single source of behaviour).
"""

from __future__ import annotations

from typing import Generic, List, Literal, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..core.models import (
    Classification,
    Importance,
    Lifecycle,
    Memory,
    MemoryContent,
    MemoryStatus,
    Relationship,
)
from ..agent.schemas import AgentChatData
from ..intelligence.schemas import Conversation, MemoryDecision

T = TypeVar("T")


class Envelope(BaseModel, Generic[T]):
    """Standard response wrapper: ``{success, message, data}``.

    Every success response uses this shape; errors use the same envelope
    with ``success=false`` and no error internals leaked into ``message``.
    """

    success: bool = True
    message: str = "ok"
    data: T


# ---------------------------------------------------------------------- #
# Memory API (4.2)                                                       #
# ---------------------------------------------------------------------- #

_MEMORY_EXAMPLE: dict = {
    "user_id": "user_001",
    "memory_id": None,
    "content": {
        "text": "My name is Sathya",
        "summary": "User's name is Sathya",
    },
    "classification": {
        "memory_type": "long_term",
        "category": "identity",
        "source": "conversation",
    },
    "importance": {"score": 0.99, "confidence": 1.0, "priority": "high"},
    "lifecycle": {"retention_policy": "permanent"},
    "status": "active",
    "entities": ["Sathya"],
    "relationships": [{"source": "Sathya", "target": "Python", "relation": "HAS_SKILL"}],
}


class MemoryCreate(Memory):
    """Create payload: the full ``Memory`` model, unchanged semantics.

    ``memory_id`` is optional — the store assigns one when omitted. The
    server pins the memory to the ``user_id`` query parameter if the body
    carries a different value.
    """

    model_config = ConfigDict(
        json_schema_extra={"examples": [{k: v for k, v in _MEMORY_EXAMPLE.items() if k != "memory_id"}]}
    )


class MemoryUpdate(BaseModel):
    """Partial update payload: only the provided fields are replaced.

    Absent fields are left untouched (same merge semantics the store
    already applies to a raw dict, now with schema validation).
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"status": "archived", "importance": {"score": 0.9, "confidence": 0.8}}
            ]
        },
    )

    content: Optional[MemoryContent] = None
    classification: Optional[Classification] = None
    importance: Optional[Importance] = None
    lifecycle: Optional[Lifecycle] = None
    status: Optional[MemoryStatus] = None
    entities: Optional[List[str]] = None
    relationships: Optional[List[Relationship]] = None


class MemoryData(BaseModel):
    """A single memory (create / get / update responses)."""

    memory: Memory


class DeleteData(BaseModel):
    """Delete confirmation."""

    memory_id: str


# ---------------------------------------------------------------------- #
# Service (4.12)                                                         #
# ---------------------------------------------------------------------- #


class HealthData(BaseModel):
    """Health-check payload: service metadata + backend reachability.

    No backend credentials or connection settings are ever included —
    NebulonDB credentials stay inside ``NebulonMind`` (the API authenticates
    only when the optional ``NMD_API_AUTH_TOKEN`` is configured).
    """

    service: str = "NebulonMind"
    version: str = ""
    provider: str = ""
    backend: Literal["up", "down", "in-memory"] = "down"
    minds: int = 0


# ---------------------------------------------------------------------- #
# Username registry (explicit registration)                              #
# ---------------------------------------------------------------------- #


class UserCreateRequest(BaseModel):
    """Input to ``POST /api/NebulonMind/user/create_user``.

    The client sends only a ``username``; the system generates the opaque
    ``unique_id`` (``user_<hex>``). Idempotent: re-calling with an existing
    username returns the stored id with ``created=False``. No password.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={"examples": [{"username": "alice"}]},
    )

    username: str = Field(..., min_length=1, description="The username to register")


class UserData(BaseModel):
    """A registered user: the opaque ``user_id`` plus whether it was created."""

    username: str
    user_id: str
    created: bool = False


# ---------------------------------------------------------------------- #
# Recall API (4.3)                                                       #
# ---------------------------------------------------------------------- #


class SearchData(BaseModel):
    """Result of a lifecycle-ranked recall (Step 3 pipeline)."""

    query: str
    top_k: int
    expand: bool
    results: List[Memory]


# ---------------------------------------------------------------------- #
# Context API (4.4)                                                      #
# ---------------------------------------------------------------------- #


class ContextData(BaseModel):
    """Bounded, provenance-carrying LLM context built from retrieved memories."""

    query: str
    context: str
    memory_ids: List[str]


# ---------------------------------------------------------------------- #
# Relationship API (4.6)                                                 #
# ---------------------------------------------------------------------- #


class RelateData(BaseModel):
    """Confirmation of a memory → entity link."""

    memory_id: str
    entity: str
    relation: str


# ---------------------------------------------------------------------- #
# Intelligence API (4.5)                                                 #
# ---------------------------------------------------------------------- #


class IntelligenceInput(BaseModel):
    """Shared input for Step 2 endpoints: ``text`` or a full ``conversation``.

    The conversation model is Step 2's own ``Conversation`` — multi-turn
    input is identical to what ``MemoryDecisionEngine`` consumes offline.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    text: Optional[str] = Field(
        None, description="Convenience: a single user message, treated as one turn"
    )
    conversation: Optional[Conversation] = Field(
        None, description="Full multi-turn conversation (Step 2 input schema)"
    )

    @model_validator(mode="after")
    def _require_input(self) -> "IntelligenceInput":
        if not self.text and (self.conversation is None or not self.conversation.turns):
            raise ValueError("provide 'text' or a non-empty 'conversation'")
        return self

    def to_conversation(self) -> Conversation:
        if self.conversation is not None and self.conversation.turns:
            return self.conversation
        return Conversation.from_user_message(str(self.text))


class IntelligenceDecideRequest(IntelligenceInput):
    """Decision-only request: Step 2 verdicts, nothing persisted."""

    include_rejected: bool = Field(
        False,
        description=(
            "Also return candidates the engine deliberately rejected "
            "(weak-signal decisions), for logging/monitoring"
        ),
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"text": "My name is Sathya. I work with Python."}]
        }
    )


class IntelligenceProcessRequest(IntelligenceInput):
    """Process request: decisions → Step 3 lifecycle gate → store."""

    persist: bool = Field(
        True,
        description=(
            "When true, STORE-approved memories are persisted; "
            "DUPLICATE / EXPIRED / INVALID candidates are reported, never stored"
        ),
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"text": "My name is Sathya. I like hiking."}]
        }
    )


class DecideData(BaseModel):
    """Step 2 verdicts (accepted always, rejected optional)."""

    decisions: List[MemoryDecision]
    rejected: Optional[List[MemoryDecision]] = None


class IngestReport(BaseModel):
    """What the Step 3 gate decided for one candidate about to be stored."""

    action: Literal["STORE", "DUPLICATE", "EXPIRED", "INVALID", "SUPERSEDE"]
    memory_id: Optional[str] = None
    existing_memory_id: Optional[str] = None
    superseded_memory_ids: List[str] = Field(default_factory=list)
    reason: str = ""


class ProcessData(BaseModel):
    """Step 2 → Step 3 outcome: decisions + per-candidate gate reports."""

    decisions: List[MemoryDecision]
    ingestions: List[IngestReport]


# ---------------------------------------------------------------------- #
# LLM status (Step 6)                                                    #
# ---------------------------------------------------------------------- #


class LLMStatusData(BaseModel):
    """Configured LLM provider status — never carries credentials.

    ``configured`` means the provider constructed successfully from the
    environment; ``error`` is a sanitized reason (key values redacted) when
    it did not. No API keys or connection secrets are ever exposed here.
    """

    provider: str = ""
    configured: bool = False
    model: Optional[str] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------- #
# Agent sessions (Step 6.9/6.12)                                         #
# ---------------------------------------------------------------------- #


class AgentSessionCreate(BaseModel):
    """Input to ``POST /api/NebulonMind/agent/session``.

    ``metadata`` is optional free-form context the client wants associated
    with the session (e.g. originating app, model, language); it is returned
    verbatim on reads.
    """

    metadata: dict = Field(default_factory=dict, description="Optional session metadata")


class AgentSessionData(BaseModel):
    """One logical agent session (runtime, in-memory)."""

    session_id: str
    user_id: str
    created_at: float
    last_activity: float
    status: str
    metadata: dict = Field(default_factory=dict)
    message_count: int = 0


class AgentSessionListData(BaseModel):
    """Active sessions for a user."""

    sessions: List[AgentSessionData] = Field(default_factory=list)


# ---------------------------------------------------------------------- #
# Evaluation (Step 13)                                                   #
# ---------------------------------------------------------------------- #


class EvaluationRunRequest(BaseModel):
    """Input to ``POST /api/NebulonMind/evaluation/run``.

    Runs the Step 6 agent over the bundled benchmark dataset (or an inline
    ``items`` list) and reports retrieval / tool-selection / answer /
    hallucination / latency / token metrics. ``seed`` pre-stores each
    dataset fact so a fresh user has something to recall.
    """

    items: Optional[List[dict]] = Field(
        None,
        description="Inline items override the bundled dataset when provided",
    )
    dataset: str = Field(
        "memory_test_v1", description="Name of the bundled dataset to run"
    )
    top_k: int = Field(5, ge=1, le=50, description="Memories recalled per question")
    max_items: int = Field(
        100, ge=1, le=500, description="Bounded number of questions to run"
    )
    seed: bool = Field(True, description="Pre-store dataset facts before running")


class EvaluationMetrics(BaseModel):
    """Aggregate metric report for one run."""

    questions: int = 0
    tool_selection_accuracy: float = 0.0
    retrieval_accuracy: float = 0.0
    answer_correctness: float = 0.0
    hallucination_rate: float = 0.0
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    avg_tokens: float = 0.0


class EvaluationItemData(BaseModel):
    """One question's outcome (retrieval, tool choice, answer, cost)."""

    id: str = ""
    question: str = ""
    type: str = "known"
    expected_tool: str = "none"
    expected_answer: str = ""
    expected_memory: str = ""
    actual_tools: List[str] = Field(default_factory=list)
    retrieval_ok: Optional[bool] = None
    tool_ok: Optional[bool] = None
    answer: str = ""
    answer_ok: bool = False
    declines_answer: bool = False
    latency_ms: float = 0.0
    tokens: int = 0
    error: Optional[str] = None


class EvaluationData(BaseModel):
    """Evaluation run payload: metrics + per-question results."""

    dataset: str = ""
    metrics: EvaluationMetrics
    results: List[EvaluationItemData] = Field(default_factory=list)


class EvaluationDatasetData(BaseModel):
    """The bundled benchmark dataset (name/description/items)."""

    name: str = ""
    description: str = ""
    items: List[dict] = Field(default_factory=list)


# ---------------------------------------------------------------------- #
# Background Agents (Step 14)                                             #
# ---------------------------------------------------------------------- #


class BackgroundMemoryRunRequest(BaseModel):
    """Input to ``POST /api/NebulonMind/background/memory/run``.

    ``mode="review"`` (default) reports the existing ``MemoryConsolidator``
    decisions without writing; ``mode="consolidate"`` additionally persists
    one merged memory per MERGE decision (sources are never deleted).
    """

    mode: Literal["review", "consolidate"] = "review"


class BackgroundTaskRunRequest(BaseModel):
    """Input to ``POST /api/NebulonMind/background/task/run``.

    ``days`` is the look-back window for the weekly summary (default 7).
    """

    days: int = Field(7, ge=1, le=365)


class BackgroundAgentReport(BaseModel):
    """A background agent's run report (agent-specific dict payload)."""

    agent: str = ""
    user_id: str = ""
    run_at: float = 0.0
    data: dict = Field(default_factory=dict)


class BackgroundJobStatus(BaseModel):
    """One scheduled background job's state (in-memory, no persistence)."""

    job_id: str = ""
    schedule: str = ""
    last_run: Optional[float] = None
    last_error: Optional[str] = None
    runs: int = 0
    last_result: Optional[dict] = None


class BackgroundSchedulerStatus(BaseModel):
    """The background scheduler snapshot for ``/background/status``."""

    running: bool = False
    poll_seconds: float = 30.0
    jobs: List[BackgroundJobStatus] = Field(default_factory=list)


# ---------------------------------------------------------------------- #
# Typed envelopes                                                        #
# ---------------------------------------------------------------------- #

HealthEnvelope = Envelope[HealthData]
UserEnvelope = Envelope[UserData]
MemoryEnvelope = Envelope[MemoryData]
DeleteEnvelope = Envelope[DeleteData]
SearchEnvelope = Envelope[SearchData]
ContextEnvelope = Envelope[ContextData]
RelateEnvelope = Envelope[RelateData]
DecideEnvelope = Envelope[DecideData]
ProcessEnvelope = Envelope[ProcessData]
AgentChatEnvelope = Envelope[AgentChatData]
AgentSessionEnvelope = Envelope[AgentSessionData]
AgentSessionListEnvelope = Envelope[AgentSessionListData]
LLMStatusEnvelope = Envelope[LLMStatusData]
EvaluationRunEnvelope = Envelope[EvaluationData]
EvaluationDatasetEnvelope = Envelope[EvaluationDatasetData]
BackgroundAgentEnvelope = Envelope[BackgroundAgentReport]
BackgroundStatusEnvelope = Envelope[BackgroundSchedulerStatus]

__all__ = [
    "AgentChatData",
    "AgentChatEnvelope",
    "AgentSessionCreate",
    "AgentSessionData",
    "AgentSessionEnvelope",
    "AgentSessionListData",
    "AgentSessionListEnvelope",
    "BackgroundAgentEnvelope",
    "BackgroundAgentReport",
    "BackgroundJobStatus",
    "BackgroundMemoryRunRequest",
    "BackgroundSchedulerStatus",
    "BackgroundStatusEnvelope",
    "BackgroundTaskRunRequest",
    "ContextData",
    "DecideData",
    "DecideEnvelope",
    "DeleteData",
    "DeleteEnvelope",
    "Envelope",
    "EvaluationData",
    "EvaluationDatasetData",
    "EvaluationDatasetEnvelope",
    "EvaluationItemData",
    "EvaluationMetrics",
    "EvaluationRunEnvelope",
    "EvaluationRunRequest",
    "HealthData",
    "HealthEnvelope",
    "IngestReport",
    "IntelligenceDecideRequest",
    "IntelligenceInput",
    "IntelligenceProcessRequest",
    "LLMStatusData",
    "LLMStatusEnvelope",
    "MemoryCreate",
    "MemoryData",
    "MemoryEnvelope",
    "MemoryUpdate",
    "ProcessData",
    "ProcessEnvelope",
    "RelateData",
    "RelateEnvelope",
    "SearchData",
    "SearchEnvelope",
    "UserCreateRequest",
    "UserData",
    "UserEnvelope",
]