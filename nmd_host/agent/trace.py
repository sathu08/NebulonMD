"""Step 12 — Observability: per-run execution trace for the agent.

The runtime records timing and metadata around its LLM calls, tool calls
and the memory recall it triggers. The trace is returned with the chat
response (``AgentChatData.trace``) so the web console can render a
Debug/Developer panel — nothing here is persisted, and it never carries
credentials or raw memory bodies by default.
"""

from __future__ import annotations

import uuid
from typing import List, Optional

from pydantic import BaseModel, Field


class LLMSpan(BaseModel):
    """One aggregated view of the LLM work in a run (summed across turns)."""

    model: str = ""
    latency_ms: float = 0.0
    tokens: int = 0


class ToolSpan(BaseModel):
    """One tool invocation inside the loop."""

    tool: str
    ok: bool = True
    latency_ms: float = 0.0
    detail: str = ""


class RecallSpan(BaseModel):
    """The memory retrieval step (tool ``recall``): query + outcome."""

    query: str = ""
    memories: int = 0
    latency_ms: float = 0.0


class ExecutionTrace(BaseModel):
    """Everything the runtime observed for one chat run.

    ``llm.latency_ms`` is the cumulative LLM wall-clock across turns;
    ``tokens`` is an approximate token estimate (word counts of the
    prompt + reply — providers that expose usage could refine it later).
    ``recall`` is set whenever the run invoked the ``recall`` tool.
    """

    trace_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    total_ms: float = 0.0
    turns: int = 0
    llm: LLMSpan = Field(default_factory=LLMSpan)
    tools: List[ToolSpan] = Field(default_factory=list)
    recall: Optional[RecallSpan] = None


__all__ = ["ExecutionTrace", "LLMSpan", "RecallSpan", "ToolSpan"]
