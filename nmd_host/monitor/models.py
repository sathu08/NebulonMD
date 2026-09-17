"""nmd_monitor models — Trace / Span / Stats (pydantic, dependency-free).

A ``MonitorTrace`` is the persisted form of one agent run: the LangSmith
``trace`` equivalent. ``MonitorSpan`` is one step inside it (the LangSmith
``run`` equivalent): grounding retrieval, one LLM turn, or one tool call.

``from_execution_trace()`` converts the existing in-memory
``ExecutionTrace`` (``nmd_host/agent/trace.py``) so the agent runtime never
changes shape — the monitor only translates + truncates.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


SpanKind = Literal["llm", "retrieval", "tool", "grounding"]


class MonitorSpan(BaseModel):
    """One step inside a trace."""

    name: str = ""
    kind: SpanKind = "tool"
    ok: bool = True
    latency_ms: float = 0.0
    input: str = ""
    output: str = ""
    error: str = ""


class MonitorTrace(BaseModel):
    """Persisted record of one ``POST /agent/chat`` run."""

    trace_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    thread_id: str = ""
    user_id: str = ""
    project: str = "default"
    request_id: str = "-"
    provider: str = ""
    model: str = ""
    input_text: str = ""
    answer: str = ""
    turns: int = 0
    latency_ms: float = 0.0
    tokens: int = 0
    tools_used: List[str] = Field(default_factory=list)
    recall_memories: int = 0
    ok: bool = True
    error: str = ""
    spans: List[MonitorSpan] = Field(default_factory=list)
    created_at_ms: int = Field(default_factory=lambda: int(time.time() * 1000))

    @classmethod
    def from_execution_trace(
        cls,
        trace: Any,
        *,
        user_id: str = "",
        thread_id: str = "",
        project: str = "default",
        request_id: str = "-",
        provider: str = "",
        model: str = "",
        input_text: str = "",
        answer: str = "",
        max_body_chars: int = 2000,
    ) -> "MonitorTrace":
        """Translate an ``ExecutionTrace`` into a storable ``MonitorTrace``.

        Truncates free text to ``max_body_chars`` (redaction contract: raw
        memory bodies are never persisted, only counts + short excerpts).
        Never raises on odd shapes — defensive for forward-compat.
        """
        def _cut(text: Any) -> str:
            text = str(text or "")
            if max_body_chars > 0 and len(text) > max_body_chars:
                return text[:max_body_chars] + "…[truncated]"
            return text

        spans: List[MonitorSpan] = []
        tools_used: List[str] = []
        recall_memories = 0
        llm_ms = 0.0
        tokens = 0
        turns = 0
        total_ms = 0.0
        trace_id = uuid.uuid4().hex
        try:
            trace_id = str(getattr(trace, "trace_id", trace_id) or trace_id)
            turns = int(getattr(trace, "turns", 0) or 0)
            total_ms = float(getattr(trace, "total_ms", 0.0) or 0.0)
            llm = getattr(trace, "llm", None)
            if llm is not None:
                llm_ms = float(getattr(llm, "latency_ms", 0.0) or 0.0)
                tokens = int(getattr(llm, "tokens", 0) or 0)
                spans.append(
                    MonitorSpan(
                        name="llm",
                        kind="llm",
                        ok=True,
                        latency_ms=llm_ms,
                        input=_cut(f"model={model or getattr(llm, 'model', '')}"),
                        output=_cut(f"tokens~{tokens}"),
                    )
                )
            recall = getattr(trace, "recall", None)
            if recall is not None:
                recall_memories = int(getattr(recall, "memories", 0) or 0)
                spans.append(
                    MonitorSpan(
                        name="recall",
                        kind="retrieval",
                        ok=True,
                        latency_ms=float(getattr(recall, "latency_ms", 0.0) or 0.0),
                        input=_cut(getattr(recall, "query", "")),
                        output=_cut(f"memories={recall_memories}"),
                    )
                )
            for tool in list(getattr(trace, "tools", []) or []):
                name = str(getattr(tool, "tool", "tool") or "tool")
                tools_used.append(name)
                spans.append(
                    MonitorSpan(
                        name=name,
                        kind="tool",
                        ok=bool(getattr(tool, "ok", True)),
                        latency_ms=float(getattr(tool, "latency_ms", 0.0) or 0.0),
                        input=_cut(name),
                        output=_cut(getattr(tool, "detail", "")),
                    )
                )
        except Exception:
            pass
        return cls(
            trace_id=trace_id,
            thread_id=thread_id or trace_id,
            user_id=user_id,
            project=project,
            request_id=request_id,
            provider=provider,
            model=model,
            input_text=_cut(input_text),
            answer=_cut(answer),
            turns=turns,
            latency_ms=total_ms,
            tokens=tokens,
            tools_used=tools_used,
            recall_memories=recall_memories,
            spans=spans,
        )


class MonitorStats(BaseModel):
    """Aggregates for ``GET /monitor/stats``."""

    traces: int = 0
    errors: int = 0
    error_rate: float = 0.0
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    avg_tokens: float = 0.0
    recall_used_rate: float = 0.0
    remember_used_rate: float = 0.0


class MonitorFeedback(BaseModel):
    """Human score attached to a trace (P1).

    ``score`` is 1-5 (5 = perfect, 1 = wrong); ``tag`` is a short label
    (e.g. ``good``, ``bad``, ``hallucination``); ``comment`` is free text.
    """

    trace_id: str = ""
    user_id: str = ""
    score: float = 0.0
    tag: str = ""
    comment: str = ""
    by: str = ""
    created_at_ms: int = Field(default_factory=lambda: int(time.time() * 1000))


__all__ = [
    "MonitorFeedback",
    "MonitorSpan",
    "MonitorStats",
    "MonitorTrace",
    "SpanKind",
]
