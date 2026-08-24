"""Step 6 — Agent Runtime schemas (conversation protocol)."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from .trace import ExecutionTrace


class AgentMessage(BaseModel):
    """One transcript entry fed to (or produced by) the runtime.

    ``role`` mirrors the LLM chat contract: ``user`` / ``assistant`` /
    ``tool``. ``tool`` messages carry the result of a memory tool call.
    """

    role: Literal["user", "assistant", "tool"]
    content: str


class AgentToolCall(BaseModel):
    """A structured tool invocation produced by the model."""

    tool: str = Field(description="Name of the tool to invoke")
    arguments: dict = Field(
        default_factory=dict, description="Tool-specific arguments"
    )


class AgentChatRequest(BaseModel):
    """Input to ``POST /api/NebulonMind/agent/chat``.

    ``text`` is the current user utterance; ``messages`` optionally carries
    the prior transcript so multi-turn conversations stay coherent. The
    runtime itself is stateless — historical turns must be passed in.
    ``session_id`` / ``conversation_id`` (optional) are stamped onto every
    memory the agent persists, so origin is always traceable.
    """

    text: str = Field(min_length=1, max_length=100_000)
    messages: List[AgentMessage] = Field(
        default_factory=list, description="Prior transcript (optional)"
    )
    session_id: Optional[str] = None
    conversation_id: Optional[str] = None


class AgentToolResult(BaseModel):
    """Outcome of one memory tool call inside the loop."""

    tool: str
    ok: bool
    detail: str


class AgentChatData(BaseModel):
    """Response payload: the final answer plus loop transparency."""

    answer: str
    turns: int = 0
    tool_calls: List[AgentToolResult] = Field(default_factory=list)
    transcript: List[AgentMessage] = Field(
        default_factory=list,
        description=(
            "Full message history produced by this run (user/assistant/tool). "
            "Clients may send it back as ``messages`` on the next call to keep "
            "a multi-turn conversation coherent; the server may persist it "
            "into an agent session."
        ),
    )
    trace: Optional[ExecutionTrace] = Field(
        default=None,
        description=(
            "Step 12 observability: timing/metadata for this run (LLM, tool "
            "calls, recall, total latency). Rendered by the console's Debug "
            "mode; omitted when observability is disabled."
        ),
    )


__all__ = [
    "AgentChatData",
    "AgentChatRequest",
    "AgentMessage",
    "AgentToolCall",
    "AgentToolResult",
    "ExecutionTrace",
]