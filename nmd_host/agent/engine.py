"""Step 6 — Agent Runtime: stateless tool-calling loop over the memory tools.

One ``AgentRuntime`` is built per request (never cached) — the process stays
stateless and any persistent state (memories, tool results) lives in
NebulonDB through the toolkit. The loop:

    system prompt + transcript (+ tool results) → LLM reply
    reply is a tool-call JSON  → execute tool → back to the LLM
    reply is plain text       → final answer

Tool calls are requested from the model in plain JSON (see
``prompt.DEFAULT_SYSTEM_PROMPT``), so no schema-constrained ``structured()``
call is needed — the runtime tolerates malformed replies by treating them as
the final answer.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Dict, List, Optional

from nmd_host.intelligence.providers import LLMProvider
from nmd_host.utils.constants import MAX_TURN_RESULT_CHARS

from .config import AgentConfig
from .prompt import build_system_prompt
from .schemas import (
    AgentChatData,
    AgentMessage,
    AgentToolCall,
    AgentToolResult,
)
from .trace import ExecutionTrace, LLMSpan, RecallSpan, ToolSpan

logger = logging.getLogger("nmd_host.agent.engine")


def _parse_tool_call(raw: str) -> Optional[AgentToolCall]:
    """Tolerantly extract a tool-call JSON object from an LLM reply."""
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    first, last = text.find("{"), text.rfind("}")
    if first < 0 or last <= first:
        return None
    try:
        data = json.loads(text[first: last + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("tool"), str):
        return None
    arguments = data.get("arguments")
    if not isinstance(arguments, dict):
        arguments = {}
    return AgentToolCall(tool=data["tool"].strip().lower(), arguments=arguments)


class AgentRuntime:
    """Tool-calling chat loop bound by ``AgentConfig.max_turns``."""

    def __init__(
        self,
        llm: LLMProvider,
        tools: List,
        config: Optional[AgentConfig] = None,
        trace: Optional[ExecutionTrace] = None,
    ) -> None:
        self._llm = llm
        self._tools = {tool.name: tool for tool in tools}
        self._config = config or AgentConfig.from_env()
        self._default_trace = trace

    def chat(
        self,
        text: str,
        messages: Optional[List[AgentMessage]] = None,
        trace: Optional[ExecutionTrace] = None,
    ) -> AgentChatData:
        """Run one conversation: prior transcript + new utterance → answer.

        Observability (Step 12): unless ``trace`` is supplied, a fresh
        ``ExecutionTrace`` is created for this run and attached to the
        returned ``AgentChatData``. Timing spans cover every LLM call and
        tool invocation; the ``recall`` tool records the query and how many
        memories were surfaced.
        """
        trace = trace or self._default_trace or ExecutionTrace()
        self._bind_trace(trace)
        user_text = text
        prior: List[AgentMessage] = list(messages or [])
        loop: List[AgentMessage] = []
        tool_results: List[AgentToolResult] = []
        system = build_system_prompt(
            {name: tool.description for name, tool in self._tools.items()},
            override=self._config.system_prompt,
        )
        model_name = getattr(self._llm, "model", None) or ""
        started = time.monotonic()
        for _ in range(self._config.max_turns):
            prompt = "\n".join(
                [
                    *(f"{m.role}: {m.content}" for m in [*prior, *loop]),
                    f"User: {user_text}",
                ]
            )
            llm_started = time.monotonic()
            reply = self._llm.complete(
                prompt,
                system=system,
                temperature=self._config.temperature,
            )
            llm_ms = (time.monotonic() - llm_started) * 1000.0
            text = reply.text if hasattr(reply, "text") else reply
            trace.llm = LLMSpan(
                model=model_name,
                latency_ms=trace.llm.latency_ms + llm_ms,
                tokens=trace.llm.tokens
                + len(prompt.split()) + len(str(text).split()),
            )
            call = _parse_tool_call(text)
            if call is None or call.tool not in self._tools:
                loop.append(AgentMessage(role="assistant", content=text))
                trace.turns = len(tool_results) + 1
                trace.total_ms = (time.monotonic() - started) * 1000.0
                return AgentChatData(
                    answer=text,
                    turns=len(tool_results) + 1,
                    tool_calls=tool_results,
                    transcript=_full_transcript(prior, user_text, loop),
                    trace=trace,
                )
            if call.tool == "recall":
                trace.recall = RecallSpan(
                    query=str(call.arguments.get("query", ""))
                )
            tool_started = time.monotonic()
            try:
                detail = self._tools[call.tool].run(call.arguments)
            except Exception as exc:  # defensive: one bad tool must not hang the loop
                logger.warning(
                    "agent tool %s failed: %s", call.tool, exc
                )
                detail = f"error: tool {call.tool} failed ({type(exc).__name__})"
            tool_ms = (time.monotonic() - tool_started) * 1000.0
            ok = not detail.startswith("error")
            trace.tools.append(
                ToolSpan(
                    tool=call.tool,
                    ok=ok,
                    latency_ms=tool_ms,
                    detail=detail[:200],
                )
            )
            if trace.recall is not None:
                trace.recall.latency_ms = tool_ms
            tool_results.append(
                AgentToolResult(tool=call.tool, ok=ok, detail=detail)
            )
            loop.append(AgentMessage(role="assistant", content=text))
            loop.append(
                AgentMessage(role="tool", content=detail[:MAX_TURN_RESULT_CHARS])
            )
        loop.append(
            AgentMessage(
                role="assistant",
                content="I could not finish answering within the tool-call limit.",
            )
        )
        trace.turns = len(tool_results)
        trace.total_ms = (time.monotonic() - started) * 1000.0
        return AgentChatData(
            answer=loop[-1].content,
            turns=len(tool_results),
            tool_calls=tool_results,
            transcript=_full_transcript(prior, user_text, loop),
            trace=trace,
        )

    def _bind_trace(self, trace: ExecutionTrace) -> None:
        """Give trace-aware tools a handle to the current run's trace.

        Tools opt in by exposing ``attach_trace(trace)`` (e.g. ``RecallTool``
        records how many memories were retrieved). Tools without the hook are
        untouched, keeping the ``AgentTool`` protocol backward compatible.
        """
        for tool in self._tools.values():
            attach = getattr(tool, "attach_trace", None)
            if callable(attach):
                attach(trace)


def _full_transcript(
    prior: List[AgentMessage],
    user_text: str,
    loop: List[AgentMessage],
) -> List[AgentMessage]:
    """Rebuild the complete clean conversation for session persistence.

    ``prior`` is the history the client (or session) supplied; ``loop``
    holds only the messages this run produced (assistant replies and tool
    results, never the ``User:``/``Available tools:`` prompt scaffolding).
    The result is the prior history followed by the new user message and the
    loop's assistant/tool messages — exactly what should be sent back as
    ``messages`` on the next turn (and what the session manager persists).
    """
    return [*prior, AgentMessage(role="user", content=user_text), *loop]


__all__ = ["AgentRuntime", "MAX_TURN_RESULT_CHARS", "_parse_tool_call"]