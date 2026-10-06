"""Step 6 — Agent Runtime: stateless tool-calling loop over the memory tools.

One ``AgentRuntime`` is built per request (never cached) — the process stays
stateless and any persistent state (memories, tool results) lives in
NebulonDB through the toolkit. The loop:

    system prompt + transcript (+ tool results) → LLM reply
    reply is a tool-call JSON  → execute tool → back to the LLM
    reply is plain text       → fallback extraction → execute remember tool if needed → back to LLM
    reply is plain text (no extraction) → final answer

Tool calls are requested from the model in plain JSON (see
``prompt.DEFAULT_SYSTEM_PROMPT``), so no schema-constrained ``structured()``
call is needed — the runtime tolerates malformed replies by treating them as
the final answer. If the LLM doesn't call tools, a fallback rule-based
extractor runs on the user's message to auto-store facts.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Dict, List, Optional

from nmd_host.intelligence.providers import LLMProvider
from nmd_host.intelligence.rules import RuleBasedExtractor
from nmd_host.intelligence.schemas import Conversation
from nmd_host.utils.constants import MAX_TURN_RESULT_CHARS

from .config import AgentConfig
from .prompt import build_system_prompt
from .router import route_question
from .schemas import (
    AgentChatData,
    AgentMessage,
    AgentToolCall,
    AgentToolResult,
)
from .trace import ExecutionTrace, LLMSpan, RecallSpan, ToolSpan

logger = logging.getLogger("nmd_host.agent.engine")


def _unwrap_answer(raw: str) -> str:
    """Unwrap a final answer mistakenly wrapped as ``{"answer": "..."}``.

    The tool-call protocol teaches models to reply in JSON, and some models
    (especially reasoning ones) over-apply it to the final answer too. A
    reply that is a JSON object with a string ``answer`` key and no ``tool``
    key is the answer itself, not a tool call — return the inner text so the
    console never shows raw ``{"answer": "..."}`` to the user.
    """
    text = (raw or "").strip()
    stripped = text
    if stripped.startswith("```"):
        stripped = stripped.strip("`").strip()
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
    if not (stripped.startswith("{") and stripped.endswith("}")):
        return raw
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        return raw
    if not isinstance(data, dict) or "tool" in data:
        return raw
    answer = data.get("answer")
    if isinstance(answer, str) and answer.strip():
        return answer.strip()
    return raw


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
        # Fallback rule-based extractor for auto-storing facts when LLM doesn't call tools
        self._fallback_extractor = RuleBasedExtractor(min_strength=0.55)

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
        fallback_used = False
        # Step 6.14 — deterministic intent router (eval suggestion #1):
        # memory questions run ``recall`` BEFORE the first LLM turn, so tool
        # selection no longer depends on model variance ("What is my name?"
        # → recall, even if the model would have answered from nothing).
        # Only ``recall`` is ever forced — never ``remember`` — so questions
        # are never auto-stored as memories.
        routed_intent = (
            route_question(user_text) if self._config.router_enabled else "none"
        )
        if routed_intent == "recall" and "recall" in self._tools:
            forced_started = time.monotonic()
            try:
                forced_detail = self._tools["recall"].run({"query": user_text})
            except Exception as exc:  # defensive: router must not break chat
                logger.warning("forced recall failed: %s", exc)
                forced_detail = f"error: tool recall failed ({type(exc).__name__})"
            forced_ms = (time.monotonic() - forced_started) * 1000.0
            forced_ok = not forced_detail.startswith("error")
            trace.tools.append(
                ToolSpan(
                    tool="recall",
                    ok=forced_ok,
                    latency_ms=forced_ms,
                    detail=forced_detail[:200],
                )
            )
            trace.recall = RecallSpan(
                query=user_text, latency_ms=forced_ms,
            )
            tool_results.append(
                AgentToolResult(tool="recall", ok=forced_ok, detail=forced_detail)
            )
            loop.append(
                AgentMessage(role="tool", content=forced_detail[:MAX_TURN_RESULT_CHARS])
            )
        # Suggestion #4 — short-answer path: routed memory questions get a
        # 1-2 sentence instruction so "What is my name?" costs one grounded
        # turn instead of thousands of reasoning tokens.
        effective_text = user_text
        if routed_intent == "recall" and self._config.concise_memory_answers:
            effective_text = (
                f"{user_text}\n"
                "(Answer in 1-2 sentences using only the recalled memories "
                "above. Answer only what the memories directly state — do "
                "not infer preferences, favorites, or unstated facts from "
                "usage. If they do not contain the answer, say you do not "
                "have it stored.)"
            )
        for _ in range(self._config.max_turns):
            prompt = "\n".join(
                [
                    *(f"{m.role}: {m.content}" for m in [*prior, *loop]),
                    f"User: {effective_text}",
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
            text = _unwrap_answer(text)
            try:
                rin = int(getattr(reply, "tokens_in", 0) or 0)
                rout = int(getattr(reply, "tokens_out", 0) or 0)
                rest = bool(getattr(reply, "estimated", True))
            except Exception:
                from ..intelligence.providers import estimate_tokens as _est
                rin, rout, rest = _est(prompt), _est(str(text)), True
            if not rin and not rout:
                from ..intelligence.providers import estimate_tokens as _est
                rin, rout, rest = _est(prompt), _est(str(text)), True
            trace.llm = LLMSpan(
                model=model_name,
                latency_ms=trace.llm.latency_ms + llm_ms,
                tokens=trace.llm.tokens + rin + rout,
                tokens_in=trace.llm.tokens_in + rin,
                tokens_out=trace.llm.tokens_out + rout,
                estimated=trace.llm.estimated and rest,
            )
            call = _parse_tool_call(text)
            if call is None or call.tool not in self._tools:
                # LLM returned plain text - try fallback extraction for remember tool
                if not fallback_used and "remember" in self._tools:
                    fallback_result = self._run_fallback_extraction(user_text)
                    if fallback_result:
                        # Execute the remember tool with extracted facts
                        tool_started = time.monotonic()
                        try:
                            detail = self._tools["remember"].run(fallback_result)
                        except Exception as exc:
                            logger.warning("fallback remember tool failed: %s", exc)
                            detail = f"error: fallback remember failed ({type(exc).__name__})"
                        tool_ms = (time.monotonic() - tool_started) * 1000.0
                        ok = not detail.startswith("error")
                        trace.tools.append(
                            ToolSpan(
                                tool="remember",
                                ok=ok,
                                latency_ms=tool_ms,
                                detail=detail[:200],
                            )
                        )
                        tool_results.append(
                            AgentToolResult(tool="remember", ok=ok, detail=detail)
                        )
                        loop.append(AgentMessage(role="assistant", content=text))
                        loop.append(
                            AgentMessage(role="tool", content=detail[:MAX_TURN_RESULT_CHARS])
                        )
                        fallback_used = True
                        # Return immediately with LLM's answer + tool result (no extra LLM call)
                        trace.turns = len(tool_results) + 1
                        trace.total_ms = (time.monotonic() - started) * 1000.0
                        return AgentChatData(
                            answer=text,
                            turns=len(tool_results) + 1,
                            tool_calls=tool_results,
                            transcript=_full_transcript(prior, user_text, loop),
                            trace=trace,
                        )
                
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

    def _run_fallback_extraction(self, user_text: str) -> Optional[dict]:
        """Run rule-based extractor on user message to auto-extract remember arguments."""
        try:
            # Create a minimal conversation with just the user's message
            conversation = Conversation.from_user_message(user_text)
            decisions = self._fallback_extractor.extract(conversation)
            
            # Filter for should_remember decisions
            remember_decisions = [d for d in decisions if d.should_remember]
            if not remember_decisions:
                return None
            
            # Combine all extracted facts into one remember call
            # Use the first (strongest) decision's text as the primary fact
            primary = remember_decisions[0].candidate
            facts = [d.candidate.text for d in remember_decisions]
            
            # Build remember arguments
            return {
                "text": " | ".join(facts) if len(facts) > 1 else primary.text,
                "category": primary.category.value if primary.category else "fact"
            }
        except Exception as exc:
            logger.warning("fallback extraction failed: %s", exc)
            return None

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