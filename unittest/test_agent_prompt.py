"""Step 6.13 — Agent system prompt & behavior policy tests (fully offline).

The prompt module (``nmd_host.agent.prompt``) defines the stable system
instruction the agent receives; these tests pin the behavior contract to the
runtime so a regression in the prompt or the loop is caught without any LLM.
A fake ``LLMProvider`` drives the loop over an in-memory repository.
"""

from __future__ import annotations

import pytest

from nmd_host.agent import (
    AgentConfig,
    AgentRuntime,
    DEFAULT_SYSTEM_PROMPT,
    build_memory_toolkit,
    build_system_prompt,
)
from nmd_host.agent.prompt import (
    AGENT_IDENTITY,
    CONFLICT_HANDLING_POLICY,
    HALLUCINATION_POLICY,
    MEMORY_PROVENANCE_REQUIREMENTS,
    RECALL_TOOL_POLICY,
    REMEMBER_TOOL_POLICY,
    RESPONSE_POLICY,
    SESSION_BEHAVIOR_POLICY,
    TERMINATION_BEHAVIOR,
    TOOL_CALL_PROTOCOL,
    TOOL_FAILURE_BEHAVIOR,
)
from nmd_host.api.service import InMemoryServiceProvider
from nmd_host.intelligence.providers import LLMResponse


class _CaptureLLM:
    """LLMProvider returning scripted replies; records prompt + system."""

    def __init__(self, replies, capture=None, systems=None):
        self._replies = list(replies)
        self.capture = capture if capture is not None else []
        self.systems = systems if systems is not None else []

    def complete(self, prompt, *, system=None, temperature=0.0):
        self.capture.append(prompt)
        self.systems.append(system)
        if self._replies:
            return LLMResponse(text=self._replies.pop(0))
        return LLMResponse(text="I have no further input.")


def _runtime(llm, user_id="user_001"):
    provider = InMemoryServiceProvider()
    provider.create_user(user_id)
    bundle = provider.bundle(user_id)
    tools = build_memory_toolkit(bundle.repository, bundle.manager, user_id)
    return AgentRuntime(llm, tools, AgentConfig(max_turns=3)), provider


# ---------------------------------------------------------------------- #
# Prompt contract                                                        #
# ---------------------------------------------------------------------- #


def test_default_prompt_defines_identity():
    assert "NebulonMind" in DEFAULT_SYSTEM_PROMPT
    assert AGENT_IDENTITY
    assert "memory" in AGENT_IDENTITY


def test_default_prompt_covers_remember_policy():
    assert "remember" in REMEMBER_TOOL_POLICY
    assert "durable" in REMEMBER_TOOL_POLICY.lower()


def test_default_prompt_covers_recall_policy():
    assert "recall" in RECALL_TOOL_POLICY
    assert "stored memory" in RECALL_TOOL_POLICY


def test_default_prompt_defines_tool_call_protocol():
    assert "JSON object" in TOOL_CALL_PROTOCOL
    assert '{"tool"' in TOOL_CALL_PROTOCOL
    assert "plain text" in TOOL_CALL_PROTOCOL


def test_default_prompt_forbids_hallucinating_memories():
    assert "never" in HALLUCINATION_POLICY.lower()
    assert "not" in HALLUCINATION_POLICY.lower()
    assert "recall" in HALLUCINATION_POLICY.lower()


def test_default_prompt_handles_conflicts():
    assert "conflict" in CONFLICT_HANDLING_POLICY.lower()
    assert "latest" in CONFLICT_HANDLING_POLICY.lower()
    assert "timestamp" in CONFLICT_HANDLING_POLICY.lower()
    assert "slot" in CONFLICT_HANDLING_POLICY.lower()
    assert "superseded" in CONFLICT_HANDLING_POLICY.lower()


def test_default_prompt_defines_response_and_provenance_policy():
    assert "plain text" in RESPONSE_POLICY
    assert "provenance" in MEMORY_PROVENANCE_REQUIREMENTS.lower()


def test_default_prompt_defines_termination_and_failure_behavior():
    assert "stop" in TERMINATION_BEHAVIOR.lower()
    assert "fail" in TOOL_FAILURE_BEHAVIOR.lower()


def test_build_system_prompt_includes_tool_list():
    result = build_system_prompt(
        {"remember": "persist a fact", "recall": "find memories"}
    )
    assert "remember: persist a fact" in result
    assert "recall: find memories" in result


def test_build_system_prompt_respects_override():
    result = build_system_prompt(
        {"recall": "find memories"}, override="CUSTOM PROMPT"
    )
    assert result.startswith("CUSTOM PROMPT")
    assert "default" not in result.lower()[:20]
    assert "recall: find memories" in result


def test_build_system_prompt_defaults_when_no_override():
    result = build_system_prompt({"recall": "find memories"})
    assert result.startswith(DEFAULT_SYSTEM_PROMPT.strip())


# ---------------------------------------------------------------------- #
# Behavior contract (changes.txt Step 6.13 tests)                        #
# ---------------------------------------------------------------------- #


def test_agent_uses_recall_when_memory_is_needed():
    llm = _CaptureLLM(
        [
            '{"tool": "recall", "arguments": {"query": "project worked on"}}',
            "Yesterday you were working on the NebulonMind agent project.",
        ]
    )
    runtime, _ = _runtime(llm)
    result = runtime.chat("What project was I working on yesterday?")
    assert result.tool_calls[0].tool == "recall"
    assert result.tool_calls[0].ok is True
    assert "NebulonMind" in result.answer


def test_agent_does_not_recall_for_simple_questions():
    llm = _CaptureLLM(["My name is fine, thanks for asking."])
    runtime, _ = _runtime(llm)
    result = runtime.chat("How are you?")
    assert result.tool_calls == []
    assert result.answer == "My name is fine, thanks for asking."


def test_agent_remembers_durable_information():
    llm = _CaptureLLM(
        [
            '{"tool": "remember", "arguments": {"text": "My name is Ada"}}',
            "Stored. I'll remember your name is Ada.",
        ]
    )
    runtime, provider = _runtime(llm)
    result = runtime.chat("Remember my name is Ada.")
    assert result.tool_calls[0].tool == "remember"
    assert result.tool_calls[0].ok is True
    stored = provider.bundle("user_001").repository.search("Ada", top_k=5)
    assert any("Ada" in m.content.text for m in stored)


def test_agent_does_not_claim_memory_that_was_not_returned():
    llm = _CaptureLLM(
        [
            '{"tool": "recall", "arguments": {"query": "favorite color"}}',
            "I don't have any memory about your favorite color.",
        ]
    )
    runtime, _ = _runtime(llm)
    result = runtime.chat("What is my favorite color?")
    assert result.tool_calls[0].tool == "recall"
    assert "no memories found" in result.tool_calls[0].detail
    assert "don't have any memory" in result.answer


def test_agent_handles_empty_recall_gracefully():
    llm = _CaptureLLM(
        [
            '{"tool": "recall", "arguments": {"query": "nothing relevant"}}',
            "I could not find anything stored about that.",
        ]
    )
    runtime, _ = _runtime(llm)
    result = runtime.chat("Do I have anything about X?")
    assert result.tool_calls[0].ok is True  # empty recall is not an error
    assert "no memories found" in result.tool_calls[0].detail
    assert result.answer == "I could not find anything stored about that."


def test_agent_handles_tool_failure():
    class _BrokenTool:
        name = "remember"
        description = "always fails"

        def run(self, arguments):
            raise RuntimeError("boom")

    llm = _CaptureLLM(
        ['{"tool": "remember", "arguments": {}}', "recovered"]
    )
    runtime = AgentRuntime(llm, [_BrokenTool()], AgentConfig(max_turns=2))
    result = runtime.chat("hi")
    assert result.tool_calls[0].ok is False
    assert "error" in result.tool_calls[0].detail
    assert result.answer == "recovered"


def test_agent_stops_after_max_turns():
    llm = _CaptureLLM(
        ['{"tool": "recall", "arguments": {"query": "x"}}'] * 10
    )
    provider = InMemoryServiceProvider()
    provider.create_user("user_001")
    bundle = provider.bundle("user_001")
    tools = build_memory_toolkit(bundle.repository, bundle.manager, "user_001")
    runtime = AgentRuntime(llm, tools, AgentConfig(max_turns=3))
    result = runtime.chat("loop")
    assert result.turns == 3
    assert "tool-call limit" in result.answer


def test_agent_follows_tool_schema():
    llm = _CaptureLLM(
        [
            '{"tool": "recall", "arguments": {"query": "name"}}',
            "Your name is Ada.",
        ]
    )
    runtime, _ = _runtime(llm)
    result = runtime.chat("What is my name?")
    assert result.tool_calls[0].tool == "recall"
    assert result.tool_calls[0].ok is True


def test_agent_system_prompt_is_passed_to_the_llm():
    llm = _CaptureLLM(["Just an answer."])
    runtime, _ = _runtime(llm)
    runtime.chat("hello")
    assert llm.systems
    assert "NebulonMind" in llm.systems[0]
    assert "remember" in llm.systems[0]


def test_agent_preserves_session_context():
    from nmd_host.agent.schemas import AgentMessage

    prior = [AgentMessage(role="user", content="My name is Ada.")]
    llm = _CaptureLLM(["Right, Ada."], capture=[])
    runtime, _ = _runtime(llm)
    result = runtime.chat("What is my name?", messages=prior)
    assert result.transcript[0].content == "My name is Ada."
    assert "Ada." in result.transcript[-1].content