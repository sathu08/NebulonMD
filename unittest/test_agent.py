"""Step 6 — Agent Runtime tests (fully offline).

A fake ``LLMProvider`` drives the loop, so no LLM, no network and no
NebulonDB are required: the runtime + tools run over
``InMemoryServiceProvider`` / ``InMemoryRepository``.
"""

import pytest
from fastapi.testclient import TestClient

from nmd_host.agent import AgentConfig, AgentRuntime, build_memory_toolkit
from nmd_host.api.server import create_app
from nmd_host.api.service import InMemoryServiceProvider
from nmd_host.core.config import ServiceConfig
from nmd_host.intelligence.providers import LLMResponse


class FakeLLM:
    """LLMProvider stand-in returning a scripted list of replies."""

    def __init__(self, replies, capture=None):
        self._replies = list(replies)
        self.capture = capture or []

    def complete(self, prompt, *, system=None, temperature=0.0):
        self.capture.append(prompt)
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
# Runtime loop                                                           #
# ---------------------------------------------------------------------- #


def test_direct_answer_without_tool_call():
    llm = FakeLLM(["My name is Sathya, as far as I remember."])
    runtime, _ = _runtime(llm)
    result = runtime.chat("What is your name?")
    assert "Sathya" in result.answer
    assert result.turns == 1
    assert result.tool_calls == []


def test_tool_call_then_final_answer():
    llm = FakeLLM(
        [
            '{"tool": "recall", "arguments": {"query": "my name"}}',
            "You previously stored that your name is Sathya.",
        ]
    )
    runtime, _ = _runtime(llm)
    result = runtime.chat("Do you know my name?")
    assert result.turns == 2
    assert result.tool_calls[0].tool == "recall"
    assert result.tool_calls[0].ok is True
    assert "Sathya" in result.answer


def test_remember_persists_through_gate_and_store():
    llm = FakeLLM(
        [
            '{"tool": "remember", "arguments": {"text": "My name is Sathya"}}',
            "Done — I stored that your name is Sathya.",
        ]
    )
    runtime, provider = _runtime(llm)
    result = runtime.chat("Remember my name.")
    assert result.tool_calls[0].tool == "remember"
    assert result.tool_calls[0].ok is True
    bundle = provider.bundle("user_001")
    stored = bundle.repository.search("Sathya", top_k=5)
    assert any("Sathya" in m.content.text for m in stored)


def test_unknown_tool_name_returns_final_answer_unless_requested():
    llm = FakeLLM(['{"tool": "fly", "arguments": {}}'])
    runtime, _ = _runtime(llm)
    result = runtime.chat("Fly for me")
    assert result.answer == '{"tool": "fly", "arguments": {}}'
    assert result.tool_calls == []


def test_malformed_tool_json_falls_back_to_answer():
    llm = FakeLLM(["no json here, just text"])
    runtime, _ = _runtime(llm)
    result = runtime.chat("hi")
    assert result.answer == "no json here, just text"


def test_tool_failure_does_not_crash_loop():
    class _BrokenTool:
        name = "remember"
        description = "always fails"

        def run(self, arguments):
            raise RuntimeError("boom")

    llm = FakeLLM(['{"tool": "remember", "arguments": {}}', "recovered"])
    tools = [_BrokenTool()]
    runtime = AgentRuntime(llm, tools, AgentConfig(max_turns=2))
    result = runtime.chat("hi")
    assert result.tool_calls[0].ok is False
    assert "error" in result.tool_calls[0].detail
    assert result.answer == "recovered"


def test_max_turns_bounds_the_loop():
    llm = FakeLLM(
        ['{"tool": "recall", "arguments": {"query": "x"}}'] * 10
    )
    provider = InMemoryServiceProvider()
    provider.create_user("user_001")
    bundle = provider.bundle("user_001")
    runtime = AgentRuntime(
        llm,
        build_memory_toolkit(
            bundle.repository,
            bundle.manager,
            "user_001",
        ),
        AgentConfig(max_turns=3),
    )
    result = runtime.chat("loop forever")
    assert result.turns == 3
    assert "tool-call limit" in result.answer


def test_remember_rejects_empty_text():
    llm = FakeLLM(
        [
            '{"tool": "remember", "arguments": {}}',
            "I need the text to remember it.",
        ]
    )
    runtime, _ = _runtime(llm)
    result = runtime.chat("remember nothing")
    assert result.tool_calls[0].ok is False
    assert "text" in result.tool_calls[0].detail


# ---------------------------------------------------------------------- #
# API endpoint                                                           #
# ---------------------------------------------------------------------- #


@pytest.fixture
def app_with_llm(monkeypatch):
    fake = FakeLLM(["That is all I know."])
    from nmd_host.api import server as server_module

    monkeypatch.setattr(
        "nmd_host.intelligence.providers.provider_from_env", lambda: fake
    )
    provider = InMemoryServiceProvider()
    provider.create_user("user_001")
    app = create_app(provider=provider, config=ServiceConfig())
    return app, fake


def test_agent_chat_endpoint_returns_envelope(app_with_llm):
    app, fake = app_with_llm
    with TestClient(app) as client:
        resp = client.post(
            "/api/NebulonMind/agent/chat",
            json={"text": "Do you know my name?"},
            params={"user_id": "user_001"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["answer"] == "That is all I know."
    assert body["data"]["turns"] == 1


def test_agent_chat_endpoint_503_without_llm_provider(monkeypatch):
    monkeypatch.setenv("NMD_LLM_PROVIDER", "")
    provider = InMemoryServiceProvider()
    provider.create_user("user_001")
    app = create_app(provider=provider, config=ServiceConfig())
    with TestClient(app) as client:
        resp = client.post(
            "/api/NebulonMind/agent/chat",
            json={"text": "hi"},
            params={"user_id": "user_001"},
        )
    assert resp.status_code == 503
    assert resp.json()["success"] is False


def test_agent_chat_endpoint_422_without_text(app_with_llm):
    app, _ = app_with_llm
    with TestClient(app) as client:
        resp = client.post(
            "/api/NebulonMind/agent/chat",
            json={},
            params={"user_id": "user_001"},
        )
    assert resp.status_code == 422