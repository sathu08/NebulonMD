"""Step 12 — Observability tests (fully offline).

Verifies the agent runtime emits an ``ExecutionTrace`` with LLM / tool /
recall / total latency, that the trace rides the ``/agent/chat`` response,
and that the memory toolkit records the recall outcome into the trace.
"""

import pytest
from fastapi.testclient import TestClient

from nmd_host.agent import AgentConfig, AgentRuntime, ExecutionTrace
from nmd_host.api.server import create_app
from nmd_host.api.service import InMemoryServiceProvider
from nmd_host.core.config import ServiceConfig
from nmd_host.intelligence.providers import LLMResponse


class FakeLLM:
    """Scripted LLMProvider (mirrors tests/test_agent.py)."""

    def __init__(self, replies):
        self._replies = list(replies)

    def complete(self, prompt, *, system=None, temperature=0.0):
        if self._replies:
            return LLMResponse(text=self._replies.pop(0))
        return LLMResponse(text="I have no further input.")


def _runtime(llm, user_id="user_001"):
    provider = InMemoryServiceProvider()
    provider.create_user(user_id)
    bundle = provider.bundle(user_id)
    from nmd_host.agent import build_memory_toolkit

    tools = build_memory_toolkit(bundle.repository, bundle.manager, user_id)
    return AgentRuntime(llm, tools, AgentConfig(max_turns=3))


def test_direct_answer_trace_has_llm_span_and_total():
    llm = FakeLLM(["My name is Sathya."])
    runtime = _runtime(llm)
    result = runtime.chat("What is your name?")
    assert isinstance(result.trace, ExecutionTrace)
    assert result.trace.trace_id
    assert result.trace.total_ms >= 0
    assert result.trace.turns == 1
    assert result.trace.llm.latency_ms > 0
    assert result.trace.llm.tokens > 0
    assert result.trace.tools == []
    assert result.trace.recall is None


def test_recall_trace_records_tool_and_memory_count():
    provider = InMemoryServiceProvider()
    provider.create_user("user_001")
    bundle = provider.bundle("user_001")
    from nmd_host.agent import build_memory_toolkit
    from nmd_host.core.models import (
        Classification,
        Importance,
        Lifecycle,
        Memory,
        MemoryContent,
        MemoryType,
        RetentionPolicy,
    )

    bundle.repository.create(
        Memory(
            user_id="user_001",
            content=MemoryContent(text="My name is Sathya"),
            classification=Classification(
                memory_type=MemoryType.SEMANTIC, category="identity"
            ),
            importance=Importance(score=0.9, confidence=1.0),
            lifecycle=Lifecycle(retention_policy=RetentionPolicy.PERMANENT),
        )
    )
    llm = FakeLLM(
        [
            '{"tool": "recall", "arguments": {"query": "my name"}}',
            "Your name is Sathya.",
        ]
    )
    tools = build_memory_toolkit(bundle.repository, bundle.manager, "user_001")
    runtime = AgentRuntime(llm, tools, AgentConfig(max_turns=3))
    result = runtime.chat("Do you know my name?")
    assert result.trace.recall is not None
    assert result.trace.recall.query == "my name"
    assert result.trace.recall.memories == 1
    assert result.trace.recall.latency_ms > 0
    assert [t.tool for t in result.trace.tools] == ["recall"]
    assert result.trace.tools[0].ok is True


def test_trace_id_stable_when_passed_in():
    trace = ExecutionTrace()
    llm = FakeLLM(["hi"])
    runtime = _runtime(llm)
    result = runtime.chat("hello", trace=trace)
    assert result.trace is trace
    assert result.trace.trace_id == trace.trace_id


@pytest.fixture
def app_with_llm(monkeypatch):
    fake = FakeLLM(["That is all I know."])
    monkeypatch.setattr(
        "nmd_host.intelligence.providers.provider_from_env", lambda: fake
    )
    provider = InMemoryServiceProvider()
    provider.create_user("user_001")
    app = create_app(provider=provider, config=ServiceConfig())
    return app


def test_agent_chat_response_carries_trace(app_with_llm):
    with TestClient(app_with_llm) as client:
        resp = client.post(
            "/api/NebulonMind/agent/chat",
            json={"text": "hi"},
            params={"user_id": "user_001"},
        )
    assert resp.status_code == 200
    body = resp.json()
    trace = body["data"]["trace"]
    assert trace is not None
    assert trace["trace_id"]
    assert trace["turns"] == 1
    assert trace["llm"]["latency_ms"] >= 0
    assert "total_ms" in trace
