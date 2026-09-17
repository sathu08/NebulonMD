"""nmd_monitor tests (fully offline, no NebulonDB / LLM required)."""

import pytest
from fastapi.testclient import TestClient

from nmd_host.agent import AgentConfig, AgentRuntime, ExecutionTrace
from nmd_host.api.server import create_app
from nmd_host.api.service import InMemoryServiceProvider
from nmd_host.core.config import ServiceConfig
from nmd_host.intelligence.providers import LLMResponse
from nmd_host.monitor import (
    InMemoryMonitorStore,
    MonitorConfig,
    MonitorRecorder,
    MonitorTrace,
    monitor_store_for,
    summarize_online,
)
from nmd_host.monitor.decorators import span


class FakeLLM:
    def __init__(self, replies):
        self._replies = list(replies)

    def complete(self, prompt, *, system=None, temperature=0.0):
        if self._replies:
            return LLMResponse(text=self._replies.pop(0))
        return LLMResponse(text="done")


def _trace_with_tools() -> ExecutionTrace:
    trace = ExecutionTrace()
    trace.turns = 2
    trace.total_ms = 12.5
    trace.llm.model = "test-model"
    trace.llm.latency_ms = 8.0
    trace.llm.tokens = 42
    from nmd_host.agent import RecallSpan, ToolSpan

    trace.recall = RecallSpan(query="my name", memories=1, latency_ms=2.0)
    trace.tools.append(ToolSpan(tool="recall", ok=True, latency_ms=2.0, detail="1 memory"))
    trace.tools.append(ToolSpan(tool="remember", ok=False, latency_ms=1.0, detail="error: skipped"))
    return trace


def test_from_execution_trace_maps_spans_and_tools():
    trace = MonitorTrace.from_execution_trace(
        _trace_with_tools(), user_id="u1", thread_id="s1", input_text="hi", answer="hello",
    )
    assert trace.user_id == "u1"
    assert trace.thread_id == "s1"
    assert trace.turns == 2
    assert trace.tokens == 42
    assert trace.tools_used == ["recall", "remember"]
    assert trace.recall_memories == 1
    assert {s.name for s in trace.spans} >= {"llm", "recall", "remember"}


def test_recorder_redacts_long_bodies():
    recorder = MonitorRecorder(MonitorConfig(enabled=True, redact=True, max_body_chars=10))
    out = recorder.from_chat(
        _trace_with_tools(), user_id="u1", thread_id="t", input_text="x" * 50, answer="y" * 50,
    )
    assert out is not None
    assert len(out.input_text) <= 30
    assert out.input_text.endswith("[truncated]")


def test_recorder_disabled_and_zero_rate_capture_nothing():
    assert MonitorRecorder(MonitorConfig(enabled=False)).from_chat(_trace_with_tools()) is None
    assert MonitorRecorder(MonitorConfig(sample_rate=0.0)).from_chat(_trace_with_tools()) is None


def test_store_list_filters_and_stats():
    store = InMemoryMonitorStore("u1")
    t1 = MonitorTrace(user_id="u1", input_text="my name is Sathya", tools_used=["recall"], ok=True)
    t2 = MonitorTrace(user_id="u1", input_text="unrelated", tools_used=["remember"], ok=False)
    store.save(t1)
    store.save(t2)
    assert store.get(t1.trace_id).trace_id == t1.trace_id
    assert [t.trace_id for t in store.list(q="sathya")] == [t1.trace_id]
    assert [t.trace_id for t in store.list(tool="remember")] == [t2.trace_id]
    assert [t.trace_id for t in store.list(ok=True)] == [t1.trace_id]
    stats = store.stats()
    assert stats.traces == 2
    assert stats.error_rate == 0.5
    assert stats.recall_used_rate == 0.5


def test_span_context_manager_records_latency_and_error():
    sinks = []
    with span("demo", "tool", sink=sinks, input_text="in"):
        pass
    assert len(sinks) == 1 and sinks[0].ok is True and sinks[0].latency_ms >= 0
    with pytest.raises(ValueError):
        with span("boom", "tool", sink=sinks):
            raise ValueError("nope")
    assert sinks[-1].ok is False and "ValueError" in sinks[-1].error


def test_summarize_online_counts_gaps():
    traces = [
        MonitorTrace(answer="a", tools_used=["recall"], recall_memories=0, ok=True),
        MonitorTrace(answer="b", tools_used=[], ok=False),
    ]
    report = summarize_online(traces)
    assert report["traces"] == 2
    assert report["tool_error_rate"] == 0.5
    assert report["answered_with_zero_recall"] == 0.5
    assert report["empty_recall_rate"] == 1.0


@pytest.fixture
def monitor_app(monkeypatch):
    fake = FakeLLM(['{"tool": "recall", "arguments": {"query": "name"}}', "Your name is Sathya."])
    monkeypatch.setattr(
        "nmd_host.intelligence.providers.provider_from_env_with_model", lambda model=None: fake
    )
    provider = InMemoryServiceProvider()
    provider.create_user("user_001")
    return create_app(provider=provider, config=ServiceConfig())


def test_monitor_routes_capture_chat_and_list(monitor_app):
    with TestClient(monitor_app) as client:
        chat = client.post(
            "/api/NebulonMind/agent/chat", json={"text": "my name?"}, params={"user_id": "user_001"}
        )
        assert chat.status_code == 200
        traces = client.get("/api/NebulonMind/monitor/traces", params={"user_id": "user_001"})
        assert traces.status_code == 200
        body = traces.json()
        assert body["success"] is True
        assert len(body["data"]["traces"]) == 1
        trace_id = body["data"]["traces"][0]["trace_id"]
        detail = client.get(
            f"/api/NebulonMind/monitor/trace/{trace_id}", params={"user_id": "user_001"}
        )
        assert detail.status_code == 200
        stats = client.get("/api/NebulonMind/monitor/stats", params={"user_id": "user_001"})
        assert stats.status_code == 200
        assert stats.json()["data"]["stats"]["traces"] == 1


def test_monitor_trace_404_and_bad_user(monitor_app):
    with TestClient(monitor_app) as client:
        missing = client.get("/api/NebulonMind/monitor/trace/nope", params={"user_id": "user_001"})
        assert missing.status_code == 404
        bad = client.get("/api/NebulonMind/monitor/traces", params={"user_id": "/bad"})
        assert bad.status_code == 400


def test_monitor_store_for_caches_per_user():
    provider = InMemoryServiceProvider()
    provider.create_user("alice")
    assert monitor_store_for(provider, "alice") is monitor_store_for(provider, "alice")


def test_agent_chat_without_monitor_config_still_works(monkeypatch):
    fake = FakeLLM(["hi"])
    monkeypatch.setattr(
        "nmd_host.intelligence.providers.provider_from_env_with_model", lambda model=None: fake
    )
    monkeypatch.setenv("NMD_MONITOR_ENABLED", "false")
    provider = InMemoryServiceProvider()
    provider.create_user("user_001")
    app = create_app(provider=provider, config=ServiceConfig())
    with TestClient(app) as client:
        resp = client.post(
            "/api/NebulonMind/agent/chat", json={"text": "hi"}, params={"user_id": "user_001"}
        )
        assert resp.status_code == 200
