"""Step 6 — provider contract + session-aware memory tests (fully offline).

Covers the open changes.txt items delivered in Step 6 phases 6.1–6.5:

* 6.1 — standardized ``LLMResponse`` (no provider-specific shapes)
* 6.2 — typed LLM errors (timeout / unavailable / rate-limit / token-limit /
        invalid response) transcoded from SDK exceptions
* 6.3 — provenance metadata (source_type / created_by / session_id /
        conversation_id) stamped through extractor → converter → store
* 6.4 — bounded rate-limit retry (backoff) and immediate surface of non
        transient errors
* 6.5 — ``GET /api/NebulonMind/llm/status`` (no credentials ever exposed)
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from nmd_host.agent import AgentConfig, AgentRuntime, build_memory_toolkit
from nmd_host.api.server import create_app
from nmd_host.api.service import InMemoryServiceProvider
from nmd_host.core.config import ServiceConfig
from nmd_host.core.models import Memory, Provenance
from nmd_host.intelligence.llm_extractor import LLMExtractor
from nmd_host.intelligence import providers as providers  # noqa: F401  (module for monkeypatch)
from nmd_host.intelligence.converter import candidate_to_memory
from nmd_host.intelligence.providers import (
    BaseLLMProvider,
    LLMInvalidResponseError,
    LLMRateLimitError,
    LLMResponse,
    LLMTimeoutError,
    LLMTokenLimitError,
    LLMUnavailableError,
    RetryingLLMProvider,
    _classify_sdk_error,
    _extract_json,
    provider_from_env,
)
from nmd_host.intelligence.schemas import Conversation, MemoryCandidate


# ---------------------------------------------------------------------- #
# 6.1 — LLMResponse                                                      #
# ---------------------------------------------------------------------- #


def test_llm_response_is_standardized_output_type():
    response = LLMResponse(text="hello", provider="openai")
    assert response.text == "hello"
    assert response.provider == "openai"


def test_base_provider_structured_accepts_llm_response():
    class _Fake(BaseLLMProvider):
        provider = "fake"

        def complete(self, prompt, *, system=None, temperature=0.0):
            return LLMResponse(text='{"decisions": []}', provider="fake")

    parsed = _Fake().structured("prompt", Conversation)
    assert parsed == {"decisions": []}


# ---------------------------------------------------------------------- #
# 6.2 — typed errors                                                     #
# ---------------------------------------------------------------------- #


class _FakeTimeout(Exception):
    pass


class _FakeRateLimit(Exception):
    pass


class _FakeConnection(Exception):
    pass


class _FakeContextLimit(Exception):
    pass


def test_sdk_timeout_maps_to_typed_error():
    error = _classify_sdk_error(_FakeTimeout("slow"))
    assert isinstance(error, LLMTimeoutError)


def test_sdk_rate_limit_maps_to_typed_error():
    error = _classify_sdk_error(_FakeRateLimit("429: quota"))
    assert isinstance(error, LLMRateLimitError)


def test_sdk_connection_error_maps_to_unavailable():
    error = _classify_sdk_error(_FakeConnection("refused"))
    assert isinstance(error, LLMUnavailableError)


def test_sdk_context_limit_maps_to_token_limit():
    error = _classify_sdk_error(_FakeContextLimit("context length exceeded"))
    assert isinstance(error, LLMTokenLimitError)


def test_invalid_json_reply_raises_typed_invalid_response():
    with pytest.raises(LLMInvalidResponseError):
        _extract_json("this is not json")


def test_typed_errors_are_provider_error_subclasses():
    from nmd_host.intelligence.providers import LLMProviderError

    for error_type in (
        LLMTimeoutError,
        LLMRateLimitError,
        LLMTokenLimitError,
        LLMUnavailableError,
        LLMInvalidResponseError,
    ):
        assert issubclass(error_type, LLMProviderError)


# ---------------------------------------------------------------------- #
# 6.4 — rate-limit retry                                                 #
# ---------------------------------------------------------------------- #


class _FlakyProvider:
    """Fails ``n_failures`` times with a rate limit, then succeeds."""

    provider = "fake"

    def __init__(self, n_failures, reply="ok"):
        self.remaining = n_failures
        self.calls = 0
        self.reply = reply

    def complete(self, prompt, *, system=None, temperature=0.0):
        self.calls += 1
        if self.remaining > 0:
            self.remaining -= 1
            raise LLMRateLimitError("slow down")
        return LLMResponse(text=self.reply, provider=self.provider)


def test_rate_limit_retries_then_succeeds():
    wrapped = RetryingLLMProvider(_FlakyProvider(2), max_retries=3, base_delay=0.0)
    response = wrapped.complete("prompt")
    assert response.text == "ok"


def test_rate_limit_exhausted_raises():
    wrapped = RetryingLLMProvider(_FlakyProvider(9), max_retries=2, base_delay=0.0)
    with pytest.raises(LLMRateLimitError):
        wrapped.complete("prompt")


def test_non_rate_errors_propagate_without_retry():
    class _TimeoutProvider:
        provider = "fake"

        def __init__(self):
            self.calls = 0

        def complete(self, prompt, *, system=None, temperature=0.0):
            self.calls += 1
            raise LLMTimeoutError("slow")

    wrapped = RetryingLLMProvider(_TimeoutProvider(), max_retries=3, base_delay=0.0)
    with pytest.raises(LLMTimeoutError):
        wrapped.complete("prompt")
    assert wrapped._provider.calls == 1


def test_provider_from_env_wraps_with_retries(monkeypatch):
    class _CannedProvider:
        provider = "openai"

        def complete(self, prompt, *, system=None, temperature=0.0):
            return LLMResponse(text="x", provider="openai")

    monkeypatch.setattr(providers, "_PROVIDERS", {"openai": _CannedProvider})
    monkeypatch.setenv("NMD_LLM_PROVIDER", "openai")
    result = provider_from_env()
    assert isinstance(result, RetryingLLMProvider)


def test_provider_from_env_zero_retries_returns_bare_adapter(monkeypatch):
    class _CannedProvider:
        provider = "openai"

        def complete(self, prompt, *, system=None, temperature=0.0):
            return LLMResponse(text="x", provider="openai")

    monkeypatch.setattr(providers, "_PROVIDERS", {"openai": _CannedProvider})
    monkeypatch.setenv("NMD_LLM_PROVIDER", "openai")
    monkeypatch.setenv("NMD_LLM_MAX_RETRIES", "0")
    result = provider_from_env()
    assert isinstance(result, _CannedProvider)


# ---------------------------------------------------------------------- #
# 6.3 — provenance metadata                                              #
# ---------------------------------------------------------------------- #


def test_candidate_to_memory_carries_provenance():
    candidate = MemoryCandidate(
        text="I live in Bangalore",
        provenance=Provenance(
            source_type="agent",
            created_by="agent",
            session_id="session_9",
            conversation_id="conv_9",
        ),
    )
    memory = candidate_to_memory(candidate, "user_001")
    assert memory.provenance.source_type == "agent"
    assert memory.provenance.created_by == "agent"
    assert memory.provenance.session_id == "session_9"
    assert memory.provenance.conversation_id == "conv_9"


def test_provenance_roundtrips_through_truth_doc():
    memory = Memory(
        user_id="user_001",
        content={"text": "I live in Bangalore"},
        provenance=Provenance(session_id="s1"),
    )
    restored = Memory.from_truth_doc(memory.to_truth_doc())
    assert restored.provenance.session_id == "s1"


class _StructuredFakeLLM:
    """LLMProvider that returns a scripted structured payload."""

    def __init__(self, payload):
        self.payload = payload

    def structured(self, prompt, schema):
        return self.payload


def test_llm_extractor_stamps_session_and_conversation_ids():
    payload = {
        "decisions": [
            {
                "candidate": {"text": "My name is Sathya", "category": "identity"},
                "should_remember": True,
                "reason": "identity",
            }
        ]
    }
    extractor = LLMExtractor(_StructuredFakeLLM(payload))
    conversation = Conversation.from_user_message("My name is Sathya")
    conversation.session_id = "session_7"
    conversation.conversation_id = "conv_7"
    decisions = extractor.extract(conversation)
    assert decisions[0].candidate.provenance.session_id == "session_7"
    assert decisions[0].candidate.provenance.conversation_id == "conv_7"


def test_agent_remember_stamps_provenance_on_stored_memory():
    provider = InMemoryServiceProvider()
    provider.create_user("user_001")
    bundle = provider.bundle("user_001")
    tools = build_memory_toolkit(
        bundle.repository,
        bundle.manager,
        "user_001",
        session_id="session_5",
        conversation_id="conv_5",
    )
    remember = next(tool for tool in tools if tool.name == "remember")
    result = remember.run({"text": "My name is Sathya"})
    assert "stored" in result
    stored = bundle.repository.search("Sathya", top_k=5)
    memory = next(m for m in stored if "Sathya" in m.content.text)
    assert memory.provenance.source_type == "agent"
    assert memory.provenance.created_by == "agent"
    assert memory.provenance.session_id == "session_5"
    assert memory.provenance.conversation_id == "conv_5"


def test_agent_runtime_passes_session_context_to_toolkit():
    llm = _ScriptedLLM(
        [
            '{"tool": "remember", "arguments": {"text": "My name is Sathya"}}',
            "Done.",
        ]
    )
    provider = InMemoryServiceProvider()
    provider.create_user("user_001")
    bundle = provider.bundle("user_001")
    tools = build_memory_toolkit(
        bundle.repository,
        bundle.manager,
        "user_001",
        session_id="session_3",
        conversation_id="conv_3",
    )
    runtime = AgentRuntime(llm, tools, AgentConfig(max_turns=3))
    result = runtime.chat("Remember this")
    assert result.tool_calls[0].ok is True
    stored = bundle.repository.search("Sathya", top_k=5)
    memory = next(m for m in stored if "Sathya" in m.content.text)
    assert memory.provenance.session_id == "session_3"
    assert memory.provenance.conversation_id == "conv_3"


class _ScriptedLLM:
    def __init__(self, replies):
        self._replies = list(replies)

    def complete(self, prompt, *, system=None, temperature=0.0):
        return LLMResponse(text=self._replies.pop(0))


# ---------------------------------------------------------------------- #
# 6.5 — /llm/status endpoint                                             #
# ---------------------------------------------------------------------- #


def _app():
    provider = InMemoryServiceProvider()
    provider.create_user("user_001")
    return create_app(provider=provider, config=ServiceConfig())


def test_llm_status_reports_not_configured(monkeypatch):
    monkeypatch.setenv("NMD_LLM_PROVIDER", "")
    with TestClient(_app()) as client:
        resp = client.get("/api/NebulonMind/llm/status")
    assert resp.status_code == 200
    assert resp.json()["success"] is True
    data = resp.json()["data"]
    assert data["provider"] == ""
    assert data["configured"] is False
    assert data["model"] is None


def test_llm_status_reports_configuration_error(monkeypatch):
    monkeypatch.setenv("NMD_LLM_PROVIDER", "sky-whale")
    with TestClient(_app()) as client:
        resp = client.get("/api/NebulonMind/llm/status")
    data = resp.json()["data"]
    assert data["provider"] == "sky-whale"
    assert data["configured"] is False
    assert "sky-whale" in data["error"]


def test_llm_status_never_leaks_api_keys(monkeypatch):
    monkeypatch.setenv("NMD_LLM_PROVIDER", "openai")
    monkeypatch.setenv("NMD_LLM_API_KEY", "sk-supersecret-1234")
    with TestClient(_app()) as client:
        resp = client.get("/api/NebulonMind/llm/status")
    body = resp.json()
    assert "sk-supersecret-1234" not in str(body)
    assert body["success"] is True


def test_llm_status_reports_configured_provider(monkeypatch):
    monkeypatch.setenv("NMD_LLM_PROVIDER", "openai")

    class _Fake:
        provider = "openai"
        model = "gpt-4o-mini"

        def complete(self, prompt, *, system=None, temperature=0.0):
            return LLMResponse(text="ok", provider="openai")

    monkeypatch.setattr(
        "nmd_host.intelligence.providers.provider_from_env", lambda: _Fake()
    )
    with TestClient(_app()) as client:
        resp = client.get("/api/NebulonMind/llm/status")
    data = resp.json()["data"]
    assert data["configured"] is True
    assert data["model"] == "gpt-4o-mini"


def test_agent_chat_returns_502_on_typed_llm_failure(monkeypatch):
    class _BoomLLM:
        def complete(self, prompt, *, system=None, temperature=0.0):
            raise LLMTimeoutError("slow provider")

    monkeypatch.setattr(
        "nmd_host.intelligence.providers.provider_from_env", lambda: _BoomLLM()
    )
    with TestClient(_app()) as client:
        resp = client.post(
            "/api/NebulonMind/agent/chat",
            json={"text": "Do you know my name?"},
            params={"user_id": "user_001"},
        )
    assert resp.status_code == 502
    assert resp.json()["success"] is False