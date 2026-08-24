"""``NMD_LLM_EXTRACTOR`` toggle tests (fully offline).

Verifies the config flag wiring in two ways:
* the builder contract (``_build_extractor``: off → rules, on → LLM, on
  without a provider → rules).
* the API end-to-end behaviour: with the toggle on a third-person fact the
  rule extractor ignores (rules.py only matches first-person patterns) is
  now decided and stored by the LLM extractor; if the LLM fails mid-call the
  pipeline falls back to the rule extractor instead of erroring.
"""

from fastapi.testclient import TestClient

from nmd_host.api.server import _build_extractor, create_app
from nmd_host.api.service import InMemoryServiceProvider
from nmd_host.core.config import ServiceConfig
from nmd_host.core.models import RetentionPolicy
from nmd_host.intelligence.converter import candidate_to_memory
from nmd_host.intelligence.llm_extractor import LLMExtractor
from nmd_host.intelligence.providers import LLMProviderError
from nmd_host.intelligence.schemas import Conversation, MemoryCandidate, MemoryCategory

THIRD_PERSON_FACT = "NebulonMind is built on NebulonDB."


class StubProvider:
    """Returns a fixed, valid structured payload (decision for the fact)."""

    def __init__(self, payload):
        self.payload = payload
        self.last_prompt = None

    def structured(self, prompt, schema):
        self.last_prompt = prompt
        return self.payload


class BrokenProvider:
    """Fails exactly like an unreachable/unavailable LLM."""

    def structured(self, prompt, schema):
        raise LLMProviderError("simulated timeout")


def _payload(category="project", relation="BUILDS"):
    return {
        "decisions": [
            {
                "candidate": {
                    "text": THIRD_PERSON_FACT,
                    "summary": "NebulonMind builds on NebulonDB",
                    "category": category,
                    "importance": 0.9,
                    "confidence": 0.95,
                    "entities": ["NebulonMind", "NebulonDB"],
                    "relationships": [
                        {
                            "source": "NebulonMind",
                            "target": "NebulonDB",
                            "relation": relation,
                        }
                    ],
                    "subject": "NebulonMind",
                },
                "should_remember": True,
                "reason": "durable fact from LLM",
            }
        ]
    }


# --------------------------------------------------------------------- #
# Builder contract                                                       #
# --------------------------------------------------------------------- #


def test_build_extractor_off_returns_none():
    assert _build_extractor(ServiceConfig(llm_extractor=False)) is None


def test_build_extractor_on_without_provider_falls_back(monkeypatch):
    monkeypatch.delenv("NMD_LLM_PROVIDER", raising=False)
    assert _build_extractor(ServiceConfig(llm_extractor=True)) is None


def test_build_extractor_on_returns_llm_extractor(monkeypatch):
    monkeypatch.setattr(
        "nmd_host.intelligence.providers.provider_from_env", lambda: StubProvider(_payload())
    )
    extractor = _build_extractor(ServiceConfig(llm_extractor=True))
    assert isinstance(extractor, LLMExtractor)


# --------------------------------------------------------------------- #
# API behaviour                                                          #
# --------------------------------------------------------------------- #


def test_default_rules_ignore_third_person_fact():
    provider = InMemoryServiceProvider()
    provider.create_user("user_001")
    with TestClient(create_app(provider=provider)) as client:
        resp = client.post(
            "/api/NebulonMind/intelligence/process",
            json={"text": THIRD_PERSON_FACT},
            params={"user_id": "user_001"},
        )
    assert resp.status_code == 200
    assert resp.json()["data"]["ingestions"] == []


def test_toggle_on_stores_third_person_fact_via_llm(monkeypatch):
    from nmd_host.intelligence import providers as providers_module

    monkeypatch.setattr(
        providers_module, "provider_from_env", lambda: StubProvider(_payload())
    )
    provider = InMemoryServiceProvider()
    provider.create_user("user_001")
    config = ServiceConfig(llm_extractor=True)
    with TestClient(create_app(provider=provider, config=config)) as client:
        resp = client.post(
            "/api/NebulonMind/intelligence/process",
            json={"text": THIRD_PERSON_FACT},
            params={"user_id": "user_001"},
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert len(data["ingestions"]) == 1
        assert data["ingestions"][0]["action"] == "STORE"
        assert resp.json()["message"] == "1 memory(ies) stored"
        memory_id = data["ingestions"][0]["memory_id"]
        # Persisted end-to-end and retrievable through the truth store API.
        got = client.get(
            f"/api/NebulonMind/memory/{memory_id}",
            params={"user_id": "user_001"},
        )
        assert got.status_code == 200
        assert (
            got.json()["data"]["memory"]["content"]["text"] == THIRD_PERSON_FACT
        )


def test_toggle_on_with_llm_failure_falls_back_to_rules(monkeypatch):
    from nmd_host.intelligence import providers as providers_module

    monkeypatch.setattr(
        providers_module,
        "provider_from_env",
        lambda: BrokenProvider(),
    )
    provider = InMemoryServiceProvider()
    provider.create_user("user_001")
    config = ServiceConfig(llm_extractor=True)
    with TestClient(create_app(provider=provider, config=config)) as client:
        resp = client.post(
            "/api/NebulonMind/intelligence/decide",
            json={"text": "My name is Sathya."},
            params={"user_id": "user_001"},
        )
    assert resp.status_code == 200
    data = resp.json()["data"]
    # Rules (identity) produced the decision, not the crashed LLM.
    categories = {d["candidate"]["category"] for d in data["decisions"]}
    assert "identity" in categories


class EmptyProvider:
    """Structured call *succeeds* but returns zero decisions (the caveat:
    the NVIDIA model once returned an empty payload)."""

    def structured(self, prompt, schema):
        return {"decisions": []}


def test_toggle_on_empty_llm_payload_falls_back_to_rules(monkeypatch):
    from nmd_host.intelligence import providers as providers_module

    monkeypatch.setattr(
        providers_module, "provider_from_env", lambda: EmptyProvider()
    )
    provider = InMemoryServiceProvider()
    provider.create_user("user_001")
    config = ServiceConfig(llm_extractor=True)
    with TestClient(create_app(provider=provider, config=config)) as client:
        resp = client.post(
            "/api/NebulonMind/intelligence/process",
            json={"text": "My name is Sathya. I like hiking."},
            params={"user_id": "user_001"},
        )
    assert resp.status_code == 200
    data = resp.json()["data"]
    # An empty LLM payload must not silently lose the turn — the rule
    # extractor decides instead and the facts still store.
    assert data["decisions"]
    assert any(i["action"] == "STORE" for i in data["ingestions"])


def test_toggle_on_remember_tool_uses_llm_extractor(monkeypatch):
    from nmd_host.agent.tools import build_memory_toolkit
    from nmd_host.agent import AgentRuntime
    from nmd_host.intelligence import providers as providers_module

    monkeypatch.setattr(
        providers_module, "provider_from_env", lambda: StubProvider(_payload())
    )
    provider = InMemoryServiceProvider()
    provider.create_user("user_001")
    bundle = provider.bundle("user_001")

    class _ScriptedLLM:
        def complete(self, prompt, *, system=None, temperature=0.0):
            return type("R", (), {
                "text": '{"tool": "remember", "arguments": {"text": "' + THIRD_PERSON_FACT + '"}}'
            })()

    llm = _ScriptedLLM()
    runtime = AgentRuntime(
        llm,
        build_memory_toolkit(
            bundle.repository,
            bundle.manager,
            "user_001",
            extractor=LLMExtractor(StubProvider(_payload())),
        ),
    )
    result = runtime.chat("Remember: " + THIRD_PERSON_FACT)
    assert result.tool_calls and result.tool_calls[0].tool == "remember"
    assert result.tool_calls[0].ok is True
    stored = bundle.repository.search("NebulonDB", top_k=5)
    assert any(m.content.text == THIRD_PERSON_FACT for m in stored)


# --------------------------------------------------------------------- #
# Caveat fix: TEMPORARY categories get a default expiry at conversion    #
# --------------------------------------------------------------------- #


def _candidate(category):
    return MemoryCandidate(
        text=THIRD_PERSON_FACT,
        category=MemoryCategory(category),
        importance=0.9,
        confidence=0.9,
        entities=["NebulonMind", "NebulonDB"],
        relationships=[
            {"source": "NebulonMind", "target": "NebulonDB", "relation": "BUILDS"}
        ],
        subject="NebulonMind",
    )


def test_temporary_categories_stamp_default_expiry():
    memory = candidate_to_memory(_candidate("knowledge"), "user_001")
    assert memory.lifecycle.retention_policy is RetentionPolicy.TEMPORARY
    assert memory.lifecycle.expires_at is not None


def test_permanent_categories_have_no_expiry():
    memory = candidate_to_memory(_candidate("identity"), "user_001")
    assert memory.lifecycle.retention_policy is RetentionPolicy.PERMANENT
    assert memory.lifecycle.expires_at is None


def test_temporary_ttl_env_override(monkeypatch):
    from nmd_host.lifecycle.lifecycle import now_utc

    monkeypatch.setenv("NMD_TEMPORARY_TTL_SECONDS", "3600")
    memory = candidate_to_memory(_candidate("knowledge"), "user_001")
    remaining = (memory.lifecycle.expires_at - now_utc()).total_seconds()
    assert 3590 <= remaining <= 3600


def test_llm_toggle_knowledge_category_now_stores(monkeypatch):
    from nmd_host.intelligence import providers as providers_module

    # KNOWLEDGE maps to TEMPORARY retention in classifier.py — previously the
    # Step 3 gate rejected it as an invalid config; the converter now stamps
    # a default expiry so the LLM-extracted fact can store and be recalled.
    monkeypatch.setattr(
        providers_module,
        "provider_from_env",
        lambda: StubProvider(_payload(category="knowledge", relation="REFERENCES")),
    )
    provider = InMemoryServiceProvider()
    provider.create_user("user_001")
    config = ServiceConfig(llm_extractor=True)
    with TestClient(create_app(provider=provider, config=config)) as client:
        resp = client.post(
            "/api/NebulonMind/intelligence/process",
            json={"text": THIRD_PERSON_FACT},
            params={"user_id": "user_001"},
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["ingestions"][0]["action"] == "STORE"
        stored = client.get(
            "/api/NebulonMind/search",
            params={"query": "NebulonDB", "user_id": "user_001"},
        ).json()["data"]["results"]
        assert any(m["content"]["text"] == THIRD_PERSON_FACT for m in stored)