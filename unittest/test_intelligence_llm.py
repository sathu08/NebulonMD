"""LLM extractor tests (2.11-2.13) using a fake provider — no network.

The fake implements the same ``LLMProvider`` contract as the real adapters,
so the extractor and validation layer are exercised end-to-end offline.
"""

import pytest

from nmd_host.intelligence.engine import MemoryDecisionEngine
from nmd_host.intelligence.llm_extractor import LLMExtractor
from nmd_host.intelligence.providers import LLMProviderError, _parse_json, provider_from_env
from nmd_host.intelligence.schemas import (
    Conversation,
    MemoryCategory,
    MemoryDecisionList,
)

VALID_PAYLOAD = {
    "decisions": [
        {
            "candidate": {
                "text": "My name is Sathya",
                "summary": "User's name is Sathya",
                "category": "identity",
                "importance": 0.99,
                "confidence": 0.95,
                "entities": ["Sathya"],
                "relationships": [],
                "subject": "Sathya",
            },
            "should_remember": True,
            "reason": "durable identity fact",
        },
        {
            "candidate": {
                "text": "I work with Python for backend services",
                "summary": "User works with Python",
                "category": "skill",
                "importance": 0.85,
                "confidence": 0.9,
                "entities": ["Python"],
                "relationships": [
                    {"source": "Sathya", "target": "Python", "relation": "HAS_SKILL"}
                ],
                "subject": "Sathya",
            },
            "should_remember": True,
            "reason": "durable skill",
        },
        {
            "candidate": {
                "text": "Let's grab coffee tomorrow",
                "category": "event",
                "importance": 0.4,
                "confidence": 0.3,
            },
            "should_remember": False,
            "reason": "transient",
        },
    ]
}


class FakeProvider:
    def __init__(self, payload):
        self.payload = payload
        self.last_prompt = None

    def structured(self, prompt, schema):
        self.last_prompt = prompt
        if isinstance(self.payload, str):
            return _parse_json(self.payload, schema)
        return self.payload


@pytest.fixture
def extractor():
    return LLMExtractor(FakeProvider(VALID_PAYLOAD))


def test_extracts_structured_decisions(extractor):
    conversation = Conversation.from_user_message(
        "My name is Sathya. I work with Python for backend services. Let's grab coffee tomorrow."
    )
    decisions = extractor.extract(conversation)
    assert len(decisions) == 3
    remembered = [d for d in decisions if d.should_remember]
    assert [d.candidate.category for d in remembered] == [
        MemoryCategory.IDENTITY,
        MemoryCategory.SKILL,
    ]
    assert remembered[1].candidate.relationships[0].relation == "HAS_SKILL"
    assert decisions[2].should_remember is False


def test_prompt_contains_conversation_and_categories(extractor):
    extractor.extract(Conversation.from_user_message("I like chess."))
    assert "I like chess." in extractor.provider.last_prompt
    assert "identity" in extractor.provider.last_prompt
    assert "HAS_SKILL" in extractor.provider.last_prompt


def test_rejects_unparseable_reply():
    provider = FakeProvider("this is not json at all")
    llm = LLMExtractor(provider)
    with pytest.raises(LLMProviderError):
        llm.extract(Conversation.from_user_message("hello"))


def test_parse_json_fenced():
    schema = MemoryDecisionList
    raw = '```json\n' + '{"decisions": []}\n' + '```'
    assert _parse_json(raw, schema).decisions == []


def test_parse_json_prose_wrapped():
    raw = 'Here you go:\n{"decisions": []}\nHope that helps.'
    assert _parse_json(raw, schema := MemoryDecisionList).decisions == []


def test_validation_drops_invalid_llm_output():
    payload = {
        "decisions": [
            {"candidate": {"text": "", "category": "fact"}},
            {"candidate": {"text": "valid memory here", "category": "fact", "importance": 7}},
            {"candidate": {"text": "Sathya knows Python", "category": "skill", "entities": ["Python"]}},
        ]
    }
    llm = LLMExtractor(FakeProvider(payload))
    decisions = llm.extract(Conversation.from_user_message("hello"))
    assert len(decisions) == 1
    assert decisions[0].candidate.text == "Sathya knows Python"


def test_min_confidence_filter():
    llm = LLMExtractor(FakeProvider(VALID_PAYLOAD), min_confidence=0.9)
    decisions = llm.extract(Conversation.from_user_message("hello"))
    assert all(d.candidate.confidence >= 0.9 for d in decisions)


def test_engine_drives_llm_extractor_end_to_end():
    engine = MemoryDecisionEngine(LLMExtractor(FakeProvider(VALID_PAYLOAD)))
    decisions = engine.extract(
        Conversation.from_user_message("My name is Sathya. Let's grab coffee tomorrow.")
    )
    assert len(decisions) == 2
    assert all(d.candidate.category in (MemoryCategory.IDENTITY, MemoryCategory.SKILL) for d in decisions)


def test_provider_from_env_requires_provider(monkeypatch):
    monkeypatch.delenv("NMD_LLM_PROVIDER", raising=False)
    with pytest.raises(LLMProviderError):
        provider_from_env()


def test_provider_from_env_rejects_unknown(monkeypatch):
    monkeypatch.setenv("NMD_LLM_PROVIDER", "sky-whale")
    with pytest.raises(LLMProviderError):
        provider_from_env()
