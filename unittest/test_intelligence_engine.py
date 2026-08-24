"""Engine + converter + validation tests (2.4, 2.9, 2.13) — offline."""

import pytest

from nmd_host.core.models import MemoryType, Priority, Relationship, RetentionPolicy
from nmd_host.intelligence.converter import candidate_to_memory
from nmd_host.intelligence.engine import MemoryDecisionEngine
from nmd_host.intelligence.schemas import (
    Conversation,
    MemoryCandidate,
    MemoryCategory,
    MemoryDecision,
)
from nmd_host.intelligence.validation import sanitize_decision, validate_decisions


@pytest.fixture
def engine():
    return MemoryDecisionEngine()


def candidate(**kwargs):
    defaults = {
        "text": "Sathya works with Python",
        "category": MemoryCategory.SKILL,
        "importance": 0.8,
        "confidence": 0.9,
        "entities": ["Sathya", "Python"],
        "relationships": [
            {"source": "Sathya", "target": "Python", "relation": "HAS_SKILL"}
        ],
    }
    defaults.update(kwargs)
    return MemoryCandidate(**defaults)


# --------------------------------------------------------------------- #
# 2.4 engine                                                            #
# --------------------------------------------------------------------- #

def test_engine_remembers_only_accepted(engine):
    conversation = Conversation.from_user_message(
        "My name is Sathya. I think the weather is fine."
    )
    decisions = engine.extract(conversation)
    assert len(decisions) == 1
    assert decisions[0].candidate.category is MemoryCategory.IDENTITY
    assert decisions[0].should_remember is True


def test_engine_decide_returns_all_verdicts(engine):
    conversation = Conversation.from_user_message(
        "My name is Sathya. I think the weather is fine."
    )
    decisions = engine.decide(conversation)
    assert len(decisions) == 2
    assert sum(1 for d in decisions if d.should_remember) == 1


# --------------------------------------------------------------------- #
# 2.9 converter                                                         #
# --------------------------------------------------------------------- #

def test_candidate_to_memory_maps_all_fields():
    memory = candidate_to_memory(candidate(), user_id="user_001")
    assert memory.user_id == "user_001"
    assert memory.content.text == "Sathya works with Python"
    assert memory.classification.category == "skill"
    assert memory.classification.memory_type is MemoryType.LONG_TERM
    assert memory.importance.score == 0.8
    assert memory.importance.confidence == 0.9
    assert memory.importance.priority is Priority.HIGH
    assert memory.lifecycle.retention_policy is RetentionPolicy.PERMANENT
    assert memory.entities == ["Sathya", "Python"]
    assert memory.relationships[0].relation == "HAS_SKILL"
    assert memory.content.structured_data["category"] == "skill"


def test_converter_category_to_type():
    event = candidate_to_memory(candidate(category=MemoryCategory.EVENT), "u")
    assert event.classification.memory_type is MemoryType.EPISODIC
    assert event.lifecycle.retention_policy is RetentionPolicy.TEMPORARY

    task = candidate_to_memory(candidate(category=MemoryCategory.TASK), "u")
    assert task.classification.memory_type is MemoryType.WORKING
    assert task.lifecycle.retention_policy is RetentionPolicy.SESSION


def test_converter_priority_boundaries():
    assert candidate_to_memory(candidate(importance=0.9), "u").importance.priority is Priority.HIGH
    assert candidate_to_memory(candidate(importance=0.6), "u").importance.priority is Priority.MEDIUM
    assert candidate_to_memory(candidate(importance=0.4), "u").importance.priority is Priority.LOW


def test_converter_persists_slots_into_structured_data():
    memory = candidate_to_memory(
        candidate(slots={"identity": {"employer": "Apple"}}), user_id="u"
    )
    assert memory.content.structured_data["slots"] == {"identity": {"employer": "Apple"}}
    assert memory.content.structured_data["category"] == "skill"


def test_converter_omits_slots_when_absent():
    memory = candidate_to_memory(candidate(), user_id="u")
    assert "slots" not in memory.content.structured_data


def test_converter_roundtrip_through_truth_doc():
    memory = candidate_to_memory(candidate(), user_id="u")
    restored = type(memory).from_truth_doc(memory.to_truth_doc())
    assert restored.content.text == memory.content.text
    assert restored.classification.category == "skill"


# --------------------------------------------------------------------- #
# 2.13 validation                                                       #
# --------------------------------------------------------------------- #

def test_validation_drops_empty_text():
    assert sanitize_decision(
        {"candidate": {"text": "   ", "category": "skill"}, "should_remember": True}
    ) is None


def test_validation_drops_out_of_range_importance():
    raw = {
        "candidate": {
            "text": "Sathya likes Python",
            "category": "preference",
            "importance": 5.0,
            "confidence": 0.9,
        },
        "should_remember": True,
    }
    assert sanitize_decision(raw) is None


def test_validation_drops_invalid_category():
    raw = {
        "candidate": {"text": "hello world", "category": "quantum"},
        "should_remember": True,
    }
    assert sanitize_decision(raw) is None


def test_validation_adds_missing_relationship_entities():
    decision = MemoryDecision(
        candidate=candidate(
            entities=[],
            relationships=[Relationship(source="Sathya", target="Python", relation="HAS_SKILL")],
        )
    )
    fixed = validate_decisions([decision])[0]
    assert set(fixed.candidate.entities) == {"Sathya", "Python"}


def test_validation_drops_relationship_with_empty_label():
    raw = {
        "candidate": {
            "text": "Sathya works with Python",
            "category": "skill",
            "relationships": [{"source": "", "target": "Python", "relation": "HAS_SKILL"}],
        },
        "should_remember": True,
    }
    assert sanitize_decision(raw) is None


def test_validation_deduplicates():
    decisions = validate_decisions(
        [MemoryDecision(candidate=candidate()), MemoryDecision(candidate=candidate())]
    )
    assert len(decisions) == 1
