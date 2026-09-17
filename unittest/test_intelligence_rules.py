"""Rule-based extractor tests: deterministic extraction, classification,
scoring, entities and relationships — all offline."""

import pytest

from nmd_host.core.models import MemoryType, RetentionPolicy
from nmd_host.intelligence.classifier import (
    CATEGORY_TO_MEMORY_TYPE,
    classify_sentence,
    memory_type_for,
    retention_for,
)
from nmd_host.intelligence.entities import extract_entities
from nmd_host.intelligence.relations import build_relationships, relation_for
from nmd_host.intelligence.rules import RuleBasedExtractor, split_sentences
from nmd_host.intelligence.schemas import Conversation, MemoryCategory
from nmd_host.intelligence.scoring import confidence_for, importance_for


@pytest.fixture
def extractor():
    return RuleBasedExtractor()


def decisions_for(text, extractor):
    return extractor.extract(Conversation.from_user_message(text))


def remembered(text, extractor):
    return [d for d in decisions_for(text, extractor) if d.should_remember]


# --------------------------------------------------------------------- #
# 2.3 rules                                                             #
# --------------------------------------------------------------------- #

def test_split_sentences():
    assert split_sentences("Hi! How are you? I am fine.") == [
        "Hi!", "How are you?", "I am fine."
    ]


def test_identity_name(extractor):
    decisions = remembered("My name is Sathya.", extractor)
    assert len(decisions) == 1
    candidate = decisions[0].candidate
    assert candidate.category is MemoryCategory.IDENTITY
    assert candidate.importance >= 0.9
    assert candidate.confidence >= 0.9
    assert "Sathya" in candidate.entities
    assert candidate.subject == "Sathya"


def test_identity_place(extractor):
    candidate = remembered("I live in Bangalore, India.", extractor)[0].candidate
    assert candidate.category is MemoryCategory.IDENTITY
    assert candidate.entities == ["Bangalore", "India"]


def test_identity_work_at_populates_employer_slot(extractor):
    candidate = remembered("I work at Apple.", extractor)[0].candidate
    assert candidate.category is MemoryCategory.IDENTITY
    assert candidate.slots == {"identity": {"employer": "Apple"}}


def test_identity_name_populates_name_slot(extractor):
    candidate = remembered("My name is Sathya.", extractor)[0].candidate
    assert candidate.slots == {"identity": {"name": "Sathya"}}


def test_non_identity_rules_have_no_slots(extractor):
    candidate = remembered("I work with Python for backend services.", extractor)[0].candidate
    assert candidate.category is MemoryCategory.SKILL
    assert candidate.slots is None


def test_skill_and_subject_resolution(extractor):
    conversation = Conversation.from_user_message(
        "My name is Sathya. I work with Python for backend services."
    )
    decisions = extractor.extract(conversation)
    assert len(decisions) == 2
    skill = next(d for d in decisions if d.candidate.category is MemoryCategory.SKILL)
    assert skill.candidate.subject == "Sathya"
    assert skill.candidate.relationships[0].source == "Sathya"
    assert skill.candidate.relationships[0].target == "Python"
    assert skill.candidate.relationships[0].relation == "HAS_SKILL"
    assert "Python" in skill.candidate.entities


def test_preference(extractor):
    candidate = remembered("I like hiking in the mountains.", extractor)[0].candidate
    assert candidate.category is MemoryCategory.PREFERENCE
    assert candidate.relationships[0].relation == "PREFERS"


def test_project(extractor):
    candidate = remembered("I'm building NebulonMind with Python right now.", extractor)[0].candidate
    assert candidate.category is MemoryCategory.PROJECT
    assert candidate.relationships[0].target == "NebulonMind"
    assert candidate.relationships[0].relation == "BUILDS"
    assert "NebulonMind" in candidate.entities


def test_goal(extractor):
    candidate = remembered("I want to learn Rust this year.", extractor)[0].candidate
    assert candidate.category is MemoryCategory.GOAL
    assert candidate.relationships[0].relation == "AIMS_TO"


def test_event(extractor):
    candidate = remembered("I just released NebulonMind 0.1.", extractor)[0].candidate
    assert candidate.category is MemoryCategory.EVENT
    assert "NebulonMind" in candidate.entities


def test_fact(extractor):
    candidate = remembered("I have 5 years of experience.", extractor)[0].candidate
    assert candidate.category is MemoryCategory.FACT


def test_weak_rule_is_not_remembered(extractor):
    decisions = decisions_for("I think the weather is nice today.", extractor)
    assert len(decisions) == 1
    assert decisions[0].should_remember is False


def test_keyword_signal_without_rule_is_rejected(extractor):
    decisions = decisions_for("I am super excited about the weekend!", extractor)
    assert all(d.should_remember is False for d in decisions)


def test_assistant_turns_ignored(extractor):
    conversation = Conversation(
        turns=[
            {"role": "assistant", "content": "I like chess and I am great."},
            {"role": "user", "content": "My name is Sathya."},
        ]
    )
    decisions = extractor.extract(conversation)
    assert len(decisions) == 1
    assert decisions[0].candidate.category is MemoryCategory.IDENTITY


def test_dedupe_repeated_sentence(extractor):
    decisions = remembered("I like coffee. I like coffee.", extractor)
    assert len(decisions) == 1


# --------------------------------------------------------------------- #
# 2.7 entities                                                          #
# --------------------------------------------------------------------- #

def test_extract_entities_capitalized():
    assert extract_entities("I work with Python and PostgreSQL") == [
        "Python", "PostgreSQL"
    ]


def test_extract_entities_skips_sentence_start_words():
    entities = extract_entities("Today I built a new feature with Rust")
    assert "Today" not in entities
    assert "Rust" in entities


def test_extract_entities_whitelist_case_insensitive():
    assert "nebulondb" in extract_entities("I use nebulondb daily")


# --------------------------------------------------------------------- #
# 2.8 relationships                                                     #
# --------------------------------------------------------------------- #

def test_relation_for_categories():
    assert relation_for(MemoryCategory.SKILL) == "HAS_SKILL"
    assert relation_for(MemoryCategory.PREFERENCE) == "PREFERS"
    assert relation_for(MemoryCategory.IDENTITY) is None


def test_build_relationships():
    rels = build_relationships(MemoryCategory.SKILL, "Python", subject="Sathya")
    assert len(rels) == 1
    assert rels[0].source == "Sathya"
    assert rels[0].target == "Python"
    assert rels[0].relation == "HAS_SKILL"


def test_build_relationships_identity_is_empty():
    assert build_relationships(MemoryCategory.IDENTITY, "Sathya") == []


def test_build_relationships_default_subject():
    rels = build_relationships(MemoryCategory.SKILL, "Python")
    assert rels[0].source == "User"


# --------------------------------------------------------------------- #
# 2.5 classification                                                    #
# --------------------------------------------------------------------- #

def test_memory_type_mapping():
    assert memory_type_for(MemoryCategory.IDENTITY) is MemoryType.LONG_TERM
    assert memory_type_for(MemoryCategory.PREFERENCE) is MemoryType.LONG_TERM
    assert memory_type_for(MemoryCategory.SKILL) is MemoryType.LONG_TERM
    assert memory_type_for(MemoryCategory.GOAL) is MemoryType.LONG_TERM
    assert memory_type_for(MemoryCategory.PROJECT) is MemoryType.SEMANTIC
    assert memory_type_for(MemoryCategory.EVENT) is MemoryType.EPISODIC
    assert memory_type_for(MemoryCategory.TASK) is MemoryType.WORKING
    assert memory_type_for(MemoryCategory.OPINION) is MemoryType.SHORT_TERM
    assert memory_type_for(MemoryCategory.KNOWLEDGE) is MemoryType.KNOWLEDGE


def test_every_memory_type_is_reachable_through_a_category():
    """Every Step 1 MemoryType must be produced by the Step 2 classifier —
    never a silently invented type.

    Note: DOC is a special document type not produced by the classifier from
    conversation — it's set via the API convenience field.
    """
    produced = set(CATEGORY_TO_MEMORY_TYPE.values())
    all_types = set(MemoryType)
    # DOC is set via API, not classifier
    assert produced == all_types - {MemoryType.DOC}


def test_knowledge_rule_extracts_knowledge_memory(extractor):
    candidate = remembered(
        "According to the docs, NebulonDB embeds vectors server-side.", extractor
    )[0].candidate
    assert candidate.category is MemoryCategory.KNOWLEDGE
    assert candidate.relationships[0].relation == "REFERENCES"
    assert memory_type_for(MemoryCategory.KNOWLEDGE) is MemoryType.KNOWLEDGE


def test_retention_mapping():
    assert retention_for(MemoryCategory.IDENTITY) is RetentionPolicy.PERMANENT
    assert retention_for(MemoryCategory.TASK) is RetentionPolicy.SESSION
    assert retention_for(MemoryCategory.EVENT) is RetentionPolicy.TEMPORARY
    assert retention_for(MemoryCategory.KNOWLEDGE) is RetentionPolicy.TEMPORARY


def test_classify_sentence_signals():
    assert classify_sentence("I prefer tea over coffee") is MemoryCategory.PREFERENCE
    assert classify_sentence("completely unrelated greeting") is None


# --------------------------------------------------------------------- #
# 2.6 scoring                                                           #
# --------------------------------------------------------------------- #

def test_scoring_bounds():
    for strength in (0.0, 0.4, 0.8, 1.0):
        assert 0.0 <= confidence_for(strength) <= 1.0
    for category in MemoryCategory:
        assert 0.0 <= importance_for(category) <= 1.0


def test_importance_ordering():
    assert importance_for(MemoryCategory.IDENTITY) > importance_for(MemoryCategory.FACT)
    assert importance_for(MemoryCategory.FACT) > importance_for(MemoryCategory.TASK)


def test_confidence_rises_with_strength_and_entities():
    assert confidence_for(0.8, has_entities=True) > confidence_for(0.8)
    assert confidence_for(0.8) > confidence_for(0.5)
