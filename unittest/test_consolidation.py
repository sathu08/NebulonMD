"""Consolidation tests (3.7): KEEP / UPDATE / MERGE candidate / IGNORE,
conflicting facts, provenance preservation. Offline."""

from nmd_host.lifecycle.consolidation import ConsolidationDecision, MemoryConsolidator
from conftest import make_memory


def _memory(text, category="skill", user_id="user_001"):
    memory = make_memory(text, user_id=user_id)
    memory.memory_id = f"mem_{abs(hash(text)) % 10**6:06d}"
    memory.classification.category = category
    return memory


def test_distinct_memories_all_keep():
    memories = [
        _memory("User works with Python", category="skill"),
        _memory("User likes hiking", category="preference"),
        _memory("User builds AI systems", category="project"),
    ]
    decisions = MemoryConsolidator().analyze(memories)
    assert len(decisions) == 3
    assert all(d.action == "KEEP" for d in decisions)
    assert all(len(d.source_memory_ids) == 1 for d in decisions)


def test_near_identical_same_category_merge_candidate():
    memories = [
        _memory("User works with Python for backend services", category="skill"),
        _memory("User works with Python for backend services and databases", category="skill"),
    ]
    decisions = MemoryConsolidator().analyze(memories)
    assert any(d.action == "MERGE" for d in decisions)
    merged = next(d for d in decisions if d.action == "MERGE")
    assert set(merged.source_memory_ids) == {m.memory_id for m in memories}


def test_conflicting_facts_are_ignored_not_merged():
    memories = [
        _memory("I like Python", category="preference"),
        _memory("I don't like Python anymore", category="preference"),
    ]
    decisions = MemoryConsolidator().analyze(memories)
    assert any(d.action == "IGNORE" for d in decisions)
    ignored = next(d for d in decisions if d.action == "IGNORE")
    assert set(ignored.source_memory_ids) == {m.memory_id for m in memories}
    assert "conflict" in ignored.reason.lower()


def _with_slots(memory, employer):
    memory.content.structured_data = {
        "category": memory.classification.category,
        "slots": {"identity": {"employer": employer}},
    }
    return memory


def test_slot_conflict_detected_without_negation():
    """Apple vs Samsung fill the same employer slot — a conflict even though
    neither statement is negated and the texts are not highly similar."""
    memories = [
        _with_slots(_memory("I work at Apple", category="identity"), "Apple"),
        _with_slots(_memory("I now work at Samsung", category="identity"), "Samsung"),
    ]
    decisions = MemoryConsolidator().analyze(memories)
    assert any(d.action == "IGNORE" for d in decisions)
    ignored = next(d for d in decisions if d.action == "IGNORE")
    assert set(ignored.source_memory_ids) == {m.memory_id for m in memories}
    assert "conflict" in ignored.reason.lower()


def test_same_slot_same_value_is_not_a_conflict():
    memories = [
        _with_slots(_memory("I work at Apple", category="identity"), "Apple"),
        _with_slots(_memory("I work for Apple", category="identity"), "Apple"),
    ]
    decisions = MemoryConsolidator().analyze(memories)
    assert not any(d.action == "IGNORE" for d in decisions)


def test_slot_conflict_requires_same_category():
    memories = [
        _with_slots(_memory("I work at Apple", category="identity"), "Apple"),
        _with_slots(_memory("I work at Samsung", category="fact"), "Samsung"),
    ]
    decisions = MemoryConsolidator().analyze(memories)
    assert not any(d.action == "IGNORE" for d in decisions)


def test_version_drift_recommends_update_but_keeps_provenance():
    memories = [
        _memory("I use Python 3.10 for my projects", category="skill"),
        _memory("I upgraded to Python 3.13 for my projects", category="skill"),
    ]
    decisions = MemoryConsolidator().analyze(memories)
    assert any(d.action == "UPDATE" for d in decisions)
    updated = next(d for d in decisions if d.action == "UPDATE")
    assert set(updated.source_memory_ids) == {m.memory_id for m in memories}
    assert "provenance" in updated.reason.lower()


def test_similar_text_different_category_keeps():
    memories = [
        _memory("User works with Python for backend services", category="skill"),
        _memory("User works with Python for backend services", category="fact"),
    ]
    decisions = MemoryConsolidator().analyze(memories)
    assert all(d.action == "KEEP" for d in decisions)


def test_no_automatic_merge_happens():
    """Consolidation only produces recommendations — nothing is deleted."""
    memories = [
        _memory("User works with Python for backend services", category="skill"),
        _memory("User works with Python for backend services and databases", category="skill"),
    ]
    decisions = MemoryConsolidator().analyze(memories)
    assert len(memories) == 2
    assert all(m.memory_id for m in memories)
    assert any(d.action == "MERGE" for d in decisions)


def test_empty_input():
    assert MemoryConsolidator().analyze([]) == []


def test_decision_is_explicit_model():
    decision = ConsolidationDecision(
        action="MERGE",
        source_memory_ids=["mem_1", "mem_2"],
        reason="similar",
    )
    assert decision.action == "MERGE"
    assert decision.source_memory_ids == ["mem_1", "mem_2"]
