"""Deduplication tests (3.3): exact / normalized / similar / different /
conflicting memories, plus the semantic layer via a searcher. Offline."""

from nmd_host.core.repository import InMemoryRepository
from nmd_host.lifecycle.deduplication import (
    DuplicateDetector,
    DuplicateResult,
    is_duplicate,
)
from conftest import make_memory


def _memories(ids_and_texts):
    memories = []
    for index, (text, category) in enumerate(ids_and_texts):
        memory = make_memory(text)
        memory.memory_id = f"mem_{index:04d}"
        memory.classification.category = category
        memories.append(memory)
    return memories


def test_exact_memory_id_duplicate():
    a, b = _memories([("Hello world", "fact"), ("Hello world", "fact")])
    b.memory_id = a.memory_id
    result = DuplicateDetector().check(a, [b])
    assert result.is_duplicate is True
    assert result.existing_memory_id == a.memory_id
    assert result.similarity == 1.0


def test_normalized_text_equality_duplicate():
    a, b = _memories([("My name is Sathya.", "identity"), ("my NAME is sathya", "identity")])
    result = DuplicateDetector().check(a, [b])
    assert result.is_duplicate is True
    assert result.similarity == 1.0


def test_reordered_tokens_duplicate():
    a, b = _memories([("My name is Sathya", "identity"), ("Sathya is my name.", "identity")])
    result = DuplicateDetector().check(a, [b])
    assert result.is_duplicate is True
    assert result.similarity == 1.0


def test_similar_content_same_category_duplicate():
    a, b = _memories([
        ("User works with Python for backend services", "skill"),
        ("User works with Python for backend services and databases", "skill"),
    ])
    result = DuplicateDetector().check(a, [b])
    assert result.is_duplicate is True
    assert result.existing_memory_id == b.memory_id


def test_different_memories_not_duplicates():
    a, b = _memories([("User likes hiking", "preference"), ("User builds AI systems", "project")])
    result = DuplicateDetector().check(a, [b])
    assert result.is_duplicate is False
    assert result.existing_memory_id is None


def test_similar_text_different_category_not_duplicate():
    a, b = _memories([
        ("User works with Python for backend services", "skill"),
        ("User works with Python for backend services now", "fact"),
    ])
    result = DuplicateDetector().check(a, [b])
    assert result.is_duplicate is False


def test_conflicting_memories_not_duplicates():
    a, b = _memories([
        ("I like Python", "preference"),
        ("I don't like Python anymore", "preference"),
    ])
    result = DuplicateDetector().check(a, [b])
    assert result.is_duplicate is False


def test_semantic_duplicate_via_searcher():
    repo = InMemoryRepository()
    a = make_memory("User prefers espresso in the morning")
    a.memory_id = "mem_aaaa"
    repo.create(a)
    b = make_memory("User enjoys coffee drinks at breakfast")
    b.memory_id = "mem_bbbb"

    detector = DuplicateDetector(searcher=repo)
    result = detector.check(a, [b])
    assert result.is_duplicate is True
    assert result.existing_memory_id == b.memory_id


def test_semantic_layer_without_searcher_gives_no_evidence():
    a, b = _memories([
        ("User prefers espresso in the morning", "preference"),
        ("User enjoys coffee drinks at breakfast", "preference"),
    ])
    result = DuplicateDetector().check(a, [b])
    assert result.is_duplicate is False


def test_find_duplicate_returns_the_existing_memory():
    a, b = _memories([("My name is Sathya", "identity"), ("My name is Sathya.", "identity")])
    found = DuplicateDetector().find_duplicate(a, [b])
    assert found is not None
    assert found.memory_id == b.memory_id


def test_find_duplicate_returns_none_when_clean():
    a, b = _memories([("User likes hiking", "preference"), ("User builds AI systems", "project")])
    assert DuplicateDetector().find_duplicate(a, [b]) is None


def test_is_duplicate_helper():
    a, b = _memories([("My name is Sathya", "identity"), ("Sathya is my name.", "identity")])
    assert is_duplicate(a, b) is True
    c = make_memory("Unrelated fact")
    assert is_duplicate(a, c) is False


def test_self_is_never_a_duplicate():
    a, _ = _memories([("My name is Sathya", "identity"), ("x", "fact")])
    result = DuplicateDetector().check(a, [a])
    assert result.is_duplicate is False


def test_explicit_result_shape():
    a, b = _memories([("User likes hiking", "preference"), ("User likes hiking", "preference")])
    result = DuplicateDetector().check(a, [b])
    assert isinstance(result, DuplicateResult)
    assert result.is_duplicate is True
    assert result.existing_memory_id == b.memory_id
