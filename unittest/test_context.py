"""Context builder tests (3.9): max items / max characters, ordering,
metadata, provenance, empty result. Offline."""

from datetime import datetime, timedelta, timezone

from nmd_host.core.models import MemoryType
from nmd_host.lifecycle.context import MemoryContextBuilder
from conftest import make_memory


def _memory(text, score=0.5, category="general", created=None):
    memory = make_memory(text, score=score)
    memory.memory_id = f"mem_{len(text)}_{score}".replace(".", "_")
    memory.classification.category = category
    memory.lifecycle.created_at = created or datetime.now(timezone.utc)
    return memory


def test_empty_result():
    assert MemoryContextBuilder().build([]) == ""


def test_max_items_respected():
    memories = [_memory(f"Fact number {i}") for i in range(10)]
    context = MemoryContextBuilder().build(memories, max_items=3)
    assert context.count("[") >= 3
    lines = context.splitlines()
    assert "[3]" in context
    assert "[4]" not in context


def test_max_characters_respected():
    memories = [_memory("A" * 400 + str(i)) for i in range(20)]
    context = MemoryContextBuilder().build(memories, max_characters=1500)
    assert len(context) <= 1500


def test_ordering_relevance_importance_recency():
    now = datetime.now(timezone.utc)
    top_relevance = _memory("User's name is Sathya", score=0.5, created=now - timedelta(days=100))
    important = _memory("User builds NebulonDB", score=0.99, created=now - timedelta(days=5))
    fresh = _memory("User is debugging HNSW today", score=0.55, created=now - timedelta(hours=1))
    builder = MemoryContextBuilder()
    context = builder.build([top_relevance, fresh, important], max_items=10)
    positions = [context.index(entry) for entry in
                 [top_relevance.content.text, important.content.text, fresh.content.text]]
    assert positions[0] < positions[1] < positions[2]  # relevance, then importance, then recency


def test_metadata_and_provenance_present():
    memory = _memory("User's name is Sathya", score=0.99)
    memory.classification.memory_type = MemoryType.LONG_TERM
    context = MemoryContextBuilder().build([memory])
    assert "memory_id: mem_" in context
    assert "type: long_term" in context
    assert "category: general" in context
    assert "importance: 0.99" in context
    assert "confidence: 0.50" in context
    assert "source: conversation" in context


def test_timestamps_present_in_provenance():
    now = datetime.now(timezone.utc)
    updated = now - timedelta(days=3)
    memory = _memory("User's name is Sathya", created=now)
    memory.lifecycle.updated_at = updated
    context = MemoryContextBuilder().build([memory])
    assert "created:" in context
    assert "updated:" in context
    assert now.strftime("%Y-%m-%dT") in context
    assert "unknown" not in context


def test_timestamp_unknown_when_missing():
    memory = _memory("User's name is Sathya")
    memory.lifecycle.created_at = None
    memory.lifecycle.updated_at = None
    context = MemoryContextBuilder().build([memory])
    assert "created: unknown" in context
    assert "updated: unknown" in context


def test_timestamp_accepts_float_epochs():
    memory = _memory("User's name is Sathya")
    memory.lifecycle.created_at = 1600000000.0
    memory.lifecycle.updated_at = None
    context = MemoryContextBuilder().build([memory])
    assert "created: 2020-09-13T" in context


def test_superseded_tag_rendered():
    from nmd_host.core.models import MemoryStatus

    memory = _memory("User works at Apple", created=datetime.now(timezone.utc))
    memory.status = MemoryStatus.ARCHIVED
    memory.content.structured_data = {"superseded_by": "mem_samsung"}
    context = MemoryContextBuilder().build([memory])
    assert "superseded: yes, by mem_samsung" in context


def test_no_superseded_tag_when_not_superseded():
    memory = _memory("User works at Apple", created=datetime.now(timezone.utc))
    context = MemoryContextBuilder().build([memory])
    assert "superseded" not in context


def test_archived_memory_is_demoted_in_ordering():
    from nmd_host.core.models import MemoryStatus

    now = datetime.now(timezone.utc)
    archived = _memory("User works at Apple", score=0.9, created=now - timedelta(days=100))
    archived.status = MemoryStatus.ARCHIVED
    current = _memory("User works at Samsung", score=0.5, created=now - timedelta(days=1))
    builder = MemoryContextBuilder()
    context = builder.build([current, archived], max_items=10)
    assert context.index("Samsung") < context.index("Apple")


def test_blocks_are_formatted_with_index():
    memories = [_memory("Fact A"), _memory("Fact B")]
    context = MemoryContextBuilder().build(memories)
    assert "[1] Fact A" in context
    assert "[2] Fact B" in context


def test_budget_cuts_later_items_first():
    memories = [_memory("Short fact"), _memory("X" * 2000)]
    context = MemoryContextBuilder().build(memories, max_characters=500)
    assert "Short fact" in context
    assert "X" * 2000 not in context


def test_builder_never_mutates_input_memories():
    memories = [
        _memory("User's name is Sathya", score=0.9),
        _memory("User builds NebulonDB", score=0.8),
    ]
    before = [m.model_dump() for m in memories]
    MemoryContextBuilder().build(memories, max_items=1, max_characters=100)
    after = [m.model_dump() for m in memories]
    assert after == before
