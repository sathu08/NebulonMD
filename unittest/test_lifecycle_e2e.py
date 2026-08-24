"""Step 3 end-to-end tests.

Offline: the full Step 2 → Step 3 pipeline over ``InMemoryRepository``.
Live (auto-skip when NebulonDB is down): conversation → extraction → Memory
→ lifecycle → NebulonMind.store() → fresh mind → retrieve → rank → context,
plus duplicate detection and temporary-memory expiration/cleanup.
"""

from datetime import datetime, timedelta, timezone

import pytest

from nmd_host.core.models import RetentionPolicy
from nmd_host.core.mind import NebulonMind
from nmd_host.core.repository import InMemoryRepository, NebulonMindRepository
from nmd_host.intelligence.bridge import MemoryIntelligence
from nmd_host.intelligence.converter import candidate_to_memory
from nmd_host.intelligence.schemas import Conversation
from nmd_host.lifecycle.manager import build_lifecycle_manager
from conftest import make_memory

# ---------------------------------------------------------------------- #
# Offline: Step 2 -> Step 3 -> repository                                 #
# ---------------------------------------------------------------------- #


def test_offline_conversation_to_context():
    repo = InMemoryRepository()
    intelligence = MemoryIntelligence(repo, user_id="user_007")
    conversation = Conversation.from_user_message(
        "My name is Sathya. I work with Python for backend services."
    )
    decisions = intelligence.decide(conversation)
    memories = [candidate_to_memory(d.candidate, "user_007") for d in decisions]

    manager = build_lifecycle_manager(repo)
    for memory in memories:
        verdict = manager.ingest(memory)
        assert verdict.action == "STORE"
        repo.create(memory)

    assert len(repo.list_all()) == 2
    results = manager.retrieve("what do I know about Sathya Python backend work?", top_k=2)
    assert len(results) == 2
    assert any("Sathya" in m.content.text for m in results)
    assert any("Python" in m.content.text for m in results)

    context = manager.build_context(results)
    assert "[1]" in context and "[2]" in context
    assert "memory_id: mem_" in context


def test_offline_duplicate_detection_on_ingest():
    repo = InMemoryRepository()
    manager = build_lifecycle_manager(repo)
    first = make_memory("My name is Sathya.")
    first.lifecycle.retention_policy = RetentionPolicy.PERMANENT
    repo.create(first)
    second = make_memory("My name is Sathya")
    second.lifecycle.retention_policy = RetentionPolicy.PERMANENT
    verdict = manager.ingest(second)
    assert verdict.action == "DUPLICATE"
    assert verdict.existing_memory is not None
    assert verdict.existing_memory.memory_id == first.memory_id


def test_offline_expired_and_invalid_ingest():
    repo = InMemoryRepository()
    manager = build_lifecycle_manager(repo)

    expired = make_memory("Old news")
    expired.lifecycle.retention_policy = RetentionPolicy.TEMPORARY
    expired.lifecycle.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)
    assert manager.ingest(expired).action == "EXPIRED"

    invalid = make_memory("Temp without expiry")
    invalid.lifecycle.retention_policy = RetentionPolicy.TEMPORARY
    verdict = manager.ingest(invalid)
    assert verdict.action == "INVALID"


def test_offline_expiry_cleanup_blocks_retrieval():
    repo = InMemoryRepository()
    manager = build_lifecycle_manager(repo)

    memory = make_memory("Debugging HNSW indexing today")
    memory.lifecycle.retention_policy = RetentionPolicy.TEMPORARY
    memory.lifecycle.expires_at = datetime.now(timezone.utc) + timedelta(days=1)
    repo.create(memory)
    assert len(manager.retrieve("hnsw indexing", top_k=5)) == 1

    memory.lifecycle.expires_at = datetime.now(timezone.utc) - timedelta(minutes=1)
    repo.update(memory.memory_id, {"lifecycle": memory.lifecycle.model_dump(mode="json")})

    assert manager.cleanup() == 1
    assert repo.get(memory.memory_id) is None
    assert manager.retrieve("hnsw indexing", top_k=5) == []


def _identity_memory(text, employer, user_id="user_007"):
    memory = make_memory(text, user_id=user_id)
    memory.classification.category = "identity"
    memory.lifecycle.retention_policy = RetentionPolicy.PERMANENT
    memory.content.structured_data = {
        "category": "identity",
        "slots": {"identity": {"employer": employer}},
    }
    return memory


def test_offline_supersede_flow():
    """Apple then Samsung: ingest flags SUPERSEDE, archive freezes the old one."""
    from nmd_host.core.models import MemoryStatus

    repo = InMemoryRepository()
    manager = build_lifecycle_manager(repo)

    apple = repo.create(_identity_memory("I work at Apple", "Apple"))
    assert apple.status is MemoryStatus.ACTIVE

    samsung = _identity_memory("I now work at Samsung", "Samsung")
    verdict = manager.ingest(samsung)
    assert verdict.action == "SUPERSEDE"
    assert verdict.superseded_memory_ids == [apple.memory_id]

    stored_samsung = repo.create(samsung)
    assert stored_samsung.status is MemoryStatus.ACTIVE
    assert manager.archive_superseded(
        stored_samsung.memory_id or "", verdict.superseded_memory_ids
    ) == 1

    old = repo.get(apple.memory_id)
    assert old.status is MemoryStatus.ARCHIVED
    assert old.content.structured_data["superseded_by"] == stored_samsung.memory_id

    results = manager.retrieve("employer samsung apple", top_k=5)
    assert any("Samsung" in m.content.text for m in results)
    assert any("Apple" in m.content.text for m in results)

    context = manager.build_context(results)
    assert "Samsung" in context
    assert "superseded: yes" in context
    assert stored_samsung.memory_id in context


def test_offline_no_supersede_without_slots():
    repo = InMemoryRepository()
    manager = build_lifecycle_manager(repo)
    first = make_memory("I work at Apple")
    first.lifecycle.retention_policy = RetentionPolicy.PERMANENT
    repo.create(first)
    second = make_memory("I now work at Samsung")
    second.lifecycle.retention_policy = RetentionPolicy.PERMANENT
    assert manager.ingest(second).action == "STORE"


# ---------------------------------------------------------------------- #
# Live: NebulonDB-backed full pipeline (auto-skips when the API is down)  #
# ---------------------------------------------------------------------- #


def build_mind(user_id, api_client=None):
    return NebulonMind(user_id=user_id, client=api_client)


def test_live_conversation_to_context(api_client, unique_user):
    mind = build_mind(unique_user, api_client)
    intelligence = MemoryIntelligence(mind, user_id=unique_user)
    manager = build_lifecycle_manager(NebulonMindRepository(mind), searcher=mind)

    conversation = Conversation.from_user_message(
        "My name is Sathya. I work with Python for backend services."
    )
    decisions = intelligence.decide(conversation)
    memories = [candidate_to_memory(d.candidate, unique_user) for d in decisions]
    assert len(memories) == 2

    for memory in memories:
        assert manager.ingest(memory).action == "STORE"
        mind.store(memory)

    fresh = build_mind(unique_user, api_client)
    fresh_manager = build_lifecycle_manager(NebulonMindRepository(fresh), searcher=fresh)
    results = fresh_manager.retrieve("Sathya Python backend", top_k=5, user_id=unique_user)
    assert len(results) == 2
    assert any("Sathya" in m.content.text for m in results)
    assert any("Python" in m.content.text for m in results)

    context = fresh_manager.build_context(results)
    assert "Sathya" in context
    assert "memory_id: mem_" in context
    assert "importance:" in context


def test_live_duplicate_detection(api_client, unique_user):
    mind = build_mind(unique_user, api_client)
    manager = build_lifecycle_manager(NebulonMindRepository(mind), searcher=mind)
    first = mind.store(make_memory("My name is Sathya", user_id=unique_user))

    duplicate = make_memory("My name is Sathya.", user_id=unique_user)
    duplicate.lifecycle.retention_policy = RetentionPolicy.PERMANENT
    verdict = manager.ingest(duplicate, user_id=unique_user)
    assert verdict.action == "DUPLICATE"
    assert verdict.existing_memory is not None
    assert verdict.existing_memory.memory_id == first.memory_id

    different = make_memory("Sathya is building NebulonDB", user_id=unique_user)
    different.lifecycle.retention_policy = RetentionPolicy.PERMANENT
    assert manager.ingest(different, user_id=unique_user).action == "STORE"


def test_live_temporary_expiry_cleanup_blocks_retrieval(api_client, unique_user):
    mind = build_mind(unique_user, api_client)
    manager = build_lifecycle_manager(NebulonMindRepository(mind), searcher=mind)

    memory = make_memory("Debugging HNSW indexing today", user_id=unique_user)
    memory.lifecycle.retention_policy = RetentionPolicy.TEMPORARY
    memory.lifecycle.expires_at = datetime.now(timezone.utc) + timedelta(days=1)
    mind.store(memory)
    assert len(manager.retrieve("hnsw indexing", top_k=5, user_id=unique_user)) == 1

    memory.lifecycle.expires_at = datetime.now(timezone.utc) - timedelta(minutes=1)
    mind.update(memory.memory_id, {"lifecycle": memory.lifecycle.model_dump(mode="json")})

    assert manager.cleanup(user_id=unique_user) == 1
    fresh = build_mind(unique_user, api_client)
    assert fresh.get(memory.memory_id) is None
    fresh_manager = build_lifecycle_manager(NebulonMindRepository(fresh), searcher=fresh)
    assert fresh_manager.retrieve("hnsw indexing", top_k=5, user_id=unique_user) == []


def test_live_permanent_memory_survives_cleanup(api_client, unique_user):
    mind = build_mind(unique_user, api_client)
    manager = build_lifecycle_manager(NebulonMindRepository(mind), searcher=mind)
    permanent = mind.store(make_memory("My name is Sathya", user_id=unique_user))
    assert manager.cleanup(user_id=unique_user) == 0
    assert mind.get(permanent.memory_id) is not None
