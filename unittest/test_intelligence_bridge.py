"""Bridge tests (2.10): MemoryIntelligence over an in-memory store —
conversation -> decisions -> Memory objects -> repository. Offline."""

from nmd_host.core.repository import InMemoryRepository
from nmd_host.intelligence.bridge import MemoryIntelligence
from nmd_host.intelligence.schemas import Conversation, MemoryCategory


def test_process_stores_only_remembered():
    repo = InMemoryRepository()
    intelligence = MemoryIntelligence(repo, user_id="user_007")
    conversation = Conversation.from_user_message(
        "My name is nmd_user_01. I think the weather is fine today."
    )
    memories = intelligence.process(conversation)

    assert len(memories) == 1
    stored = repo.search("nmd_user_01 name", top_k=5)
    assert len(stored) == 1
    assert stored[0].classification.category == "identity"


def test_process_persists_all_categories():
    repo = InMemoryRepository()
    intelligence = MemoryIntelligence(repo, user_id="user_007")
    conversation = Conversation.from_user_message(
        "My name is Sathya. I work with Python. I like hiking."
    )
    memories = intelligence.process(conversation)

    assert len(memories) == 3
    categories = {m.classification.category for m in memories}
    assert categories == {"identity", "skill", "preference"}
    assert len(repo.search("python", top_k=5)) >= 1


def test_process_returns_memories_without_storing():
    repo = InMemoryRepository()
    intelligence = MemoryIntelligence(repo, user_id="user_007")
    memories = intelligence.process_text("My name is nmd_user_01.", persist=False)
    assert len(memories) == 1
    assert repo.search("nmd_user_01", top_k=5) == []


def test_accepts_mind_like_store_with_store_method():
    class FakeMind:
        def __init__(self):
            self.stored = []

        def store(self, memory):
            self.stored.append(memory)

    mind = FakeMind()
    intelligence = MemoryIntelligence(mind, user_id="user_007")
    intelligence.process_text("I work with Python.")
    assert len(mind.stored) == 1
    assert mind.stored[0].entities == ["Python"]


def test_accepts_repository_like_store():
    repo = InMemoryRepository()
    intelligence = MemoryIntelligence(repo, user_id="user_007")
    intelligence.process_text("I like hiking.")
    assert len(repo._memories) == 1


def test_decide_does_not_persist():
    repo = InMemoryRepository()
    intelligence = MemoryIntelligence(repo, user_id="user_007")
    decisions = intelligence.decide(Conversation.from_user_message("I like hiking."))
    assert len(decisions) == 1
    assert decisions[0].candidate.category is MemoryCategory.PREFERENCE
    assert repo.search("hiking", top_k=5) == []


def test_invalid_store_raises():
    import pytest

    with pytest.raises(TypeError):
        MemoryIntelligence("not a store", user_id="u")
