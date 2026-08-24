"""Step 2 end-to-end: conversation -> decision -> Memory -> NebulonMind
storage -> recall. Backed by the live NebulonDB API (auto-skips when down).
"""

from nmd_host.core.mind import NebulonMind
from nmd_host.intelligence.bridge import MemoryIntelligence
from nmd_host.intelligence.schemas import Conversation


def build_mind(user_id, api_client=None):
    return NebulonMind(user_id=user_id, client=api_client)


def test_conversation_to_memory_to_recall(api_client, unique_user):
    mind = build_mind(unique_user, api_client)
    intelligence = MemoryIntelligence(mind, user_id=unique_user)

    conversation = Conversation.from_user_message(
        "My name is Sathya. I work with Python for backend services. I like hiking."
    )
    memories = intelligence.process(conversation)

    assert len(memories) == 3
    categories = {m.classification.category for m in memories}
    assert categories == {"identity", "skill", "preference"}
    assert all(m.importance.score > 0.7 for m in memories)

    fresh = build_mind(unique_user, api_client)
    assert len(fresh.truth.read_all()) == 3
    assert fresh.vector.count() == 3

    results = fresh.recall("sathya python backend", top_k=3)
    assert len(results) == 3
    skill = next(r for r in results if r.classification.category == "skill")
    assert "Python" in skill.entities
    assert any(
        r.source == "Sathya" and r.target == "Python" and r.relation == "HAS_SKILL"
        for r in skill.relationships
    )


def test_weak_memories_are_not_stored(api_client, unique_user):
    mind = build_mind(unique_user, api_client)
    intelligence = MemoryIntelligence(mind, user_id=unique_user)
    conversation = Conversation.from_user_message(
        "Hello there! I think the weather is nice today. My name is Sathya."
    )
    intelligence.process(conversation)

    fresh = build_mind(unique_user, api_client)
    assert len(fresh.truth.read_all()) == 1
    stored = fresh.recall("weather nice", top_k=5)
    assert all("weather" not in m.content.text for m in stored)


def test_graph_has_relationship_edges(api_client, unique_user):
    mind = build_mind(unique_user, api_client)
    intelligence = MemoryIntelligence(mind, user_id=unique_user)
    intelligence.process_text("My name is Sathya. I work with Python.")

    fresh = build_mind(unique_user, api_client)
    edges = fresh.graph.count_edges()
    assert edges >= 4  # HAS_ENTITY x3 + HAS_SKILL (edges idempotent across reads)

    skill = next(
        m for m in fresh.recall("python skill", top_k=5)
        if m.classification.category == "skill"
    )
    assert set(fresh.graph.entities_of(skill.memory_id)) >= {"Sathya", "Python"}
