"""End-to-end tests: store -> fresh mind -> recall, via the NebulonDB API.

Every test uses a unique user id so segments never collide on the shared
server.
"""

import pytest

from nmd_host.core.mind import NebulonMind
from nmd_host.core.models import Relationship
from conftest import make_memory


def build_mind(user_id, api_client=None):
    return NebulonMind(user_id=user_id, client=api_client)


def sample_memories(user_id="user_001"):
    return [
        make_memory(
            "My name is nmd001 and I build AI systems",
            memory_type="long_term",
            entities=["nmd001", "AI"],
            relationships=[Relationship(source="nmd001", target="AI", relation="BUILDS")],
            user_id=user_id,
        ),
        make_memory(
            "nmd001 works with Python for backend services",
            memory_type="semantic",
            entities=["nmd001", "Python"],
            user_id=user_id,
        ),
        make_memory(
            "User likes hiking in the mountains on weekends",
            memory_type="episodic",
            entities=["Hiking"],
            user_id=user_id,
        ),
    ]


def test_store_recall_e2e(api_client, unique_user):
    mind = build_mind(unique_user, api_client)
    memories = sample_memories(unique_user)
    stored = [mind.store(m) for m in memories]
    assert all(m.memory_id for m in stored)
    assert len({m.memory_id for m in stored}) == 3

    results = mind.recall("nmd001 python backend", top_k=3)
    assert len(results) >= 1
    assert any("Python" in r.content.text for r in results)
    assert any("nmd001" in r.content.text for r in results)

    no_hit = mind.recall("quantum physics research", top_k=1)
    assert len(no_hit) == 1  # nearest neighbour still returned


def test_store_fresh_mind_no_duplicates_loss(api_client, unique_user):
    mind = build_mind(unique_user, api_client)
    stored = [mind.store(m) for m in sample_memories(unique_user)]

    fresh = build_mind(unique_user, api_client)
    assert fresh.vector.count() == 3
    assert len(fresh.truth.read_all()) == 3
    assert len(fresh.recall("nmd001 python", top_k=5)) >= 1

    for m in stored:
        retrieved = fresh.get(m.memory_id)
        assert retrieved is not None
        assert retrieved.content.text == m.content.text

    assert fresh.graph.count_edges() == 6  # 1 BUILDS + (2+2+1) HAS_ENTITY


def test_recall_graph_expansion(api_client, unique_user):
    mind = build_mind(unique_user, api_client)
    for m in sample_memories(unique_user):
        mind.store(m)

    expanded = mind.recall("python backend services", top_k=1, expand=True)
    ids = {m.memory_id for m in expanded}
    assert len(ids) >= 2  # top hit + memory sharing entity "Python"
    assert any("My name is nmd001" in m.content.text for m in expanded)


def test_store_rollback_on_vector_failure(api_client, unique_user, monkeypatch):
    mind = build_mind(unique_user, api_client)

    def boom(memory_id, text):
        raise RuntimeError("vector exploded")

    monkeypatch.setattr(mind.vector, "update", boom)
    memory = make_memory("will fail", user_id=unique_user)
    with pytest.raises(RuntimeError):
        mind.store(memory)

    assert len(mind.truth.read_all()) == 0
    assert mind.vector.count() == 0
    assert mind.graph.count_edges() == 0


def test_store_rollback_on_graph_failure(api_client, unique_user, monkeypatch):
    mind = build_mind(unique_user, api_client)

    def boom(memory):
        raise RuntimeError("graph exploded")

    monkeypatch.setattr(mind.graph, "relate_memory", boom)
    memory = make_memory("graph fail", entities=["G"], user_id=unique_user)
    with pytest.raises(RuntimeError):
        mind.store(memory)

    assert len(mind.truth.read_all()) == 0
    assert mind.vector.count() == 0
    assert mind.graph.count_edges() == 0


def test_multi_user_isolation(api_client, unique_user):
    user_a = unique_user
    user_b = "user_" + user_a[5:][::-1]

    mind_a = build_mind(user_a, api_client)
    mind_a.store(make_memory("Alice likes chess", user_id=user_a))
    mind_b = build_mind(user_b, api_client)
    mind_b.store(make_memory("Bob likes hiking", user_id=user_b))

    reopened_a = build_mind(user_a, api_client)
    assert len(reopened_a.truth.read_all()) == 1
    assert len(reopened_a.recall("chess", top_k=1)) == 1
    assert "chess" in reopened_a.recall("chess", top_k=1)[0].content.text

    reopened_b = build_mind(user_b, api_client)
    assert len(reopened_b.truth.read_all()) == 1
    assert "hiking" in reopened_b.recall("hiking", top_k=1)[0].content.text
    assert "chess" not in reopened_b.recall("chess", top_k=1)[0].content.text


def test_update_and_delete(api_client, unique_user):
    mind = build_mind(unique_user, api_client)
    stored = mind.store(make_memory("user likes coffee", entities=["Coffee"], user_id=unique_user))
    updated = mind.update(stored.memory_id, {"content": {"text": "user likes tea"}})
    assert mind.get(stored.memory_id).content.text == "user likes tea"
    assert mind.vector.count() == 1

    assert mind.delete(stored.memory_id) is True
    assert mind.get(stored.memory_id) is None
    assert mind.vector.count() == 0
    assert len(mind.truth.read_all()) == 0
    assert mind.graph.entities_of(stored.memory_id) == []
    assert mind.graph.count_nodes() == 1  # entity node "Coffee" persists


def test_relate_links_memory_to_entity(api_client, unique_user):
    mind = build_mind(unique_user, api_client)
    stored = mind.store(make_memory("Sathya builds AI systems", entities=["Sathya"], user_id=unique_user))
    mind.relate(stored.memory_id, "NebulonMind")
    assert set(mind.graph.entities_of(stored.memory_id)) == {"Sathya", "NebulonMind"}


def test_provenance_source_memory_id_persisted(api_client, unique_user):
    mind = build_mind(unique_user, api_client)
    stored = mind.store(
        make_memory(
            "I am learning Python",
            entities=["Python"],
            relationships=[Relationship(source="Sathya", target="Python", relation="HAS_SKILL")],
            user_id=unique_user,
        )
    )
    retrieved = mind.get(stored.memory_id)
    assert len(retrieved.relationships) == 1
    assert retrieved.relationships[0].source_memory_id == stored.memory_id

    fresh = build_mind(unique_user, api_client)
    again = fresh.get(stored.memory_id)
    assert again.relationships[0].source_memory_id == stored.memory_id


def test_delete_keeps_shared_entities_and_other_memories(api_client, unique_user):
    mind = build_mind(unique_user, api_client)
    memory_a = mind.store(
        make_memory("Sathya works with Python", entities=["Sathya", "Python"], user_id=unique_user)
    )
    memory_b = mind.store(
        make_memory("I use Python for NebulonDB", entities=["Python", "NebulonDB"], user_id=unique_user)
    )
    assert mind.graph.count_nodes() >= 4  # Sathya + Python + NebulonDB + memory nodes

    assert mind.delete(memory_a.memory_id) is True

    fresh = build_mind(unique_user, api_client)
    assert fresh.get(memory_a.memory_id) is None
    assert fresh.get(memory_b.memory_id) is not None
    assert "Python" in fresh.graph.entities_of(memory_b.memory_id)
    assert fresh.vector.count() == 1
    hits = fresh.recall("python nebulondb", top_k=1)
    assert hits and hits[0].memory_id == memory_b.memory_id
