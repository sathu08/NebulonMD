"""Smoke tests for the three stores against the NebulonDB API (per README order).

Every test uses a unique user id so segments never collide on the shared
server.
"""

from nmd_host.core.models import Memory, Relationship
from nmd_host.stores import GraphStore, TruthStore, VectorStore
from conftest import make_memory


# --------------------------------------------------------------------- #
# TruthStore (COSMOS)                                                   #
# --------------------------------------------------------------------- #


def test_truth_store_crud_and_restore(api_client, unique_user):
    truth = TruthStore(api_client, unique_user)
    memory = make_memory("My name is Sathya", user_id=unique_user)
    memory.memory_id = "mem_001"

    truth.create(memory.to_truth_doc())
    assert truth.get("mem_001")["content"]["text"] == "My name is Sathya"
    assert len(truth.read_all()) == 1

    truth.update({**memory.to_truth_doc(), "content": {"text": "My name is Sathya Kumar"}})
    assert truth.get("mem_001")["content"]["text"] == "My name is Sathya Kumar"
    assert len(truth.read_all()) == 1  # update, not duplicate insert

    # A fresh store sees the same data (server-side persistence).
    fresh = TruthStore(api_client, unique_user)
    assert fresh.get("mem_001")["content"]["text"] == "My name is Sathya Kumar"


def test_truth_store_create_is_idempotent(api_client, unique_user):
    truth = TruthStore(api_client, unique_user)
    memory = make_memory("dup check", user_id=unique_user)
    memory.memory_id = "mem_001"
    truth.create(memory.to_truth_doc())
    truth.create(memory.to_truth_doc())
    assert len(truth.read_all()) == 1


def test_truth_store_delete(api_client, unique_user):
    truth = TruthStore(api_client, unique_user)
    memory = make_memory("tmp", user_id=unique_user)
    memory.memory_id = "mem_x"
    truth.create(memory.to_truth_doc())
    assert truth.delete("mem_x") is True
    assert truth.get("mem_x") is None
    assert truth.delete("mem_x") is False


# --------------------------------------------------------------------- #
# VectorStore (ORBIT)                                                   #
# --------------------------------------------------------------------- #


def test_vector_store_insert_search_delete(api_client, unique_user):
    store = VectorStore(api_client, unique_user)

    store.insert("mem_001", "user likes python programming")
    store.insert("mem_002", "user drives a red car")
    assert store.count() == 2

    hits = store.search("python coding", top_k=2)
    assert hits
    assert hits[0]["metadata"]["memory_id"] == "mem_001"
    assert hits[0]["score"] > 0.0

    assert store.record_id_of("mem_002") is not None
    assert store.delete("mem_002") is True
    assert store.count() == 1
    assert store.record_id_of("mem_002") is None


def test_vector_store_update_is_idempotent(api_client, unique_user):
    store = VectorStore(api_client, unique_user)
    store.insert("mem_001", "first version of the text")
    store.update("mem_001", "second version of the text")
    assert store.count() == 1  # update, not duplicate insert
    hits = store.search("second version", top_k=1)
    assert hits[0]["metadata"]["memory_id"] == "mem_001"


def test_vector_store_search_hits_carry_memory_id(api_client, unique_user):
    store = VectorStore(api_client, unique_user)
    store.insert("mem_001", "persistent memory about Sathya")
    hits = store.search("persistent thing", top_k=1)
    assert hits[0]["metadata"]["memory_id"] == "mem_001"


# --------------------------------------------------------------------- #
# GraphStore (ORBIT Mesh)                                               #
# --------------------------------------------------------------------- #


def _shared_graph(api_client, unique_user):
    truth = TruthStore(api_client, unique_user)
    graph = GraphStore(api_client, unique_user, truth=truth)
    return truth, graph


def test_graph_store_relationships_and_idempotency(api_client, unique_user):
    truth, graph = _shared_graph(api_client, unique_user)
    memory = make_memory(
        "Sathya builds AI systems",
        entities=["Sathya", "Python"],
        relationships=[Relationship(source="Sathya", target="Python", relation="HAS_SKILL")],
        user_id=unique_user,
    )
    memory.memory_id = "mem_001"
    truth.create(memory.to_truth_doc())
    graph.relate_memory(memory)
    assert graph.count_edges() == 3  # 1 HAS_SKILL + 2 HAS_ENTITY

    graph.relate_memory(memory)  # re-store must not duplicate
    assert graph.count_edges() == 3
    assert graph.count_nodes() == 3

    assert set(graph.entities_of("mem_001")) == {"Sathya", "Python"}
    assert set(graph.memories_of("Python")) == {"mem_001"}


def test_graph_store_delete_memory(api_client, unique_user):
    truth, graph = _shared_graph(api_client, unique_user)
    vector = VectorStore(api_client, unique_user)
    memory = make_memory("A relates to B", entities=["A", "B"], user_id=unique_user)
    memory.memory_id = "mem_001"
    truth.create(memory.to_truth_doc())
    vector.insert("mem_001", memory.content.text)
    graph.relate_memory(memory)

    # Deleting the vector record removes the memory node + its edges.
    assert vector.delete("mem_001") is True
    truth.delete("mem_001")
    assert graph.entities_of("mem_001") == []
    assert set(graph.memories_of("B")) == set()


def test_graph_store_memories_of_scans_truth(api_client, unique_user):
    truth, graph = _shared_graph(api_client, unique_user)
    m1 = make_memory("one", entities=["Shared"], user_id=unique_user)
    m2 = make_memory("two", entities=["Shared"], user_id=unique_user)
    m1.memory_id, m2.memory_id = "mem_001", "mem_002"
    truth.create(m1.to_truth_doc())
    truth.create(m2.to_truth_doc())
    assert set(graph.memories_of("Shared")) == {"mem_001", "mem_002"}
