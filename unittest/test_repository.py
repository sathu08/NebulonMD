from nmd_host.core.mind import NebulonMind
from nmd_host.core.models import Relationship
from nmd_host.core.repository import InMemoryRepository, NebulonMindRepository
from conftest import make_memory


def test_crud():
    repo = InMemoryRepository()
    memory = make_memory("My name is Sathya", entities=["Sathya"])
    created = repo.create(memory)
    assert created.memory_id is not None

    assert repo.get(created.memory_id).content.text == "My name is Sathya"

    updated = repo.update(created.memory_id, {"content": {"text": "My name is Sathya Kumar"}})
    assert updated.content.text == "My name is Sathya Kumar"

    assert repo.delete(created.memory_id) is True
    assert repo.get(created.memory_id) is None
    assert repo.delete(created.memory_id) is False


def test_search_overlap():
    repo = InMemoryRepository()
    repo.create(make_memory("User works with Python for AI systems"))
    repo.create(make_memory("User drives a red car"))
    repo.create(make_memory("Python is used for backend services"))

    results = repo.search("python backend")
    assert results
    assert "python" in results[0].content.text.lower()


def test_search_top_k():
    repo = InMemoryRepository()
    for i in range(10):
        repo.create(make_memory(f"shared topic note number {i}"))
    results = repo.search("shared topic", top_k=3)
    assert len(results) == 3


def test_relate():
    repo = InMemoryRepository()
    memory = repo.create(make_memory("Sathya builds AI systems", entities=["Sathya"]))
    repo.relate(memory.memory_id, "NebulonMind")
    assert "NebulonMind" in repo.get(memory.memory_id).entities


def test_search_does_not_mutate_store():
    repo = InMemoryRepository()
    memory = repo.create(make_memory("Likes hiking in the mountains"))
    result = repo.search("hiking")[0]
    result.content.text = "changed"
    assert repo.get(memory.memory_id).content.text == "Likes hiking in the mountains"


def test_provenance_stamped_on_create():
    repo = InMemoryRepository()
    memory = make_memory(
        "Sathya works with Python",
        entities=["Sathya", "Python"],
        relationships=[Relationship(source="Sathya", target="Python", relation="HAS_SKILL")],
    )
    created = repo.create(memory)
    assert created.memory_id is not None
    for relationship in created.relationships:
        assert relationship.source_memory_id == created.memory_id


def test_provenance_stamped_on_update():
    repo = InMemoryRepository()
    created = repo.create(make_memory("Sathya knows Python"))
    updated = repo.update(
        created.memory_id,
        {
            "relationships": [
                {"source": "Sathya", "target": "Python", "relation": "HAS_SKILL"}
            ]
        },
    )
    assert updated.relationships[0].source_memory_id == created.memory_id


def test_api_repository_crud_search_relate(api_client, unique_user):
    mind = NebulonMind(user_id=unique_user, client=api_client)
    repo = NebulonMindRepository(mind)

    memory = make_memory("My name is nmd001", entities=["nmd001"], user_id=unique_user)
    created = repo.create(memory)
    assert created.memory_id is not None
    assert repo.get(created.memory_id).content.text == "My name is nmd001"

    updated = repo.update(created.memory_id, {"content": {"text": "My name is nmd001 Kumar"}})
    assert repo.get(created.memory_id).content.text == "My name is nmd001 Kumar"

    repo.relate(created.memory_id, "NebulonMind")
    assert "NebulonMind" in repo.get(created.memory_id).entities

    hits = repo.search("nmd001 kumar", top_k=3)
    assert hits and hits[0].memory_id == created.memory_id

    assert repo.delete(created.memory_id) is True
    assert repo.get(created.memory_id) is None
    repo.close()
