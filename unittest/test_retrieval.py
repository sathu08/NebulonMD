"""Retrieval tests (3.5): top-k, candidate expansion, expired filtering,
ranking, deduplication, user isolation. Offline (InMemoryRepository)."""

from datetime import datetime, timedelta, timezone

from nmd_host.core.models import RetentionPolicy
from nmd_host.core.repository import InMemoryRepository
from nmd_host.lifecycle.ranking import MemoryRanker, RankingConfig
from nmd_host.lifecycle.retrieval import MemoryRetriever, RetrievalConfig
from conftest import make_memory


def _store(repo, text, user_id="user_001", score=0.5, policy=None, expires_at=None):
    memory = make_memory(text, user_id=user_id, score=score)
    if policy:
        memory.lifecycle.retention_policy = policy
    if expires_at:
        memory.lifecycle.expires_at = expires_at
    return repo.create(memory)


def _retriever(repo, **config_kwargs):
    config = RetrievalConfig(**config_kwargs)
    ranker = MemoryRanker(RankingConfig())
    return MemoryRetriever(repo, ranker=ranker, config=config)


def test_top_k_is_respected():
    repo = InMemoryRepository()
    for text in ["User likes hiking", "User works with Python", "User builds AI systems"]:
        _store(repo, text)
    retriever = _retriever(repo, top_k=2, candidate_multiplier=3)
    results = retriever.retrieve("user python hiking ai", top_k=2)
    assert len(results) == 2


def test_candidate_expansion_requested_from_store():
    requested = {}

    class RecordingRepository(InMemoryRepository):
        def search(self, query, top_k=5):
            requested["top_k"] = top_k
            return super().search(query, top_k=top_k)

    recording = RecordingRepository()
    for index in range(10):
        _store(recording, f"User works on topic number {index}")

    retriever = _retriever(recording, top_k=3, candidate_multiplier=4)
    retriever.retrieve("user works topic number", top_k=3)
    assert requested["top_k"] == 12  # 3 x 4, not 3


def test_expired_candidates_are_filtered_out():
    repo = InMemoryRepository()
    past = datetime.now(timezone.utc) - timedelta(hours=1)
    _store(repo, "User works with Python", policy=RetentionPolicy.TEMPORARY, expires_at=past)
    _store(repo, "User works with Python", score=0.9)
    retriever = _retriever(repo, top_k=5, candidate_multiplier=3)
    results = retriever.retrieve("python")
    assert all(m.lifecycle.expires_at is None for m in results)


def test_include_expired_keeps_them():
    repo = InMemoryRepository()
    past = datetime.now(timezone.utc) - timedelta(hours=1)
    expired = _store(repo, "User works with Python", policy=RetentionPolicy.TEMPORARY, expires_at=past)
    _store(repo, "User works with Rust")
    retriever = _retriever(repo, top_k=5, candidate_multiplier=3, include_expired=True)
    results = retriever.retrieve("python")
    assert any(m.memory_id == expired.memory_id for m in results)


def test_ranking_applied_to_candidates():
    repo = InMemoryRepository()
    _store(repo, "User works with Python", score=0.2)
    important = _store(repo, "User works with Python", score=0.95)
    retriever = _retriever(repo, top_k=1, candidate_multiplier=3)
    results = retriever.retrieve("user works with python")
    assert len(results) == 1
    assert results[0].memory_id == important.memory_id


def test_duplicates_collapsed_during_retrieval():
    repo = InMemoryRepository()
    _store(repo, "My name is Sathya")
    dup = _store(repo, "My name is Sathya.")
    retriever = _retriever(repo, top_k=5, candidate_multiplier=3)
    results = retriever.retrieve("sathya name")
    assert all(m.memory_id != dup.memory_id for m in results)


def test_similar_but_distinct_facts_survive_dedup():
    repo = InMemoryRepository()
    first = _store(repo, "User works with Python")
    second = _store(repo, "User works with Python and Rust")
    retriever = _retriever(repo, top_k=5, candidate_multiplier=3)
    results = retriever.retrieve("user works with python")
    ids = {m.memory_id for m in results}
    assert first.memory_id in ids
    assert second.memory_id in ids


def test_user_isolation():
    repo = InMemoryRepository()
    _store(repo, "Alice likes chess", user_id="user_alice")
    _store(repo, "Bob loves hiking", user_id="user_bob")
    retriever = _retriever(repo, top_k=5, candidate_multiplier=3)
    results = retriever.retrieve("chess", user_id="user_alice")
    assert all(m.user_id == "user_alice" for m in results)
    assert all("chess" in m.content.text for m in results)
    assert not any(m.user_id == "user_bob" for m in results)


def test_user_isolation_never_leaks_other_user():
    repo = InMemoryRepository()
    _store(repo, "Alice likes chess", user_id="user_alice")
    _store(repo, "Bob loves chess as well", user_id="user_bob")
    retriever = _retriever(repo, top_k=5, candidate_multiplier=3)
    results = retriever.retrieve("chess", user_id="user_alice")
    assert len(results) == 1
    assert results[0].user_id == "user_alice"


def test_config_validation():
    try:
        RetrievalConfig(top_k=0)
        assert False, "top_k=0 must be rejected"
    except ValueError:
        pass
    try:
        RetrievalConfig(candidate_multiplier=0)
        assert False, "multiplier=0 must be rejected"
    except ValueError:
        pass
