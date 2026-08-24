"""Forgetting tests (3.8): expired deletion, permanent protection, session
cleanup, repository delegation. Offline."""

from datetime import datetime, timedelta, timezone

from nmd_host.core.models import MemoryStatus, RetentionPolicy
from nmd_host.core.repository import InMemoryRepository
from nmd_host.lifecycle.forgetting import ForgettingManager
from nmd_host.lifecycle.retention import RetentionEvaluator
from conftest import make_memory


def _memory(text, policy=RetentionPolicy.PERMANENT, expires_at=None, user_id="user_001"):
    memory = make_memory(text, user_id=user_id)
    memory.memory_id = f"mem_{text.replace(' ', '_')}"
    memory.lifecycle.retention_policy = policy
    if expires_at:
        memory.lifecycle.expires_at = expires_at
    return memory


def _expired_temp(text, **kwargs):
    return _memory(
        text,
        policy=RetentionPolicy.TEMPORARY,
        expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        **kwargs,
    )


def test_expired_temporary_is_forgotten():
    repo = InMemoryRepository()
    expired = _expired_temp("Debugging HNSW today")
    repo.create(expired)
    manager = ForgettingManager(repo)
    assert [m.memory_id for m in manager.find_expired()] == [expired.memory_id]
    assert manager.cleanup() == 1
    assert repo.get(expired.memory_id) is None


def test_permanent_memory_is_protected():
    repo = InMemoryRepository()
    permanent = _memory("My name is Sathya", policy=RetentionPolicy.PERMANENT)
    repo.create(permanent)
    manager = ForgettingManager(repo)
    assert manager.find_expired() == []
    assert manager.cleanup() == 0
    assert repo.get(permanent.memory_id) is not None


def test_old_permanent_is_never_forgotten():
    repo = InMemoryRepository()
    old = _memory("My name is Sathya", policy=RetentionPolicy.PERMANENT)
    old.lifecycle.created_at = datetime.now(timezone.utc) - timedelta(days=3650)
    repo.create(old)
    manager = ForgettingManager(repo)
    assert manager.cleanup() == 0
    assert repo.get(old.memory_id) is not None


def test_session_cleanup_only_when_session_ended():
    repo = InMemoryRepository()
    session = _memory("Continue the database architecture discussion", policy=RetentionPolicy.SESSION)
    repo.create(session)
    manager = ForgettingManager(repo)
    assert manager.cleanup() == 0  # session still alive
    assert repo.get(session.memory_id) is not None
    assert manager.cleanup(session_ended=True) == 1
    assert repo.get(session.memory_id) is None


def test_cleanup_mixes_policies_correctly():
    repo = InMemoryRepository()
    expired = _expired_temp("Temp fact")
    permanent = _memory("Permanent fact", policy=RetentionPolicy.PERMANENT)
    repo.create(expired)
    repo.create(permanent)
    manager = ForgettingManager(repo)
    assert manager.cleanup() == 1
    assert repo.get(expired.memory_id) is None
    assert repo.get(permanent.memory_id) is not None


def test_archived_memory_is_never_forgotten():
    repo = InMemoryRepository()
    archived = _expired_temp("Old fact")
    archived.status = MemoryStatus.ARCHIVED
    repo.create(archived)
    manager = ForgettingManager(repo)
    assert manager.cleanup() == 0
    assert repo.get(archived.memory_id) is not None


def test_forget_delegates_to_repository():
    repo = InMemoryRepository()
    memory = _memory("Something", policy=RetentionPolicy.TEMPORARY)
    repo.create(memory)
    manager = ForgettingManager(repo)
    assert manager.forget(memory.memory_id) is True
    assert repo.get(memory.memory_id) is None
    assert manager.forget(memory.memory_id) is False


def test_find_expired_filters_by_user():
    repo = InMemoryRepository()
    user_a = _expired_temp("A fact", user_id="user_a")
    user_b = _expired_temp("B fact", user_id="user_b")
    repo.create(user_a)
    repo.create(user_b)
    manager = ForgettingManager(repo)
    assert {m.user_id for m in manager.find_expired(user_id="user_a")} == {"user_a"}


def test_low_importance_is_never_a_deletion_reason():
    repo = InMemoryRepository()
    low = _memory("Trivial detail", policy=RetentionPolicy.PERMANENT, expires_at=None)
    low.importance.score = 0.05
    repo.create(low)
    manager = ForgettingManager(repo)
    assert manager.cleanup() == 0
    assert repo.get(low.memory_id) is not None
