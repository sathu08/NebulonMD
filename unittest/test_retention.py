"""Retention evaluator tests (3.2): permanent → retained, expired temporary →
deleted, future expiry → retained, missing expiry → safe behavior. Offline."""

from datetime import datetime, timedelta, timezone

from nmd_host.core.models import MemoryStatus, RetentionPolicy
from nmd_host.lifecycle.retention import RetentionEvaluator
from conftest import make_memory


def _evaluator(now=None):
    return RetentionEvaluator(clock=(lambda: now) if now else None)


def _in_past(hours: int) -> datetime:
    return datetime.now(timezone.utc) - timedelta(hours=hours)


def _memory(policy: RetentionPolicy, **lifecycle):
    memory = make_memory("Some fact")
    memory.lifecycle.retention_policy = policy
    for key, value in lifecycle.items():
        setattr(memory.lifecycle, key, value)
    return memory


def test_permanent_retained_forever():
    evaluator = _evaluator()
    memory = _memory(RetentionPolicy.PERMANENT, expires_at=_in_past(1000))
    assert evaluator.is_expired(memory) is False
    assert evaluator.should_retain(memory) is True
    assert evaluator.should_delete(memory) is False


def test_expired_temporary_is_eligible_for_deletion():
    evaluator = _evaluator()
    memory = _memory(RetentionPolicy.TEMPORARY, expires_at=_in_past(1))
    assert evaluator.is_expired(memory) is True
    assert evaluator.should_retain(memory) is False
    assert evaluator.should_delete(memory) is True


def test_future_expiry_is_retained():
    evaluator = _evaluator()
    memory = _memory(RetentionPolicy.TEMPORARY, expires_at=_in_past(-24))
    assert evaluator.should_retain(memory) is True
    assert evaluator.should_delete(memory) is False


def test_missing_expiry_retains_but_flags_invalid_configuration():
    evaluator = _evaluator()
    memory = _memory(RetentionPolicy.TEMPORARY)
    assert evaluator.is_valid_configuration(memory) is False
    assert evaluator.should_retain(memory) is True  # safe: retain, never invent
    assert evaluator.should_delete(memory) is False


def test_permanent_and_session_are_always_valid_configurations():
    evaluator = _evaluator()
    assert evaluator.is_valid_configuration(_memory(RetentionPolicy.PERMANENT)) is True
    assert evaluator.is_valid_configuration(_memory(RetentionPolicy.SESSION)) is True


def test_session_retained_until_session_ends():
    evaluator = _evaluator()
    memory = _memory(RetentionPolicy.SESSION)
    assert evaluator.should_retain(memory) is True
    assert evaluator.should_retain(memory, session_ended=True) is False
    assert evaluator.should_delete(memory, session_ended=True) is True


def test_old_but_valid_temporary_is_never_deleted_for_age():
    evaluator = _evaluator()
    memory = _memory(RetentionPolicy.TEMPORARY, expires_at=_in_past(-30 * 24))
    assert evaluator.should_retain(memory) is True
    assert evaluator.should_delete(memory) is False


def test_archived_is_never_deleted():
    evaluator = _evaluator()
    memory = _memory(RetentionPolicy.TEMPORARY, expires_at=_in_past(1))
    memory.status = MemoryStatus.ARCHIVED
    assert evaluator.is_expired(memory) is False
    assert evaluator.should_retain(memory) is True
    assert evaluator.should_delete(memory) is False


def test_clock_injection_makes_expiry_deterministic():
    fixed_now = datetime(2026, 1, 15, tzinfo=timezone.utc)
    evaluator = _evaluator(now=fixed_now)
    memory = _memory(
        RetentionPolicy.TEMPORARY,
        expires_at=datetime(2026, 1, 18, tzinfo=timezone.utc),
    )
    assert evaluator.is_expired(memory) is False

    later = datetime(2026, 1, 20, tzinfo=timezone.utc)
    evaluator_later = _evaluator(now=later)
    assert evaluator_later.is_expired(memory) is True
