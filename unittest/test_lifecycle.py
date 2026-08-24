"""Lifecycle state tests (3.1): active/expired/permanent/temporary/session,
age and inactivity calculation. Offline."""

from datetime import datetime, timedelta, timezone

from nmd_host.core.models import MemoryStatus, RetentionPolicy
from nmd_host.lifecycle.lifecycle import (
    MemoryState,
    get_state,
    is_expired,
    memory_age,
    now_utc,
    time_since_access,
)
from conftest import make_memory


def _memory(policy: RetentionPolicy = RetentionPolicy.PERMANENT, **lifecycle):
    memory = make_memory("My name is Sathya")
    memory.lifecycle.retention_policy = policy
    for key, value in lifecycle.items():
        setattr(memory.lifecycle, key, value)
    return memory


def _in_past(hours: int) -> datetime:
    return now_utc() - timedelta(hours=hours)


def test_permanent_memory_is_active():
    memory = _memory(RetentionPolicy.PERMANENT)
    assert is_expired(memory) is False
    assert get_state(memory) is MemoryState.ACTIVE


def test_expired_temporary_memory():
    memory = _memory(RetentionPolicy.TEMPORARY, expires_at=_in_past(1))
    assert is_expired(memory) is True
    assert get_state(memory) is MemoryState.EXPIRED


def test_temporary_future_expiry_is_active():
    memory = _memory(RetentionPolicy.TEMPORARY, expires_at=_in_past(-1))
    assert is_expired(memory) is False
    assert get_state(memory) is MemoryState.ACTIVE


def test_permanent_never_expires_even_with_past_expires_at():
    memory = _memory(RetentionPolicy.PERMANENT, expires_at=_in_past(30))
    assert is_expired(memory) is False
    assert get_state(memory) is MemoryState.ACTIVE


def test_session_memory_state_and_clock_immunity():
    memory = _memory(RetentionPolicy.SESSION, expires_at=_in_past(1))
    assert get_state(memory) is MemoryState.SESSION
    assert is_expired(memory) is False  # session lifetime is not the clock's job


def test_archived_memory_is_frozen_not_expired():
    memory = _memory(RetentionPolicy.TEMPORARY, expires_at=_in_past(1))
    memory.status = MemoryStatus.ARCHIVED
    assert get_state(memory) is MemoryState.ARCHIVED
    assert is_expired(memory) is False


def test_missing_expires_at_is_not_expired():
    memory = _memory(RetentionPolicy.TEMPORARY)
    assert is_expired(memory) is False
    assert get_state(memory) is MemoryState.ACTIVE


def test_age_calculation():
    memory = _memory(RetentionPolicy.PERMANENT, created_at=_in_past(2))
    age = memory_age(memory)
    assert 1.5 * 3600 < age < 2.5 * 3600


def test_age_zero_when_no_timestamp():
    memory = _memory(RetentionPolicy.PERMANENT)
    assert memory_age(memory) == 0.0


def test_time_since_access_uses_last_accessed():
    memory = _memory(RetentionPolicy.PERMANENT, last_accessed=_in_past(5))
    idle = time_since_access(memory)
    assert 4.5 * 3600 < idle < 5.5 * 3600


def test_time_since_access_falls_back_to_updated():
    memory = _memory(RetentionPolicy.PERMANENT, updated_at=_in_past(1))
    idle = time_since_access(memory)
    assert 0.5 * 3600 < idle < 1.5 * 3600
