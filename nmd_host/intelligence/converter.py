"""Conversion: Step 2 decision output -> the Step 1 ``Memory`` model.

Nothing in Step 1 changes; this is the only place the two models meet.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict

from nmd_host.core.models import (
    Classification,
    Importance,
    Lifecycle,
    Memory,
    MemoryContent,
    Provenance,
    RetentionPolicy,
)

from nmd_host.utils.constants import DEFAULT_TEMPORARY_TTL_SECONDS
from nmd_host.utils.env_helpers import env_int

from nmd_host.lifecycle.lifecycle import now_utc
from .classifier import memory_type_for, retention_for
from .scoring import priority_from_score

from .schemas import MemoryCandidate


def _temporary_expires_at() -> datetime:
    """Default expiry for TEMPORARY-mapped categories (``NMD_TEMPORARY_TTL_SECONDS``).

    The Step 3 gate treats a TEMPORARY memory without ``expires_at`` as an
    invalid configuration (it refuses to invent an expiry decision). The
    decision pipeline is the layer that labels a category temporary, so it
    supplies the expiry here to keep Step 1 / Step 3 self-consistent.
    """
    ttl = env_int("NMD_TEMPORARY_TTL_SECONDS", DEFAULT_TEMPORARY_TTL_SECONDS)
    if ttl < 1:
        ttl = DEFAULT_TEMPORARY_TTL_SECONDS
    return now_utc() + timedelta(seconds=ttl)


def candidate_to_memory(
    candidate: MemoryCandidate,
    user_id: str,
    source: str = "conversation",
) -> Memory:
    """Map a decision candidate onto the Step 1 ``Memory`` model."""
    content = MemoryContent(
        text=candidate.text,
        summary=candidate.summary,
        structured_data=_structured_data(candidate),
    )
    classification = Classification(
        memory_type=memory_type_for(candidate.category),
        category=candidate.category.value,
        source=source,
    )
    importance = Importance(
        score=candidate.importance,
        confidence=candidate.confidence,
        priority=priority_from_score(candidate.importance),
    )
    retention = retention_for(candidate.category)
    lifecycle = Lifecycle(
        retention_policy=retention,
        expires_at=(
            _temporary_expires_at()
            if retention is RetentionPolicy.TEMPORARY
            else None
        ),
    )
    return Memory(
        user_id=user_id,
        content=content,
        classification=classification,
        provenance=candidate.provenance or Provenance(),
        importance=importance,
        lifecycle=lifecycle,
        entities=list(dict.fromkeys(candidate.entities)),
        relationships=list(candidate.relationships),
    )


def _structured_data(candidate: MemoryCandidate) -> Dict[str, Any]:
    data: Dict[str, Any] = {"category": candidate.category.value}
    if candidate.summary:
        data["summary"] = candidate.summary
    if candidate.subject:
        data["subject"] = candidate.subject
    if candidate.reasoning:
        data["reasoning"] = candidate.reasoning
    if candidate.slots:
        data["slots"] = candidate.slots
    return data


__all__ = ["candidate_to_memory"]
