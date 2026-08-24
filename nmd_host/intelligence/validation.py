"""Validation layer: garbage from an LLM (or any extractor) never reaches
storage. Decisions are re-parsed through pydantic, structurally sanitized,
and de-duplicated; anything unparseable is dropped, not stored.
"""

from __future__ import annotations

from pydantic import ValidationError
from typing import Iterable, List, Optional

from .schemas import MemoryCandidate, MemoryDecision


def validate_decisions(decisions: Iterable[MemoryDecision]) -> List[MemoryDecision]:
    """Return only structurally sound, de-duplicated decisions."""
    seen: set = set()
    valid: List[MemoryDecision] = []
    for decision in decisions:
        sanitized = sanitize_decision(decision)
        if sanitized is None:
            continue
        candidate = sanitized.candidate
        key = (candidate.category.value, candidate.text.lower())
        if key in seen:
            continue
        seen.add(key)
        valid.append(sanitized)
    return valid


def sanitize_decision(decision: MemoryDecision) -> Optional[MemoryDecision]:
    """Fix minor issues and drop invalid decisions (returns ``None``)."""
    if not isinstance(decision, MemoryDecision):
        try:
            decision = MemoryDecision.model_validate(decision)
        except ValidationError:
            return None

    candidate = decision.candidate
    if not isinstance(candidate, MemoryCandidate):
        try:
            candidate = MemoryCandidate.model_validate(candidate)
        except ValidationError:
            return None

    text = (candidate.text or "").strip()
    if len([t for t in text.split() if len(t) >= 2]) < 1:
        return None

    entities = [e.strip() for e in candidate.entities if e and e.strip()]
    relationships = [r for r in candidate.relationships if _relationship_ok(r)]
    seen_entities: set = set()
    for relationship in relationships:
        for label in (relationship.source, relationship.target):
            if label and label not in seen_entities and not _is_placeholder(label):
                seen_entities.add(label)
                if label not in entities:
                    entities.append(label)

    candidate = candidate.model_copy(
        update={
            "text": text,
            "entities": entities,
            "relationships": relationships,
        }
    )
    return decision.model_copy(update={"candidate": candidate})


def _is_placeholder(label: str) -> bool:
    """Generic subject labels that should never become graph entity nodes."""
    return label.strip().lower() == "user"


def _relationship_ok(relationship) -> bool:
    source = (relationship.source or "").strip()
    target = (relationship.target or "").strip()
    return bool(source) and bool(target)


__all__ = ["sanitize_decision", "validate_decisions"]
