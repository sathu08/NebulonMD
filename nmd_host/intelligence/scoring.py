"""Importance & confidence scoring for memory candidates.

Importance answers "how valuable is this to remember?", confidence answers
"how reliable is the extraction?". Both are floats in [0, 1]; the LLM
extractor fills them itself and this module only shapes rule-based scores.
"""

from __future__ import annotations

from nmd_host.core.models import Priority

from .schemas import MemoryCategory


_CATEGORY_IMPORTANCE = {
    MemoryCategory.IDENTITY: 0.95,
    MemoryCategory.SKILL: 0.85,
    MemoryCategory.PREFERENCE: 0.80,
    MemoryCategory.PROJECT: 0.80,
    MemoryCategory.GOAL: 0.75,
    MemoryCategory.FACT: 0.65,
    MemoryCategory.EVENT: 0.55,
    MemoryCategory.OPINION: 0.55,
    MemoryCategory.KNOWLEDGE: 0.70,
    MemoryCategory.TASK: 0.45,
}

def importance_for(category: MemoryCategory, has_entities: bool = False) -> float:
    base = _CATEGORY_IMPORTANCE.get(category, 0.5)
    score = base + (0.05 if has_entities else 0.0)
    return min(1.0, round(score, 2))


def confidence_for(strength: float, has_entities: bool = False) -> float:
    """``strength`` is the rule quality in [0, 1]; entities raise confidence."""
    score = 0.45 + 0.5 * max(0.0, min(1.0, strength)) + (0.05 if has_entities else 0.0)
    return min(1.0, round(score, 2))


def priority_from_score(score: float) -> Priority:
    if score >= 0.8:
        return Priority.HIGH
    if score >= 0.55:
        return Priority.MEDIUM
    return Priority.LOW


__all__ = ["confidence_for", "importance_for", "priority_from_score"]
