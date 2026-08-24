"""Memory classification: every candidate receives a category, which in turn
resolves to a Step 1 ``MemoryType`` and a retention policy.
"""

from __future__ import annotations

from typing import Optional

from nmd_host.core.models import MemoryType, RetentionPolicy
from .schemas import MemoryCategory

CATEGORY_TO_MEMORY_TYPE = {
    MemoryCategory.IDENTITY: MemoryType.LONG_TERM,
    MemoryCategory.PREFERENCE: MemoryType.LONG_TERM,
    MemoryCategory.SKILL: MemoryType.LONG_TERM,
    MemoryCategory.GOAL: MemoryType.LONG_TERM,
    MemoryCategory.PROJECT: MemoryType.SEMANTIC,
    MemoryCategory.FACT: MemoryType.SEMANTIC,
    MemoryCategory.EVENT: MemoryType.EPISODIC,
    MemoryCategory.TASK: MemoryType.WORKING,
    MemoryCategory.OPINION: MemoryType.SHORT_TERM,
    MemoryCategory.KNOWLEDGE: MemoryType.KNOWLEDGE,
}

CATEGORY_TO_RETENTION = {
    MemoryCategory.IDENTITY: RetentionPolicy.PERMANENT,
    MemoryCategory.PREFERENCE: RetentionPolicy.PERMANENT,
    MemoryCategory.SKILL: RetentionPolicy.PERMANENT,
    MemoryCategory.GOAL: RetentionPolicy.PERMANENT,
    MemoryCategory.PROJECT: RetentionPolicy.PERMANENT,
    MemoryCategory.FACT: RetentionPolicy.TEMPORARY,
    MemoryCategory.EVENT: RetentionPolicy.TEMPORARY,
    MemoryCategory.TASK: RetentionPolicy.SESSION,
    MemoryCategory.OPINION: RetentionPolicy.TEMPORARY,
    MemoryCategory.KNOWLEDGE: RetentionPolicy.TEMPORARY,
}

_SIGNALS: dict = {
    MemoryCategory.IDENTITY: ["my name", "i am", "i'm", "i live in", "i work at"],
    MemoryCategory.PREFERENCE: ["i like", "i love", "i prefer", "i enjoy", "i don't like"],
    MemoryCategory.SKILL: ["i work with", "i use", "good at", "experience with"],
    MemoryCategory.GOAL: ["i want to", "my goal", "i plan to", "trying to"],
    MemoryCategory.PROJECT: ["my project", "i'm building", "i am building", "developing"],
    MemoryCategory.FACT: ["i have", "i own", "years old", "my email"],
    MemoryCategory.EVENT: ["i created", "i finished", "i completed", "i started"],
    MemoryCategory.TASK: ["i need to", "i have to", "i should"],
    MemoryCategory.OPINION: ["i think", "i believe", "in my opinion"],
    MemoryCategory.KNOWLEDGE: ["according to", "the documentation", "docs say", "research shows"],
}


def memory_type_for(category: MemoryCategory) -> MemoryType:
    return CATEGORY_TO_MEMORY_TYPE.get(category, MemoryType.SEMANTIC)


def retention_for(category: MemoryCategory) -> RetentionPolicy:
    return CATEGORY_TO_RETENTION.get(category, RetentionPolicy.TEMPORARY)


def classify_sentence(text: str) -> Optional[MemoryCategory]:
    """Keyword-signal fallback classifier (used when no rule matched).

    Returns the category with the strongest keyword hit, or ``None``.
    """
    lowered = text.lower()
    best: Optional[MemoryCategory] = None
    best_score = 0.0
    for category, signals in _SIGNALS.items():
        hits = sum(1 for signal in signals if signal in lowered)
        if hits and hits > best_score:
            best_score = hits
            best = category
    return best


__all__ = [
    "classify_sentence",
    "memory_type_for",
    "retention_for",
]
