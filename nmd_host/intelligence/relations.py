"""Relationship extraction: turn a classified sentence into
``subject -RELATION-> object`` triples for the Step 1 graph store.
"""

from __future__ import annotations

from typing import List, Optional

from nmd_host.core.models import Relationship
from .schemas import MemoryCategory


RELATION_FOR_CATEGORY = {
    MemoryCategory.IDENTITY: None,
    MemoryCategory.PREFERENCE: "PREFERS",
    MemoryCategory.SKILL: "HAS_SKILL",
    MemoryCategory.GOAL: "AIMS_TO",
    MemoryCategory.PROJECT: "BUILDS",
    MemoryCategory.FACT: "HAS",
    MemoryCategory.EVENT: "HAPPENED",
    MemoryCategory.TASK: "NEEDS_TO",
    MemoryCategory.OPINION: "OPINES",
    MemoryCategory.KNOWLEDGE: "REFERENCES",
}

DEFAULT_SUBJECT = "User"


def relation_for(category: MemoryCategory) -> Optional[str]:
    return RELATION_FOR_CATEGORY.get(category)


def build_relationships(
    category: MemoryCategory,
    object_hint: Optional[str],
    subject: Optional[str] = None,
) -> List[Relationship]:
    """Build the triple for a rule match.

    ``subject`` defaults to the user label (or the resolved user name);
    ``object_hint`` is the object captured by the matching rule.
    """
    relation = relation_for(category)
    obj = (object_hint or "").strip()
    if not relation or not obj:
        return []
    return [Relationship(source=subject or DEFAULT_SUBJECT, target=obj, relation=relation)]


__all__ = ["DEFAULT_SUBJECT", "build_relationships", "relation_for"]
