"""Deterministic entity extraction: capitalized tokens + a tech whitelist.

Rule-based NER is deliberately conservative: entities are proper nouns or
known domain terms (Python, NebulonDB, HNSW, ...). The LLM extractor
supplies its own entities; this module backs the rule-based pipeline.
"""

from __future__ import annotations

import re
from typing import List

_CAPITALIZED = re.compile(r"\b[A-Z][\w.'-]{1,}\b")

TECH_WHITELIST = [
    "nebulondb",
    "nebulonmind",
    "hnsw",
    "postgresql",
    "mysql",
    "sqlite",
    "redis",
    "mongodb",
    "fastapi",
    "pydantic",
    "python",
    "typescript",
    "javascript",
    "react",
    "node",
    "rust",
    "go",
    "kubernetes",
    "docker",
    "git",
    "graphql",
    "sql",
    "llm",
    "pytorch",
    "tensorflow",
    "flask",
    "django",
    "aws",
    "gcp",
]

_STOPWORDS = {
    "i", "i'm", "my", "me", "the", "a", "an", "and", "or", "but",
    "it", "this", "that", "these", "those", "we", "you", "they", "he",
    "she", "there", "here", "today", "yesterday", "tomorrow", "last",
    "next", "this", "my", "just", "actually", "however", "so", "now",
    "when", "what", "who", "how", "why", "where", "to", "with", "from",
}


def _exact_case(text: str, term: str) -> str:
    """Return ``term`` as it actually appears in ``text``."""
    pos = text.lower().find(term.lower())
    return text[pos: pos + len(term)] if pos >= 0 else term


def extract_entities(text: str) -> List[str]:
    """Extract entities from a sentence, de-duplicated, in order of appearance."""
    entities: List[str] = []
    seen = set()
    for match in _CAPITALIZED.finditer(text):
        word = match.group(0)
        if word.lower() in _STOPWORDS or word.lower() in seen:
            continue
        seen.add(word.lower())
        entities.append(word)

    lowered = text.lower()
    for term in TECH_WHITELIST:
        if re.search(rf"\b{re.escape(term)}\b", lowered) and term not in seen:
            seen.add(term)
            entities.append(_exact_case(text, term))
    return entities


__all__ = ["TECH_WHITELIST", "extract_entities"]
