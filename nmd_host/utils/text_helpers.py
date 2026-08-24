"""Token-level text helpers shared by deduplication and ranking.

Moved here from ``lifecycle.deduplication`` so the lexical similarity used
by both the deduplication and ranking layers lives in one place; the
deduplication module re-imports these names to keep its public API stable.
"""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"[a-z0-9]+")


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    return " ".join(_WORD_RE.findall((text or "").lower()))


def token_set(text: str) -> set:
    return set(_WORD_RE.findall((text or "").lower()))


def text_similarity(text_a: str, text_b: str) -> float:
    """Token overlap in [0, 1], tolerant of extra words and reordering.

    Uses the overlap coefficient (|A ∩ B| / min(|A|, |B|)) so "My name is
    Sathya" vs "Sathya is my name" scores 1.0 while "My name is Sathya"
    vs "My name is Sathya and I build AI" also scores 1.0 (subset).
    """
    tokens_a, tokens_b = token_set(text_a), token_set(text_b)
    if not tokens_a or not tokens_b:
        return 1.0 if tokens_a == tokens_b else 0.0
    overlap = len(tokens_a & tokens_b)
    return overlap / min(len(tokens_a), len(tokens_b))


def jaccard_similarity(text_a: str, text_b: str) -> float:
    """Jaccard index in [0, 1] — penalizes subsets and extra words.

    The conservative cousin of ``text_similarity``: used for retrieval-time
    deduplication where "User works with Python" and "User works with
    Python and Rust" must survive as distinct memories.
    """
    tokens_a, tokens_b = token_set(text_a), token_set(text_b)
    if not tokens_a and not tokens_b:
        return 1.0
    union = tokens_a | tokens_b
    if not union:
        return 0.0
    return len(tokens_a & tokens_b) / len(union)


__all__ = [
    "jaccard_similarity",
    "normalize_text",
    "text_similarity",
    "token_set",
]
