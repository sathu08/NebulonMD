"""Step 13.1 — Grounding verdicts (eval suggestions #2 + #3).

Keyword decline-matching (``_declines``) fails both ways: it missed real
declines (``q7``: "stored memories do not indicate ... no reference")
*and* flagged correct answers ("confirmed by two stored memories").
This module replaces wording heuristics with a single question:

    Is every factual claim in the answer supported by retrieved memory?

Labels:

* ``SUPPORTED`` — the answer's claims appear in the retrieved memories,
  or it correctly abstains on an unknown (decline + nothing to support).
* ``UNSUPPORTED_INFERENCE`` — a definitive claim with no supporting
  memory (e.g. "favorite is Python" from "writes projects in Python").
* ``CONTRADICTED`` — the answer denies/contradicts a retrieved fact
  (e.g. "you never specified a language" while "Python" is stored).
* ``UNKNOWN`` — no retrieved memories and no decline signal to judge
  (empty answer, infra error).

Pure functions, fully offline — no LLM-as-judge cost on the hot path.
"""

from __future__ import annotations

from typing import Any, List, Sequence

SUPPORTED = "SUPPORTED"
UNSUPPORTED_INFERENCE = "UNSUPPORTED_INFERENCE"
CONTRADICTED = "CONTRADICTED"
UNKNOWN = "UNKNOWN"

GROUNDING_LABELS = (
    SUPPORTED,
    UNSUPPORTED_INFERENCE,
    CONTRADICTED,
    UNKNOWN,
)

# Denial language: the answer claims the fact was never provided.
_DENIAL_MARKERS = (
    "has not",
    "have not",
    "never specified",
    "not previously",
    "no technical details",
    "isn't stored",
    "is not stored",
    "not discussed",
)

# Decline language (single source of truth; runner re-exports it):
# abstaining instead of claiming.
_DECLINE_MARKERS = (
    "don't know",
    "dont know",
    "do not know",
    "don't have",
    "dont have",
    "do not have",
    "n't have",
    "no memory",
    "no memories",
    "no stored",
    "not stored",
    "nothing stored",
    "do not indicate",
    "does not indicate",
    "dont indicate",
    "doesnt indicate",
    "do not contain",
    "does not contain",
    "dont contain",
    "doesnt contain",
    "no reference",
    "no mention",
    "no indication",
    "no evidence",
    "no explicit",
    "no information",
    "not have",
    "insufficient",
    "cannot",
    "can't say",
    "can't confirm",
    "cannot confirm",
    "not sure",
    "no way to know",
    "no data",
    "haven't",
    "have not",
    "no record",
)


def _normalize(text: Any) -> str:
    cleaned = (
        str(text or "")
        .replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
        .lower()
    )
    return " ".join(cleaned.split())


def _contains(haystack: str, needle: str) -> bool:
    needle = _normalize(needle)
    return bool(needle) and needle in _normalize(haystack)


def declines(answer: str) -> bool:
    """True when the answer abstains instead of making a factual claim."""
    normalized = _normalize(answer)
    return any(m in normalized for m in _DECLINE_MARKERS)


def grounding_verdict(
    answer: str,
    retrieved_texts: Sequence[str],
    expected: str = "",
    question_type: str = "known",
) -> str:
    """Judge one answer against its retrieved memories.

    ``retrieved_texts`` are the memory texts surfaced for the question
    (independent retrieval, not the agent's claims). ``expected`` is the
    dataset's ``expected_memory``/``expected_answer`` ("" for unknowns).
    """
    normalized_answer = _normalize(answer)
    if not normalized_answer:
        return UNKNOWN
    blob = " ".join(_normalize(t) for t in (retrieved_texts or []))
    expected_norm = _normalize(expected)
    if not blob:
        # Nothing retrieved: abstaining is correct, claiming is not —
        # except greetings / general-knowledge items (known, no expected
        # anchor) which need no memory support at all.
        if declines(answer):
            return SUPPORTED
        if question_type == "unknown":
            return UNSUPPORTED_INFERENCE
        if expected_norm:
            return UNSUPPORTED_INFERENCE
        return SUPPORTED
    expected_in_memory = bool(expected_norm) and expected_norm in blob
    expected_in_answer = bool(expected_norm) and expected_norm in normalized_answer
    is_decline = declines(answer)

    if question_type == "unknown":
        # No supporting memory should exist; any definitive claim is a
        # hallucination, any abstention is correct.
        return SUPPORTED if is_decline else UNSUPPORTED_INFERENCE

    # Known question: the expected fact should be stored.
    if expected_in_memory and expected_in_answer:
        return SUPPORTED
    if expected_in_memory and not expected_in_answer:
        if is_decline or any(m in normalized_answer for m in _DENIAL_MARKERS):
            # Memory exists but the agent says it was never provided.
            return CONTRADICTED
        return UNSUPPORTED_INFERENCE
    if not expected_in_memory and expected_in_answer:
        # Asserts the fact without retrieved support.
        return UNSUPPORTED_INFERENCE
    # No expected anchor at all (e.g. greeting): nothing to contradict.
    if is_decline:
        return SUPPORTED
    return SUPPORTED if normalized_answer else UNKNOWN


def is_supported(verdict: str) -> bool:
    """True only for correctly grounded answers / correct abstentions."""
    return verdict == SUPPORTED


def grounding_accuracy(results: List[dict]) -> float:
    """Fraction of scored items whose grounding verdict is SUPPORTED.

    Items with infra errors are excluded (failures, not hallucinations).
    """
    scored = [r for r in results if not r.get("error") and r.get("grounding_verdict")]
    if not scored:
        return 0.0
    return round(
        sum(1 for r in scored if r.get("grounding_verdict") == SUPPORTED)
        / len(scored),
        4,
    )


def unsupported_rate(results: List[dict]) -> float:
    """Fraction of scored items that are UNSUPPORTED_INFERENCE/CONTRADICTED."""
    scored = [r for r in results if not r.get("error") and r.get("grounding_verdict")]
    if not scored:
        return 0.0
    return round(
        sum(
            1
            for r in scored
            if r.get("grounding_verdict")
            in (UNSUPPORTED_INFERENCE, CONTRADICTED)
        )
        / len(scored),
        4,
    )


__all__ = [
    "CONTRADICTED",
    "GROUNDING_LABELS",
    "SUPPORTED",
    "UNKNOWN",
    "UNSUPPORTED_INFERENCE",
    "declines",
    "grounding_accuracy",
    "grounding_verdict",
    "is_supported",
    "unsupported_rate",
]
