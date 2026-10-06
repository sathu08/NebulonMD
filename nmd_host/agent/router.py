"""Step 6.14 — Deterministic intent router (eval suggestion #1).

The live-LLM tool loop is non-deterministic: across three runs and two
models the agent skipped ``recall`` on ``q1/q3/q4`` (``actual_tools: []``)
even though ``manager.retrieve`` found the memory every time
(``retrieval_ok: true``). This module fixes tool selection *before* the
LLM is consulted:

    "What is my name?"  →  intent = memory  →  tool = recall

Rules (no LLM, fully offline-testable):

* greetings / small talk / bare general-knowledge questions (no
  first-person memory reference) → ``"none"`` — answer directly.
* memory questions (first-person reference: my/mine/I/me, ``do i``,
  ``am i``, ``favorite``, ``prefer``, ``know``, ``remember`` ...) →
  ``"recall"``.
* durable self-statements (``my name is ...``, ``remember this ...``,
  ``note that i ...``) → ``"remember"``. The runtime only ever
  *forces* ``recall``; ``remember`` is reported for scoring so eval
  questions are never auto-stored.

``route_question`` returns ``"recall" | "remember" | "none"``.
"""

from __future__ import annotations

import re

# Greetings / small talk / thanks — never memory work.
_GREETING_RE = re.compile(
    r"^\s*(hi|hii+|hello|hey|yo|good\s*(morning|afternoon|evening)|"
    r"thanks?|thank\s*you|bye|goodbye|ok|okay|yes|no|sure)\s*[!.,?]*\s*$",
    re.IGNORECASE,
)

# Explicit store requests / durable self-statements.
_REMEMBER_RE = re.compile(
    r"\b(remember\s+(this|that)|don't\s+forget|note\s+that|"
    r"my\s+name\s+is|i\s+am\s+[A-Z]|born\s+on|i\s+live\s+in|"
    r"my\s+(favorite|favourite)\s+\w+\s+is|i\s+(like|love|prefer)\s+\w+)",
    re.IGNORECASE,
)

# First-person memory references → the answer may depend on stored memory.
_MEMORY_RE = re.compile(
    r"\b(my|mine|me\b.*\b(question|project|name|language|database)|"
    r"\bi\s+(am|know|use|like|love|prefer|build|work|have)|"
    r"\bdo\s+i\b|\bam\s+i\b|favorite|favourite|remember|recall|"
    r"what\s+(is\s+my|project\s+am\s+i|database\s+does)|"
    r"which\s+\w+\s+do\s+i|who\s+am\s+i)",
    re.IGNORECASE,
)


def route_question(text: str) -> str:
    """Classify one user utterance to ``recall`` / ``remember`` / ``none``.

    Never raises: unparseable input routes to ``"none"`` (answer directly).
    """
    cleaned = str(text or "").strip()
    if not cleaned:
        return "none"
    if _GREETING_RE.match(cleaned):
        return "none"
    if _REMEMBER_RE.search(cleaned):
        return "remember"
    if _MEMORY_RE.search(cleaned):
        return "recall"
    return "none"


def should_force_recall(text: str) -> bool:
    """True when the runtime must run ``recall`` before the first LLM turn."""
    try:
        return route_question(text) == "recall"
    except Exception:
        return False


__all__ = ["route_question", "should_force_recall"]
