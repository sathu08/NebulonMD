"""Step 13 — Evaluation metrics (pure functions, fully offline).

Each metric reads a list of *result records* produced by
``nmd_host.evaluation.runner`` and returns a float in ``[0, 1]`` (or an
averaged latency/token value). Metrics are deliberately cheap and
dependency-free so they can be unit-tested without an LLM.

A result record carries at least:

* ``type`` — ``known`` (an answer exists in memory) or ``unknown``
* ``expected_tool`` — ``recall`` / ``remember`` / ``none``
* ``tool_ok`` — the agent chose the expected tool (or none)
* ``retrieval_ok`` — expected memory text was surfaced by the pipeline
  (only meaningful for recall items)
* ``answer_ok`` — the final answer matched the expected answer (known items)
* ``declines_answer`` — the answer admitted it had no supporting memory
  (only meaningful for unknown items)
* ``latency_ms``, ``tokens`` — per-question cost
"""

from __future__ import annotations

from typing import List

from .grounding import grounding_accuracy, unsupported_rate


def _safe_div(numerator: float, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _avg(key: str, results: List[dict]) -> float:
    values = [float(r.get(key) or 0.0) for r in results]
    return round(sum(values) / len(values), 2) if values else 0.0


def tool_selection_accuracy(results: List[dict]) -> float:
    """Fraction of questions where the agent picked the expected tool."""
    return _safe_div(
        sum(1 for r in results if r.get("tool_ok")), len(results)
    )


def retrieval_accuracy(results: List[dict]) -> float:
    """Fraction of recall questions where the expected memory was retrieved.

    This measures the memory pipeline independent of the final answer:
    retrieval can be correct while the answer is wrong.
    """
    recall_items = [
        r for r in results
        if r.get("type") == "known" and r.get("expected_tool") == "recall"
    ]
    return _safe_div(
        sum(1 for r in recall_items if r.get("retrieval_ok")),
        len(recall_items),
    )


def answer_correctness(results: List[dict]) -> float:
    """Fraction of known questions answered correctly (any expected tool)."""
    known = [r for r in results if r.get("type") == "known"]
    return _safe_div(
        sum(1 for r in known if r.get("answer_ok")), len(known)
    )


def hallucination_rate(results: List[dict]) -> float:
    """Fraction of unknown questions where the agent made an unsupported claim.

    An unknown question has no supporting memory; answering it with a
    definitive claim (i.e. the answer did *not* decline) is a hallucination.
    Items that errored (e.g. 429 rate-limit with empty answer) are infra
    failures, not hallucinations, so they are excluded from the rate.
    So are blank answers: with no claim made, there is nothing to judge
    (they fail answer-side, never here).
    """
    unknown = [
        r for r in results
        if r.get("type") == "unknown" and not r.get("error")
        and str(r.get("answer") or "").strip()
    ]
    return _safe_div(
        sum(1 for r in unknown if not r.get("declines_answer")),
        len(unknown),
    )


def latency_stats(results: List[dict]) -> dict:
    """Per-run latency (ms): average and max across all questions."""
    values = [float(r.get("latency_ms") or 0.0) for r in results]
    if not values:
        return {"avg_ms": 0.0, "max_ms": 0.0}
    return {
        "avg_ms": round(sum(values) / len(values), 2),
        "max_ms": round(max(values), 2),
    }


def token_stats(results: List[dict]) -> dict:
    """Per-run token usage: average (approximate estimate)."""
    return {"avg_tokens": _avg("tokens", results)}


def router_forced_rate(results: List[dict]) -> float:
    """Fraction of routed-recall questions where recall actually ran.

    Suggestion #1's axis: the intent router said ``recall`` — did the
    agent (forced pre-step or its own call) run it? Excludes infra errors.
    """
    routed = [
        r for r in results
        if r.get("router_says") == "recall" and not r.get("error")
    ]
    return _safe_div(
        sum(1 for r in routed if r.get("router_forced")), len(routed)
    )


def summarize(results: List[dict]) -> dict:
    """One compact report for the dashboard/API endpoint."""
    return {
        "questions": len(results),
        "tool_selection_accuracy": tool_selection_accuracy(results),
        "retrieval_accuracy": retrieval_accuracy(results),
        "answer_correctness": answer_correctness(results),
        "hallucination_rate": hallucination_rate(results),
        # Step 13.1 — separated axes (suggestion #5): grounding verdict
        # accuracy, unsupported-claim rate, and router reliability sit
        # alongside the legacy keyword metrics (kept for compat).
        "grounding_accuracy": grounding_accuracy(results),
        "unsupported_rate": unsupported_rate(results),
        "router_forced_rate": router_forced_rate(results),
        "avg_latency_ms": latency_stats(results)["avg_ms"],
        "max_latency_ms": latency_stats(results)["max_ms"],
        "avg_tokens": token_stats(results)["avg_tokens"],
    }


__all__ = [
    "answer_correctness",
    "grounding_accuracy",
    "hallucination_rate",
    "latency_stats",
    "retrieval_accuracy",
    "router_forced_rate",
    "summarize",
    "token_stats",
    "tool_selection_accuracy",
    "unsupported_rate",
]