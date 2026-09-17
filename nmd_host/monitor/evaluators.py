"""nmd_monitor online evaluators — heuristic scores over stored traces.

Offline scoring (ground truth) stays in ``evaluation/metrics.py``. These
helpers score *live* traffic with no expected answer: the LangSmith
"online eval" equivalent for P3 dashboards.
"""

from __future__ import annotations

from typing import Iterable


def _safe_div(numerator: float, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def answered_with_zero_recall(traces: Iterable) -> float:
    """Fraction of answered runs that never called ``recall`` (grounding gap)."""
    items = list(traces)
    answered = [t for t in items if getattr(t, "answer", "")]
    return _safe_div(
        sum(1 for t in answered if "recall" not in (getattr(t, "tools_used", []) or [])),
        len(answered),
    )


def tool_error_rate(traces: Iterable) -> float:
    """Fraction of traces with ``ok=False`` (any tool/LLM failure)."""
    items = list(traces)
    return _safe_div(sum(1 for t in items if not getattr(t, "ok", True)), len(items))


def empty_recall_rate(traces: Iterable) -> float:
    """Fraction of recall runs that surfaced zero memories."""
    items = [t for t in traces if "recall" in (getattr(t, "tools_used", []) or [])]
    return _safe_div(
        sum(1 for t in items if int(getattr(t, "recall_memories", 0) or 0) == 0),
        len(items),
    )


def summarize_online(traces: Iterable) -> dict:
    """One compact online-eval report for dashboards."""
    items = list(traces)
    return {
        "traces": len(items),
        "tool_error_rate": tool_error_rate(items),
        "answered_with_zero_recall": answered_with_zero_recall(items),
        "empty_recall_rate": empty_recall_rate(items),
    }


__all__ = [
    "answered_with_zero_recall",
    "empty_recall_rate",
    "summarize_online",
    "tool_error_rate",
]
