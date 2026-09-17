"""nmd_monitor decorators — the ``@traceable`` equivalent for custom code."""

from __future__ import annotations

import functools
import time
from contextlib import contextmanager
from typing import Any, Callable, Generator, List, Optional

from .models import MonitorSpan


@contextmanager
def span(
    name: str,
    kind: str = "tool",
    *,
    sink: Optional[List[MonitorSpan]] = None,
    input_text: str = "",
) -> Generator[MonitorSpan, None, None]:
    """Time one block as a span; appends to ``sink`` on exit (fail-open)."""
    current = MonitorSpan(name=name, kind=kind, input=input_text)
    started = time.monotonic()
    try:
        yield current
        current.ok = True
    except Exception as exc:
        current.ok = False
        current.error = f"{type(exc).__name__}: {exc}"[:300]
        raise
    finally:
        current.latency_ms = (time.monotonic() - started) * 1000.0
        try:
            if sink is not None:
                sink.append(current)
        except Exception:
            pass


def monitored(name: str = "", kind: str = "tool") -> Callable:
    """Decorate any function so failures + latency are recorded as a span.

    The span is appended to ``self._monitor_spans`` when present, else
    discarded — the return value / exceptions never change.
    """
    def decorator(fn: Callable) -> Callable:
        span_name = name or fn.__name__

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            sink = None
            if args:
                sink = getattr(args[0], "_monitor_spans", None)
            with span(span_name, kind, sink=sink):
                return fn(*args, **kwargs)

        return wrapper

    return decorator


__all__ = ["monitored", "span"]
