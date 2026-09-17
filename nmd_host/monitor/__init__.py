"""nmd_monitor — own LangSmith-equivalent for NebulonMind (P0).

Local-first observability: every ``POST /agent/chat`` produces one
``MonitorTrace`` (trace_id + spans + latency/tokens) persisted in
NebulonDB corpus ``mind_traces`` (segment ``user_{id}``), listed via
``/api/NebulonMind/monitor/*``.

Fail-open by contract: a monitor failure never fails a chat.
"""

from .config import MonitorConfig
from .decorators import monitored, span
from .evaluators import summarize_online
from .models import MonitorSpan, MonitorStats, MonitorTrace
from .recorder import MonitorRecorder
from .store import (
    InMemoryMonitorStore,
    NebulonDBMonitorStore,
    monitor_store_for,
)

__all__ = [
    "InMemoryMonitorStore",
    "MonitorConfig",
    "MonitorRecorder",
    "MonitorSpan",
    "MonitorStats",
    "MonitorTrace",
    "NebulonDBMonitorStore",
    "monitor_store_for",
    "monitored",
    "span",
    "summarize_online",
]
