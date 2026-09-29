"""nmd_monitor store — trace persistence (NebulonDB or in-memory).

Mirrors ``stores/chat_history_store.py``: one JSON doc per trace in the
``mind_traces`` COSMOS corpus (``doc_type="monitor_trace"``), segment
``user_{partition}`` for multi-tenant isolation. Full trace JSON lives in
the ``text`` column so no NebulonDB schema change is needed.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from .models import MonitorFeedback, MonitorStats, MonitorTrace

logger = logging.getLogger("nmd_host.monitor.store")


class InMemoryMonitorStore:
    """Dict-backed store for offline tests (same interface as DB store)."""

    def __init__(self, user_id: str = "") -> None:
        self._user_id = user_id
        self._traces: Dict[str, MonitorTrace] = {}

    def save(self, trace: MonitorTrace) -> MonitorTrace:
        self._traces[trace.trace_id] = trace
        return trace

    def get(self, trace_id: str) -> Optional[MonitorTrace]:
        return self._traces.get(trace_id)

    def list(
        self,
        *,
        q: str = "",
        tool: str = "",
        ok: Optional[bool] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[MonitorTrace]:
        items = list(self._traces.values())
        items.sort(key=lambda t: t.created_at_ms, reverse=True)
        needle = (q or "").strip().lower()
        if needle:
            items = [
                t for t in items
                if needle in (t.input_text or "").lower()
                or needle in (t.answer or "").lower()
            ]
        if tool:
            items = [t for t in items if tool in (t.tools_used or [])]
        if ok is not None:
            items = [t for t in items if t.ok is ok]
        return items[offset: offset + max(0, limit)]

    def stats(self, *, since_ms: int = 0) -> MonitorStats:
        items = [t for t in self._traces.values() if t.created_at_ms >= since_ms]
        total = len(items)
        if not total:
            return MonitorStats()
        errors = sum(1 for t in items if not t.ok)
        lat = [t.latency_ms for t in items]
        toks = [t.tokens for t in items]
        recall = sum(1 for t in items if "recall" in (t.tools_used or []))
        remember = sum(1 for t in items if "remember" in (t.tools_used or []))
        return MonitorStats(
            traces=total,
            errors=errors,
            error_rate=round(errors / total, 4),
            avg_latency_ms=round(sum(lat) / total, 2),
            max_latency_ms=round(max(lat), 2),
            avg_tokens=round(sum(toks) / total, 2),
            recall_used_rate=round(recall / total, 4),
            remember_used_rate=round(remember / total, 4),
        )

    def save_feedback(self, feedback: MonitorFeedback) -> bool:
        """Save feedback to in-memory store."""
        try:
            key = f"{feedback.user_id}_{feedback.trace_id}_{feedback.created_at_ms}"
            self._traces[key] = feedback
            return True
        except Exception:
            return False

    def get_feedback(self, trace_id: str = "", limit: int = 50, offset: int = 0) -> List[MonitorFeedback]:
        """Get feedback entries, optionally filtered by trace_id."""
        items = [t for t in self._traces.values() if isinstance(t, MonitorFeedback)]
        if trace_id:
            items = [t for t in items if t.trace_id == trace_id]
        items.sort(key=lambda f: f.created_at_ms, reverse=True)
        return items[offset: offset + max(0, limit)]


class NebulonDBMonitorStore:
    """NebulonDB-backed store (production path)."""

    CORPUS = "mind_traces"
    FEEDBACK_CORPUS = "mind_feedback"
    DOC_TYPE = "monitor_trace"

    def __init__(self, client: Any, user_id: str, username: str = "") -> None:
        self._api = client
        self._user_id = user_id
        self._username = username or user_id
        self._segment = f"user_{user_id}"

    @staticmethod
    def _parse_record(record: Dict[str, Any]) -> Optional[MonitorTrace]:
        try:
            payload = json.loads(record.get("text", ""))
        except (TypeError, ValueError):
            return None
        if not isinstance(payload, dict) or not payload.get("trace_id"):
            return None
        try:
            return MonitorTrace.model_validate(payload)
        except Exception:
            return None

    def _parse_all(self) -> List[MonitorTrace]:
        try:
            records = self._api.get_data(self.CORPUS, self._segment, "cosmos")
        except Exception as exc:
            logger.warning("monitor trace read failed: %s", exc)
            return []
        items = []
        for record in records or []:
            trace = self._parse_record(record)
            if trace is not None:
                items.append(trace)
        return items

    def save(self, trace: MonitorTrace) -> MonitorTrace:
        # Defensive auto-create: startup bootstrap provisions mind_traces,
        # but older servers / raced first-writes still hit "Corpus
        # 'mind_traces' not found". Best-effort ensure keeps the turn alive;
        # server.py already treats capture as fail-open (warning only).
        try:
            ensure = getattr(self._api, "ensure_corpus", None)
            if callable(ensure):
                ensure(self.CORPUS, "cosmos")
        except Exception:
            pass
        self._api.load_segment(
            self.CORPUS,
            self._segment,
            "cosmos",
            records=[{"text": trace.model_dump_json()}],
            set_columns=["text"],
            lang_type="en",
            doc_type=self.DOC_TYPE,
            metadata={"app": "nmd_monitor", "ok": trace.ok},
        )
        return trace

    def get(self, trace_id: str) -> Optional[MonitorTrace]:
        for trace in self._parse_all():
            if trace.trace_id == trace_id:
                return trace
        return None

    def list(
        self,
        *,
        q: str = "",
        tool: str = "",
        ok: Optional[bool] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[MonitorTrace]:
        items = self._parse_all()
        items.sort(key=lambda t: t.created_at_ms, reverse=True)
        needle = (q or "").strip().lower()
        if needle:
            items = [
                t for t in items
                if needle in (t.input_text or "").lower()
                or needle in (t.answer or "").lower()
            ]
        if tool:
            items = [t for t in items if tool in (t.tools_used or [])]
        if ok is not None:
            items = [t for t in items if t.ok is ok]
        return items[offset: offset + max(0, limit)]

    def stats(self, *, since_ms: int = 0) -> MonitorStats:
        mem = InMemoryMonitorStore(self._user_id)
        for trace in self._parse_all():
            mem.save(trace)
        return mem.stats(since_ms=since_ms)

    def save_feedback(self, feedback: MonitorFeedback) -> bool:
        """Save a feedback entry (same load_segment/get_data API as traces).

        Was broken: used self.client.upsert/query which don't exist on
        NebulonDBClient (always AttributeError → silent False). Now one JSON
        doc per feedback in mind_feedback cosmos, segment user_<id>.
        """
        try:
            ensure = getattr(self._api, "ensure_corpus", None)
            if callable(ensure):
                ensure(self.FEEDBACK_CORPUS, "cosmos")
        except Exception:
            pass
        try:
            self._api.load_segment(
                self.FEEDBACK_CORPUS,
                f"user_{self._user_id}",
                "cosmos",
                records=[{"text": feedback.model_dump_json()}],
                set_columns=["text"],
                lang_type="en",
                doc_type="monitor_feedback",
                metadata={"app": "nmd_monitor"},
            )
            return True
        except Exception:
            return False

    def get_feedback(self, trace_id: str = "", limit: int = 50, offset: int = 0) -> List[MonitorFeedback]:
        """Get feedback entries, optionally filtered by trace_id."""
        try:
            records = self._api.get_data(
                self.FEEDBACK_CORPUS, f"user_{self._user_id}", "cosmos")
        except Exception:
            return []
        items: List[MonitorFeedback] = []
        for record in records or []:
            try:
                payload = json.loads(record.get("text", ""))
            except (TypeError, ValueError):
                continue
            if not isinstance(payload, dict):
                continue
            try:
                fb = MonitorFeedback.model_validate(payload)
            except Exception:
                continue
            if not trace_id or fb.trace_id == trace_id:
                items.append(fb)
        items.sort(key=lambda f: f.created_at_ms, reverse=True)
        return items[offset: offset + max(0, limit)]


def monitor_store_for(provider: Any, username: str):
    """Monitor store for a username under any provider.

    Same resolution as ``history_store_for``: registered usernames use
    their opaque ``unique_id`` partition, anything else falls back to the
    name (lenient, never an error). DB-backed when the provider exposes a
    backend ``client``, in-memory otherwise (cached on the provider).
    """
    try:
        partition = provider.registry().resolve(username)
    except Exception:
        partition = username
    client = getattr(provider, "client", None)
    if client is not None:
        return NebulonDBMonitorStore(client, partition, username)
    stores = provider.__dict__.setdefault("_monitor_stores", {})
    if partition not in stores:
        stores[partition] = InMemoryMonitorStore(partition)
    return stores[partition]


def retention_cutoff_ms(days: int) -> int:
    """Epoch-ms cutoff for the P3 retention sweep (traces older -> delete)."""
    return int((time.time() - max(0, days) * 86400) * 1000)


__all__ = [
    "InMemoryMonitorStore",
    "NebulonDBMonitorStore",
    "monitor_store_for",
    "retention_cutoff_ms",
    "MonitorFeedback",
]
