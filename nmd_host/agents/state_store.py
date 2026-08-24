"""Background job state persisted over NebulonDB (optional).

``BackgroundScheduler`` keeps job state (last run, last result, error, runs)
in process memory; ``NebulonDBJobStateStore`` persists it in a COSMOS corpus
so a service restart does not reset the scheduler to "never ran".

Layout mirrors the session store: corpus ``mind_background``, segment
``jobs``, one JSON document per job id (the backend API has no upsert, so
saves are delete-then-insert). Persistence is best-effort — the scheduler
wraps every call with a fallback to in-memory state.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional, Protocol

from ..api import NebulonDBClient

logger = logging.getLogger("nmd_host.agents.state_store")


class JobStateStore(Protocol):
    """Storage for scheduler job state, keyed by job id."""

    def load(self) -> Dict[str, dict]:
        ...  # pragma: no cover - protocol

    def save(self, job_id: str, state: Dict[str, Any]) -> None:
        ...  # pragma: no cover - protocol


class InMemoryJobStateStore:
    """Dict-backed fake for offline tests (same contract as the network store)."""

    def __init__(self) -> None:
        self._state: Dict[str, dict] = {}

    def load(self) -> Dict[str, dict]:
        return dict(self._state)

    def save(self, job_id: str, state: Dict[str, Any]) -> None:
        self._state[job_id] = dict(state)


class NebulonDBJobStateStore:
    """Job state persisted in COSMOS ``mind_background`` (one doc per job)."""

    CORPUS = "mind_background"
    SEGMENT = "jobs"

    def __init__(self, client: NebulonDBClient) -> None:
        self._api = client

    @staticmethod
    def _parse_doc(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            doc = json.loads(record["text"])
        except (KeyError, TypeError, ValueError):
            return None
        return doc if isinstance(doc, dict) else None

    def _find_record(self, job_id: str) -> Optional[Dict[str, Any]]:
        for record in self._api.get_data(self.CORPUS, self.SEGMENT, "cosmos"):
            doc = self._parse_doc(record)
            if doc and doc.get("job_id") == job_id:
                return record
        return None

    def load(self) -> Dict[str, dict]:
        states: Dict[str, dict] = {}
        try:
            records = self._api.get_data(self.CORPUS, self.SEGMENT, "cosmos")
        except Exception:  # segment/backend absent → no persisted state
            return states
        for record in records:
            doc = self._parse_doc(record)
            if doc and doc.get("job_id"):
                states[doc["job_id"]] = {
                    key: doc[key]
                    for key in ("last_run", "last_result", "last_error", "runs")
                    if key in doc
                }
        return states

    def save(self, job_id: str, state: Dict[str, Any]) -> None:
        record = self._find_record(job_id)
        if record is not None:
            self._api.delete_record(
                self.CORPUS, self.SEGMENT, "cosmos", record["_id"]
            )
        payload = {"job_id": job_id}
        payload.update(state)
        self._api.load_segment(
            self.CORPUS,
            self.SEGMENT,
            "cosmos",
            records=[{"text": json.dumps(payload)}],
            set_columns=["text"],
        )


__all__ = [
    "InMemoryJobStateStore",
    "JobStateStore",
    "NebulonDBJobStateStore",
]