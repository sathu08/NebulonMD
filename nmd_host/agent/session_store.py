"""Step 6 — durable agent sessions over NebulonDB (optional).

The ``AgentSessionManager`` is a process-local registry; it holds session
state for the running process only. ``NebulonDBSessionStore`` persists
sessions in a COSMOS corpus so a service restart does not lose multi-turn
continuity: clients keep passing the same ``session_id`` and the manager
re-hydrates the session on the next access.

Layout mirrors ``TruthStore``: a COSMOS corpus (``mind_sessions``) with one
``user_{user_id}`` segment per user and one JSON document per session. The
backend API has no upsert, so updates rewrite the document (delete-then-
insert, idempotent re-saves).

Durability is best-effort by design: every call is wrapped defensively by
the manager, so a backend outage degrades to the historical in-memory
behaviour instead of breaking chat.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Protocol

from ..api import NebulonDBClient
from .schemas import AgentMessage
from .session import AgentSession

logger = logging.getLogger("nmd_host.agent.session_store")


class SessionStore(Protocol):
    """Persistence contract an ``AgentSessionManager`` may be given."""

    def get(self, user_id: str, session_id: str) -> Optional[AgentSession]:
        ...  # pragma: no cover - protocol

    def save(self, session: AgentSession) -> None:
        ...  # pragma: no cover - protocol

    def delete(self, user_id: str, session_id: str) -> bool:
        ...  # pragma: no cover - protocol


class InMemorySessionStore:
    """Dict-backed fake for offline tests (same contract as the network store)."""

    def __init__(self) -> None:
        self._sessions: Dict[tuple, AgentSession] = {}

    def get(self, user_id: str, session_id: str) -> Optional[AgentSession]:
        return self._sessions.get((user_id, session_id))

    def save(self, session: AgentSession) -> None:
        self._sessions[(session.user_id, session.session_id)] = session

    def delete(self, user_id: str, session_id: str) -> bool:
        return self._sessions.pop((user_id, session_id), None) is not None


class NebulonDBSessionStore:
    """Sessions persisted in COSMOS ``mind_sessions`` (per-user segments)."""

    CORPUS = "mind_sessions"

    def __init__(self, client: NebulonDBClient) -> None:
        self._api = client

    @staticmethod
    def _parse_doc(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            doc = json.loads(record["text"])
        except (KeyError, TypeError, ValueError):
            return None
        return doc if isinstance(doc, dict) else None

    def _segment(self, user_id: str) -> str:
        return f"user_{user_id}"

    def _load(self, user_id: str) -> List[Dict[str, Any]]:
        return self._api.get_data(self.CORPUS, self._segment(user_id), "cosmos")

    def _find_record(self, user_id: str, session_id: str) -> Optional[Dict[str, Any]]:
        for record in self._load(user_id):
            doc = self._parse_doc(record)
            if doc and doc.get("session_id") == session_id:
                return record
        return None

    def get(self, user_id: str, session_id: str) -> Optional[AgentSession]:
        record = self._find_record(user_id, session_id)
        if record is None:
            return None
        doc = self._parse_doc(record)
        try:
            return AgentSession.model_validate(doc)
        except Exception:  # corrupt/older doc — drop it rather than crash chat
            logger.warning(
                "dropping unparseable durable session %s for user %s",
                session_id, user_id,
            )
            return None

    def save(self, session: AgentSession) -> None:
        segment = self._segment(session.user_id)
        self.delete(session.user_id, session.session_id)
        self._api.load_segment(
            self.CORPUS,
            segment,
            "cosmos",
            records=[{"text": json.dumps(session.model_dump(mode="json"))}],
            set_columns=["text"],
        )

    def delete(self, user_id: str, session_id: str) -> bool:
        record = self._find_record(user_id, session_id)
        if record is None:
            return False
        return self._api.delete_record(
            self.CORPUS, self._segment(user_id), "cosmos", record["_id"]
        )


__all__ = [
    "InMemorySessionStore",
    "NebulonDBSessionStore",
    "SessionStore",
]