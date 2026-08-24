"""Step 6 — Agent session management (changes.txt 6.9).

Per the changes.txt architectural decision, no SQLite or Redis is
introduced for Step 6: NebulonDB stays the only persistent store. The
``AgentSessionManager`` is a short-lived *runtime* registry — sessions are
process-local memory (transcript + metadata), bounded by the agent config.
Anything worth keeping across restarts already flows to NebulonDB through
the memory tools (remember / recall), never through a session table.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from .schemas import AgentMessage

logger = logging.getLogger("nmd_host.agent.session")


class AgentSession(BaseModel):
    """Logical agent conversation carried in process memory.

    ``status`` is ``active`` until ``close()``; a closed (or expired)
    session is no longer returned by the manager. ``transcript`` holds the
    running message history so multi-turn chats stay coherent without the
    client resending history every request.
    """

    session_id: str
    user_id: str
    created_at: float = Field(description="Unix timestamp of creation")
    last_activity: float = Field(description="Unix timestamp of last touch")
    status: Literal["active", "closed"] = "active"
    metadata: dict = Field(default_factory=dict)
    transcript: List[AgentMessage] = Field(default_factory=list)


class AgentSessionManager:
    """Thread-safe, bounded registry of ``AgentSession`` objects.

    Sessions are scoped by ``(user_id, session_id)`` so one user can never
    read or close another user's session (per-user isolation, same policy as
    the memory bundles). Capacity is bounded per user
    (``max_sessions_per_user``); stale sessions are expired lazily on access
    (``session_ttl_seconds``, 0 disables expiry).

    An optional ``store`` (``SessionStore``) lifts the registry from a
    process-local cache to a durable one: creations/transcript appends are
    persisted and misses on ``get`` are re-hydrated from the store, so
    multi-turn sessions survive a service restart. Persistence is
    best-effort — any store failure is logged and degrades to in-memory
    behaviour.
    """

    def __init__(
        self,
        max_sessions_per_user: int = 100,
        session_ttl_seconds: float = 3600.0,
        store: Optional["SessionStore"] = None,
    ) -> None:
        self._max_sessions_per_user = max(1, max_sessions_per_user)
        self._ttl = max(0.0, session_ttl_seconds)
        self._store = store
        self._sessions: Dict[tuple, AgentSession] = {}
        self._lock = threading.Lock()

    @property
    def store(self) -> Optional["SessionStore"]:
        return self._store

    def _store_save(self, session: AgentSession) -> None:
        if self._store is None:
            return
        try:
            self._store.save(session)
        except Exception as exc:  # best-effort durability
            logger.warning(
                "failed to persist agent session %s for user %s: %s",
                session.session_id, session.user_id, exc,
            )

    def _store_delete(self, user_id: str, session_id: str) -> None:
        if self._store is None:
            return
        try:
            self._store.delete(user_id, session_id)
        except Exception as exc:  # best-effort durability
            logger.warning(
                "failed to delete durable session %s for user %s: %s",
                session_id, user_id, exc,
            )

    # ------------------------------------------------------------------ #
    # lifecycle: create / get / close / touch                            #
    # ------------------------------------------------------------------ #

    def create(
        self,
        user_id: str,
        metadata: Optional[dict] = None,
        session_id: Optional[str] = None,
    ) -> AgentSession:
        """Create a new active session for ``user_id``.

        Raises ``ValueError`` when the user already holds
        ``max_sessions_per_user`` sessions (capacity is bounded so an
        operator-controlled leak cannot grow the registry unboundedly).
        """
        with self._lock:
            session_id = session_id or self._generate_id()
            key = self._key(user_id, session_id)
            if key in self._sessions:
                logger.warning(
                    "agent session %s already exists for user %s",
                    session_id, user_id,
                )
                raise ValueError(
                    f"agent session {session_id!r} already exists for user {user_id!r}"
                )
            self._prune_expired_locked(user_id)
            if self._active_count_locked(user_id) >= self._max_sessions_per_user:
                logger.warning(
                    "user %s reached max agent sessions (%d)",
                    user_id, self._max_sessions_per_user,
                )
                raise ValueError(
                    f"user {user_id!r} reached max agent sessions "
                    f"({self._max_sessions_per_user}); close one first"
                )
            now = time.time()
            session = AgentSession(
                session_id=session_id,
                user_id=user_id,
                created_at=now,
                last_activity=now,
                metadata=dict(metadata or {}),
            )
            self._sessions[key] = session
        self._store_save(session)
        return session

    def get(self, user_id: str, session_id: str) -> Optional[AgentSession]:
        """Return the session if it exists, is active and not expired.

        On an in-memory miss, tries to re-hydrate the session from the
        durable store (so ``session_id`` survives restarts).
        """
        with self._lock:
            key = self._key(user_id, session_id)
            session = self._sessions.get(key)
            if session is not None:
                if session.status != "active" or self._expired(session):
                    self._sessions.pop(key, None)
                    session = None
        if session is None and self._store is not None:
            try:
                stored = self._store.get(user_id, session_id)
            except Exception as exc:  # best-effort durability
                logger.warning(
                    "failed to rehydrate session %s for user %s: %s",
                    session_id, user_id, exc,
                )
                stored = None
            if stored is not None:
                with self._lock:
                    if stored.status != "active" or self._expired(stored):
                        self._store_delete(user_id, session_id)
                    else:
                        self._sessions[key] = stored
                        session = stored
        return session

    def touch(self, user_id: str, session_id: str) -> bool:
        """Refresh ``last_activity`` (returns False when not found/closed)."""
        with self._lock:
            session = self._sessions.get(self._key(user_id, session_id))
            if session is None or session.status != "active":
                return False
            session.last_activity = time.time()
            return True

    def close(self, user_id: str, session_id: str) -> bool:
        """Close and remove the session (returns False when not found)."""
        with self._lock:
            removed = (
                self._sessions.pop(self._key(user_id, session_id), None) is not None
            )
        if removed:
            self._store_delete(user_id, session_id)
        return removed

    def append_messages(
        self,
        user_id: str,
        session_id: str,
        messages: List[AgentMessage],
    ) -> bool:
        """Extend a session's transcript (returns False when not found/closed)."""
        if not messages:
            return True
        with self._lock:
            session = self._sessions.get(self._key(user_id, session_id))
            if session is None or session.status != "active":
                return False
            session.transcript.extend(messages)
            session.last_activity = time.time()
        self._store_save(session)
        return True

    # ------------------------------------------------------------------ #
    # introspection                                                      #
    # ------------------------------------------------------------------ #

    def active_count(self, user_id: str) -> int:
        """Number of live sessions for a user (expiry swept lazily)."""
        with self._lock:
            self._prune_expired_locked(user_id)
            return self._active_count_locked(user_id)

    def list_active(self, user_id: str) -> List[AgentSession]:
        """Snapshot of the user's active (non-expired) sessions."""
        with self._lock:
            self._prune_expired_locked(user_id)
            return [
                session
                for (uid, _sid), session in self._sessions.items()
                if uid == user_id and session.status == "active"
            ]

    def clear(self) -> None:
        """Drop every session (used at app shutdown)."""
        with self._lock:
            self._sessions.clear()

    # ------------------------------------------------------------------ #
    # internals                                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _key(user_id: str, session_id: str) -> tuple:
        return (user_id, session_id)

    @staticmethod
    def _generate_id() -> str:
        return uuid.uuid4().hex

    def _expired(self, session: AgentSession) -> bool:
        return self._ttl > 0 and time.time() - session.last_activity > self._ttl

    def _active_count_locked(self, user_id: str) -> int:
        return sum(
            1
            for (uid, _sid), s in self._sessions.items()
            if uid == user_id and s.status == "active"
        )

    def _prune_expired_locked(self, user_id: str) -> None:
        for key, session in list(self._sessions.items()):
            if key[0] == user_id and self._expired(session):
                self._sessions.pop(key, None)


__all__ = ["AgentSession", "AgentSessionManager"]
