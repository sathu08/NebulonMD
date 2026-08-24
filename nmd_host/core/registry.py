"""Username → user_id registry (explicit registration, no auto-register).

NebulonMind's identity layer lives in a dedicated ``nmd_Secrets``
(COSMOS) corpus under the ``Authentication`` segment, holding rows of
``{username, unique_id}``. Only this segment is identity-only; memory data
stays in ``mind_truth`` / ``mind_semantic`` under ``user_{unique_id}``.

Registration is explicit: a client calls ``/user/create_user`` first (or
``UserRegistry.create_user`` directly) and the system generates the opaque
``unique_id`` (``user_<hex>``). No code path auto-registers a username —
an unregistered username is rejected everywhere (the registry raises
``UserNotFoundError``, surfaced as HTTP 401/403 by the API gate).

Two implementations ship:

* ``NebulonDBUserRegistry`` — persistence over the NebulonDB REST API
  (COSMOS ``nmd_Secrets`` / ``Authentication``), used by the production
  ``DefaultServiceProvider``.
* ``InMemoryUserRegistry`` — a dict-backed fake for offline tests so no
  NebulonDB is required (mirrors the ``InMemoryServiceProvider``).
"""

from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from ..api import NebulonDBClient


class UserNotFoundError(RuntimeError):
    """Raised when a username is used without having been registered.

    Registration is explicit via ``create_user``; there is no auto-register,
    so an unknown username is always an error (surfaced as HTTP 401/403).
    """

    def __init__(self, username: str) -> None:
        super().__init__(f"username {username!r} not registered; call /user/create_user")
        self.username = username


def _random_user_id() -> str:
    return f"user_{uuid.uuid4().hex[:8]}"


class UserRegistry(ABC):
    """Identity contract: bootstrap / create_user / resolve / verify."""

    def __init__(
        self, corpus: str = "nmd_Secrets", segment: str = "Authentication"
    ) -> None:
        self.corpus = corpus
        self.segment = segment

    @abstractmethod
    def bootstrap(self) -> None:
        """Ensure the identity corpus + segment exist (idempotent)."""

    @abstractmethod
    def create_user(self, username: str) -> Tuple[str, bool]:
        """Register a user, returning ``(unique_id, created)``.

        Idempotent: re-calling with an already-registered username returns
        the stored id with ``created=False``.
        """

    @abstractmethod
    def resolve(self, username: str) -> str:
        """Map a registered username to its ``unique_id``.

        Raises ``UserNotFoundError`` when the username is unregistered.
        """

    def verify(self, username: str) -> str:
        """Gate for every operation: resolve (raise on unregistered)."""
        return self.resolve(username)


class NebulonDBUserRegistry(UserRegistry):
    """Production registry persisted via the NebulonDB REST API."""

    def __init__(
        self,
        client: NebulonDBClient,
        corpus: str = "nmd_Secrets",
        segment: str = "Authentication",
    ) -> None:
        super().__init__(corpus=corpus, segment=segment)
        self._api = client

    def bootstrap(self) -> None:
        self._api.ensure_segment(
            self.corpus, self.segment, "cosmos", set_columns=["text"]
        )

    def _records(self) -> List[Dict]:
        """Decode the identity rows stored in the COSMOS `text` column.

        The COSMOS engine persists a single ``text`` column per record (plus
        a few metadata fields and the backend-assigned ``_id``), so each
        identity ``{username, unique_id}`` pair is JSON-encoded into ``text``
        exactly like ``TruthStore`` does for memory documents. Rows that are
        not valid JSON objects (e.g. a seeded placeholder or legacy flat
        rows) are skipped.
        """
        rows: List[Dict] = []
        for record in self._api.get_data(self.corpus, self.segment, "cosmos"):
            try:
                doc = json.loads(record.get("text") or "")
            except (TypeError, ValueError):
                continue
            if isinstance(doc, dict):
                rows.append(doc)
        return rows

    def create_user(self, username: str) -> Tuple[str, bool]:
        username = (username or "").strip()
        existing = self._lookup(username)
        if existing is not None:
            return existing, False
        unique_id = _random_user_id()
        self._api.load_segment(
            self.corpus,
            self.segment,
            "cosmos",
            records=[
                {"text": json.dumps({"username": username, "unique_id": unique_id})}
            ],
            set_columns=["text"],
        )
        return unique_id, True

    def resolve(self, username: str) -> str:
        existing = self._lookup((username or "").strip())
        if existing is None:
            raise UserNotFoundError(username)
        return existing

    def _lookup(self, username: str) -> Optional[str]:
        if not username:
            raise UserNotFoundError(username)
        for record in self._records():
            if record.get("username") == username:
                unique_id = record.get("unique_id")
                if unique_id:
                    return unique_id
        return None


class InMemoryUserRegistry(UserRegistry):
    """Dict-backed registry for offline tests (no NebulonDB)."""

    def __init__(
        self, corpus: str = "nmd_Secrets", segment: str = "Authentication"
    ) -> None:
        super().__init__(corpus=corpus, segment=segment)
        self._users: Dict[str, str] = {}

    def bootstrap(self) -> None:
        # No-op: identity state is held in memory.
        pass

    def create_user(self, username: str) -> Tuple[str, bool]:
        username = (username or "").strip()
        existing = self._users.get(username)
        if existing is not None:
            return existing, False
        unique_id = _random_user_id()
        self._users[username] = unique_id
        return unique_id, True

    def resolve(self, username: str) -> str:
        username = (username or "").strip()
        unique_id = self._users.get(username)
        if unique_id is None:
            raise UserNotFoundError(username)
        return unique_id

    @property
    def users(self) -> Dict[str, str]:
        """Registered username → unique_id mapping (test introspection)."""
        return dict(self._users)


__all__ = [
    "InMemoryUserRegistry",
    "NebulonDBUserRegistry",
    "UserNotFoundError",
    "UserRegistry",
]
