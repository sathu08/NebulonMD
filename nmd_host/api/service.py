"""Phase 4 — per-user service bundles and injectable providers.

Routes never construct ``NebulonMind`` / repositories themselves; they only
talk to the ``ServiceBundle`` their user's provider hands out, so CRUD goes
through the repository and recall/context through the Step 3 lifecycle
manager (the existing phases stay the single source of behaviour).

Two providers ship:

* ``DefaultServiceProvider`` — the production stack: ``NebulonMind`` over
  the NebulonDB REST API (NebulonDB credentials stay inside ``NebulonMind``,
  never exposed by the API).
* ``InMemoryServiceProvider`` — the identical interfaces over an
  ``InMemoryRepository``; used by unit tests so no NebulonDB is required.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from ..core.config import NebulonDBConfig
from ..core.models import Memory
from ..core.registry import (
    InMemoryUserRegistry,
    NebulonDBUserRegistry,
    UserNotFoundError,
    UserRegistry,
)
from ..core.repository import InMemoryRepository, NebulonMindRepository
from ..lifecycle.manager import MemoryLifecycleManager, build_lifecycle_manager
from .client import NebulonDBClient

logger = logging.getLogger("nmd_host.api.service")


@dataclass
class ServiceBundle:
    """Everything the API needs for one user.

    ``expand_fn`` / ``relate_fn`` are injected by the provider so the routes
    stay storage-agnostic: the default provider backs them with
    ``NebulonMind`` graph features, the in-memory provider with repository
    equivalents (or a plain fallback).
    """

    user_id: str
    repository: Any
    manager: MemoryLifecycleManager
    expand_fn: Optional[Callable[[str, int], List[Memory]]] = None
    relate_fn: Optional[Callable[[str, str, str], None]] = None

    def recall_candidates(
        self, query: str, top_k: int, expand: bool = False
    ) -> List[Memory]:
        """Candidate memories for ``expand=True`` searches (graph expansion
        delegated to the provider); otherwise plain repository search."""
        if expand and self.expand_fn is not None:
            return self.expand_fn(query, top_k)
        return self.repository.search(query, top_k=top_k)

    def relate(self, memory_id: str, entity: str, relation: str = "HAS_ENTITY") -> None:
        """Link a memory to an entity (existing ``relate()`` functionality)."""
        if self.relate_fn is not None:
            self.relate_fn(memory_id, entity, relation)
        else:
            self.repository.relate(memory_id, entity)


class ServiceProvider(ABC):
    """Hands out per-user ``ServiceBundle``s; owns resource lifecycle.

    The argument to ``bundle`` is a *username* (identity) which the provider
    resolves through its ``UserRegistry`` to the real ``user_id``
    (``unique_id``). An unregistered username raises ``UserNotFoundError`` —
    never auto-registered.
    """

    name: str = "abstract"

    @abstractmethod
    def bundle(self, user_id: str) -> ServiceBundle:
        """Return (creating if needed) the service bundle for a username.

        The username must already be registered (``create_user``); otherwise
        ``UserNotFoundError`` is raised.
        """

    @abstractmethod
    def registry(self) -> UserRegistry:
        """The provider's identity registry (bootstrap / create_user)."""

    def user_count(self) -> int:
        return 0

    def create_user(self, username: str) -> tuple:
        """Register a username, returning ``(unique_id, created)`` (idempotent)."""
        return self.registry().create_user(username)

    def bootstrap(self) -> None:
        """Ensure identity corpus/segment exist (idempotent, called on startup)."""
        self.registry().bootstrap()

    def verify_backend(self) -> bool:
        """True when the persistence backend is reachable (best-effort)."""
        return False

    def close(self) -> None:
        """Release all resources (called on app shutdown)."""


class DefaultServiceProvider(ServiceProvider):
    """Production provider: one shared ``NebulonDBClient`` per app, a
    ``NebulonMind`` per user (cached for the process lifetime). Usernames are
    resolved to their ``unique_id`` through the behind-the-scenes
    ``NebulonDBUserRegistry``; unregistered usernames are rejected."""

    name = "nebulondb"

    @property
    def client(self) -> "NebulonDBClient":
        """The shared backend HTTP client (used for durable sessions/job state)."""
        return self._client

    def __init__(
        self,
        client: Optional[NebulonDBClient] = None,
        backend_retries: int = 2,
        registry: Optional[UserRegistry] = None,
    ) -> None:
        self._client = client or NebulonDBClient(
            NebulonDBConfig.from_env(), backend_retries=backend_retries
        )
        self._registry = registry or NebulonDBUserRegistry(self._client)
        self._bundles: Dict[str, ServiceBundle] = {}

    def registry(self) -> UserRegistry:
        return self._registry

    def bundle(self, username: str) -> ServiceBundle:
        user_id = self._registry.resolve(username)
        bundle = self._bundles.get(username)
        if bundle is None:
            # Imported lazily: ``NebulonMind`` imports ``..api`` eagerly, so
            # importing it at module level would be circular.
            from ..core.mind import NebulonMind

            mind = NebulonMind(user_id=user_id, client=self._client)
            repository = NebulonMindRepository(mind)
            manager = build_lifecycle_manager(repository, searcher=mind)
            bundle = ServiceBundle(
                user_id=user_id,
                repository=repository,
                manager=manager,
                expand_fn=lambda query, k: mind.recall(query, top_k=k, expand=True),
                relate_fn=lambda m, e, r: mind.relate(m, e, r),
            )
            self._bundles[username] = bundle
            logger.info("spawned NebulonMind bundle for %s (user_id=%s)", username, user_id)
        return bundle

    def user_count(self) -> int:
        return len(self._bundles)

    def verify_backend(self) -> bool:
        try:
            self._client.verify()
            return True
        except Exception:
            return False

    def bootstrap(self) -> None:
        """Ensure the identity corpus/segment AND the storage corpora exist.

        Storage corpora (``mind_truth`` COSMOS, ``mind_semantic`` ORBIT) are
        provisioned here so the first memory write/recall never hits a
        missing-corpus failure; segments inside them materialise lazily on
        the backend. Any failure propagates to the caller (the server
        lifespan logs it and readiness reports not-ready).
        """
        super().bootstrap()
        self._client.ensure_storage_corpora()

    def close(self) -> None:
        for bundle in self._bundles.values():
            try:
                bundle.repository.close()
            except Exception:  # pragma: no cover - defensive shutdown
                pass
        self._bundles.clear()


class InMemoryServiceProvider(ServiceProvider):
    """Offline provider: identical interfaces over ``InMemoryRepository``.
    Unit tests use this so no NebulonDB is required (Phase 4.10). Usernames
    resolve through an in-memory registry; unregistered usernames are
    rejected exactly like the production provider."""

    name = "in-memory"

    def __init__(self, registry: Optional[UserRegistry] = None) -> None:
        self._registry = registry or InMemoryUserRegistry()
        self._bundles: Dict[str, ServiceBundle] = {}

    def registry(self) -> UserRegistry:
        return self._registry

    def bundle(self, username: str) -> ServiceBundle:
        user_id = self._registry.resolve(username)
        bundle = self._bundles.get(username)
        if bundle is None:
            repository = InMemoryRepository()
            manager = build_lifecycle_manager(repository, searcher=repository)
            bundle = ServiceBundle(
                user_id=user_id,
                repository=repository,
                manager=manager,
                expand_fn=None,
                relate_fn=lambda m, e, r: repository.relate(m, e),
            )
            self._bundles[username] = bundle
        return bundle

    def user_count(self) -> int:
        return len(self._bundles)

    def close(self) -> None:
        for bundle in self._bundles.values():
            bundle.repository.close()
        self._bundles.clear()


__all__ = [
    "DefaultServiceProvider",
    "InMemoryServiceProvider",
    "ServiceBundle",
    "ServiceProvider",
    "UserNotFoundError",
    "UserRegistry",
]