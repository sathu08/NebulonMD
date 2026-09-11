"""HTTP client for the NebulonDB REST API (no ``ndb_host`` imports).

All persistence in nmd_host goes through NebulonDB's FastAPI service
(see ``.env`` / ``nmd_host/core/config.py``). The client speaks the service's
StandardResponse envelope: ``{success, message, data, ...}``.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import requests

from ..core.config import NebulonDBConfig
from .tracing import request_id_var

logger = logging.getLogger("nmd_host.api.client")


class NebulonDBError(RuntimeError):
    """Raised when the NebulonDB API rejects or fails a request."""

    pass


class NebulonDBClient:
    """Thin HTTP client over the NebulonDB API v1 routes.

    Auth is HTTP Basic (username/password from config). Reads and writes
    target per-user segments inside fixed corpora, e.g.
    ``mind_truth`` (cosmos) and ``mind_semantic`` (orbit).
    """

    def __init__(self, config: NebulonDBConfig, backend_retries: int = 2) -> None:
        self._config = config
        self._backend_retries = max(0, backend_retries)
        self._session = requests.Session()
        self._session.auth = (config.username, config.password)

    @property
    def config(self) -> NebulonDBConfig:
        return self._config

    # ------------------------------------------------------------------ #
    # Low-level                                                          #
    # ------------------------------------------------------------------ #

    def _url(self, path: str) -> str:
        return f"{self._config.base_url}{path}"

    def _timeout(self, method: str) -> tuple:
        """Explicit connect/read/write timeouts (P1).

        Writes (load/produce side-effects) get the longer ``write_timeout``;
        reads and verifies the ``read_timeout``. The connect timeout always
        bounds the handshake phase.
        """
        config = self._config
        if method in ("POST", "PUT", "PATCH", "DELETE"):
            return (config.connect_timeout, config.write_timeout)
        return (config.connect_timeout, config.read_timeout)

    @staticmethod
    def _is_transient(exc: NebulonDBError, status: Optional[int]) -> bool:
        """True when the failure is worth a bounded retry:
        transport errors (unreachable/timeout/reset) or backend 5xx."""
        if status is not None:
            return status >= 500
        message = str(exc)
        return any(
            marker in message.lower()
            for marker in ("unreachable", "timeout", "connection", "reset")
        )

    def _request(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        retries: int = 0,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Perform one request with explicit timeouts and bounded retries.

        ``retries`` only applies to callers that opted in (idempotent
        operations). Each retry backs off ``0.1 * 2**attempt`` seconds
        (max ~ 0.8s for the default 3 attempts). HTTP 4xx and backend
        ``success=false`` rejections are never retried.
        """
        headers = {}
        if request_id:
            headers["X-Request-ID"] = request_id
        else:
            current = request_id_var.get()
            if current and current != "-":
                headers["X-Request-ID"] = current
        attempts = retries + 1
        last_error: Optional[NebulonDBError] = None
        last_cause: Optional[BaseException] = None
        last_status: Optional[int] = None
        for attempt in range(attempts):
            try:
                if params:
                    resp = self._session.request(
                        method,
                        self._url(path),
                        json=payload,
                        params=params,
                        timeout=timeout or self._timeout(method),
                        headers=headers,
                    )
                else:
                    resp = self._session.request(
                        method,
                        self._url(path),
                        json=payload,
                        timeout=timeout or self._timeout(method),
                        headers=headers,
                    )
            except requests.RequestException as exc:
                last_error = NebulonDBError(
                    f"NebulonDB API unreachable ({self._config.base_url}): {exc}"
                )
                last_cause = exc
                last_status = None
            else:
                try:
                    body = resp.json()
                except ValueError:
                    body = {}
                if not isinstance(body, dict):
                    body = {}
                if resp.status_code >= 400:
                    last_error = NebulonDBError(
                        body.get("message")
                        or f"NebulonDB API returned HTTP {resp.status_code} for {method} {path}"
                    )
                    last_status = resp.status_code
                elif body.get("success") is False:
                    last_error = NebulonDBError(
                        body.get("message") or f"NebulonDB API rejected {method} {path}"
                    )
                    last_status = resp.status_code if resp.status_code >= 500 else None
                else:
                    return body
            if attempt < attempts - 1:
                if not self._is_transient(last_error, last_status):
                    break  # deterministic failure (4xx / rejection) — never retried
                backoff = 0.1 * (2 ** attempt)
                logger.warning(
                    "NebulonDB %s %s transient failure (attempt %d/%d); retrying in %.1fs: %s",
                    method, path, attempt + 1, attempts, backoff, last_error,
                )
                time.sleep(backoff)
        assert last_error is not None
        if last_cause is not None:
            raise last_error from last_cause
        raise last_error

    # ------------------------------------------------------------------ #
    # Auth                                                               #
    # ------------------------------------------------------------------ #

    def verify(self) -> Dict[str, Any]:
        """Confirm service availability and credentials; returns the user record."""
        return self._request(
            "GET", "/auth/verify", retries=self._backend_retries
        )

    # ------------------------------------------------------------------ #
    # Corpus                                                             #
    # ------------------------------------------------------------------ #

    def list_corpus(self) -> List[Dict[str, Any]]:
        return self._request(
            "GET",
            "/corpus/list_corpus",
            retries=self._backend_retries,
        ).get("data", {}).get("corpus_list", [])

    def create_corpus(self, name: str, ndb_type: str) -> None:
        self._request(
            "POST",
            "/corpus/create_corpus",
            {"corpus_name": name, "ndb_type": ndb_type},
        )

    def ensure_corpus(self, name: str, ndb_type: str) -> None:
        """Create the corpus on first use; no-op when it already exists."""
        if not any(c.get("name") == name for c in self.list_corpus()):
            self.create_corpus(name, ndb_type)

    def ensure_storage_corpora(self) -> None:
        """Ensure every storage corpus exists (idempotent, startup).

        Runs at service startup so the first memory write/recall never
        fails with ``Corpus 'mind_truth'/'mind_semantic'/'mind_chats' not
        found in metadata``. NebulonDB materialises the segments inside
        each corpus lazily on first use, so only the corpus itself needs
        provisioning here.
        """
        from ..stores.truth_store import TruthStore
        from ..stores.vector_store import VectorStore
        from ..stores.chat_history_store import ChatHistoryStore

        self.ensure_corpus(TruthStore.CORPUS, "cosmos")
        self.ensure_corpus(VectorStore.CORPUS, "orbit")
        self.ensure_corpus(ChatHistoryStore.CORPUS, "cosmos")

    def list_segment(
        self, corpus: str, ndb_type: str
    ) -> List[Dict[str, Any]]:
        """List the segments registered under a corpus (metadata records)."""
        return (
            self._request(
                "GET",
                "/segment/list_segment",
                params={"corpus_name": corpus},
                retries=self._backend_retries,
            )
            .get("data", {})
            .get("segment_list", [])
        )

    def ensure_segment(
        self,
        corpus: str,
        segment: str,
        ndb_type: str,
        set_columns: Optional[List[str]] = None,
        is_precomputed: Optional[bool] = None,
    ) -> None:
        """Ensure a corpus segment exists; no-op when already present.

        The CORSMOS engine materialises a segment on its first ``load_segment``
        write, and the backend registers it in corpus metadata only after at
        least one record is inserted. So ``ensure_segment`` guarantees the
        corpus exists and, when the segment is not yet registered, seeds it
        with a single empty record so identity lookups have an addressable
        segment. Subsequent loads use ``load_segment``/``get_data`` with the
        full record set (idempotent — re-loading never duplicates).
        """
        self.ensure_corpus(corpus, ndb_type)
        names = {
            str(entry.get("name"))
            for entry in self.list_segment(corpus, ndb_type)
            if isinstance(entry, dict)
        }
        if segment in names:
            return
        self.load_segment(
            corpus,
            segment,
            ndb_type,
            records=[{"text": '{"_seed": true}'}],
            set_columns=set_columns or ["text"],
            is_precomputed=is_precomputed,
        )

    # ------------------------------------------------------------------ #
    # Segment (records / vectors)                                        #
    # ------------------------------------------------------------------ #

    def load_segment(
        self,
        corpus: str,
        segment: str,
        ndb_type: str,
        records: List[Dict[str, Any]],
        set_columns: Optional[List[str]] = None,
        is_precomputed: Optional[bool] = None,
        lang_type: Optional[str] = None,
        doc_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Insert records into a segment (cosmos: docs, orbit: text to embed).

        ``metadata`` (optional) is merged into every record server-side;
        a per-record ``metadata`` dict inside ``records`` wins per row.
        """
        payload: Dict[str, Any] = {
            "corpus_name": corpus,
            "segment_name": segment,
            "ndb_type": ndb_type,
            "segment_dataset": records,
            "set_columns": set_columns or ["text"],
        }
        if is_precomputed is not None:
            payload["is_precomputed"] = is_precomputed
        if lang_type is not None:
            payload["lang_type"] = lang_type
        if doc_type is not None:
            payload["doc_type"] = doc_type
        if metadata is not None:
            payload["metadata"] = metadata
        logger.debug(f"load_segment payload: {payload}")
        return self._request("POST", "/segment/load_segment", payload)

    def get_data(
        self,
        corpus: str,
        segment: str,
        ndb_type: str,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        payload: Dict[str, Any] = {
            "corpus_name": corpus,
            "segment_name": segment,
            "ndb_type": ndb_type,
        }
        if limit is not None:
            payload["limit"] = limit
        return (
            self._request(
                "POST",
                "/segment/get_data",
                payload,
                retries=self._backend_retries,
            )
            .get("data", {})
            .get("records", [])
        )

    def search_segment(
        self,
        corpus: str,
        segment: str,
        ndb_type: str,
        search_item: str,
        top_matches: int = 5,
        mode: str = "auto",
        rank: bool = False,
    ) -> List[Dict[str, Any]]:
        """Semantic search; the query is embedded server-side."""
        return self._request(
            "POST",
            "/segment/search_segment",
            {
                "corpus_name": corpus,
                "segment_name": segment,
                "ndb_type": ndb_type,
                "search_item": search_item,
                "top_matches": top_matches,
                "mode": mode,
                "rank": rank,
            },
            retries=self._backend_retries,
        ).get("data", [])

    def delete_record(
        self, corpus: str, segment: str, ndb_type: str, record_id: int
    ) -> bool:
        body = self._request(
            "POST",
            "/segment/delete_record",
            {
                "corpus_name": corpus,
                "segment_name": segment,
                "ndb_type": ndb_type,
                "record_id": record_id,
            },
            retries=self._backend_retries,
        )
        return bool(body.get("exists", True))

    def segment_stats(
        self, corpus: str, segment: str, ndb_type: str
    ) -> Dict[str, Any]:
        return self._request(
            "POST",
            "/segment/segment_stats",
            {"corpus_name": corpus, "segment_name": segment, "ndb_type": ndb_type},
            retries=self._backend_retries,
        ).get("data", {})

    # ------------------------------------------------------------------ #
    # Mesh graph                                                         #
    # ------------------------------------------------------------------ #

    def mesh_load_graph(
        self,
        corpus: str,
        segment: str,
        ndb_type: str,
        edges: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Write edges; node labels are auto-resolved (idempotent inserts)."""
        return self._request(
            "POST",
            "/segment/mesh_load_graph",
            {
                "corpus_name": corpus,
                "segment_name": segment,
                "ndb_type": ndb_type,
                "edges": edges,
            },
        )
