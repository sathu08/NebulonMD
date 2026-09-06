"""Step 5 — reliability / production-hardening tests.

Exercises existing components against injected failures — no new business
logic. Everything here runs offline (no live NebulonDB needed):

* backend unavailable / timeout / malformed responses
* repository and lifecycle exceptions surfacing as the 500 envelope
* store failure compensation (truth/vector/graph partial writes)
* per-user resource lifecycle, shutdown, user isolation
* concurrent requests sharing per-user ``ServiceBundle``s
* idempotency of repeated store / update / delete
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import requests
from fastapi.testclient import TestClient

from nmd_host.api import NebulonDBClient, NebulonDBError
from nmd_host.api.server import create_app
from nmd_host.api.service import (
    DefaultServiceProvider,
    InMemoryServiceProvider,
    ServiceBundle,
)
from nmd_host.core.config import NebulonDBConfig
from nmd_host.core.mind import NebulonMind
from conftest import make_memory

API = "/api/NebulonMind"


def _register(provider, *usernames: str) -> None:
    """Explicit registration: every ``user_id`` is now a registered username."""
    for name in usernames:
        provider.create_user(name)


# ---------------------------------------------------------------------- #
# Fake NebulonDB backend (offline)                                       #
# ---------------------------------------------------------------------- #


class _FakeBackend:
    """Minimal in-process NebulonDB: record storage + injected failures."""

    def __init__(self, fail: str | None = None) -> None:
        self.fail = fail
        self._records: dict = {}
        self._next_id = 1
        self.loaded: list = []
        self.edges: list = []

    def _maybe_fail(self, method: str) -> None:
        if self.fail == method:
            raise RuntimeError(f"injected {method} failure")

    def _records_of(self, corpus: str, segment: str) -> list:
        return [
            r
            for r in self._records.values()
            if r["corpus"] == corpus and r["segment"] == segment
        ]

    def load_segment(self, corpus, segment, ndb_type, records, set_columns=None, is_precomputed=None):
        loaded: list = []
        for record in records:
            self._next_id += 1
            rec = {
                "corpus": corpus,
                "segment": segment,
                "ndb_type": ndb_type,
                "_id": self._next_id,
                "id": self._next_id,
                **{k: v for k, v in record.items()},
            }
            rec.setdefault("name", record.get("name"))
            rec.setdefault("label", record.get("name"))
            rec.setdefault("text", record.get("text", ""))
            rec["metadata"] = {"label": rec.get("name"), "text": rec.get("text", "")}
            self._records[rec["_id"]] = rec
            loaded.append(rec)
        self.loaded.extend(loaded)
        return {"success": True, "data": loaded}

    def get_data(self, corpus, segment, ndb_type, limit=None):
        self._maybe_fail("get_data")
        records = self._records_of(corpus, segment)
        if limit is not None:
            records = records[:limit]
        return [dict(r) for r in records]

    def delete_record(self, corpus, segment, ndb_type, record_id):
        self._maybe_fail("delete_record")
        rec = self._records.pop(record_id, None)
        return {"success": True, "data": {}, "exists": rec is not None}

    def search_segment(self, corpus, segment, ndb_type, search_item, top_matches=5, mode="auto", rank=False):
        self._maybe_fail("search_segment")
        hits = [
            {"id": r["_id"], "score": 1.0, "text": r["text"], "metadata": r["metadata"]}
            for r in self._records_of(corpus, segment)
        ][:top_matches]
        return {"success": True, "data": hits}

    def segment_stats(self, corpus, segment, ndb_type):
        self._maybe_fail("segment_stats")
        records = self._records_of(corpus, segment)
        return {"success": True, "data": {"vector_count": len(records), "edge_count": 0, "node_count": 0}}

    def mesh_load_graph(self, corpus, segment, ndb_type, edges):
        self._maybe_fail("mesh_load_graph")
        self.edges.extend(edges)
        return {"success": True, "data": {}}

    def verify(self):
        self._maybe_fail("verify")
        return {"success": True, "data": {"user": "fake"}}

    def list_corpus(self):
        return {"success": True, "data": {"corpus_list": []}}

    def create_corpus(self, name, ndb_type):
        return {"success": True, "data": {}}

    def ensure_corpus(self, name, ndb_type):
        return None

    def list_segment(self, corpus, ndb_type):
        return {"success": True, "data": {"segment_list": []}}

    def ensure_segment(self, corpus, segment, ndb_type, set_columns=None, is_precomputed=None):
        return None


class _FakeClient:
    """NebulonDBClient-shaped fake: duck-typed backend for stores/minds."""

    def __init__(self, backend: _FakeBackend) -> None:
        self.backend = backend

    def _url(self, path: str) -> str:
        return f"fake://{path}"

    def load_segment(self, *args, **kwargs):
        return self.backend.load_segment(*args, **kwargs)

    def get_data(self, *args, **kwargs):
        return self.backend.get_data(*args, **kwargs)

    def delete_record(self, *args, **kwargs):
        return self.backend.delete_record(*args, **kwargs)

    def search_segment(self, *args, **kwargs):
        return self.backend.search_segment(*args, **kwargs)

    def segment_stats(self, *args, **kwargs):
        return self.backend.segment_stats(*args, **kwargs)

    def mesh_load_graph(self, *args, **kwargs):
        return self.backend.mesh_load_graph(*args, **kwargs)

    def verify(self):
        return self.backend.verify()

    def list_corpus(self):
        return self.backend.list_corpus()

    def create_corpus(self, name, ndb_type):
        return self.backend.create_corpus(name, ndb_type)

    def ensure_corpus(self, name, ndb_type):
        return None

    def list_segment(self, corpus, ndb_type):
        return self.backend.list_segment(corpus, ndb_type)

    def ensure_segment(self, corpus, segment, ndb_type, set_columns=None, is_precomputed=None):
        return self.backend.ensure_segment(
            corpus, segment, ndb_type, set_columns=set_columns, is_precomputed=is_precomputed
        )


class _FakeResponse:
    """requests.Response stand-in: scripted payload or JSON failure."""

    def __init__(self, payload=None, status_code=200, json_error=None) -> None:
        self._payload = payload
        self.status_code = status_code
        self._json_error = json_error

    def json(self):
        if self._json_error is not None:
            raise self._json_error
        return self._payload


class _FakeSession:
    """requests.Session stand-in for injecting transport failures.

    The real ``NebulonDBClient._request`` runs against this session, so the
    wrap-into-``NebulonDBError`` behaviour under test is the production one.
    """

    auth = None

    def __init__(self, *, response=None, exc=None) -> None:
        self._response = response
        self._exc = exc

    def request(self, method, url, json=None, timeout=None, headers=None):
        if self._exc is not None:
            raise self._exc
        return self._response


class _TimeoutClient(NebulonDBClient):
    """Client whose HTTP transport raises ``requests.Timeout``."""

    def __init__(self, config: NebulonDBConfig) -> None:
        super().__init__(config)
        self._session = _FakeSession(exc=requests.exceptions.Timeout("read timed out"))


class _RejectingClient(NebulonDBClient):
    """Client whose backend answers every request with ``success=false``."""

    def __init__(self, config: NebulonDBConfig) -> None:
        super().__init__(config)
        self._session = _FakeSession(
            response=_FakeResponse({"success": False, "message": "rejected by backend"})
        )


def _dead_config() -> NebulonDBConfig:
    """Connection settings guaranteed to fail fast (port 1, connection refused)."""
    return NebulonDBConfig(host="127.0.0.1", port=1, username="u", password="p")


# ---------------------------------------------------------------------- #
# 5.1 / 5.9 — failure contract at the client                              #
# ---------------------------------------------------------------------- #


def test_client_unreachable_backend_is_controlled_error():
    client = NebulonDBClient(_dead_config())
    with pytest.raises(NebulonDBError, match="unreachable"):
        client.verify()


def test_client_timeout_is_controlled_error():
    client = _TimeoutClient(NebulonDBConfig(host="localhost", port=6969))
    with pytest.raises(NebulonDBError, match="timed out"):
        client.verify()


def test_client_rejection_is_controlled_error():
    client = _RejectingClient(NebulonDBConfig(host="localhost", port=6969))
    with pytest.raises(NebulonDBError, match="rejected by backend"):
        client.verify()


def test_client_server_error_without_message_is_controlled_error():
    client = NebulonDBClient(NebulonDBConfig(host="localhost", port=6969))
    client._session = _FakeSession(
        response=_FakeResponse("server blew up", status_code=502)
    )
    with pytest.raises(NebulonDBError, match="HTTP 502"):
        client.verify()


# ---------------------------------------------------------------------- #
# 5.2 — store failure consistency (compensation)                          #
# ---------------------------------------------------------------------- #


def test_store_compensates_when_graph_write_fails():
    backend = _FakeBackend(fail="mesh_load_graph")
    mind = NebulonMind(user_id="user_rel", client=_FakeClient(backend))
    memory = make_memory(
        "Sathya uses Python for backend work",
        user_id="user_rel",
        entities=["Sathya", "Python"],
    )

    with pytest.raises(RuntimeError, match="mesh_load_graph"):
        mind.store(memory)

    assert mind.get(memory.memory_id) is None  # truth rolled back
    assert mind.truth.read_all() == []
    assert backend._records == {}  # truth + vector rows gone too


def test_store_compensates_when_vector_write_fails(monkeypatch):
    backend = _FakeBackend()
    mind = NebulonMind(user_id="user_rel", client=_FakeClient(backend))
    memory = make_memory(
        "vector write fails",
        user_id="user_rel",
        entities=["Sathya"],
    )

    def boom(memory_id, text):
        raise RuntimeError("vector exploded")

    monkeypatch.setattr(mind.vector, "update", boom)
    with pytest.raises(RuntimeError, match="vector exploded"):
        mind.store(memory)

    assert mind.get(memory.memory_id) is None
    assert mind.truth.read_all() == []


# ---------------------------------------------------------------------- #
# 5.9 — API failure envelopes (no traceback / credentials leak)           #
# ---------------------------------------------------------------------- #


def test_health_reports_backend_down_when_unreachable():
    provider = DefaultServiceProvider(NebulonDBClient(_dead_config()))
    with TestClient(create_app(provider=provider)) as client:
        body = client.get(f"{API}/health").json()
    assert body["success"] is True
    assert body["data"]["backend"] == "down"


def test_store_with_unreachable_backend_returns_500_envelope():
    provider = DefaultServiceProvider(NebulonDBClient(_dead_config()))
    with TestClient(create_app(provider=provider), raise_server_exceptions=False) as client:
        resp = client.post(f"{API}/memory", json={"user_id": "u", "content": {"text": "x"}})
    body = resp.json()
    assert resp.status_code == 500
    assert body["success"] is False
    assert body["message"] == "internal server error"
    leaked = str(body).lower()
    assert "nmd_user_01" not in leaked
    assert "127.0.0.1" not in leaked
    assert "traceback" not in leaked


def test_store_with_timing_out_backend_returns_500_envelope():
    provider = DefaultServiceProvider(_TimeoutClient(NebulonDBConfig(host="localhost", port=6969)))
    with TestClient(create_app(provider=provider), raise_server_exceptions=False) as client:
        resp = client.post(f"{API}/memory", json={"user_id": "u", "content": {"text": "x"}})
    assert resp.status_code == 500
    body = resp.json()
    assert body["success"] is False
    assert "timed out" not in str(body)  # internals never leak


def test_rejected_backend_verify_does_not_crash_health():
    provider = DefaultServiceProvider(_RejectingClient(NebulonDBConfig(host="localhost", port=6969)))
    with TestClient(create_app(provider=provider), raise_server_exceptions=False) as client:
        body = client.get(f"{API}/health").json()
    assert body["success"] is True
    assert body["data"]["backend"] == "down"


def test_repository_exception_returns_500_envelope_without_leak():
    provider = InMemoryServiceProvider()
    app = create_app(provider=provider)
    _register(provider, "user_001")

    def boom(memory):
        raise RuntimeError("boom: my-secret-psk")

    provider.bundle("user_001").repository.create = boom
    with TestClient(app, raise_server_exceptions=False) as client:
        body = client.post(
            f"{API}/memory",
            json={"user_id": "user_001", "content": {"text": "x"}},
            params={"user_id": "user_001"},
        ).json()
    assert body["success"] is False
    assert body["message"] == "internal server error"
    assert "my-secret-psk" not in str(body)


def test_lifecycle_exception_on_search_returns_500_envelope():
    provider = InMemoryServiceProvider()
    app = create_app(provider=provider)
    _register(provider, "user_001")

    def boom(query, top_k=5, user_id=None):
        raise RuntimeError("retriever exploded")

    provider.bundle("user_001").manager.retrieve = boom
    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.get(f"{API}/search", params={"query": "x", "user_id": "user_001"})
    body = resp.json()
    assert resp.status_code == 500
    assert body["success"] is False
    assert "retriever exploded" not in str(body)


def test_malformed_backend_search_response_returns_controlled_500():
    backend = _FakeBackend()
    client = _FakeClient(backend)
    provider = DefaultServiceProvider(client)
    _register(provider, "user_001")
    with TestClient(create_app(provider=provider), raise_server_exceptions=False) as c:
        c.post(f"{API}/memory", json={"user_id": "user_001", "content": {"text": "hello world"}})
        backend.search_segment = lambda *a, **k: {"success": True, "data": 42}
        resp = c.get(f"{API}/search", params={"query": "hello", "user_id": "user_001"})
    assert resp.status_code == 500
    assert resp.json()["success"] is False


# ---------------------------------------------------------------------- #
# 5.3 — per-user resource lifecycle                                       #
# ---------------------------------------------------------------------- #


def test_database_provider_caches_one_bundle_per_user():
    backend = _FakeBackend()
    provider = DefaultServiceProvider(_FakeClient(backend))
    _register(provider, "user_cache")
    first = provider.bundle("user_cache")
    for _ in range(5):
        assert provider.bundle("user_cache") is first
    assert provider.user_count() == 1
    assert isinstance(first, ServiceBundle)


def test_in_memory_provider_isolates_users_and_caches_bundles():
    provider = InMemoryServiceProvider()
    _register(provider, "user_a", "user_b")
    a = provider.bundle("user_a")
    b = provider.bundle("user_b")
    assert a is provider.bundle("user_a")
    assert b is provider.bundle("user_b")
    assert a.repository is not b.repository
    a.repository.create(make_memory("Alice secret", user_id="user_a"))
    assert b.repository.list_all() == []
    assert a.repository.list_all(user_id="user_a")[0].content.text == "Alice secret"


def test_shutdown_closes_all_cached_resources():
    provider = InMemoryServiceProvider()
    _register(provider, "user_a", "user_b")
    bundle_a = provider.bundle("user_a")
    bundle_a.repository.create(make_memory("data", user_id="user_a"))
    provider.bundle("user_b")
    with TestClient(create_app(provider=provider)) as client:
        assert client.get(f"{API}/health").json()["data"]["minds"] == 2

    provider.close()  # lifespan shutdown path
    assert provider.user_count() == 0
    assert bundle_a.repository.list_all() == []


# ---------------------------------------------------------------------- #
# 5.3 — concurrency                                                       #
# ---------------------------------------------------------------------- #


def test_concurrent_requests_share_one_bundle_and_keep_state():
    provider = InMemoryServiceProvider()
    _register(provider, "user_conc")
    bundle = provider.bundle("user_conc")
    app = create_app(provider=provider)
    errors: list = []

    # No ``with``: each TestClient would run lifespan shutdown (provider.close)
    # independently, which would clear the shared bundle cache. Plain clients
    # exercise request handling only, like multiple connections to one app.
    def worker(offset: int):
        client = TestClient(app, raise_server_exceptions=False)
        for i in range(4):
            text = f"concurrent memory {offset}-{i} about stars"
            resp = client.post(
                f"{API}/memory",
                params={"user_id": "user_conc"},
                json={
                    "user_id": "user_conc",
                    "memory_id": f"mem_conc_{offset}_{i}",
                    "content": {"text": text},
                },
            )
            if resp.status_code != 201:
                errors.append(resp.text)

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(worker, range(6)))
    assert errors == []
    # one logical resource set shared by all requests
    assert provider.bundle("user_conc") is bundle
    assert provider.user_count() == 1
    assert len(bundle.repository.list_all()) == 24


def test_concurrent_requests_keep_users_isolated():
    provider = InMemoryServiceProvider()
    app = create_app(provider=provider)
    errors: list = []
    leaked: list = []

    users = [f"user_iso_{i}" for i in range(8)]
    _register(provider, *users)

    def worker(user_id: str):
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            f"{API}/memory",
            params={"user_id": user_id},
            json={
                "user_id": user_id,
                "memory_id": f"mem_iso_{user_id}",
                "content": {"text": f"only {user_id} knows this fact"},
            },
        )
        if resp.status_code != 201:
            errors.append(resp.text)
        hits = client.get(
            f"{API}/search", params={"query": "only", "user_id": user_id}
        ).json()["data"]["results"]
        if any(user_id not in h["content"]["text"] for h in hits):
            leaked.append(hits)

    users = [f"user_iso_{i}" for i in range(8)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(worker, users))
    assert errors == []
    assert leaked == []
    assert provider.user_count() == len(users)


# ---------------------------------------------------------------------- #
# 5.7 — idempotent repeated operations                                    #
# ---------------------------------------------------------------------- #


def test_repeated_store_same_memory_id_converges():
    provider = InMemoryServiceProvider()
    app = create_app(provider=provider)
    _register(provider, "user_idem")
    with TestClient(app, raise_server_exceptions=False) as client:
        payload = {
            "user_id": "user_idem",
            "memory_id": "mem_idem",
            "content": {"text": "same memory every time"},
        }
        first = client.post(f"{API}/memory", params={"user_id": "user_idem"}, json=payload)
        second = client.post(f"{API}/memory", params={"user_id": "user_idem"}, json=payload)
        assert first.status_code == 201 and second.status_code == 201
        stored = provider.bundle("user_idem").repository.list_all()
        assert len(stored) == 1
        assert stored[0].memory_id == "mem_idem"


def test_repeated_update_converges_to_last_value():
    provider = InMemoryServiceProvider()
    app = create_app(provider=provider)
    _register(provider, "user_idem")
    with TestClient(app, raise_server_exceptions=False) as client:
        client.post(
            f"{API}/memory",
            params={"user_id": "user_idem"},
            json={"user_id": "user_idem", "content": {"text": "original"}},
        )
        memory = provider.bundle("user_idem").repository.list_all()[0]
        mid = memory.memory_id
        client.put(f"{API}/memory/{mid}", json={"status": "archived"}, params={"user_id": "user_idem"})
        client.put(f"{API}/memory/{mid}", json={"status": "active"}, params={"user_id": "user_idem"})
        final = client.get(f"{API}/memory/{mid}", params={"user_id": "user_idem"}).json()
    assert final["data"]["memory"]["status"] == "active"


def test_repeated_delete_first_succeeds_then_404_envelope():
    provider = InMemoryServiceProvider()
    app = create_app(provider=provider)
    _register(provider, "user_idem")
    with TestClient(app, raise_server_exceptions=False) as client:
        client.post(
            f"{API}/memory",
            params={"user_id": "user_idem"},
            json={"user_id": "user_idem", "content": {"text": "to be deleted"}},
        )
        memory = provider.bundle("user_idem").repository.list_all()[0]
        mid = memory.memory_id
        first = client.delete(f"{API}/memory/{mid}", params={"user_id": "user_idem"})
        second = client.delete(f"{API}/memory/{mid}", params={"user_id": "user_idem"})
    assert first.status_code == 200
    assert first.json()["success"] is True
    assert second.status_code == 404
    assert second.json()["success"] is False


def test_store_then_search_then_delete_then_not_found_is_stable():
    provider = InMemoryServiceProvider()
    app = create_app(provider=provider)
    _register(provider, "user_idem")
    with TestClient(app, raise_server_exceptions=False) as client:
        for _ in range(3):
            client.post(
                f"{API}/memory",
                params={"user_id": "user_idem"},
                json={"user_id": "user_idem", "content": {"text": "stable cycle"}},
            )
        memory = provider.bundle("user_idem").repository.list_all()[0]
        mid = memory.memory_id
        for _ in range(3):
            assert client.get(f"{API}/memory/{mid}", params={"user_id": "user_idem"}).status_code == 200
        client.delete(f"{API}/memory/{mid}", params={"user_id": "user_idem"})
        for _ in range(3):
            assert client.get(f"{API}/memory/{mid}", params={"user_id": "user_idem"}).status_code == 404