"""Username registry tests (offline, no NebulonDB).

Covers the explicit-registration doctrine on the in-memory registry and the
API gate built on top of it:

* ``create_user`` generates an opaque ``user_<hex>`` id and is idempotent.
* ``resolve`` returns the stored id for a known username.
* ``resolve`` on an unknown username raises (no auto-register) and the API
  surfaces it as a 403.
* the offline test provider gates every per-user route on the registry.
* bootstrap is idempotent.
"""

import json

import pytest
from fastapi.testclient import TestClient

from nmd_host.api.server import create_app
from nmd_host.api.service import (
    DefaultServiceProvider,
    InMemoryServiceProvider,
    UserNotFoundError,
)
from nmd_host.core.registry import InMemoryUserRegistry, NebulonDBUserRegistry

USER_URL = "/api/NebulonMind/user/create_user"
SETUP_URL = "/api/NebulonMind/user/setup"
RESOLVE_URL = "/api/NebulonMind/user/resolve"
MEMORY_URL = "/api/NebulonMind/memory"


@pytest.fixture
def registry():
    return InMemoryUserRegistry()


@pytest.fixture
def provider(registry):
    return InMemoryServiceProvider(registry=registry)


@pytest.fixture
def client(provider):
    app = create_app(provider=provider)
    with TestClient(app) as test_client:
        yield test_client


# ---------------------------------------------------------------------- #
# Registry behaviour (unit)                                              #
# ---------------------------------------------------------------------- #


def test_create_user_generates_opaque_hex_id(registry):
    unique_id, created = registry.create_user("alice")
    assert created is True
    assert unique_id.startswith("user_")
    assert len(unique_id) == len("user_") + 8
    assert unique_id[5:]  # non-empty hex suffix
    # id is opaque: not derived by encoding the username
    assert "alice" not in unique_id


def test_create_user_is_idempotent_no_duplicate(registry):
    first_id, first_created = registry.create_user("alice")
    second_id, second_created = registry.create_user("alice")
    assert first_created is True
    assert second_created is False
    assert second_id == first_id
    # a distinct username gets a distinct id
    bob_id, bob_created = registry.create_user("bob")
    assert bob_created is True
    assert bob_id != first_id


def test_resolve_returns_stored_id_for_known_username(registry):
    unique_id, _ = registry.create_user("alice")
    assert registry.resolve("alice") == unique_id
    assert registry.verify("alice") == unique_id


def test_resolve_unknown_username_raises_no_auto_register(registry):
    with pytest.raises(UserNotFoundError):
        registry.resolve("ghost")
    # resolution never creates the user (no auto-register)
    assert registry.users == {}


def test_bootstrap_is_idempotent(registry):
    registry.bootstrap()
    registry.bootstrap()
    registry.bootstrap()


# ---------------------------------------------------------------------- #
# NebulonDB registry over a real-shaped COSMOS backend                   #
# ---------------------------------------------------------------------- #


class _CosmosFake:
    """Mimics the real COSMOS engine: records keep ONLY ``text`` + ``_id``.

    This is the behaviour a live NebulonDB backend actually exhibits (each
    identity row is JSON-encoded into the ``text`` column — any other columns
    are dropped), so a registry bug here fails fast instead of passing only
    against a lenient fake.
    """

    def __init__(self):
        self._records: dict = {}
        self._next_id = 0
        self._segments: dict = {}
        self._corpora: set = set()

    def list_corpus(self):
        return [{"name": n} for n in self._corpora]

    def create_corpus(self, name, ndb_type):
        self._corpora.add(name)
        self._segments.setdefault(name, [])
        return {"success": True, "message": "created"}

    def ensure_corpus(self, name, ndb_type):
        self.create_corpus(name, ndb_type)

    def ensure_storage_corpora(self):
        self.ensure_corpus("mind_truth", "cosmos")
        self.ensure_corpus("mind_semantic", "orbit")

    def list_segment(self, corpus, ndb_type):
        names = self._segments.get(corpus, [])
        return [{"name": s, "inserted": len(names[s])} for s in names]

    def ensure_segment(self, corpus, segment, ndb_type, set_columns=None, is_precomputed=None):
        self.ensure_corpus(corpus, ndb_type)
        if segment not in self._segments.setdefault(corpus, []):
            self.load_segment(
                corpus,
                segment,
                ndb_type,
                [{"text": "{}"}],
                set_columns=set_columns or ["text"],
                is_precomputed=is_precomputed,
            )

    def load_segment(self, corpus, segment, ndb_type, records, set_columns=None, is_precomputed=None):
        key = (corpus, segment)
        stored = self._records.setdefault(key, [])
        for record in records:
            self._next_id += 1
            stored.append({"text": record.get("text", ""), "_id": self._next_id})
        if segment not in self._segments.setdefault(corpus, []):
            self._segments[corpus].append(segment)
        return {"success": True, "data": [{"_id": self._next_id}]}

    def get_data(self, corpus, segment, ndb_type, limit=None):
        rows = list(self._records.get((corpus, segment), []))
        if limit is not None:
            rows = rows[:limit]
        return rows

    def delete_record(self, corpus, segment, ndb_type, record_id):
        key = (corpus, segment)
        rows = self._records.setdefault(key, [])
        self._records[key] = [r for r in rows if r["_id"] != record_id]
        return {"success": True}


def test_bootstrap_storage_corpora_provisioned_on_startup():
    backend = _CosmosFake()
    provider = DefaultServiceProvider(
        client=backend, registry=NebulonDBUserRegistry(client=backend)
    )
    provider.bootstrap()
    # identity corpus/segment plus both storage corpora exist afterwards
    assert "nmd_Secrets" in backend._corpora
    assert "mind_truth" in backend._corpora
    assert "mind_semantic" in backend._corpora
    # idempotent — a second startup does not error or duplicate
    provider.bootstrap()
    assert len(backend._corpora) == 3


def test_nebulondb_registry_round_trips_over_real_cosmos_shape():
    client = _CosmosFake()
    registry = NebulonDBUserRegistry(client=client)
    registry.bootstrap()

    unique_id, created = registry.create_user("alice")
    assert created is True
    assert unique_id.startswith("user_")

    # idempotent: second registration returns the stored id
    again, created_again = registry.create_user("alice")
    assert created_again is False
    assert again == unique_id

    # resolve reads it back through the JSON-in-text column
    assert registry.resolve("alice") == unique_id
    with pytest.raises(UserNotFoundError):
        registry.resolve("ghost")

    # the identity row really is a single JSON doc in the text column
    rows = client.get_data("nmd_Secrets", "Authentication", "cosmos")
    decoded = [json.loads(r["text"]) for r in rows if isinstance(json.loads(r["text"]), dict)]
    assert {"username": "alice", "unique_id": unique_id} in decoded


def test_nebulondb_registry_skips_non_json_seed_rows():
    client = _CosmosFake()
    registry = NebulonDBUserRegistry(client=client)
    registry.bootstrap()
    # a legacy flattened row (pre-JSON format) must not break lookups
    client._records[("nmd_Secrets", "Authentication")].append({"text": "alice", "_id": 999})
    assert registry._lookup("ghost") is None
    registry.create_user("alice")
    assert registry.resolve("alice").startswith("user_")


# ---------------------------------------------------------------------- #
# API gate (offline HTTP)                                                #
# ---------------------------------------------------------------------- #


def test_create_user_endpoint_returns_user_id(client, provider):
    resp = client.post(USER_URL, json={"username": "alice"})
    assert resp.status_code == 201
    body = resp.json()
    assert body["success"] is True
    data = body["data"]
    assert data["username"] == "alice"
    assert data["user_id"].startswith("user_")
    assert data["created"] is True


def test_create_user_endpoint_idempotent(client):
    first = client.post(USER_URL, json={"username": "alice"}).json()["data"]
    second = client.post(USER_URL, json={"username": "alice"}).json()["data"]
    assert second["user_id"] == first["user_id"]
    assert second["created"] is False


def test_create_user_endpoint_strip_and_validate(client):
    resp = client.post(USER_URL, json={"username": ""})
    assert resp.status_code == 422


def test_setup_endpoint_activates_registered_user(client):
    created = client.post(USER_URL, json={"username": "alice"}).json()["data"]
    # setup is a POST twin of resolve and never creates
    resp = client.post(SETUP_URL, json={"username": "alice"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["username"] == "alice"
    assert body["data"]["user_id"] == created["user_id"]
    assert body["data"]["created"] is False


def test_setup_endpoint_rejects_unregistered(client, provider):
    resp = client.post(SETUP_URL, json={"username": "ghost"})
    assert resp.status_code == 404
    body = resp.json()
    assert body["success"] is False
    assert "not registered" in body["message"]
    # no auto-register side effect
    assert "ghost" not in provider.registry().users


def test_setup_endpoint_validates_username(client):
    resp = client.post(SETUP_URL, json={"username": ""})
    assert resp.status_code == 422


def test_resolve_endpoint_returns_stored_user_id(client):
    created = client.post(USER_URL, json={"username": "alice"}).json()["data"]
    resp = client.get(RESOLVE_URL, params={"username": "alice"})
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["user_id"] == created["user_id"]
    assert data["created"] is False


def test_resolve_endpoint_rejects_unregistered(client, provider):
    resp = client.get(RESOLVE_URL, params={"username": "ghost"})
    assert resp.status_code == 404
    assert resp.json()["success"] is False
    assert "ghost" not in provider.registry().users


def test_unregistered_username_rejected_with_403(client):
    resp = client.get(MEMORY_URL + "/mem_x", params={"user_id": "ghost"})
    assert resp.status_code == 403
    body = resp.json()
    assert body["success"] is False
    assert "not registered" in body["message"]
    assert "create_user" in body["message"]


def test_registered_user_can_use_endpoints(client):
    created = client.post(USER_URL, json={"username": "alice"}).json()["data"]
    assert created["created"] is True
    resp = client.post(
        MEMORY_URL,
        json={"user_id": "alice", "content": {"text": "My name is Sathya"}},
        params={"user_id": "alice"},
    )
    assert resp.status_code == 201
    memory = resp.json()["data"]["memory"]
    # memory is pinned to the resolved opaque user id
    assert memory["user_id"] == created["user_id"]
    assert memory["content"]["text"] == "My name is Sathya"


def test_cross_user_isolation(client):
    client.post(USER_URL, json={"username": "alice"})
    client.post(USER_URL, json={"username": "bob"})
    created_alice = client.post(
        MEMORY_URL,
        json={"user_id": "alice", "content": {"text": "Alice secret"}},
        params={"user_id": "alice"},
    ).json()["data"]["memory"]["memory_id"]
    created_bob = client.post(
        MEMORY_URL,
        json={"user_id": "bob", "content": {"text": "Bob secret"}},
        params={"user_id": "bob"},
    ).json()["data"]["memory"]["memory_id"]
    # memories are pinned to distinct resolved user ids
    alice_id = client.get(
        MEMORY_URL + "/" + created_alice,
        params={"user_id": "alice"},
    ).json()["data"]["memory"]["user_id"]
    bob_id = client.get(
        MEMORY_URL + "/" + created_bob,
        params={"user_id": "bob"},
    ).json()["data"]["memory"]["user_id"]
    assert alice_id != bob_id
    # each user only recalls their own memory
    alice_memories = client.get(
        "/api/NebulonMind/search",
        params={"query": "secret", "user_id": "alice"},
    ).json()["data"]["results"]
    bob_memories = client.get(
        "/api/NebulonMind/search",
        params={"query": "secret", "user_id": "bob"},
    ).json()["data"]["results"]
    assert any(m["content"]["text"] == "Alice secret" for m in alice_memories)
    assert all(m["content"]["text"] != "Alice secret" for m in bob_memories)