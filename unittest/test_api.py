"""Phase 4.10 — API endpoint tests.

Every endpoint (and its failure paths) is exercised against the
``InMemoryServiceProvider``, so no NebulonDB is required. Also verifies
Phase 4.7 (standard error envelope, no leaked internals),
4.12 (backend credentials never visible) and 4.9 (OpenAPI docs).
"""

import pytest
from fastapi.testclient import TestClient

from nmd_host.api.server import create_app
from nmd_host.api.service import InMemoryServiceProvider
from nmd_host.core.models import Memory, MemoryStatus
from nmd_host.core.repository import InMemoryRepository
from nmd_host.lifecycle.context import MemoryContextBuilder

MEMORY_URL = "/api/NebulonMind/memory"


@pytest.fixture
def provider():
    return InMemoryServiceProvider()


@pytest.fixture
def client(provider):
    app = create_app(provider=provider)
    with TestClient(app) as test_client:
        # Explicit registration: every user_id below is a registered username.
        for name in ("user_001", "user_empty", "user_a", "user_b", "user_api"):
            test_client.post(
                "/api/NebulonMind/user/create_user", json={"username": name}
            )
        yield test_client


def store(client, text="My name is Sathya", user_id="user_001", gate=False, **body):
    payload = {"user_id": user_id, "content": {"text": text}}
    payload.update(body)
    params = {"user_id": user_id}
    if gate:
        params["gate"] = "true"
    return client.post(MEMORY_URL, json=payload, params=params)


# ---------------------------------------------------------------------- #
# Health (4.12)                                                          #
# ---------------------------------------------------------------------- #


def test_health_envelope(provider, client):
    resp = client.get("/api/NebulonMind/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["backend"] == "in-memory"
    assert body["data"]["provider"] == "in-memory"
    assert body["data"]["minds"] == 0
    # after a store the user count grows
    store(client)
    assert client.get("/api/NebulonMind/health").json()["data"]["minds"] == 1


def test_health_never_leaks_backend_credentials(client):
    resp = client.get("/api/NebulonMind/health")
    text = str(resp.json())
    for secret in ("nmd_user_01", "6969", "NEBULONDB_", "password"):
        assert secret not in text


def test_openapi_docs_complete(provider, client):
    spec = client.get("/openapi.json").json()
    paths = spec["paths"]
    for path in (
        "/api/NebulonMind/health",
        "/api/NebulonMind/memory",
        "/api/NebulonMind/memory/{memory_id}",
        "/api/NebulonMind/search",
        "/api/NebulonMind/memory/context",
        "/api/NebulonMind/intelligence/decide",
        "/api/NebulonMind/intelligence/process",
        "/api/NebulonMind/memory/{memory_id}/relate",
    ):
        assert path in paths, f"missing route {path}"
    # every path documents an operation summary + an envelope response model
    for path, item in paths.items():
        methods = [op for method, op in item.items() if method in ("get", "post", "put", "delete", "patch")]
        assert methods, f"{path} defines no HTTP operation"
        for operation in methods:
            assert operation.get("summary"), f"{path} lacks a summary"


def test_openapi_never_exposes_credentials_in_descriptions(client):
    spec_text = str(client.get("/openapi.json").json())
    for secret in ("NEBULONDB_USERNAME", "NEBULONDB_PASSWORD", "6969", "nmd_user_01"):
        assert secret not in spec_text


# ---------------------------------------------------------------------- #
# Memory CRUD (4.2)                                                      #
# ---------------------------------------------------------------------- #


def test_create_memory_201_envelope(client, provider):
    resp = store(client)
    assert resp.status_code == 201
    body = resp.json()
    assert body["success"] is True
    memory = body["data"]["memory"]
    assert memory["memory_id"].startswith("mem_")
    assert memory["content"]["text"] == "My name is Sathya"
    # memory is pinned to the registered username's resolved opaque user id
    assert memory["user_id"] == provider.registry().resolve("user_001")


def test_create_memory_pins_user_id_from_query(client, provider):
    resp = store(client, user_id="user_api")
    memory = resp.json()["data"]["memory"]
    assert memory["user_id"] == provider.registry().resolve("user_api")
    # body user_id is overridden by the query param's resolved identity
    resp = client.post(
        MEMORY_URL,
        json={"user_id": "other_user", "content": {"text": "stored under query user"}},
        params={"user_id": "user_api"},
    )
    assert resp.json()["data"]["memory"]["user_id"] == provider.registry().resolve("user_api")


def test_memory_default_bypasses_gate_stores_duplicate(client):
    store(client, text="My name is Sathya.")
    # raw POST /memory (no ?gate=true) keeps raw repository semantics: the
    # same fact stored again creates a second memory (documented caveat).
    resp = store(client, text="My name is Sathya.")
    assert resp.status_code == 201
    assert resp.json()["data"]["memory"]["memory_id"].startswith("mem_")


def test_memory_gate_true_dedups_reseed(client):
    first = store(client, text="My name is Sathya.").json()["data"]["memory"]
    duplicate = store(client, text="My name is Sathya.", gate=True)
    assert duplicate.status_code == 200          # idempotent, not an error
    body = duplicate.json()
    assert body["success"] is True
    assert "duplicate" in body["message"]
    assert body["data"]["memory"]["memory_id"] == first["memory_id"]
    results = client.get(
        "/api/NebulonMind/search",
        params={"query": "Sathya", "user_id": "user_001"},
    ).json()["data"]["results"]
    hits = [m for m in results if "Sathya" in m["content"]["text"]]
    assert len(hits) == 1
    assert hits[0]["memory_id"] == first["memory_id"]


def test_memory_gate_true_stamps_temporary_expiry(client):
    resp = store(
        client,
        text="NebulonDB embeds vectors server-side.",
        lifecycle={"retention_policy": "temporary"},
        gate=True,
    )
    assert resp.status_code == 201
    memory = resp.json()["data"]["memory"]
    assert memory["lifecycle"]["retention_policy"] == "temporary"
    assert memory["lifecycle"]["expires_at"] is not None


def test_memory_gate_true_refuses_already_expired(client):
    expired = store(
        client,
        text="This should never be stored.",
        lifecycle={"retention_policy": "temporary", "expires_at": "2000-01-01T00:00:00Z"},
        gate=True,
    )
    assert expired.status_code == 422
    body = expired.json()
    assert body["success"] is False
    assert "expired" in body["message"].lower()


def test_get_memory(client):
    memory_id = store(client).json()["data"]["memory"]["memory_id"]
    resp = client.get(f"{MEMORY_URL}/{memory_id}", params={"user_id": "user_001"})
    assert resp.status_code == 200
    assert resp.json()["data"]["memory"]["memory_id"] == memory_id


def test_get_memory_missing_404_envelope(client):
    resp = client.get(f"{MEMORY_URL}/mem_nope", params={"user_id": "user_001"})
    assert resp.status_code == 404
    body = resp.json()
    assert body["success"] is False
    assert "not found" in body["message"]


def test_get_memory_missing_404_envelope_when_user_has_none(client, provider):
    resp = client.get(f"{MEMORY_URL}/mem_nope", params={"user_id": "user_empty"})
    assert resp.status_code == 404


def test_update_memory_partial(client):
    memory_id = store(client).json()["data"]["memory"]["memory_id"]
    resp = client.put(
        f"{MEMORY_URL}/{memory_id}",
        json={"status": "archived"},
        params={"user_id": "user_001"},
    )
    assert resp.status_code == 200
    memory = resp.json()["data"]["memory"]
    assert memory["status"] == MemoryStatus.ARCHIVED.value
    assert memory["content"]["text"] == "My name is Sathya"  # untouched


def test_update_memory_missing_404(client):
    resp = client.put(
        f"{MEMORY_URL}/mem_nope", json={"status": "archived"},
        params={"user_id": "user_001"},
    )
    assert resp.status_code == 404
    assert resp.json()["success"] is False


def test_update_memory_invalid_payload_422(client):
    memory_id = store(client).json()["data"]["memory"]["memory_id"]
    resp = client.put(
        f"{MEMORY_URL}/{memory_id}",
        json={"importance": {"score": 5.0}},  # out of range 0..1
        params={"user_id": "user_001"},
    )
    assert resp.status_code == 422
    body = resp.json()
    assert body["success"] is False
    assert "errors" in body["data"]
    assert body["data"]["errors"][0]["loc"]

def test_update_memory_unknown_field_422(client):
    memory_id = store(client).json()["data"]["memory"]["memory_id"]
    resp = client.put(
        f"{MEMORY_URL}/{memory_id}",
        json={"bogus": 1},
        params={"user_id": "user_001"},
    )
    assert resp.status_code == 422


def test_delete_memory_then_404(client):
    memory_id = store(client).json()["data"]["memory"]["memory_id"]
    resp = client.delete(f"{MEMORY_URL}/{memory_id}", params={"user_id": "user_001"})
    assert resp.status_code == 200
    assert resp.json()["data"]["memory_id"] == memory_id
    assert client.get(f"{MEMORY_URL}/{memory_id}", params={"user_id": "user_001"}).status_code == 404


def test_delete_memory_missing_404(client):
    resp = client.delete(f"{MEMORY_URL}/mem_nope", params={"user_id": "user_001"})
    assert resp.status_code == 404
    assert resp.json()["success"] is False


# ---------------------------------------------------------------------- #
# Recall API (4.3)                                                       #
# ---------------------------------------------------------------------- #


def test_search_returns_ranked_results(client):
    store(client, text="My name is Sathya")
    store(client, text="I work with Python")
    resp = client.get(
        "/api/NebulonMind/search",
        params={"query": "Sathya", "top_k": 5, "user_id": "user_001"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["query"] == "Sathya"
    texts = [m["content"]["text"] for m in body["data"]["results"]]
    assert "My name is Sathya" in texts
    assert body["data"]["expand"] is False


def test_search_expand_flag_accepted_without_graph(client):
    store(client, text="My name is Sathya")
    resp = client.get(
        "/api/NebulonMind/search",
        params={"query": "Sathya", "expand": "true", "user_id": "user_001"},
    )
    assert resp.status_code == 200
    assert resp.json()["data"]["expand"] is True


def test_search_isolates_users(client):
    store(client, text="My name is Sathya", user_id="user_a")
    resp = client.get(
        "/api/NebulonMind/search",
        params={"query": "Sathya", "user_id": "user_b"},
    )
    assert resp.json()["data"]["results"] == []


def test_search_missing_query_422(client):
    resp = client.get("/api/NebulonMind/search", params={"user_id": "user_001"})
    assert resp.status_code == 422
    assert resp.json()["success"] is False


def test_search_top_k_out_of_range_422(client):
    for bad in (0, 51):
        resp = client.get(
            "/api/NebulonMind/search",
            params={"query": "x", "top_k": bad, "user_id": "user_001"},
        )
        assert resp.status_code == 422, f"top_k={bad} must be rejected"


# ---------------------------------------------------------------------- #
# Context API (4.4)                                                      #
# ---------------------------------------------------------------------- #


def test_context_builds_bounded_provenance_context(client):
    stored = store(client, text="My name is Sathya").json()["data"]["memory"]
    resp = client.post(
        "/api/NebulonMind/memory/context",
        params={
            "query": "Sathya",
            "top_k": 5,
            "max_items": 10,
            "max_characters": 6000,
            "user_id": "user_001",
        },
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert isinstance(data["context"], str)
    assert "Sathya" in data["context"]
    assert data["memory_ids"] == [stored["memory_id"]]


def test_context_respects_bounds(client):
    builder = MemoryContextBuilder()
    repo = InMemoryRepository()
    text = "x" * 400
    for i in range(20):
        repo.create(Memory(user_id="u", content={"text": text + str(i)}))
    context = builder.build(repo.list_all(), max_items=1, max_characters=50)
    assert len(context) <= 50 + 3  # capped block markers
    assert context  # never empty for valid input


def test_context_invalid_bounds_422(client):
    store(client)
    resp = client.post(
        "/api/NebulonMind/memory/context",
        params={"query": "x", "top_k": 5, "max_items": 0, "user_id": "user_001"},
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------- #
# Intelligence API (4.5)                                                 #
# ---------------------------------------------------------------------- #


def test_decide_returns_validated_decisions(client):
    resp = client.post(
        "/api/NebulonMind/intelligence/decide",
        json={"text": "My name is Sathya. I work with Python."},
        params={"user_id": "user_001"},
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert len(data["decisions"]) == 2
    categories = {d["candidate"]["category"] for d in data["decisions"]}
    assert {"identity", "skill"} <= categories
    assert all(d["should_remember"] for d in data["decisions"])


def test_decide_include_rejected(client):
    resp = client.post(
        "/api/NebulonMind/intelligence/decide",
        json={"text": "I should install the app.", "include_rejected": True},
        params={"user_id": "user_001"},
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["decisions"] == []          # TASK rule is below min strength
    assert len(data["rejected"]) == 1
    assert data["rejected"][0]["should_remember"] is False


def test_decide_multi_turn_conversation(client):
    resp = client.post(
        "/api/NebulonMind/intelligence/decide",
        json={
            "conversation": {
                "turns": [
                    {"role": "user", "content": "My name is Sathya"},
                    {"role": "assistant", "content": "Nice to meet you!"},
                    {"role": "user", "content": "I like hiking."},
                ]
            }
        },
        params={"user_id": "user_001"},
    )
    assert resp.status_code == 200
    categories = {d["candidate"]["category"] for d in resp.json()["data"]["decisions"]}
    assert {"identity", "preference"} <= categories


def test_decide_never_persists(client):
    client.post(
        "/api/NebulonMind/intelligence/decide",
        json={"text": "My name is Sathya."},
        params={"user_id": "user_001"},
    )
    results = client.get(
        "/api/NebulonMind/search", params={"query": "Sathya", "user_id": "user_001"}
    ).json()["data"]["results"]
    assert results == []


def test_process_stores_with_lifecycle_gate(client):
    resp = client.post(
        "/api/NebulonMind/intelligence/process",
        json={"text": "My name is Sathya. I like hiking."},
        params={"user_id": "user_001"},
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert len(data["ingestions"]) == 2
    assert all(i["action"] == "STORE" for i in data["ingestions"])
    assert all(i["memory_id"] for i in data["ingestions"])
    # and they really are stored — recall finds them
    resp = client.get(
        "/api/NebulonMind/search", params={"query": "hiking", "user_id": "user_001"}
    )
    assert any("hiking" in m["content"]["text"] for m in resp.json()["data"]["results"])


def test_process_duplicate_detects_existing_memory(client):
    first = client.post(
        "/api/NebulonMind/intelligence/process",
        json={"text": "My name is Sathya."},
        params={"user_id": "user_001"},
    ).json()["data"]["ingestions"][0]
    second = client.post(
        "/api/NebulonMind/intelligence/process",
        json={"text": "My name is Sathya."},
        params={"user_id": "user_001"},
    ).json()["data"]["ingestions"][0]
    assert second["action"] == "DUPLICATE"
    assert second["existing_memory_id"] == first["memory_id"]


def test_process_temporary_categories_get_default_expiry_and_store(client):
    resp = client.post(
        "/api/NebulonMind/intelligence/process",
        json={"text": "According to the docs, NebulonDB embeds vectors server-side."},
        params={"user_id": "user_001"},
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["ingestions"][0]["action"] == "STORE"
    # KNOWLEDGE maps to TEMPORARY retention; the pipeline stamps a default
    # expiry so the Step 3 gate accepts it instead of rejecting it INVALID.
    memory_id = data["ingestions"][0]["memory_id"]
    assert memory_id is not None
    stored = client.get(
        f"/api/NebulonMind/memory/{memory_id}",
        params={"user_id": "user_001"},
    ).json()["data"]["memory"]
    assert stored["lifecycle"]["retention_policy"] == "temporary"
    assert stored["lifecycle"]["expires_at"] is not None


def test_process_persist_false_decides_without_storing(client):
    resp = client.post(
        "/api/NebulonMind/intelligence/process",
        json={"text": "I like hiking.", "persist": False},
        params={"user_id": "user_001"},
    )
    assert resp.status_code == 200
    assert resp.json()["data"]["ingestions"] == []
    results = client.get(
        "/api/NebulonMind/search", params={"query": "hiking", "user_id": "user_001"}
    ).json()["data"]["results"]
    assert results == []


def test_process_and_decide_empty_input_422(client):
    for url in ("/api/NebulonMind/intelligence/decide", "/api/NebulonMind/intelligence/process"):
        resp = client.post(url, json={}, params={"user_id": "user_001"})
        assert resp.status_code == 422
        resp = client.post(url, json={"text": ""}, params={"user_id": "user_001"})
        assert resp.status_code == 422


# ---------------------------------------------------------------------- #
# Relationship API (4.6)                                                 #
# ---------------------------------------------------------------------- #


def test_relate_links_memory_to_entity(client):
    memory_id = store(client).json()["data"]["memory"]["memory_id"]
    resp = client.post(
        f"{MEMORY_URL}/{memory_id}/relate",
        params={"entity": "Python", "relation": "HAS_SKILL", "user_id": "user_001"},
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["entity"] == "Python"
    assert data["relation"] == "HAS_SKILL"
    # entity visible on the memory afterwards
    memory = client.get(f"{MEMORY_URL}/{memory_id}", params={"user_id": "user_001"}).json()["data"]["memory"]
    assert "Python" in memory["entities"]


def test_relate_missing_memory_404(client):
    resp = client.post(
        f"{MEMORY_URL}/mem_nope/relate",
        params={"entity": "Python", "user_id": "user_001"},
    )
    assert resp.status_code == 404
    assert resp.json()["success"] is False


def test_relate_missing_entity_422(client):
    memory_id = store(client).json()["data"]["memory"]["memory_id"]
    resp = client.post(
        f"{MEMORY_URL}/{memory_id}/relate",
        params={"entity": "", "user_id": "user_001"},
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------- #
# Error handling (4.7)                                                   #
# ---------------------------------------------------------------------- #


def test_unknown_route_uses_standard_envelope(client):
    resp = client.get("/api/NebulonMind/not-a-route")
    assert resp.status_code == 404
    body = resp.json()
    assert body["success"] is False
    assert isinstance(body["message"], str)


def test_unknown_route_404_does_not_leak_exception(client):
    body = client.get("/api/NebulonMind/not-a-route").json()
    assert "exception" not in body["message"].lower()
    assert "traceback" not in str(body).lower()


class _BrokenRepository:
    """Minimal repository-compatible object whose reads blow up."""

    def __init__(self):
        from nmd_host.core.repository import InMemoryRepository

        self._inner = InMemoryRepository()

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def get(self, memory_id):
        raise RuntimeError("boom: internal connection string secret")


def test_unhandled_errors_become_generic_500():
    from nmd_host.api.service import ServiceBundle

    provider = InMemoryServiceProvider()
    provider.create_user("user_001")
    bundle = provider.bundle("user_001")
    broken = _BrokenRepository()
    provider._bundles["user_001"] = ServiceBundle(
        user_id="user_001",
        repository=broken,
        manager=bundle.manager,
    )
    app = create_app(provider=provider)
    # raise_server_exceptions=False: we are testing the error *envelope*
    # returned for an unhandled error, not the re-raised exception.
    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.get(f"{MEMORY_URL}/mem_any", params={"user_id": "user_001"})
    assert resp.status_code == 500
    body = resp.json()
    assert body["success"] is False
    assert body["message"] == "internal server error"
    assert "boom" not in str(body)  # internal exception never leaked


# ---------------------------------------------------------------------- #
# User isolation across endpoints                                        #
# ---------------------------------------------------------------------- #


def test_per_user_isolation_everywhere(client, provider):
    store(client, text="My name is Sathya", user_id="user_a")
    other = client.get(
        "/api/NebulonMind/search", params={"query": "Sathya", "user_id": "user_b"}
    ).json()["data"]["results"]
    assert other == []
    missing = client.get(f"{MEMORY_URL}/mem_x", params={"user_id": "user_b"})
    assert missing.status_code == 404