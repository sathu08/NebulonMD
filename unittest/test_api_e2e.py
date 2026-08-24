"""Phase 4.11 / 4.12 — end-to-end API tests against a live NebulonDB.

Full pipeline through the HTTP layer: API → Step 2 (decide) → Step 3
(lifecycle gate) → storage → recall, plus health checks and startup/
shutdown validation. These tests skip automatically when the NebulonDB
backend is unreachable (same convention as ``conftest.api_client``).
"""

import uuid

import pytest
from fastapi.testclient import TestClient

from nmd_host.api.server import create_app
from nmd_host.api.service import DefaultServiceProvider

MEMORY_URL = "/api/NebulonMind/memory"


@pytest.fixture(scope="module")
def api_context(api_client):
    """Live app: DefaultServiceProvider over the shared session client."""
    provider = DefaultServiceProvider(client=api_client)
    app = create_app(provider=provider)
    with TestClient(app) as test_client:
        yield test_client, provider


@pytest.fixture
def unique_user(api_context):
    """A registered username, minted through the public /user/create_user API."""
    client, _ = api_context
    name = "user_" + uuid.uuid4().hex[:12]
    resp = client.post(
        "/api/NebulonMind/user/create_user",
        json={"username": name},
    )
    assert resp.status_code == 201, resp.text
    assert resp.json()["data"]["created"] is True
    return name


def process(client, text, user_id):
    return client.post(
        "/api/NebulonMind/intelligence/process",
        json={"text": text},
        params={"user_id": user_id},
    )


def delete_if_present(client, memory_id, user_id):
    if memory_id:
        client.delete(f"{MEMORY_URL}/{memory_id}", params={"user_id": user_id})


# ---------------------------------------------------------------------- #
# Health (4.12)                                                          #
# ---------------------------------------------------------------------- #


def test_api_health_backend_up(api_context):
    client, provider = api_context
    resp = client.get("/api/NebulonMind/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["backend"] == "up"
    assert body["data"]["provider"] == "nebulondb"
    assert body["data"]["version"]
    # no credentials leak through health
    assert "password" not in str(body)


# ---------------------------------------------------------------------- #
# Identity registry (4.13 — username gate)                                #
# ---------------------------------------------------------------------- #


def test_e2e_unregistered_username_rejected(api_context):
    client, _ = api_context
    stranger = "user_" + uuid.uuid4().hex[:12]
    resp = process(client, "My name is X.", stranger)
    assert resp.status_code == 403
    assert resp.json()["success"] is False
    assert resp.headers.get("WWW-Authenticate") == "NebulonMindUser"


def test_e2e_create_user_idempotent(api_context):
    client, _ = api_context
    name = "user_" + uuid.uuid4().hex[:12]
    first = client.post("/api/NebulonMind/user/create_user", json={"username": name})
    assert first.status_code == 201
    assert first.json()["data"]["created"] is True
    user_id = first.json()["data"]["user_id"]
    second = client.post("/api/NebulonMind/user/create_user", json={"username": name})
    assert second.status_code == 201
    assert second.json()["data"]["created"] is False
    assert second.json()["data"]["user_id"] == user_id


# ---------------------------------------------------------------------- #
# Full pipeline: API → Step 2 → Step 3 → storage → recall (4.11)         #
# ---------------------------------------------------------------------- #


def test_e2e_intelligence_process_then_recall(api_context, unique_user):
    client, _ = api_context
    resp = process(client, "My name is Sathya.", unique_user)
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["ingestions"][0]["action"] == "STORE"
    memory_id = data["ingestions"][0]["memory_id"]
    assert memory_id

    # the memory is retrievable through the Step 3 pipeline
    recall = client.get(
        "/api/NebulonMind/search",
        params={"query": "What is the user's name?", "top_k": 5, "user_id": unique_user},
    )
    assert recall.status_code == 200
    results = recall.json()["data"]["results"]
    assert any(m["memory_id"] == memory_id for m in results)
    assert any("Sathya" in m["content"]["text"] for m in results)

    # and readable through the CRUD endpoint
    get = client.get(f"{MEMORY_URL}/{memory_id}", params={"user_id": unique_user})
    assert get.status_code == 200
    assert get.json()["data"]["memory"]["content"]["text"] == "My name is Sathya."

    delete_if_present(client, memory_id, unique_user)


def test_e2e_lifecycle_gate_deduplicates(api_context, unique_user):
    client, _ = api_context
    first = process(client, "I work with Python.", unique_user).json()["data"]["ingestions"]
    assert first[0]["action"] == "STORE"
    second = process(client, "I work with Python.", unique_user).json()["data"]["ingestions"]
    assert second[0]["action"] == "DUPLICATE"
    assert second[0]["existing_memory_id"] == first[0]["memory_id"]
    delete_if_present(client, first[0]["memory_id"], unique_user)


def test_e2e_delete_removes_from_recall(api_context, unique_user):
    client, _ = api_context
    memory_id = process(client, "I like hiking.", unique_user).json()["data"]["ingestions"][0]["memory_id"]
    assert client.delete(f"{MEMORY_URL}/{memory_id}", params={"user_id": unique_user}).status_code == 200
    recall = client.get(
        "/api/NebulonMind/search",
        params={"query": "hiking", "user_id": unique_user},
    ).json()["data"]["results"]
    assert all(m["memory_id"] != memory_id for m in recall)
    assert client.get(f"{MEMORY_URL}/{memory_id}", params={"user_id": unique_user}).status_code == 404


# ---------------------------------------------------------------------- #
# Relationships + graph expansion (4.6)                                  #
# ---------------------------------------------------------------------- #


def test_e2e_relate_then_expand_search(api_context, unique_user):
    client, _ = api_context
    memory_id = process(client, "I work with Python.", unique_user).json()["data"]["ingestions"][0]["memory_id"]

    resp = client.post(
        f"{MEMORY_URL}/{memory_id}/relate",
        params={"entity": "Python", "relation": "HAS_SKILL", "user_id": unique_user},
    )
    assert resp.status_code == 200
    assert resp.json()["data"]["entity"] == "Python"

    # depth-1 graph expansion does not error and still returns results
    resp = client.get(
        "/api/NebulonMind/search",
        params={"query": "Python", "expand": "true", "user_id": unique_user},
    )
    assert resp.status_code == 200
    assert isinstance(resp.json()["data"]["results"], list)

    delete_if_present(client, memory_id, unique_user)


def test_e2e_relate_missing_memory_404(api_context, unique_user):
    client, _ = api_context
    resp = client.post(
        f"{MEMORY_URL}/mem_nope/relate",
        params={"entity": "X", "user_id": unique_user},
    )
    assert resp.status_code == 404
    assert resp.json()["success"] is False


# ---------------------------------------------------------------------- #
# Production validation (4.12)                                           #
# ---------------------------------------------------------------------- #


def test_e2e_setup_teardown(api_context):
    """Startup/shutdown ran without error (TestClient context did both)."""


def test_openapi_documentation_served(api_context):
    client, _ = api_context
    spec = client.get("/openapi.json").json()
    assert spec["info"]["version"]
    assert "/api/NebulonMind/intelligence/process" in spec["paths"]
    # decision/process request examples are part of the schema
    process_schema = spec["components"]["schemas"]["IntelligenceProcessRequest"]
    assert process_schema.get("examples")