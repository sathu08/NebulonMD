"""Per-user chat transcript API tests (fully offline)."""

import pytest
from fastapi.testclient import TestClient

from nmd_host.api.server import create_app
from nmd_host.api.service import InMemoryServiceProvider
from nmd_host.core.config import ServiceConfig


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("NEBULONMD_HOME", str(tmp_path))
    provider = InMemoryServiceProvider()
    provider.create_user("user_001")
    app = create_app(provider=provider, config=ServiceConfig())
    with TestClient(app) as c:
        yield c


def _chat(_id="c1", title="Hello", messages=None):
    return {
        "id": _id,
        "title": title,
        "messages": messages or [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
    }


def test_save_and_list_chats(client):
    resp = client.post(
        "/api/NebulonMind/chats",
        json={"user_id": "user_001", "chat": _chat()},
    )
    assert resp.status_code == 201
    assert resp.json()["success"] is True

    resp = client.get("/api/NebulonMind/chats", params={"user_id": "user_001"})
    assert resp.status_code == 200
    chats = resp.json()["data"]["chats"]
    assert len(chats) == 1
    assert chats[0]["id"] == "c1"
    assert chats[0]["title"] == "Hello"
    assert chats[0]["messages"][0]["content"] == "hi"


def test_history_is_per_user(client):
    client.post("/api/NebulonMind/chats", json={"user_id": "user_001", "chat": _chat()})
    other = client.get("/api/NebulonMind/chats", params={"user_id": "someone_else"})
    assert other.status_code == 200
    assert other.json()["data"]["chats"] == []


def test_get_and_delete_chat(client):
    client.post("/api/NebulonMind/chats", json={"user_id": "user_001", "chat": _chat()})

    resp = client.get("/api/NebulonMind/chats/c1", params={"user_id": "user_001"})
    assert resp.status_code == 200
    assert resp.json()["data"]["chat"]["id"] == "c1"

    resp = client.get("/api/NebulonMind/chats/missing", params={"user_id": "user_001"})
    assert resp.status_code == 404

    resp = client.delete("/api/NebulonMind/chats/c1", params={"user_id": "user_001"})
    assert resp.status_code == 200
    assert resp.json()["data"]["deleted"] is True
    assert client.get("/api/NebulonMind/chats", params={"user_id": "user_001"}).json()[
        "data"
    ]["chats"] == []


def test_upsert_overwrites_chat(client):
    client.post(
        "/api/NebulonMind/chats",
        json={"user_id": "user_001", "chat": _chat(messages=[{"role": "user", "content": "v1"}])},
    )
    client.post(
        "/api/NebulonMind/chats",
        json={"user_id": "user_001", "chat": _chat(messages=[{"role": "user", "content": "v2"}])},
    )
    chats = client.get("/api/NebulonMind/chats", params={"user_id": "user_001"}).json()[
        "data"
    ]["chats"]
    assert len(chats) == 1
    assert chats[0]["messages"][0]["content"] == "v2"


def test_invalid_user_id_rejected(client):
    for path in ("/api/NebulonMind/chats", "/api/NebulonMind/chats/c1"):
        resp = client.get(path, params={"user_id": "/create"})
        assert resp.status_code == 400


def test_transcript_to_chat_keeps_user_assistant_only():
    from nmd_host.agent.schemas import AgentMessage
    from nmd_host.stores.chat_history_store import transcript_to_chat

    chat = transcript_to_chat(
        "sess_1",
        "user_001",
        [
            AgentMessage(role="user", content="My name is Sathya, I work with Python"),
            AgentMessage(role="tool", content="remembered 1 memory"),
            AgentMessage(role="assistant", content="Noted!"),
        ],
    )
    assert chat["id"] == "sess_1"
    assert chat["title"].startswith("My name is Sathya")
    assert [m["role"] for m in chat["messages"]] == ["user", "assistant"]


def test_history_store_for_unknown_user_is_lenient():
    from nmd_host.stores.chat_history_store import InMemoryChatHistoryStore, history_store_for

    provider = InMemoryServiceProvider()
    store = history_store_for(provider, "ghost_user")
    assert isinstance(store, InMemoryChatHistoryStore)
    record = store.save({"id": "g1", "messages": [{"role": "user", "content": "hi"}]})
    assert record["id"] == "g1"
    # same provider + name resolves to the same store (upsert, not duplicate)
    assert history_store_for(provider, "ghost_user").list() == [record]