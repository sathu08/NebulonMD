"""Step 6.9 — Agent session management tests (fully offline).

Covers the ``AgentSessionManager`` (in-memory, bounded, per-user isolated,
TTL-expired) and the session API surface:
``POST /agent/session``, ``GET /agent/session/{id}``,
``GET /agent/sessions``, ``DELETE /agent/session/{id}``, plus the
multi-turn continuity of ``POST /agent/chat`` with ``session_id``.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from nmd_host.agent import AgentSessionManager
from nmd_host.agent.schemas import AgentMessage
from nmd_host.api.server import create_app
from nmd_host.api.service import InMemoryServiceProvider
from nmd_host.core.config import ServiceConfig
from nmd_host.intelligence.providers import LLMResponse


# ---------------------------------------------------------------------- #
# AgentSessionManager (unit)                                             #
# ---------------------------------------------------------------------- #


def test_create_returns_active_session():
    manager = AgentSessionManager()
    session = manager.create("user_001", metadata={"app": "test"})
    assert session.session_id
    assert session.user_id == "user_001"
    assert session.status == "active"
    assert session.metadata == {"app": "test"}
    assert session.created_at > 0
    assert session.last_activity >= session.created_at


def test_get_roundtrips_and_touch_updates_activity():
    manager = AgentSessionManager()
    session = manager.create("user_001")
    time.sleep(0.001)
    assert manager.touch("user_001", session.session_id) is True
    fetched = manager.get("user_001", session.session_id)
    assert fetched is not None
    assert fetched.last_activity > session.created_at


def test_get_unknown_returns_none():
    manager = AgentSessionManager()
    assert manager.get("user_001", "nope") is None


def test_close_removes_session():
    manager = AgentSessionManager()
    session = manager.create("user_001")
    assert manager.close("user_001", session.session_id) is True
    assert manager.get("user_001", session.session_id) is None
    assert manager.close("user_001", session.session_id) is False


def test_sessions_are_isolated_per_user():
    manager = AgentSessionManager()
    s1 = manager.create("user_001")
    s2 = manager.create("user_002")
    assert manager.get("user_001", s2.session_id) is None
    assert manager.get("user_002", s1.session_id) is None
    assert manager.close("user_001", s2.session_id) is False


def test_max_sessions_per_user_is_bounded():
    manager = AgentSessionManager(max_sessions_per_user=2)
    manager.create("user_001")
    manager.create("user_001")
    with pytest.raises(ValueError):
        manager.create("user_001")
    # a different user is unaffected
    manager.create("user_002")


def test_duplicate_session_id_rejected():
    manager = AgentSessionManager()
    session = manager.create("user_001")
    with pytest.raises(ValueError):
        manager.create("user_001", session_id=session.session_id)


def test_ttl_expires_sessions():
    manager = AgentSessionManager(session_ttl_seconds=0.05)
    session = manager.create("user_001")
    assert manager.get("user_001", session.session_id) is not None
    time.sleep(0.08)
    assert manager.get("user_001", session.session_id) is None
    assert manager.active_count("user_001") == 0


def test_list_active_returns_only_live_sessions():
    manager = AgentSessionManager()
    s1 = manager.create("user_001")
    s2 = manager.create("user_001")
    manager.create("user_002")
    ids = {s.session_id for s in manager.list_active("user_001")}
    assert ids == {s1.session_id, s2.session_id}


def test_append_messages_updates_transcript():
    manager = AgentSessionManager()
    session = manager.create("user_001")
    assert manager.append_messages(
        "user_001",
        session.session_id,
        [AgentMessage(role="user", content="hi"), AgentMessage(role="assistant", content="yo")],
    ) is True
    fetched = manager.get("user_001", session.session_id)
    assert fetched is not None
    assert len(fetched.transcript) == 2
    assert fetched.transcript[0].content == "hi"


# ---------------------------------------------------------------------- #
# Session API                                                            #
# ---------------------------------------------------------------------- #


def _app():
    provider = InMemoryServiceProvider()
    for name in ("user_001", "user_002"):
        provider.create_user(name)
    return create_app(provider=provider, config=ServiceConfig()), provider


def _resolved(app, username):
    return app.state.provider.registry().resolve(username)


def test_create_get_delete_session_endpoints():
    app, _ = _app()
    with TestClient(app) as client:
        resp = client.post(
            "/api/NebulonMind/agent/session",
            json={"metadata": {"app": "cli"}},
            params={"user_id": "user_001"},
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["success"] is True
        session_id = body["data"]["session_id"]
        assert body["data"]["user_id"] == _resolved(app, "user_001")
        assert body["data"]["status"] == "active"
        assert body["data"]["metadata"] == {"app": "cli"}

        got = client.get(
            f"/api/NebulonMind/agent/session/{session_id}",
            params={"user_id": "user_001"},
        )
        assert got.status_code == 200
        assert got.json()["data"]["session_id"] == session_id

        deleted = client.delete(
            f"/api/NebulonMind/agent/session/{session_id}",
            params={"user_id": "user_001"},
        )
        assert deleted.status_code == 200
        assert deleted.json()["data"]["memory_id"] == session_id

        gone = client.get(
            f"/api/NebulonMind/agent/session/{session_id}",
            params={"user_id": "user_001"},
        )
        assert gone.status_code == 404


def test_list_sessions_endpoint_isolates_users():
    app, _ = _app()
    with TestClient(app) as client:
        client.post("/api/NebulonMind/agent/session", params={"user_id": "user_001"})
        client.post("/api/NebulonMind/agent/session", params={"user_id": "user_001"})
        client.post("/api/NebulonMind/agent/session", params={"user_id": "user_002"})
        resp = client.get("/api/NebulonMind/agent/sessions", params={"user_id": "user_001"})
        assert resp.status_code == 200
        sessions = resp.json()["data"]["sessions"]
        assert len(sessions) == 2
        assert all(s["user_id"] == _resolved(app, "user_001") for s in sessions)


def test_session_404_for_other_user():
    app, _ = _app()
    with TestClient(app) as client:
        created = client.post(
            "/api/NebulonMind/agent/session", params={"user_id": "user_001"}
        ).json()["data"]
        resp = client.get(
            f"/api/NebulonMind/agent/session/{created['session_id']}",
            params={"user_id": "user_002"},
        )
        assert resp.status_code == 404


def test_create_session_capacity_400():
    provider = InMemoryServiceProvider()
    provider.create_user("user_001")
    app = create_app(provider=provider, config=ServiceConfig())
    app.state.sessions._max_sessions_per_user = 1
    with TestClient(app) as tiny:
        tiny.post("/api/NebulonMind/agent/session", params={"user_id": "user_001"})
        resp = tiny.post("/api/NebulonMind/agent/session", params={"user_id": "user_001"})
        assert resp.status_code == 400
        assert resp.json()["success"] is False


class _ScriptedLLM:
    """LLMProvider returning a scripted list of replies."""

    def __init__(self, replies, capture=None):
        self._replies = list(replies)
        self.capture = capture or []

    def complete(self, prompt, *, system=None, temperature=0.0):
        self.capture.append(prompt)
        if self._replies:
            return LLMResponse(text=self._replies.pop(0))
        return LLMResponse(text="I have nothing else to say.")


def test_chat_with_session_persists_transcript_and_continues(monkeypatch):
    fake = _ScriptedLLM(
        ["First turn answer.", "Second turn answer."]
    )
    monkeypatch.setattr(
        "nmd_host.intelligence.providers.provider_from_env", lambda: fake
    )
    with TestClient(_app()[0]) as client:
        session_id = client.post(
            "/api/NebulonMind/agent/session", params={"user_id": "user_001"}
        ).json()["data"]["session_id"]

        first = client.post(
            "/api/NebulonMind/agent/chat",
            json={"text": "hello", "session_id": session_id},
            params={"user_id": "user_001"},
        )
        assert first.status_code == 200
        assert first.json()["data"]["answer"] == "First turn answer."

        # the session now holds the transcript
        got = client.get(
            f"/api/NebulonMind/agent/session/{session_id}",
            params={"user_id": "user_001"},
        )
        assert got.json()["data"]["message_count"] >= 2  # user + assistant

        second = client.post(
            "/api/NebulonMind/agent/chat",
            json={"text": "hi again", "session_id": session_id},
            params={"user_id": "user_001"},
        )
        assert second.status_code == 200
        assert second.json()["data"]["answer"] == "Second turn answer."
        # the second prompt included the first turn's history
        prompt = fake.capture[-1]
        assert "hello" in prompt
        assert "First turn answer." in prompt


def test_chat_with_unknown_session_is_stateless(monkeypatch):
    fake = _ScriptedLLM(["lonely answer."])
    monkeypatch.setattr(
        "nmd_host.intelligence.providers.provider_from_env", lambda: fake
    )
    with TestClient(_app()[0]) as client:
        resp = client.post(
            "/api/NebulonMind/agent/chat",
            json={"text": "hello", "session_id": "does-not-exist"},
            params={"user_id": "user_001"},
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["answer"] == "lonely answer."


# ---------------------------------------------------------------------- #
# Durable sessions (optional SessionStore re-hydration)                  #
# ---------------------------------------------------------------------- #


def test_durable_session_rehydrates_after_restart():
    from nmd_host.agent.session_store import InMemorySessionStore

    store = InMemorySessionStore()
    first = AgentSessionManager(store=store)
    session = first.create("user_001")
    session_id = session.session_id
    first.append_messages(
        "user_001",
        session_id,
        [
            AgentMessage(role="user", content="hello"),
            AgentMessage(role="assistant", content="hi there!"),
        ],
    )

    # simulate a service restart: new manager, same persistent store
    restarted = AgentSessionManager(store=store)
    got = restarted.get("user_001", session_id)
    assert got is not None
    assert [m.content for m in got.transcript] == ["hello", "hi there!"]

    # durable appends keep persisting across further restarts
    restarted.append_messages(
        "user_001", session_id, [AgentMessage(role="user", content="next")]
    )
    again = AgentSessionManager(store=store).get("user_001", session_id)
    assert [m.content for m in again.transcript] == ["hello", "hi there!", "next"]


def test_durable_session_close_removes_from_store():
    from nmd_host.agent.session_store import InMemorySessionStore

    store = InMemorySessionStore()
    manager = AgentSessionManager(store=store)
    session = manager.create("user_001")
    assert manager.close("user_001", session.session_id) is True
    assert store.get("user_001", session.session_id) is None
    # a fresh manager cannot re-hydrate a closed session
    assert AgentSessionManager(store=store).get("user_001", session.session_id) is None


def test_sessions_stay_process_local_without_a_store():
    manager = AgentSessionManager()
    session = manager.create("user_001")
    assert manager.get("user_001", session.session_id) is not None
    # without a SessionStore a fresh manager knows nothing about it
    assert AgentSessionManager().get("user_001", session.session_id) is None
