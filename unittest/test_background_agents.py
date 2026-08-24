"""Step 14 — Background Agents tests (fully offline).

The agents run over ``InMemoryServiceProvider`` (real repository + real
Step 3 lifecycle manager), so decisions come from the real
``MemoryConsolidator``; the scheduler cron logic and lifecycle are tested
directly, and the API endpoints are covered via ``TestClient`` without a
real clock (manual triggers).
"""

import time

import pytest
from fastapi.testclient import TestClient

from nmd_host.agents import BackgroundScheduler, CronSpec, MemoryAgent, TaskAgent
from nmd_host.api.server import create_app
from nmd_host.api.service import InMemoryServiceProvider
from nmd_host.core.config import ServiceConfig
from nmd_host.core.models import (
    Classification,
    Importance,
    Lifecycle,
    Memory,
    MemoryContent,
    MemoryStatus,
    MemoryType,
    Provenance,
    RetentionPolicy,
)


def _memory(text, category="work", by="user", at=None, retention=RetentionPolicy.PERMANENT):
    return Memory(
        user_id="user_001",
        content=MemoryContent(text=text),
        classification=Classification(
            memory_type=MemoryType.LONG_TERM,
            category=category,
            source="user",
        ),
        provenance=Provenance(
            source_type="conversation", created_by=by, created_at=at or time.time()
        ),
        importance=Importance(score=0.7, confidence=0.9, priority="medium"),
        lifecycle=Lifecycle(
            retention_policy=retention,
            created_at=at or time.time(),
            updated_at=at or time.time(),
        ),
        status=MemoryStatus.ACTIVE,
    )


def _provider(username):
    provider = InMemoryServiceProvider()
    provider.create_user(username)
    return provider


# ---------------------------------------------------------------------- #
# Cron / scheduler                                                       #
# ---------------------------------------------------------------------- #


def test_cron_parse_and_match():
    nightly = CronSpec.parse("0 2 * * *")
    assert nightly.matches(time.localtime()) in (True, False)
    assert nightly.matches(time.struct_time((2026, 1, 1, 2, 0, 0, 3, 1, 0)))
    assert not nightly.matches(time.struct_time((2026, 1, 1, 3, 0, 0, 3, 1, 0)))


def test_cron_supports_lists_and_steps():
    every_5 = CronSpec.parse("*/5 * * * *")
    assert 0 in every_5.minute and 55 in every_5.minute and 57 not in every_5.minute
    spec = CronSpec.parse("0 9,17 * * 1-5")
    assert spec.hour == {9, 17}
    assert spec.dow == {1, 2, 3, 4, 5}


def test_scheduler_registers_and_runs_now():
    scheduler = BackgroundScheduler()
    calls = []
    scheduler.register("memory", lambda: calls.append(1) or {"agent": "memory"}, "0 2 * * *")
    assert scheduler.snapshot()["jobs"][0]["schedule"] == "0 2 * * *"
    result = scheduler.run_now("memory")
    assert result == {"agent": "memory"}
    assert calls == [1]
    status = scheduler.snapshot()["jobs"][0]
    assert status["runs"] == 1
    assert status["last_run"] is not None
    assert status["last_error"] is None


def test_scheduler_job_due_once_per_minute():
    job = BackgroundScheduler().register  # noqa: F841 - just exercising import
    from nmd_host.agents.scheduler import BackgroundJob, CronSpec as CS

    bj = BackgroundJob(
        job_id="j", fn=lambda: {}, cron=CS.parse("* * * * *")
    )
    now = time.localtime()
    assert bj.due(now) is True
    bj.mark_done(now, {"ok": 1}, None)
    assert bj.due(now) is False  # already ran this minute


def test_scheduler_start_stop_lifecycle():
    import asyncio

    scheduler = BackgroundScheduler()
    scheduler.register("memory", lambda: {"agent": "memory"}, "0 2 * * *")

    async def drive():
        await scheduler.start()
        assert scheduler.snapshot()["running"] is True
        await scheduler.stop()
        assert scheduler.snapshot()["running"] is False

    asyncio.run(drive())


# ---------------------------------------------------------------------- #
# Memory Agent — real consolidator over the real lifecycle manager       #
# ---------------------------------------------------------------------- #


def test_memory_agent_review_reports_decisions():
    provider = _provider("user_001")
    bundle = provider.bundle("user_001")
    for text in ("I am learning Python.", "I started learning Python."):
        bundle.repository.create(
            _memory(text, category="learning", by="user")
        )
    agent = MemoryAgent(bundle.repository, bundle.manager, user_id="user_001")
    report = agent.run(mode="review")
    assert report["agent"] == "memory"
    assert report["memories_reviewed"] >= 2
    # singletons missing similarity links -> KEEP; nothing deleted ever
    assert report["deleted_memory_ids"] == []
    assert report["created_memory_ids"] == []
    actions = report["by_action"]
    assert all(k in ("KEEP", "UPDATE", "MERGE", "IGNORE") for k in actions)


def test_memory_agent_consolidate_never_deletes_and_keeps_sources():
    provider = _provider("user_001")
    bundle = provider.bundle("user_001")
    ids = [
        bundle.repository.create(m).memory_id
        for m in (
            # one is a token-subset of the other → consolidator MERGE
            _memory("I am learning Python.", category="learning"),
            _memory("I am learning Python and I practice daily.", category="learning"),
        )
    ]
    agent = MemoryAgent(bundle.repository, bundle.manager, user_id="user_001")
    report = agent.run(mode="consolidate")
    # sources remain readable (agent never deletes)
    for mid in ids:
        assert bundle.repository.get(mid) is not None
    assert report["created_memory_ids"], "expected a merged memory to be created"
    merged = bundle.repository.get(report["created_memory_ids"][0])
    assert merged is not None
    assert merged.provenance.created_by == "memory_agent"


def test_memory_agent_empty_user():
    provider = _provider("nobody")
    bundle = provider.bundle("nobody")
    report = MemoryAgent(
        bundle.repository, bundle.manager, user_id="nobody"
    ).run(mode="consolidate")
    assert report["memories_reviewed"] == 0
    assert report["created_memory_ids"] == []


# ---------------------------------------------------------------------- #
# Task Agent — weekly summary over the real repository                   #
# ---------------------------------------------------------------------- #


def test_task_agent_groups_and_stores_summary():
    provider = _provider("user_001")
    bundle = provider.bundle("user_001")
    recent = time.time() - 100
    old = time.time() - 86400 * 20
    for text in ("Shipped the recall API.", "Fixed the ranking bug."):
        bundle.repository.create(_memory(text, category="work", at=recent))
    bundle.repository.create(
        _memory("Old fact from three weeks ago.", category="old", at=old)
    )
    agent = TaskAgent(bundle.repository, bundle.manager, user_id="user_001", days=7)
    report = agent.run()
    assert report["agent"] == "task"
    assert report["period_days"] == 7
    assert report["memories_scanned"] == 3
    assert report["memories_covered"] == 2
    assert report["groups"].get("work") == 2
    assert "Shipped the recall API." in report["summary"]
    assert report["stored_memory_id"] is not None
    stored = bundle.repository.get(report["stored_memory_id"])
    assert stored.classification.category == "summary"
    assert stored.provenance.created_by == "task_agent"


def test_task_agent_empty_window():
    provider = _provider("user_001")
    bundle = provider.bundle("user_001")
    bundle.repository.create(_memory("an old memory", at=time.time() - 86400 * 30))
    report = TaskAgent(
        bundle.repository, bundle.manager, user_id="user_001", days=7
    ).run()
    assert report["memories_covered"] == 0
    assert report["stored_memory_id"] is None  # nothing to summarize


def test_task_agent_optional_llm_rewrite():
    provider = _provider("user_001")
    bundle = provider.bundle("user_001")
    bundle.repository.create(
        _memory("Worked on the dashboard.", category="work", at=time.time() - 50)
    )
    calls = []

    class FakeLLM:
        def complete(self, prompt, *, system=None, temperature=0.0):
            calls.append(True)
            from nmd_host.intelligence.providers import LLMResponse

            return LLMResponse(text="[LLM] Dashboard work happened.")

    agent = TaskAgent(
        bundle.repository,
        bundle.manager,
        user_id="user_001",
        llm=FakeLLM(),
    )
    report = agent.run()
    assert calls == [True]
    assert report["summary"] == "[LLM] Dashboard work happened."


# ---------------------------------------------------------------------- #
# API                                                                     #
# ---------------------------------------------------------------------- #


@pytest.fixture
def app(monkeypatch):
    provider = InMemoryServiceProvider()
    provider.create_user("user_001")
    app = create_app(provider=provider, config=ServiceConfig())
    with TestClient(app) as client:
        yield client


def test_background_memory_run_endpoint(app):
    resp = app.post(
        "/api/NebulonMind/background/memory/run",
        json={"mode": "consolidate"},
        params={"user_id": "user_001"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["agent"] == "memory"


def test_background_task_run_endpoint(app):
    resp = app.post(
        "/api/NebulonMind/background/task/run",
        json={"days": 7},
        params={"user_id": "user_001"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["agent"] == "task"


def test_background_status_endpoint(app):
    resp = app.get("/api/NebulonMind/background/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert "jobs" in body["data"]
    job_ids = {j["job_id"] for j in body["data"]["jobs"]}
    assert "memory" in job_ids and "task_weekly_summary" in job_ids


def test_console_background_assets_served(app):
    index = app.get("/api/NebulonMind/dashboard/")
    assert index.status_code == 200
    html = index.text
    assert "NebulonMind" in html
    assert "terminal-container" in html
    resp = app.get("/api/NebulonMind/dashboard/js/main.js")
    assert resp.status_code == 200
    assert "background/status" in resp.text.lower()


# ---------------------------------------------------------------------- #
# Durable job state (optional JobStateStore)                             #
# ---------------------------------------------------------------------- #


def test_job_state_store_roundtrip():
    from nmd_host.agents.state_store import InMemoryJobStateStore

    store = InMemoryJobStateStore()
    store.save(
        "memory",
        {"last_run": 1.0, "runs": 2, "last_error": None, "last_result": {"x": 1}},
    )
    state = store.load().get("memory")
    assert state is not None
    assert state["runs"] == 2
    assert state["last_run"] == 1.0
    assert store.load().get("nonexistent") is None


def test_job_state_restores_after_restart():
    from nmd_host.agents.state_store import InMemoryJobStateStore

    def job_fn():
        return {"summary": "weekly report"}

    store = InMemoryJobStateStore()
    first = BackgroundScheduler(store=store)
    first.register("task_weekly_summary", job_fn, "* * * * *")
    result = first.run_now("task_weekly_summary")
    assert result == {"summary": "weekly report"}

    # simulate a restart: new scheduler object, same persistent store
    restarted = BackgroundScheduler(store=store)
    restarted.register("task_weekly_summary", job_fn, "* * * * *")
    restarted._hydrate_state()
    snap = {j["job_id"]: j for j in restarted.snapshot()["jobs"]}[
        "task_weekly_summary"
    ]
    assert snap["runs"] == 1
    assert snap["last_run"] is not None
    assert snap["last_result"] == {"summary": "weekly report"}


def test_job_state_absent_without_store():
    scheduler = BackgroundScheduler()
    scheduler.register("memory", lambda: {"summary": "x"}, "* * * * *")
    scheduler.run_now("memory")
    snap = scheduler.snapshot()["jobs"][0]
    assert snap["runs"] == 1
    # a store-less scheduler has nothing to rehydrate on a restart
    fresh = BackgroundScheduler()
    fresh.register("memory", lambda: {"summary": "x"}, "* * * * *")
    fresh._hydrate_state()
    assert fresh.snapshot()["jobs"][0]["runs"] == 0