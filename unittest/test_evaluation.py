"""Step 13 — Agent Evaluation tests (fully offline).

Metrics are pure functions tested directly; the runner is exercised over an
``InMemoryServiceProvider`` with a scripted ``LLMProvider``; the API
endpoint is covered end-to-end via ``TestClient``.
"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from nmd_host.api.server import create_app
from nmd_host.api.service import InMemoryServiceProvider
from nmd_host.core.config import ServiceConfig
from nmd_host.evaluation.metrics import (
    answer_correctness,
    hallucination_rate,
    retrieval_accuracy,
    summarize,
    tool_selection_accuracy,
)
from nmd_host.evaluation.runner import (
    DATASET_PATH,
    EvaluationRunner,
    load_dataset,
    save_report,
)
from nmd_host.intelligence.providers import LLMResponse


# ---------------------------------------------------------------------- #
# metrics                                                                #
# ---------------------------------------------------------------------- #


def test_metrics_are_zero_on_empty():
    assert summarize([])["questions"] == 0
    assert tool_selection_accuracy([]) == 0.0
    assert retrieval_accuracy([]) == 0.0


def test_tool_selection_accuracy():
    results = [
        {"tool_ok": True}, {"tool_ok": True}, {"tool_ok": False},
    ]
    assert tool_selection_accuracy(results) == pytest.approx(2 / 3, abs=1e-3)


def test_retrieval_accuracy_counts_only_known_recall_items():
    results = [
        {"type": "known", "expected_tool": "recall", "retrieval_ok": True},
        {"type": "known", "expected_tool": "recall", "retrieval_ok": False},
        {"type": "known", "expected_tool": "none", "retrieval_ok": None},
        {"type": "unknown", "expected_tool": "recall", "retrieval_ok": False},
    ]
    assert retrieval_accuracy(results) == 0.5


def test_hallucination_rate_counts_unsupported_claims():
    results = [
        {"type": "unknown", "declines_answer": False},
        {"type": "unknown", "declines_answer": True},
    ]
    assert hallucination_rate(results) == 0.5


def test_summarize_shape():
    results = [
        {
            "type": "known", "expected_tool": "recall", "tool_ok": True,
            "retrieval_ok": True, "answer_ok": True,
            "latency_ms": 1000.0, "tokens": 500,
        },
    ]
    report = summarize(results)
    assert report["questions"] == 1
    assert report["tool_selection_accuracy"] == 1.0
    assert report["retrieval_accuracy"] == 1.0
    assert report["answer_correctness"] == 1.0
    assert report["hallucination_rate"] == 0.0
    assert report["avg_latency_ms"] == 1000.0
    assert report["avg_tokens"] == 500.0


# ---------------------------------------------------------------------- #
# dataset + runner                                                       #
# ---------------------------------------------------------------------- #


def test_bundled_dataset_loads():
    assert DATASET_PATH.exists()
    items = load_dataset()
    assert len(items) >= 6
    assert all("question" in item for item in items)


def test_report_save(tmp_path):
    report = {"metrics": {"questions": 1}, "results": []}
    path = save_report(report, dataset_name="t")
    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8"))["metrics"]["questions"] == 1


class _ScriptedLLM:
    """Question → (tool json | answer) scripted by keyword match."""

    def __init__(self):
        self.calls = 0

    def complete(self, prompt, *, system=None, temperature=0.0):
        self.calls += 1
        lowered = prompt.lower()
        # Second pass: the recursive prompt already embeds a tool result.
        if "tool: " in lowered:
            if "nmd_user_01" in lowered or "my name" in lowered:
                return LLMResponse(text="Your name is nmd_user_01.")
            return LLMResponse(text="I do not have a memory confirming that.")
        if "what is my name" in lowered:
            return LLMResponse(
                text='{"tool": "recall", "arguments": {"query": "my name"}}'
            )
        if "what project" in lowered or "database" in lowered \
                or "language do i use" in lowered:
            return LLMResponse(
                text='{"tool": "recall", "arguments": {"query": "project"}}'
            )
        if "do i know rust" in lowered or "favorite programming" in lowered:
            return LLMResponse(
                text='{"tool": "recall", "arguments": {"query": "rust"}}'
            )
        if "hello" in lowered:
            return LLMResponse(text="Hello! How can I help?")
        if "what is python" in lowered:
            return LLMResponse(text="Python is a programming language.")
        return LLMResponse(text="I do not know.")


def test_runner_over_scripted_llm():
    provider = InMemoryServiceProvider()
    provider.create_user("user_001")
    bundle = provider.bundle("user_001")
    runner = EvaluationRunner(
        _ScriptedLLM(), bundle.repository, bundle.manager,
        user_id="user_001", top_k=5,
    )
    items = [
        {
            "id": "n1", "type": "known", "question": "What is my name?",
            "seed": "My name is Sathya.",
            "expected_memory": "Sathya", "expected_tool": "recall",
            "expected_answer": "Sathya",
        },
        {
            "id": "n2", "type": "known", "question": "Hello!",
            "seed": "", "expected_memory": "", "expected_tool": "none",
            "expected_answer": "",
        },
        {
            "id": "n3", "type": "unknown", "question": "Do I know Rust?",
            "seed": "", "expected_memory": "", "expected_tool": "recall",
            "expected_answer": "",
        },
    ]
    report = runner.run(items, seed=True)
    metrics = report["metrics"]
    assert metrics["questions"] == 3
    assert metrics["tool_selection_accuracy"] == 1.0
    assert metrics["retrieval_accuracy"] == 1.0
    rank = {r["id"]: r for r in report["results"]}
    assert rank["n1"]["answer_ok"] is True
    assert rank["n1"]["retrieval_ok"] is True
    assert rank["n2"]["tool_ok"] is True
    assert rank["n3"]["declines_answer"] is True
    assert rank["n3"]["answer_ok"] is None  # unknown: not an answer-check


# ---------------------------------------------------------------------- #
# API endpoint                                                           #
# ---------------------------------------------------------------------- #


@pytest.fixture
def app_with_llm(monkeypatch):
    monkeypatch.setattr(
        "nmd_host.intelligence.providers.provider_from_env",
        lambda: _ScriptedLLM(),
    )
    provider = InMemoryServiceProvider()
    provider.create_user("user_001")
    app = create_app(provider=provider, config=ServiceConfig())
    return app


def test_evaluation_dataset_endpoint(app_with_llm):
    with TestClient(app_with_llm) as client:
        resp = client.get("/api/NebulonMind/evaluation/dataset")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["name"] == "memory_test_v1"
    assert len(body["data"]["items"]) >= 6


def test_evaluation_run_endpoint(app_with_llm):
    with TestClient(app_with_llm) as client:
        resp = client.post(
            "/api/NebulonMind/evaluation/run",
            json={"max_items": 3, "seed": True},
            params={"user_id": "user_001"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["metrics"]["questions"] == 3
    assert body["data"]["metrics"]["tool_selection_accuracy"] > 0
    assert len(body["data"]["results"]) == 3
    assert body["data"]["results"][0]["question"]


def test_evaluation_run_503_without_llm(monkeypatch):
    monkeypatch.setenv("NMD_LLM_PROVIDER", "")
    provider = InMemoryServiceProvider()
    provider.create_user("user_001")
    app = create_app(provider=provider, config=ServiceConfig())
    with TestClient(app) as client:
        resp = client.post(
            "/api/NebulonMind/evaluation/run",
            json={"max_items": 2},
            params={"user_id": "user_001"},
        )
    assert resp.status_code == 503
    assert resp.json()["success"] is False


def test_console_evaluation_assets_served(app_with_llm):
    with TestClient(app_with_llm) as client:
        index = client.get("/api/NebulonMind/dashboard/")
        assert index.status_code == 200
        html = index.text
        assert "NebulonMind" in html
        assert "terminal-container" in html
        for asset in (
            "/api/NebulonMind/dashboard/js/main.js",
            "/api/NebulonMind/dashboard/css/main.css",
        ):
            resp = client.get(asset)
            assert resp.status_code == 200, asset
        js = client.get("/api/NebulonMind/dashboard/js/main.js").text
        assert "/evaluation/run" in js