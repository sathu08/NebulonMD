"""Step 13 — Agent evaluation runner (changes.txt 13.1–13.7).

Drives the real Step 6 agent over a benchmark dataset and produces a
metric report. For each question it:

1. (recall items) measures *retrieval* independently — did the Step 3
   pipeline surface the expected memory? — via ``manager.retrieve``;
2. runs the agent chat loop over the user's memory toolkit and records the
   chosen tool(s), final answer, latency and approximate tokens;
3. checks answer correctness (known items) and hallucination behaviour
   (unknown items).

Everything here is offline-testable with a scripted ``LLMProvider`` and an
``InMemoryServiceProvider``; the API endpoint wires the real provider.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .metrics import summarize
from nmd_host.core.config import _nmd_home
from nmd_host.agent import AgentConfig, AgentRuntime, build_memory_toolkit

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


REPO_ROOT = _nmd_home()
DATASET_PATH = REPO_ROOT / "unittest" / "evaluation" / "dataset.json"
REPORTS_DIR = REPO_ROOT / "evaluation_reports"

_DECLINE_MARKERS = (
    "don't know",
    "dont know",
    "do not know",
    "no memory",
    "no memories",
    "not have",
    "insufficient",
    "no information",
    "cannot",
    "can't say",
    "can't confirm",
    "cannot confirm",
    "not sure",
    "no way to know",
    "no data",
    "haven't",
    "no record",
)


def _normalize(text: Any) -> str:
    return " ".join(str(text or "").lower().split())


def _answer_ok(expected: str, answer: str) -> bool:
    if not expected:
        return True
    return _normalize(expected) in _normalize(answer)


def _declines(answer: str) -> bool:
    normalized = _normalize(answer)
    return any(marker in normalized for marker in _DECLINE_MARKERS)


def load_dataset(path: Optional[Path] = None) -> List[dict]:
    """Load the bundled benchmark dataset (JSON file → item list)."""
    dataset_path = Path(path) if path else DATASET_PATH
    if not dataset_path.exists():
        return []
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return list(data.get("items", []))
    return list(data)


def _safe_user(user_id: str) -> str:
    cleaned = "".join(
        c if (c.isalnum() or c in ("-", "_")) else "_" for c in str(user_id or "")
    ).strip("_")
    return cleaned or "anonymous"


def save_report(
    report: dict, dataset_name: str = "dataset", user_id: str = ""
) -> Path:
    """Persist a run's report as ``evaluation_reports/<user>/<dataset>-<timestamp>.json``."""
    user_dir = REPORTS_DIR / _safe_user(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    path = user_dir / f"{dataset_name or 'dataset'}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


class EvaluationRunner:
    """Evaluate the agent over a list of dataset items.

    ``llm`` may be any ``LLMProvider`` (scripted fake or real); ``seeder``
    is an optional callable ``(text) -> None`` used to pre-store the
    dataset's known facts so retrieval has something to find on a fresh
    user.
    """

    def __init__(
        self,
        llm: Any,
        repository: Any,
        manager: Any,
        user_id: str = "user_001",
        top_k: int = 5,
        agent_config: Optional[AgentConfig] = None,
        seeder: Optional[Any] = None,
    ) -> None:
        self._llm = llm
        self._repository = repository
        self._manager = manager
        self._user_id = user_id
        self._top_k = top_k
        self._agent_config = agent_config or AgentConfig(max_turns=3)
        self._seeder = seeder

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def run(self, items: List[dict], seed: bool = True) -> Dict[str, Any]:
        """Run the agent over ``items`` and return ``{metrics, results}``.

        When ``seed`` is true, each ``seed`` text from the dataset is stored
        (through the existing Step 3 gate → repository) before evaluation so
        a fresh user has the benchmark facts to recall.
        """
        if seed:
            self._seed(items)
        results = [self._run_one(item) for item in items]
        return {"metrics": summarize(results), "results": results}

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #

    def _seed(self, items: List[dict]) -> None:
        if self._seeder is not None:
            for item in items:
                text = str(item.get("seed", "")).strip()
                if text:
                    self._seeder(text)
            return
        seen: set = set()
        for item in items:
            text = str(item.get("seed", "")).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            self._store_memory(text)

    def _store_memory(self, text: str, category: str = "fact") -> None:
        memory = Memory(
            user_id=self._user_id,
            content=MemoryContent(text=text, summary=text),
            classification=Classification(
                memory_type=MemoryType.SEMANTIC,
                category=category,
                source="evaluation",
            ),
            provenance=Provenance(source_type="api", created_by="evaluation"),
            importance=Importance(score=0.9, confidence=1.0),
            lifecycle=Lifecycle(retention_policy=RetentionPolicy.PERMANENT),
            status=MemoryStatus.ACTIVE,
        )
        verdict = self._manager.ingest(memory, user_id=self._user_id)
        if verdict.action == "STORE":
            self._repository.create(memory)

    def _run_one(self, item: dict) -> Dict[str, Any]:
        question = str(item.get("question", ""))
        expected_tool = str(item.get("expected_tool", "none"))
        expected_memory = str(item.get("expected_memory", ""))
        expected_answer = str(item.get("expected_answer", ""))
        result: Dict[str, Any] = {
            "id": str(item.get("id", "")),
            "question": question,
            "type": str(item.get("type", "known")),
            "expected_tool": expected_tool,
            "expected_answer": expected_answer,
            "expected_memory": expected_memory,
            "actual_tools": [],
            "retrieval_ok": None,
            "tool_ok": None,
            "answer": "",
            "answer_ok": False,
            "declines_answer": False,
            "latency_ms": 0.0,
            "tokens": 0,
            "error": None,
        }

        # Retrieval accuracy measured independently of the agent.
        if expected_tool == "recall":
            try:
                memories = self._manager.retrieve(
                    question, top_k=self._top_k, user_id=self._user_id
                )
                result["retrieval_ok"] = self._memory_hit(
                    expected_memory or expected_answer, memories
                )
            except Exception as exc:  # defensive: never fail the whole run
                result["retrieval_ok"] = False
                result["error"] = f"retrieval: {type(exc).__name__}: {exc}"

        started = time.monotonic()
        try:
            tools = build_memory_toolkit(
                self._repository, self._manager, self._user_id
            )
            runtime = AgentRuntime(self._llm, tools, self._agent_config)
            data = runtime.chat(question)
            result["answer"] = data.answer
            result["actual_tools"] = [t.tool for t in data.tool_calls]
            result["latency_ms"] = round(
                (time.monotonic() - started) * 1000.0, 2
            )
            if data.trace is not None and data.trace.llm is not None:
                result["tokens"] = data.trace.llm.tokens
            result["tool_ok"] = self._tool_ok(
                expected_tool, result["actual_tools"]
            )
            # answer_ok is only meaningful where an expected answer exists
            # (unknown questions are scored by hallucination/decline instead).
            result["answer_ok"] = (
                _answer_ok(expected_answer, data.answer)
                if item.get("type") == "known"
                else None
            )
            result["declines_answer"] = _declines(data.answer)
        except Exception as exc:  # defensive: one bad item must not kill the run
            result["error"] = f"agent: {type(exc).__name__}: {exc}"
            result["latency_ms"] = round(
                (time.monotonic() - started) * 1000.0, 2
            )
        return result

    @staticmethod
    def _tool_ok(expected_tool: str, actual_tools: List[str]) -> bool:
        if expected_tool == "none":
            return not actual_tools
        return any(t == expected_tool for t in actual_tools)

    @staticmethod
    def _memory_hit(expected: str, memories: List[Memory]) -> bool:
        if not expected:
            return True
        normalized = _normalize(expected)
        for memory in memories:
            haystacks = [
                memory.content.text,
                memory.content.summary or "",
                memory.memory_id or "",
            ]
            if any(normalized in _normalize(h) for h in haystacks):
                return True
        return False


__all__ = [
    "DATASET_PATH",
    "EvaluationRunner",
    "REPORTS_DIR",
    "load_dataset",
    "save_report",
]