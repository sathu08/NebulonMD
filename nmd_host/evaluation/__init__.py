"""Step 13 — Agent evaluation: metrics + runner (see changes.txt 13).

Run the Step 6 agent over a benchmark dataset and produce retrieval / tool /
answer / hallucination / latency / token metrics. The data artifacts
(dataset.json, reports/) live at the repository root ``evaluation/``.
"""

from .metrics import (
    answer_correctness,
    hallucination_rate,
    latency_stats,
    retrieval_accuracy,
    summarize,
    token_stats,
    tool_selection_accuracy,
)
from .runner import (
    DATASET_PATH,
    EvaluationRunner,
    REPORTS_DIR,
    load_dataset,
    save_report,
)

__all__ = [
    "DATASET_PATH",
    "EvaluationRunner",
    "REPORTS_DIR",
    "answer_correctness",
    "hallucination_rate",
    "latency_stats",
    "load_dataset",
    "retrieval_accuracy",
    "save_report",
    "summarize",
    "token_stats",
    "tool_selection_accuracy",
]
