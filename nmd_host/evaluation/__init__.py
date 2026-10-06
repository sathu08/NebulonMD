"""Step 13 — Agent evaluation: metrics + runner (see changes.txt 13).

Run the Step 6 agent over a benchmark dataset and produce retrieval / tool /
answer / hallucination / latency / token metrics. The data artifacts
(dataset.json, reports/) live at the repository root ``evaluation/``.
"""

from .grounding import (
    CONTRADICTED,
    GROUNDING_LABELS,
    SUPPORTED,
    UNKNOWN,
    UNSUPPORTED_INFERENCE,
    grounding_accuracy,
    grounding_verdict,
    is_supported,
    unsupported_rate,
)
from .metrics import (
    answer_correctness,
    grounding_accuracy,
    hallucination_rate,
    latency_stats,
    retrieval_accuracy,
    router_forced_rate,
    summarize,
    token_stats,
    tool_selection_accuracy,
    unsupported_rate,
)
from .runner import (
    DATASET_PATH,
    EvaluationRunner,
    REPORTS_DIR,
    load_dataset,
    save_report,
)

__all__ = [
    "CONTRADICTED",
    "DATASET_PATH",
    "EvaluationRunner",
    "GROUNDING_LABELS",
    "REPORTS_DIR",
    "SUPPORTED",
    "UNKNOWN",
    "UNSUPPORTED_INFERENCE",
    "answer_correctness",
    "grounding_accuracy",
    "grounding_verdict",
    "hallucination_rate",
    "is_supported",
    "latency_stats",
    "load_dataset",
    "retrieval_accuracy",
    "router_forced_rate",
    "save_report",
    "summarize",
    "token_stats",
    "tool_selection_accuracy",
    "unsupported_rate",
]
