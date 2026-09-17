"""Step 3 — Memory Lifecycle & Retrieval Intelligence.

Manages memory over time and decides what is relevant now: retention,
expiration, deduplication, ranking, retrieval, consolidation, forgetting
and context building. Consumes Step 1 repositories (never ``ndb_host``)
and already-extracted Step 2 ``Memory`` objects.
"""

from .consolidation import ConsolidationDecision, MemoryConsolidator
from .context import ContextConfig, MemoryContextBuilder
from .deduplication import (
    DuplicateDetector,
    DuplicateResult,
    is_duplicate,
    normalize_text,
    text_similarity,
)
from .forgetting import ForgettingManager
from .lifecycle import MemoryState, get_state, is_expired, memory_age, now_utc, time_since_access
from .manager import IngestionResult, MemoryLifecycleManager, build_lifecycle_manager
from .ranking import (
    LexicalSemanticScorer,
    MemoryRanker,
    RankingConfig,
    SemanticScorer,
    confidence_score,
    importance_score,
    recency_score,
)
from .retention import RetentionError, RetentionEvaluator
from .retrieval import MemoryRetriever, RetrievalConfig

__all__ = [
    "ConsolidationDecision",
    "ContextConfig",
    "DuplicateDetector",
    "DuplicateResult",
    "ForgettingManager",
    "IngestionResult",
    "LexicalSemanticScorer",
    "MemoryContextBuilder",
    "MemoryLifecycleManager",
    "MemoryRanker",
    "MemoryRetriever",
    "MemoryState",
    "RankingConfig",
    "RetentionError",
    "RetentionEvaluator",
    "RetrievalConfig",
    "SemanticScorer",
    "build_lifecycle_manager",
    "confidence_score",
    "get_state",
    "importance_score",
    "is_duplicate",
    "is_expired",
    "memory_age",
    "normalize_text",
    "now_utc",
    "recency_score",
    "text_similarity",
    "time_since_access",
]
