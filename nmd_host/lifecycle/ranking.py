"""Ranking (Step 3).

Combines four independent signals into one final score:

    final = semantic*w_semantic + importance*w_importance
          + confidence*w_confidence + recency*w_recency

* Semantic relevance — query-to-memory similarity (lexical by default, or an
  injected scorer; the vector store already selected candidates).
* Importance — ``importance.score`` (what the fact is worth).
* Confidence — ``importance.confidence`` (how reliable it is).
* Recency — bounded exponential decay on ``updated_at``; recent memories
  score near 1, old memories decay toward 0 but never collapse. This is
  deliberately independent from importance: an important permanent memory
  stays retrievable because its importance weight carries it, not because
  it was recently touched.

Retention acts as a hard *gate* before ranking (see ``retrieval.py``), not
as a ranking signal. Weights are validated to sum to 1.0 and are
configurable — no magic numbers scattered through the code.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Protocol

from dataclasses import dataclass

from nmd_host.core.models import Memory
from nmd_host.utils.constants import SECONDS_PER_DAY
from nmd_host.utils.env_helpers import env_float as _env_float

from .deduplication import text_similarity


@dataclass
class RankingConfig:
    """Blend weights for the four ranking signals (sum must be 1.0)."""

    semantic_weight: float = 0.50
    importance_weight: float = 0.20
    confidence_weight: float = 0.15
    recency_weight: float = 0.15
    recency_half_life_days: float = 30.0

    def __post_init__(self) -> None:
        weights = [
            self.semantic_weight,
            self.importance_weight,
            self.confidence_weight,
            self.recency_weight,
        ]
        if any(w < 0.0 for w in weights):
            raise ValueError("ranking weights must be non-negative")
        total = sum(weights)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"ranking weights must sum to 1.0 (got {total})"
            )
        if self.recency_half_life_days <= 0.0:
            raise ValueError("recency_half_life_days must be positive")

    @classmethod
    def from_env(cls) -> "RankingConfig":
        return cls(
            semantic_weight=_env_float("NMD_RANKING_SEMANTIC_WEIGHT", 0.50),
            importance_weight=_env_float("NMD_RANKING_IMPORTANCE_WEIGHT", 0.20),
            confidence_weight=_env_float("NMD_RANKING_CONFIDENCE_WEIGHT", 0.15),
            recency_weight=_env_float("NMD_RANKING_RECENCY_WEIGHT", 0.15),
            recency_half_life_days=_env_float(
                "NMD_RANKING_RECENCY_HALF_LIFE_DAYS", 30.0
            ),
        )


class SemanticScorer(Protocol):
    """Scores how relevant one memory is to a query, in [0, 1]."""

    def score(self, query: str, memory: Memory) -> float: ...


class LexicalSemanticScorer:
    """Default scorer: token overlap between query and memory text."""

    def score(self, query: str, memory: Memory) -> float:
        text = " ".join(
            filter(None, [memory.content.text, memory.content.summary])
        )
        return text_similarity(query, text)


def recency_score(
    memory: Memory,
    now: Optional[datetime] = None,
    half_life_days: float = 30.0,
) -> float:
    """Bounded exponential decay on recency, in [0, 1].

    ``score = 2 ** (-age_days / half_life_days)`` — recent memories score
    near 1, old memories decay toward 0. Never ``1/age`` (stable), never
    negative. Anchored on ``updated_at`` (the last time the fact changed).
    """
    from .lifecycle import memory_age

    if half_life_days <= 0.0:
        raise ValueError("half_life_days must be positive")
    age_days = memory_age(memory, now=now) / SECONDS_PER_DAY
    return float(2.0 ** (-age_days / half_life_days))


def importance_score(memory: Memory) -> float:
    return float(memory.importance.score)


def confidence_score(memory: Memory) -> float:
    return float(memory.importance.confidence)


class MemoryRanker:
    """Blends the four signals into a single score and orders memories."""

    def __init__(
        self,
        config: Optional[RankingConfig] = None,
        semantic_scorer: Optional[SemanticScorer] = None,
    ) -> None:
        self.config = config or RankingConfig()
        self.semantic_scorer = semantic_scorer or LexicalSemanticScorer()

    def score(
        self,
        query: str,
        memory: Memory,
        now: Optional[datetime] = None,
    ) -> float:
        cfg = self.config
        return (
            cfg.semantic_weight * self.semantic_scorer.score(query, memory)
            + cfg.importance_weight * importance_score(memory)
            + cfg.confidence_weight * confidence_score(memory)
            + cfg.recency_weight
            * recency_score(
                memory,
                now=now,
                half_life_days=cfg.recency_half_life_days,
            )
        )

    def rank(
        self,
        query: str,
        memories: List[Memory],
        now: Optional[datetime] = None,
    ) -> List[Memory]:
        """Return memories ordered best-first (stable within ties)."""
        return sorted(
            memories,
            key=lambda m: self.score(query, m, now),
            reverse=True,
        )


__all__ = [
    "LexicalSemanticScorer",
    "MemoryRanker",
    "RankingConfig",
    "SemanticScorer",
    "confidence_score",
    "importance_score",
    "recency_score",
]
