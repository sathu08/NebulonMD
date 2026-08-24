"""Ranking tests (3.4): semantic / importance / confidence / recency signals,
combined score and weight validation. Offline."""

from datetime import datetime, timedelta, timezone

import pytest

from nmd_host.lifecycle.ranking import (
    LexicalSemanticScorer,
    MemoryRanker,
    RankingConfig,
    recency_score,
)
from conftest import make_memory


def _memory(text, score=0.5, confidence=0.5, created=None, updated=None):
    memory = make_memory(text, score=score, confidence=confidence)
    if created:
        memory.lifecycle.created_at = created
    if updated:
        memory.lifecycle.updated_at = updated
    return memory


def test_semantic_score_from_scorer():
    ranker = MemoryRanker(RankingConfig(semantic_weight=1.0, importance_weight=0.0, confidence_weight=0.0, recency_weight=0.0))
    assert ranker.score("python backend", _memory("User works with Python")) > ranker.score(
        "python backend", _memory("User likes hiking")
    )


def test_importance_signal():
    ranker = MemoryRanker(RankingConfig(semantic_weight=0.0, importance_weight=1.0, confidence_weight=0.0, recency_weight=0.0))
    important = ranker.score("q", _memory("fact", score=0.95))
    trivial = ranker.score("q", _memory("fact", score=0.2))
    assert important > trivial


def test_confidence_signal():
    ranker = MemoryRanker(RankingConfig(semantic_weight=0.0, importance_weight=0.0, confidence_weight=1.0, recency_weight=0.0))
    confident = ranker.score("q", _memory("fact", confidence=0.95))
    uncertain = ranker.score("q", _memory("fact", confidence=0.3))
    assert confident > uncertain


def test_recency_signal_favours_fresh_memories():
    ranker = MemoryRanker(RankingConfig(semantic_weight=0.0, importance_weight=0.0, confidence_weight=0.0, recency_weight=1.0))
    now = datetime.now(timezone.utc)
    fresh = _memory("fact", updated=now - timedelta(hours=1))
    old = _memory("fact", updated=now - timedelta(days=400))
    assert ranker.score("q", fresh) > ranker.score("q", old)


def test_recency_is_bounded_and_stable():
    now = datetime.now(timezone.utc)
    ancient = _memory("fact", updated=now - timedelta(days=3650))
    score = recency_score(ancient, now=now)
    assert 0.0 < score < 1.0  # never 1/age blow-ups, never zero


def test_combined_score_uses_all_weights():
    ranker = MemoryRanker(RankingConfig())
    query = "user works with python"
    memory = _memory("User works with Python", score=0.9, confidence=0.9, updated=datetime.now(timezone.utc))
    total = ranker.score(query, memory)
    cfg = ranker.config
    semantic = LexicalSemanticScorer().score(query, memory)
    expected = (
        cfg.semantic_weight * semantic
        + cfg.importance_weight * 0.9
        + cfg.confidence_weight * 0.9
        + cfg.recency_weight * 1.0  # freshly updated
    )
    assert total == pytest.approx(expected, abs=1e-6)


def test_rank_orders_best_first():
    ranker = MemoryRanker(RankingConfig(semantic_weight=0.8, importance_weight=0.2, confidence_weight=0.0, recency_weight=0.0))
    now = datetime.now(timezone.utc)
    low = _memory("User works with Python", score=0.3, updated=now)
    high = _memory("User works with Python", score=0.95, updated=now)
    ranked = ranker.rank("python", [low, high])
    assert ranked[0].memory_id == high.memory_id


def test_weight_validation_rejects_non_unit_sum():
    with pytest.raises(ValueError):
        RankingConfig(semantic_weight=0.5, importance_weight=0.2, confidence_weight=0.1, recency_weight=0.1)
    with pytest.raises(ValueError):
        RankingConfig(semantic_weight=1.0, importance_weight=1.0, confidence_weight=-0.5, recency_weight=-0.5)


def test_weight_validation_rejects_negatives():
    with pytest.raises(ValueError):
        RankingConfig(semantic_weight=-0.1, importance_weight=0.4, confidence_weight=0.3, recency_weight=0.4)


def test_recency_half_life_must_be_positive():
    with pytest.raises(ValueError):
        RankingConfig(recency_half_life_days=0.0)
    with pytest.raises(ValueError):
        recency_score(_memory("x"), half_life_days=-1)


def test_important_permanent_memory_survives_recency_decay():
    ranker = MemoryRanker(RankingConfig(
        semantic_weight=0.0, importance_weight=0.6,
        confidence_weight=0.0, recency_weight=0.4,
    ))
    now = datetime.now(timezone.utc)
    important_old = _memory("User's name is Sathya", score=0.99, updated=now - timedelta(days=365))
    trivial_fresh = _memory("User mentioned the weather", score=0.3, updated=now - timedelta(minutes=1))
    assert ranker.score("name", important_old) > ranker.score("name", trivial_fresh)
