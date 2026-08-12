from __future__ import annotations

from kawaneen.normalization.metrics import (
    aggregate_query_metrics,
    paired_confidence_interval,
    query_metrics,
)


def test_metrics_support_multi_relevant_qrels() -> None:
    metrics = query_metrics(("wrong", "target-b", "target-a"), ("target-a", "target-b"))
    assert metrics["recall_at_1"] == 0.0
    assert metrics["recall_at_5"] == 1.0
    assert metrics["mrr_at_10"] == 0.5
    assert metrics["ndcg_at_10"] > 0.0


def test_aggregate_metrics_and_paired_interval_are_deterministic() -> None:
    rows = [
        query_metrics(("target",), ("target",)),
        query_metrics(("wrong",), ("target",)),
    ]
    summary = aggregate_query_metrics(rows)
    assert summary["recall_at_1"] == 0.5
    interval = paired_confidence_interval([1.0, 0.0], [0.0, 0.0], seed=7, replicates=200)
    assert interval == paired_confidence_interval([1.0, 0.0], [0.0, 0.0], seed=7, replicates=200)
    assert interval["estimate"] == 0.5
    assert interval["lower"] <= interval["estimate"] <= interval["upper"]
