from kawaneen.retrieval.metrics import (
    complete_evidence_recall_at_k,
    mrr_at_k,
    ndcg_at_k,
    paired_bootstrap,
    precision_at_k,
    recall_at_k,
    wins_ties_losses,
)


def test_binary_metrics_and_mrr_use_positive_qrel_grades() -> None:
    retrieved = ("a", "b", "c")
    qrels = {"a": 2, "c": 1}

    assert recall_at_k(retrieved, qrels, 1) == 0.5
    assert recall_at_k(retrieved, qrels, 5) == 1.0
    assert precision_at_k(retrieved, qrels, 2) == 0.5
    assert mrr_at_k(retrieved, qrels, 10) == 1.0


def test_ndcg_uses_graded_gain() -> None:
    assert ndcg_at_k(("low", "high"), {"high": 2, "low": 1}, 2) < 1.0


def test_complete_evidence_requires_every_group() -> None:
    groups = {"g1": frozenset({"a"}), "g2": frozenset({"b"})}

    assert complete_evidence_recall_at_k(("a", "b"), groups, 2) == 1.0
    assert complete_evidence_recall_at_k(("a", "c"), groups, 2) == 0.0


def test_bootstrap_is_seeded_and_reports_wins_ties_losses() -> None:
    left = (1.0, 0.0, 1.0)
    right = (0.0, 0.0, 1.0)

    first = paired_bootstrap(left, right, seed=20260815, replicates=2000)
    second = paired_bootstrap(left, right, seed=20260815, replicates=2000)

    assert first == second
    assert wins_ties_losses(left, right) == {"wins": 1, "ties": 2, "losses": 0}


def test_metrics_reject_invalid_or_empty_inputs() -> None:
    import pytest

    with pytest.raises(ValueError, match="positive"):
        precision_at_k(("a",), {"a": 1}, 0)
    assert complete_evidence_recall_at_k(("a",), {}, 1) == 0.0
    with pytest.raises(ValueError, match="equal non-empty"):
        paired_bootstrap((1.0,), (), seed=1)
    with pytest.raises(ValueError, match="bootstrap settings"):
        paired_bootstrap((1.0,), (0.0,), seed=1, replicates=0)
