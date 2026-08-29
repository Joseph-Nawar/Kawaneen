from __future__ import annotations

from kawaneen.phase15.runner import evaluate_dev_rankings, summarize_ranking_runs


def test_evaluate_dev_rankings_returns_paired_metric_rows() -> None:
    records = (
        {
            "query_id": "q1",
            "answerability": "answerable",
            "chunk_qrels": [{"chunk_id": "c1", "grade": 2}],
            "evidence_groups": [{"spans": [{"unit_id": "u1"}]}],
        },
        {"query_id": "q2", "answerability": "unanswerable", "chunk_qrels": []},
    )
    chunks = ({"chunk_id": "c1", "source_unit_ids": ["u1"]},)
    result = evaluate_dev_rankings(
        records,
        {"q1": ("c1",), "q2": ()},
        chunks,
    )
    assert result.query_ids == ("q1",)
    assert result.metrics["Recall@1"] == (1.0,)
    assert result.metrics["CompleteEvidenceRecall@10"] == (1.0,)


def test_summarize_ranking_runs_includes_paired_deltas() -> None:
    result = summarize_ranking_runs(
        {
            "baseline": {"Recall@10": (0.0, 1.0)},
            "candidate": {"Recall@10": (1.0, 1.0)},
        }
    )
    assert result["systems"]["candidate"]["Recall@10"]["mean"] == 1.0
    assert result["paired_deltas"]["candidate-vs-baseline"]["Recall@10"]["wins"] == 1
