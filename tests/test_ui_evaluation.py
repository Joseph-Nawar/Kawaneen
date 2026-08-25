from pathlib import Path

import pytest

from kawaneen.ui.evaluation import (
    aggregate_latency,
    build_evaluation_snapshot,
    validate_source_path,
)


ROOT = Path(__file__).parents[1]


def test_snapshot_contains_required_tracked_metrics_and_hashes() -> None:
    snapshot = build_evaluation_snapshot(ROOT)

    assert snapshot.generation["ValidCitationRate"] == 1.0
    assert snapshot.generation["final_answer_coverage"] == 0.59375
    assert snapshot.extraction["micro_f1"] == pytest.approx(0.16417910447761197)
    assert snapshot.extraction["full_rule_exact_f1"] == pytest.approx(0.02127659574468085)
    assert snapshot.sources
    assert all(len(source.sha256) == 64 for source in snapshot.sources)


def test_snapshot_source_validation_rejects_private_or_outside_paths() -> None:
    with pytest.raises(ValueError, match="allowlist"):
        validate_source_path(ROOT, Path("artifacts/private/secret.json"))
    with pytest.raises(ValueError, match="repository"):
        validate_source_path(ROOT, Path("../outside.json"))


def test_latency_aggregation_keeps_last_fifty_and_computes_percentiles() -> None:
    summary = aggregate_latency(list(range(1, 61)))

    assert summary.count == 50
    assert summary.values[0] == 11
    assert summary.values[-1] == 60
    assert summary.median == pytest.approx(35.5)
    assert summary.p95 == pytest.approx(57.55)
