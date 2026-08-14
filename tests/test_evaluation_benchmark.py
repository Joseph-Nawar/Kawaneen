from __future__ import annotations

from pathlib import Path

from kawaneen.evaluation.validation import benchmark_source_status


def test_benchmark_source_is_unavailable_without_permitted_instances(tmp_path: Path) -> None:
    result = benchmark_source_status(tmp_path / "missing")
    assert result == {
        "status": "unavailable",
        "reason": "no permitted benchmark query/relevance instances are present",
        "fabricated_from_metadata": False,
    }


def test_metadata_only_benchmark_source_is_not_eligible(tmp_path: Path) -> None:
    path = tmp_path / "benchmark"
    path.write_text("metadata only", encoding="utf-8")
    result = benchmark_source_status(path)
    assert result["status"] == "blocked_schema_unverified"
    assert result["fabricated_from_metadata"] is False
