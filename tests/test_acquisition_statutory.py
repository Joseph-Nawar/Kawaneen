from __future__ import annotations

import csv
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from kawaneen.acquisition.models import FileExpectation, SourceSpecification
from kawaneen.acquisition.statutory import audit_statutory_quality


def test_statutory_quality_metrics_are_counts_only(tmp_path: Path) -> None:
    pq.write_table(
        pa.table(
            {
                "text": ["Arabic text", "Arabic text", "x"],
                "article_number": ["1", "1", "2"],
                "law_name": ["Law A", "Law A", "Law B"],
                "law_type": ["regulation", "regulation", "law"],
                "source": ["MOJ", "MOJ", "MOJ"],
            }
        ),
        tmp_path / "data.parquet",
    )
    spec = SourceSpecification(
        schema_version=1,
        source_id="saudi-moj-derived",
        version="test",
        revision="test",
        provider="fixture",
        identifier="fixture",
        licence="CC BY 4.0",
        expected_records=3,
        files=(
            FileExpectation(
                path="data.parquet",
                format="parquet",
                expected_records=3,
                expected_columns=("text", "article_number", "law_name", "law_type", "source"),
            ),
        ),
    )
    result = audit_statutory_quality(spec, tmp_path)
    assert result.total_rows == 3
    assert result.unique_rows == 2
    assert result.unique_law_names == 2
    assert result.duplicate_law_article_keys == 1
    assert result.extremely_short_records == 3
    assert not hasattr(result, "text")


def test_reconciliation_template_is_sanitized_and_manual_reviewable() -> None:
    path = Path("data/manifests/reconciliation/core-commercial-civil-v1.csv")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 12
    assert all(row["eligible_for_kawaneen_v1_statutory_corpus"] == "not_eligible" for row in rows)
    assert all(
        row["reconciliation_status"] == "present_partial_reviewed_not_eligible" for row in rows
    )
    assert all(row["reviewer_type"] == "independent_ai_source_review" for row in rows)
    assert all(row["human_verified"] == "false" for row in rows)
    assert all(row["manual_review_notes"].strip() for row in rows)
    assert all(row["dataset_article_count"].isdigit() for row in rows)
    assert "article text" not in path.read_text(encoding="utf-8").lower()
