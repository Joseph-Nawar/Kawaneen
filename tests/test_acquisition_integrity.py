from __future__ import annotations

import csv
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from kawaneen.acquisition.integrity import IntegrityError, verify_specification
from kawaneen.acquisition.models import FileExpectation, SourceSpecification


def _spec() -> SourceSpecification:
    return SourceSpecification(
        schema_version=1,
        source_id="alarb",
        version="test",
        revision="test",
        provider="fixture",
        identifier="fixture",
        licence="test",
        expected_records=3,
        expected_splits={"train": 2, "test": 1},
        allowed_purposes=("evaluation",),
        files=(
            FileExpectation(
                path="train.parquet", format="parquet", expected_records=2, split="train"
            ),
            FileExpectation(
                path="test.parquet", format="parquet", expected_records=1, split="test"
            ),
        ),
    )


def test_parquet_integrity_reports_schema_and_overlap_without_mutation(tmp_path: Path) -> None:
    table = pa.table({"id": ["a", "b"], "text": ["fiction one", "fiction two"]})
    pq.write_table(table, tmp_path / "train.parquet")
    pq.write_table(pa.table({"id": ["c"], "text": ["fiction three"]}), tmp_path / "test.parquet")
    result = verify_specification(_spec(), tmp_path)
    assert result.row_counts == {"train.parquet": 2, "test.parquet": 1}
    assert result.split_overlap_count == 0
    assert result.files[0].sha256
    assert (tmp_path / "train.parquet").exists()


def test_csv_row_count_and_utf8_are_strict(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "text"])
        writer.writerow(["1", "fiction"])
    spec = SourceSpecification(
        schema_version=1,
        source_id="arabiccr",
        version="test",
        revision="test",
        provider="fixture",
        identifier="fixture",
        licence="test",
        expected_records=2,
        files=(FileExpectation(path="data.csv", format="csv", expected_records=2),),
    )
    with pytest.raises(IntegrityError, match="row count"):
        verify_specification(spec, tmp_path)


def test_exact_duplicate_rows_and_split_overlap_are_reported(tmp_path: Path) -> None:
    table = pa.table({"id": ["same", "same"], "text": ["x", "x"]})
    pq.write_table(table, tmp_path / "train.parquet")
    pq.write_table(pa.table({"id": ["same"], "text": ["x"]}), tmp_path / "test.parquet")
    result = verify_specification(_spec().model_copy(update={"expected_records": 3}), tmp_path)
    assert result.duplicate_row_count == 2
    assert result.split_overlap_count == 1


def test_total_expected_record_count_is_checked(tmp_path: Path) -> None:
    pq.write_table(pa.table({"id": ["a", "b"], "text": ["x", "y"]}), tmp_path / "train.parquet")
    pq.write_table(pa.table({"id": ["c"], "text": ["z"]}), tmp_path / "test.parquet")
    spec = _spec().model_copy(update={"expected_records": 4})
    with pytest.raises(IntegrityError, match="total row count"):
        verify_specification(spec, tmp_path)
