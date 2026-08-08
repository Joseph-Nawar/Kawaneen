"""Deterministic, masked local privacy screening."""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path
from typing import Any, cast

import pyarrow.parquet as pq

from kawaneen.acquisition.models import (
    PrivacyFinding,
    PrivacyResult,
    PrivacySummary,
    SourceSpecification,
)

_PATTERNS = {
    "email": re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
    "phone": re.compile(r"(?<!\d)(?:\+?\d[\d ()-]{7,}\d)(?!\d)"),
    "iban_like": re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b"),
}
_IDENTITY_COLUMNS = re.compile(
    r"(?:national|identity|civil|passport|iqama|id)[ _-]?(?:number|no|id)?", re.I
)
_ADDRESS_COLUMNS = re.compile(r"address|street|location|residence", re.I)


def _rows(path: Path, file_format: str) -> tuple[list[str], list[dict[str, Any]]]:
    if file_format == "csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            return list(reader.fieldnames or []), list(reader)
    table: Any = cast(Any, pq.read_table)(path)  # pyright: ignore[reportUnknownMemberType]
    return list(table.column_names), table.to_pylist()


def screen_privacy(specification: SourceSpecification, root: Path) -> PrivacyResult:
    """Scan configured tabular files and return masked findings, never clearance."""

    findings: list[PrivacyFinding] = []
    for expected in specification.files:
        if expected.format not in {"csv", "parquet"}:
            continue
        path = root.joinpath(*Path(expected.path).parts)
        columns, rows = _rows(path, expected.format)
        for row_number, row in enumerate(rows, start=2):
            for column in columns:
                value = str(row.get(column, ""))
                if not value:
                    continue
                for detector, pattern in _PATTERNS.items():
                    if pattern.search(value):
                        findings.append(
                            PrivacyFinding(
                                detector=detector,
                                column=column,
                                file_path=expected.path,
                                row_number=row_number,
                                masked_value="[REDACTED]",
                            )
                        )
                if _IDENTITY_COLUMNS.search(column) or _ADDRESS_COLUMNS.search(column):
                    findings.append(
                        PrivacyFinding(
                            detector="identifier_or_address_column",
                            column=column,
                            file_path=expected.path,
                            row_number=row_number,
                            masked_value="[REDACTED]",
                        )
                    )
    ordered = tuple(
        sorted(
            findings, key=lambda item: (item.file_path, item.row_number, item.column, item.detector)
        )
    )
    return PrivacyResult(
        source_id=specification.source_id,
        finding_count=len(ordered),
        findings=ordered,
        legal_clearance=False,
        review_status="pending_manual_review",
    )


def summarize_privacy(result: PrivacyResult, sample_cap: int = 100) -> PrivacySummary:
    """Aggregate only counts and categories; never include values or legal text."""

    detector_counts = Counter(item.detector for item in result.findings)
    column_counts = Counter(item.column for item in result.findings)
    affected_records = {(item.file_path, item.row_number) for item in result.findings}
    return PrivacySummary(
        source_id=result.source_id,
        findings_by_detector=dict(sorted(detector_counts.items())),
        affected_record_count=len(affected_records),
        findings_by_column=dict(sorted(column_counts.items())),
        deterministic_review_sample_size=min(sample_cap, len(affected_records)),
        confirmed_pii_count=None,
        likely_false_positive_count=None,
        unresolved_categories=tuple(sorted(detector_counts)),
        manual_review_status=result.review_status,
    )


def write_private_review_bundle(result: PrivacyResult, destination: Path) -> Path:
    """Write only masked findings below the caller-provided private artifact root."""

    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{result.source_id}-privacy-review.json"
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return path
