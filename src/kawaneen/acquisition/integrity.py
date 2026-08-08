"""Deterministic file, schema, row-integrity, and exact-duplicate checks."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, cast

import pyarrow.parquet as pq

from kawaneen.acquisition.models import FileDigest, IntegrityResult, SourceSpecification
from kawaneen.acquisition.storage import digest_file


class IntegrityError(ValueError):
    """Raised when an acquired file does not match its specification."""


def schema_fingerprint(names: list[str], types: list[str]) -> str:
    """Hash ordered column names and logical types."""

    payload = json.dumps(
        {"names": names, "types": types}, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical_row(row: dict[str, Any]) -> str:
    normalized = {str(key): row[key] for key in sorted(row)}
    return json.dumps(
        normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )


def _csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]], list[str]]:
    findings: list[str] = []
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        findings.append(f"UTF-8 BOM detected: {path.name}")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise IntegrityError(f"CSV is not valid UTF-8: {path}") from exc
    reader = csv.DictReader(text.splitlines())
    if reader.fieldnames is None or any(not name for name in reader.fieldnames):
        raise IntegrityError(f"CSV has an invalid header: {path}")
    rows = list(reader)
    if any(None in row for row in rows):
        raise IntegrityError(f"CSV has rows with inconsistent column counts: {path}")
    return list(reader.fieldnames), rows, findings


def _parquet_rows(path: Path) -> tuple[list[str], list[dict[str, Any]], list[str]]:
    try:
        table: Any = cast(Any, pq.read_table)(path)  # pyright: ignore[reportUnknownMemberType]
    except Exception as exc:
        raise IntegrityError(f"invalid Parquet file: {path}") from exc
    names = list(table.column_names)
    rows = table.to_pylist()
    return names, rows, []


def _read_rows(
    path: Path, file_format: str
) -> tuple[list[str], list[dict[str, Any]], list[str], list[str]]:
    if file_format == "csv":
        names, rows, findings = _csv_rows(path)
        types = ["string"] * len(names)
    elif file_format == "parquet":
        names, rows, findings = _parquet_rows(path)
        schema: Any = cast(Any, pq.read_schema)(path)  # pyright: ignore[reportUnknownMemberType]
        types = [str(schema.field(name).type) for name in names]
    else:
        return [], [], [], []
    return names, rows, findings, types


def verify_specification(specification: SourceSpecification, root: Path) -> IntegrityResult:
    """Verify every specified file and report exact duplicate counts only."""

    digests: list[FileDigest] = []
    row_counts: dict[str, int] = {}
    fingerprints: dict[str, str] = {}
    findings: list[str] = []
    seen_hashes: Counter[str] = Counter()
    all_rows: list[str] = []
    split_rows: dict[str, set[str]] = {}

    for expected in specification.files:
        path = root.joinpath(*Path(expected.path).parts)
        if not path.is_file() or path.is_symlink():
            raise IntegrityError(f"expected file is missing or unsafe: {expected.path}")
        digest = digest_file(path)
        digests.append(FileDigest(path=expected.path, sha256=digest.sha256, size=digest.size))
        seen_hashes[digest.sha256] += 1
        if expected.format in {"csv", "parquet"}:
            names, rows, file_findings, types = _read_rows(path, expected.format)
            if expected.expected_records is not None and len(rows) != expected.expected_records:
                raise IntegrityError(
                    f"row count mismatch for {expected.path}: "
                    f"{len(rows)} != {expected.expected_records}"
                )
            if expected.expected_columns and tuple(names) != expected.expected_columns:
                raise IntegrityError(f"schema columns mismatch for {expected.path}")
            row_counts[expected.path] = len(rows)
            fingerprints[expected.path] = schema_fingerprint(names, types)
            findings.extend(file_findings)
            canonical = [_canonical_row(row) for row in rows]
            all_rows.extend(canonical)
            if expected.split:
                split_rows[expected.split] = set(canonical)

    actual_records = sum(row_counts.values())
    if actual_records != specification.expected_records:
        raise IntegrityError(
            f"total row count mismatch: {actual_records} != {specification.expected_records}"
        )

    duplicate_rows = len(all_rows) - len(set(all_rows))
    overlap = len(split_rows.get("train", set()) & split_rows.get("test", set()))
    physical_duplicates = sum(count - 1 for count in seen_hashes.values() if count > 1)
    return IntegrityResult(
        source_id=specification.source_id,
        files=tuple(digests),
        row_counts=row_counts,
        schema_fingerprints=fingerprints,
        physical_duplicate_count=physical_duplicates,
        duplicate_row_count=duplicate_rows,
        split_overlap_count=overlap,
        findings=tuple(sorted(findings)),
    )
