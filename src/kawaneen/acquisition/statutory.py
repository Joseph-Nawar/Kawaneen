"""Sanitized quality metrics for article-level statutory seed datasets."""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, cast

import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field

from kawaneen.acquisition.models import SourceSpecification


class StatutoryQualityResult(BaseModel):
    """Counts-only statutory quality report; never stores article text."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    source_id: str
    total_rows: int = Field(ge=0)
    unique_rows: int = Field(ge=0)
    unique_law_names: int = Field(ge=0)
    law_type_counts: dict[str, int] = Field(default_factory=dict)
    unique_article_labels: int = Field(ge=0)
    missing_or_empty_fields: dict[str, int] = Field(default_factory=dict)
    exact_duplicate_rows: int = Field(ge=0)
    duplicate_law_article_keys: int = Field(ge=0)
    extremely_short_records: int = Field(ge=0)
    extremely_long_records: int = Field(ge=0)
    likely_malformed_or_reversed_arabic: int = Field(ge=0)
    suspicious_character_sequences: int = Field(ge=0)
    likely_merged_word_or_ocr_artifacts: int = Field(ge=0)
    field_availability: dict[str, bool] = Field(default_factory=dict)
    schema_columns: tuple[str, ...] = ()


_ARABIC = re.compile(r"[\u0600-\u06ff]")
_LETTERS = re.compile(r"[\w\u0600-\u06ff]", re.UNICODE)
_SUSPICIOUS = re.compile(r"[\ufffd\x00\u202a-\u202e]|(?:\.{4,}|-{4,}|_{4,})")
_OCR = re.compile(r"ـ{2,}|[A-Za-z]{2,}[\u0600-\u06ff]|[\u0600-\u06ff][A-Za-z]{2,}")


def _rows(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    table: Any = cast(Any, pq.read_table)(path)  # pyright: ignore[reportUnknownMemberType]
    return list(table.column_names), table.to_pylist()


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(_text(item) for item in cast(list[Any], value))
    return str(value)


def audit_statutory_quality(
    specification: SourceSpecification, root: Path
) -> StatutoryQualityResult:
    """Compute deterministic quality counts for the first Parquet data file."""

    expected = next((item for item in specification.files if item.format == "parquet"), None)
    if expected is None:
        raise ValueError("statutory quality audit requires a Parquet data file")
    path = root.joinpath(*Path(expected.path).parts)
    columns, rows = _rows(path)
    required = ("text", "law_name", "law_type", "article_number")
    missing = {
        column: sum(not _text(row.get(column)).strip() for row in rows)
        for column in required
        if column in columns
    }
    for column in required:
        if column not in columns:
            missing[column] = len(rows)
    text_values = [_text(row.get("text")) for row in rows]
    canonical_rows = [
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
        for row in rows
    ]
    law_names = [
        _text(row.get("law_name")).strip() for row in rows if _text(row.get("law_name")).strip()
    ]
    law_types = [
        _text(row.get("law_type")).strip() for row in rows if _text(row.get("law_type")).strip()
    ]
    article_labels = [
        _text(row.get("article_number")).strip()
        for row in rows
        if _text(row.get("article_number")).strip()
    ]
    keys = [
        (_text(row.get("law_name")).strip(), _text(row.get("article_number")).strip())
        for row in rows
        if _text(row.get("law_name")).strip() and _text(row.get("article_number")).strip()
    ]
    key_counts = Counter(keys)
    short = sum(0 < len(value) < 20 for value in text_values)
    long = sum(len(value) > 10000 for value in text_values)
    malformed = sum(
        bool(value) and (len(_ARABIC.findall(value)) / max(1, len(_LETTERS.findall(value))) < 0.2)
        for value in text_values
    )
    suspicious = sum(bool(_SUSPICIOUS.search(value)) for value in text_values)
    ocr = sum(bool(_OCR.search(value)) for value in text_values)
    availability = {
        field: field in columns
        for field in (
            "judgment_date",
            "publication_date",
            "effective_date",
            "status",
            "amendments",
            "details_url",
            "official_url",
        )
    }
    return StatutoryQualityResult(
        source_id=specification.source_id,
        total_rows=len(rows),
        unique_rows=len(set(canonical_rows)),
        unique_law_names=len(set(law_names)),
        law_type_counts=dict(sorted(Counter(law_types).items())),
        unique_article_labels=len(set(article_labels)),
        missing_or_empty_fields=dict(sorted(missing.items())),
        exact_duplicate_rows=len(rows) - len(set(canonical_rows)),
        duplicate_law_article_keys=sum(count - 1 for count in key_counts.values() if count > 1),
        extremely_short_records=short,
        extremely_long_records=long,
        likely_malformed_or_reversed_arabic=malformed,
        suspicious_character_sequences=suspicious,
        likely_merged_word_or_ocr_artifacts=ocr,
        field_availability=availability,
        schema_columns=tuple(columns),
    )


def write_statutory_summary(
    result: StatutoryQualityResult, directory: Path = Path("data/manifests")
) -> None:
    """Merge one counts-only quality result deterministically."""

    path = directory / "statutory_quality_summaries.json"
    summaries: list[dict[str, Any]] = []
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            summaries = cast(list[dict[str, Any]], payload)
    summaries = [item for item in summaries if item.get("source_id") != result.source_id]
    summaries.append(result.model_dump())
    partial = path.with_name(f"{path.name}.partial")
    directory.mkdir(parents=True, exist_ok=True)
    partial.write_text(
        json.dumps(
            sorted(summaries, key=lambda item: item["source_id"]), ensure_ascii=False, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)
