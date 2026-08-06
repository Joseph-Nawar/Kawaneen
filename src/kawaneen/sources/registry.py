"""Offline CSV registry loading, validation, and deterministic summaries."""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from kawaneen.sources.models import SourceRecord

DEFAULT_REGISTRY_PATH = Path("data/manifests/source_registry.csv")


class RegistryValidationError(ValueError):
    """Raised when a registry cannot be safely loaded or validated."""


def load_registry(path: Path = DEFAULT_REGISTRY_PATH) -> list[SourceRecord]:
    """Load and validate a registry CSV without network or filesystem writes."""

    if not path.is_file():
        raise RegistryValidationError(f"registry file does not exist: {path}")

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RegistryValidationError("registry file has no header")
        records: list[SourceRecord] = []
        errors: list[str] = []
        seen_ids: set[str] = set()
        for row_number, row in enumerate(reader, start=2):
            try:
                record = SourceRecord.model_validate(row)
            except ValidationError as exc:
                errors.extend(f"row {row_number}: {error['msg']}" for error in exc.errors())
                continue
            if record.source_id in seen_ids:
                errors.append(f"row {row_number}: duplicate source_id {record.source_id}")
            seen_ids.add(record.source_id)
            records.append(record)

    if errors:
        raise RegistryValidationError("; ".join(errors))
    if not records:
        raise RegistryValidationError("registry contains no records")
    return records


def summarize_registry(records: list[SourceRecord]) -> dict[str, Any]:
    """Return JSON-compatible counts for governance and technical dimensions."""

    return {
        "source_count": len(records),
        "decisions": _counts(record.decision.value for record in records),
        "roles": _counts(record.source_role.value for record in records),
        "jurisdictions": _counts(record.jurisdiction for record in records),
        "privacy_risks": _counts(record.privacy_risk.value for record in records),
        "access_statuses": _counts(record.access_status.value for record in records),
        "file_formats": _counts(record.file_format for record in records),
    }


def _counts(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def format_summary(summary: dict[str, Any]) -> str:
    """Render a compact human-readable registry summary."""

    lines = [f"Sources: {summary['source_count']}"]
    for label in (
        "decisions",
        "roles",
        "jurisdictions",
        "privacy_risks",
        "access_statuses",
        "file_formats",
    ):
        title = label.replace("_", " ").title()
        values = ", ".join(f"{key}={value}" for key, value in summary[label].items())
        lines.append(f"{title}: {values}")
    return "\n".join(lines)
