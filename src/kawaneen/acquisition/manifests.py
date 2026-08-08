"""Deterministic version-controlled acquisition manifest helpers."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field

from kawaneen.acquisition.models import (
    IntegrityResult,
    PrivacyResult,
    PrivacySummary,
    StageEligibility,
)


class ManifestError(ValueError):
    """Raised for malformed or unsafe manifests."""


class DatasetSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    source_id: str
    version: str
    purpose: str
    record_count: int = Field(ge=0)
    modeling_split: str
    canonical_source: str = "unspecified"
    acquisition_method: str = "unspecified"
    stage_eligibility: StageEligibility = Field(default_factory=StageEligibility)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial")
    try:
        partial.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(partial, path)
    except Exception:
        if partial.exists():
            partial.unlink()
        raise


def build_manifests(
    result: IntegrityResult,
    privacy: PrivacyResult,
    privacy_summary: PrivacySummary,
    source_version: str,
    purpose: str,
    directory: Path = Path("data/manifests"),
    canonical_source: str = "unspecified",
    acquisition_method: str = "unspecified",
) -> None:
    """Write all requested manifests with deterministic ordering."""

    files = [item.model_dump() for item in sorted(result.files, key=lambda item: item.path)]
    lock_path = directory / "acquisition_lock.json"
    existing_lock = _read_json(lock_path, {})
    existing_sources = existing_lock.get("sources", [])
    if not existing_sources and existing_lock.get("source_id"):
        existing_sources = [
            {
                "source_id": existing_lock["source_id"],
                "version": existing_lock["version"],
                "files": existing_lock.get("files", []),
            }
        ]
    sources = [source for source in existing_sources if source.get("source_id") != result.source_id]
    sources.append(
        {
            "source_id": result.source_id,
            "version": source_version,
            "canonical_source": canonical_source,
            "acquisition_method": acquisition_method,
            "files": files,
        }
    )
    sources.sort(key=lambda source: (source["source_id"], source["version"]))
    lock = {"schema_version": 2, "sources": sources}
    _atomic_json(directory / "acquisition_lock.json", lock)
    snapshots = _read_json(directory / "dataset_snapshots.json", [])
    snapshots = [item for item in snapshots if item.get("source_id") != result.source_id]
    _atomic_json(
        directory / "dataset_snapshots.json",
        sorted(
            [
                *snapshots,
                DatasetSnapshot(
                    source_id=result.source_id,
                    version=source_version,
                    purpose=purpose,
                    record_count=sum(result.row_counts.values()),
                    modeling_split="official" if result.source_id == "alarb" else "none_approved",
                    canonical_source=canonical_source,
                    acquisition_method=acquisition_method,
                    stage_eligibility=StageEligibility(
                        legal_clearance=False,
                        authorized_for_local_parsing=True,
                        authorized_for_evaluation=result.source_id in {"alarb", "arabiccr"},
                        authorized_for_training=False,
                        authorized_for_public_display=False,
                    ),
                ).model_dump(),
            ],
            key=lambda item: (item["source_id"], item["version"]),
        ),
    )
    _write_privacy_manifest(
        directory / "privacy_review_status.csv", privacy, privacy_summary, directory
    )
    _write_raw_manifest(directory / "raw_file_manifest.csv", result, directory)


def _read_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_raw_manifest(path: Path, result: IntegrityResult, directory: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, Any]] = []
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            existing = list(reader)
    normalized: list[dict[str, Any]] = []
    for item in existing:
        if "source_id" not in item:
            item["source_id"] = result.source_id
            item["version"] = "unknown"
        normalized.append(item)
    normalized = [item for item in normalized if item.get("source_id") != result.source_id]
    rows = [
        {
            "schema_version": 2,
            "source_id": result.source_id,
            "version": _source_version_from_lock(directory, result.source_id),
            **item.model_dump(),
        }
        for item in sorted(result.files, key=lambda item: item.path)
    ]
    all_rows: list[dict[str, Any]] = normalized + rows
    all_rows.sort(key=lambda item: (item["source_id"], item["version"], item["path"]))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["schema_version", "source_id", "version", "path", "size", "sha256"]
        )
        writer.writeheader()
        writer.writerows(all_rows)


def _source_version_from_lock(directory: Path, source_id: str) -> str:
    lock = _read_json(directory / "acquisition_lock.json", {})
    for source in lock.get("sources", []):
        if source.get("source_id") == source_id:
            return str(source["version"])
    return "unknown"


def _write_privacy_manifest(
    path: Path, privacy: PrivacyResult, summary: PrivacySummary, directory: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, Any]] = []
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))
    existing = [item for item in existing if item.get("source_id") != privacy.source_id]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "schema_version",
                "source_id",
                "status",
                "finding_count",
                "affected_record_count",
                "review_sample_size",
                "confirmed_pii_count",
                "likely_false_positive_count",
                "legal_clearance",
            ],
        )
        writer.writeheader()
        rows: list[dict[str, Any]] = [
            *existing,
            {
                "schema_version": 1,
                "source_id": privacy.source_id,
                "status": privacy.review_status,
                "finding_count": privacy.finding_count,
                "affected_record_count": summary.affected_record_count,
                "review_sample_size": summary.deterministic_review_sample_size,
                "confirmed_pii_count": summary.confirmed_pii_count
                if summary.confirmed_pii_count is not None
                else "not_reviewed",
                "likely_false_positive_count": summary.likely_false_positive_count
                if summary.likely_false_positive_count is not None
                else "not_reviewed",
                "legal_clearance": privacy.legal_clearance,
            },
        ]
        rows.sort(key=lambda item: item["source_id"])
        writer.writerows(rows)
    summaries = _read_json(directory / "privacy_summaries.json", [])
    summaries = [item for item in summaries if item.get("source_id") != summary.source_id]
    summaries.append(summary.model_dump())
    _atomic_json(
        directory / "privacy_summaries.json", sorted(summaries, key=lambda item: item["source_id"])
    )


def validate_manifests(directory: Path = Path("data/manifests")) -> None:
    """Validate required JSON/CSV manifest presence and basic schema fields."""

    for filename in ("acquisition_lock.json", "dataset_snapshots.json"):
        path = directory / filename
        if not path.is_file():
            raise ManifestError(f"missing manifest: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if filename == "acquisition_lock.json" and not isinstance(payload, dict):
            raise ManifestError(f"acquisition lock must be a JSON object: {path}")
        if filename == "dataset_snapshots.json" and not isinstance(payload, list):
            raise ManifestError(f"dataset snapshots must be a JSON list: {path}")
        if filename == "acquisition_lock.json":
            lock_payload = cast(dict[str, Any], payload)
            sources_payload = cast(list[dict[str, Any]], lock_payload.get("sources", []))
            for source in sources_payload:
                if not source.get("source_id") or not source.get("version"):
                    raise ManifestError("acquisition lock has an incomplete source identity")
                for item in source.get("files", []):
                    _validate_manifest_path(item.get("path", ""))
    raw_manifest = directory / "raw_file_manifest.csv"
    if not raw_manifest.is_file():
        raise ManifestError(f"missing manifest: {raw_manifest}")
    with raw_manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"schema_version", "source_id", "version", "path", "size", "sha256"}
        if not required.issubset(reader.fieldnames or set()):
            raise ManifestError("raw file manifest has an invalid header")
        for row in reader:
            _validate_manifest_path(row.get("path", ""))
    privacy_manifest = directory / "privacy_review_status.csv"
    if not privacy_manifest.is_file():
        raise ManifestError(f"missing manifest: {privacy_manifest}")
    with privacy_manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "schema_version",
            "source_id",
            "status",
            "finding_count",
            "affected_record_count",
            "review_sample_size",
            "confirmed_pii_count",
            "likely_false_positive_count",
            "legal_clearance",
        }
        if not required.issubset(reader.fieldnames or set()):
            raise ManifestError("privacy review manifest has an invalid header")
    summary_manifest = directory / "privacy_summaries.json"
    if not summary_manifest.is_file() or not isinstance(_read_json(summary_manifest, None), list):
        raise ManifestError(f"missing or invalid manifest: {summary_manifest}")


def _validate_manifest_path(value: str) -> None:
    candidate = PurePosixPath(value)
    if not value or candidate.is_absolute() or ".." in candidate.parts:
        raise ManifestError("manifest paths must be repository-relative")
