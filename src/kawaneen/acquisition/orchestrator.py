"""Offline-first orchestration for gated acquisition and inspection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kawaneen.acquisition.adapters import (
    AdapterError,
    HuggingFaceAdapter,
    LocalFileAdapter,
)
from kawaneen.acquisition.integrity import verify_specification
from kawaneen.acquisition.manifests import build_manifests, validate_manifests
from kawaneen.acquisition.models import AcquisitionOperation, AcquisitionPurpose
from kawaneen.acquisition.policy import authorize_source
from kawaneen.acquisition.privacy import (
    screen_privacy,
    summarize_privacy,
    write_private_review_bundle,
)
from kawaneen.acquisition.specs import load_specifications
from kawaneen.acquisition.statutory import audit_statutory_quality, write_statutory_summary
from kawaneen.acquisition.storage import clean_partials, source_root

DEFAULT_RAW_ROOT = Path("data/raw")
DEFAULT_MANIFEST_ROOT = Path("data/manifests")
DEFAULT_PRIVATE_ROOT = Path("artifacts/private")


def _spec(source_id: str):
    specifications = load_specifications()
    try:
        return specifications[source_id]
    except KeyError as exc:
        raise ValueError(f"no acquisition specification for source: {source_id}") from exc


def plan() -> list[dict[str, Any]]:
    """Return deterministic source authorization plans without network access."""

    rows: list[dict[str, Any]] = []
    for source_id, specification in sorted(load_specifications().items()):
        rows.append(
            {
                "source_id": source_id,
                "version": specification.version,
                "identifier": specification.identifier,
                "revision": specification.revision,
                "allowed_purposes": [purpose.value for purpose in specification.allowed_purposes],
                "files": [item.path for item in specification.files],
                "decision": "permitted_source"
                if source_id in {"alarb", "arabiccr", "saudi-moj-derived"}
                else "not_specified",
            }
        )
    return rows


def acquire_source(
    source_id: str,
    purpose: AcquisitionPurpose,
    raw_root: Path = DEFAULT_RAW_ROOT,
) -> tuple[Any, ...]:
    """Acquire only through a source-specific authorized adapter."""

    authorization = authorize_source(source_id, AcquisitionOperation.ACQUIRE, purpose)
    if not authorization.allowed:
        raise PermissionError(authorization.reason)
    specification = _spec(source_id)
    clean_partials(raw_root)
    if specification.provider == "huggingface":
        return HuggingFaceAdapter().acquire(specification, raw_root)
    raise AdapterError(
        "no stable public API adapter is configured for this provider; use import-local"
    )


def import_local(
    source_id: str,
    file_path: Path,
    purpose: AcquisitionPurpose,
    raw_root: Path = DEFAULT_RAW_ROOT,
) -> tuple[Any, ...]:
    """Import one local file only after source-specific policy authorization."""

    authorization = authorize_source(source_id, AcquisitionOperation.IMPORT_LOCAL, purpose)
    if not authorization.allowed:
        raise PermissionError(authorization.reason)
    specification = _spec(source_id)
    clean_partials(raw_root)
    return LocalFileAdapter().import_file(specification, file_path, raw_root)


def verify_source(source_id: str, purpose: AcquisitionPurpose = AcquisitionPurpose.INTEGRITY):
    """Verify installed raw files against the version-controlled specification."""

    authorization = authorize_source(source_id, AcquisitionOperation.VERIFY, purpose)
    if not authorization.allowed:
        raise PermissionError(authorization.reason)
    specification = _spec(source_id)
    root = source_root(DEFAULT_RAW_ROOT, source_id, specification.version)
    return verify_specification(specification, root)


def audit_source(
    source_id: str, purpose: AcquisitionPurpose = AcquisitionPurpose.PRIVACY_INSPECTION
):
    """Run masked local privacy screening and write only an ignored review bundle."""

    authorization = authorize_source(source_id, AcquisitionOperation.AUDIT, purpose)
    if not authorization.allowed:
        raise PermissionError(authorization.reason)
    specification = _spec(source_id)
    root = source_root(DEFAULT_RAW_ROOT, source_id, specification.version)
    result = screen_privacy(specification, root)
    write_private_review_bundle(result, DEFAULT_PRIVATE_ROOT)
    return result


def build_manifest(
    source_id: str, purpose: AcquisitionPurpose = AcquisitionPurpose.INTEGRITY
) -> None:
    """Verify, audit, and update deterministic manifests for one source."""

    authorization = authorize_source(source_id, AcquisitionOperation.MANIFEST_BUILD, purpose)
    if not authorization.allowed:
        raise PermissionError(authorization.reason)
    specification = _spec(source_id)
    integrity = verify_specification(
        specification, source_root(DEFAULT_RAW_ROOT, source_id, specification.version)
    )
    privacy = audit_source(source_id, AcquisitionPurpose.PRIVACY_INSPECTION)
    privacy_summary = summarize_privacy(privacy)
    build_manifests(
        integrity,
        privacy,
        privacy_summary,
        specification.version,
        purpose.value,
        DEFAULT_MANIFEST_ROOT,
        specification.canonical_source,
        specification.acquisition_method,
    )


def audit_statutory(source_id: str):
    """Run a counts-only statutory quality audit for a local Parquet seed."""

    authorization = authorize_source(
        source_id, AcquisitionOperation.AUDIT, AcquisitionPurpose.INTEGRITY
    )
    if not authorization.allowed:
        raise PermissionError(authorization.reason)
    specification = _spec(source_id)
    result = audit_statutory_quality(
        specification, source_root(DEFAULT_RAW_ROOT, source_id, specification.version)
    )
    write_statutory_summary(result, DEFAULT_MANIFEST_ROOT)
    return result


def status() -> list[dict[str, Any]]:
    """Return deterministic local raw availability status."""

    result: list[dict[str, Any]] = []
    for source_id, specification in sorted(load_specifications().items()):
        root = source_root(DEFAULT_RAW_ROOT, source_id, specification.version)
        result.append(
            {
                "source_id": source_id,
                "version": specification.version,
                "raw_present": root.is_dir(),
                "expected_files": len(specification.files),
            }
        )
    return result


def validate_manifest() -> None:
    """Validate the version-controlled manifest set."""

    validate_manifests(DEFAULT_MANIFEST_ROOT)


def rebuild_auto() -> list[dict[str, Any]]:
    """Rebuild only derived manifests for already-installed permitted raw data."""

    rows: list[dict[str, Any]] = []
    for source_id in sorted(load_specifications()):
        root = source_root(DEFAULT_RAW_ROOT, source_id, _spec(source_id).version)
        if root.is_dir():
            build_manifest(source_id)
            rows.append({"source_id": source_id, "rebuilt": True})
    return rows
