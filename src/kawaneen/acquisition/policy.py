"""Fail-closed source and purpose authorization."""

from __future__ import annotations

from pathlib import Path

from kawaneen.acquisition.models import (
    AcquisitionOperation,
    AcquisitionPurpose,
    Authorization,
)
from kawaneen.sources.registry import RegistryValidationError, load_registry

_INSPECTION_OPERATIONS = {
    AcquisitionOperation.VERIFY,
    AcquisitionOperation.AUDIT,
    AcquisitionOperation.MANIFEST_BUILD,
    AcquisitionOperation.MANIFEST_VALIDATE,
    AcquisitionOperation.STATUS,
    AcquisitionOperation.REBUILD,
}
_DENIED_PURPOSES = {
    AcquisitionPurpose.TRAINING,
    AcquisitionPurpose.PUBLISHING,
    AcquisitionPurpose.PUBLIC_DISPLAY,
    AcquisitionPurpose.PUBLIC_DEMO,
}


def authorize_source(
    source_id: str,
    operation: AcquisitionOperation,
    purpose: AcquisitionPurpose,
    registry_path: Path = Path("data/manifests/source_registry.csv"),
) -> Authorization:
    """Authorize one operation using the Phase 1 registry and no bypass path."""

    if purpose in _DENIED_PURPOSES or operation in {
        AcquisitionOperation.TRAIN,
        AcquisitionOperation.PUBLISH,
        AcquisitionOperation.PUBLIC_DISPLAY,
        AcquisitionOperation.PUBLIC_DEMO,
    }:
        return Authorization(
            allowed=False,
            source_id=source_id,
            operation=operation,
            purpose=purpose,
            reason="training, publishing, public display, and public demo operations are denied",
        )

    try:
        records = {record.source_id: record for record in load_registry(registry_path)}
    except RegistryValidationError as exc:
        return Authorization(
            allowed=False,
            source_id=source_id,
            operation=operation,
            purpose=purpose,
            reason=f"source registry is invalid: {exc}",
        )
    record = records.get(source_id)
    if record is None:
        return Authorization(
            allowed=False,
            source_id=source_id,
            operation=operation,
            purpose=purpose,
            reason="source is absent from the Phase 1 registry",
        )

    if source_id == "alarb":
        allowed_purposes: set[AcquisitionPurpose] = {
            AcquisitionPurpose.EVALUATION,
            AcquisitionPurpose.LOCAL_PARSING,
            AcquisitionPurpose.INTEGRITY,
            AcquisitionPurpose.DUPLICATE_ANALYSIS,
            AcquisitionPurpose.PRIVACY_INSPECTION,
        }
    elif source_id == "arabiccr":
        allowed_purposes = {
            AcquisitionPurpose.LOCAL_RESEARCH,
            AcquisitionPurpose.LOCAL_PARSING,
            AcquisitionPurpose.INSPECTION,
            AcquisitionPurpose.INTEGRITY,
            AcquisitionPurpose.DUPLICATE_ANALYSIS,
            AcquisitionPurpose.PRIVACY_INSPECTION,
        }
    elif source_id == "saudi-moj-derived":
        allowed_purposes = {
            AcquisitionPurpose.LOCAL_RESEARCH,
            AcquisitionPurpose.LOCAL_PARSING,
            AcquisitionPurpose.INTEGRITY,
            AcquisitionPurpose.DUPLICATE_ANALYSIS,
            AcquisitionPurpose.PRIVACY_INSPECTION,
        }
    else:
        allowed_purposes = set()

    if source_id not in {"alarb", "arabiccr", "saudi-moj-derived"}:
        reason = "Phase 1 decision does not authorize acquisition for this source"
        allowed = False
    elif operation in _INSPECTION_OPERATIONS and purpose in allowed_purposes:
        reason = "registry-authorized local inspection operation"
        allowed = True
    elif (
        operation in {AcquisitionOperation.ACQUIRE, AcquisitionOperation.IMPORT_LOCAL}
        and purpose in allowed_purposes
    ):
        reason = "registry-authorized source-specific acquisition purpose"
        allowed = True
    elif operation is AcquisitionOperation.PARSE and purpose is AcquisitionPurpose.LOCAL_PARSING:
        reason = "registry-authorized private local parsing; public use remains denied"
        allowed = source_id in {"alarb", "arabiccr", "saudi-moj-derived"}
    else:
        reason = "operation and purpose are not authorized for this source"
        allowed = False
    return Authorization(
        allowed=allowed,
        source_id=record.source_id,
        operation=operation,
        purpose=purpose,
        reason=reason,
    )
