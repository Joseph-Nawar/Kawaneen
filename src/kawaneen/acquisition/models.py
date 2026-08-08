"""Typed contracts for gated acquisition and deterministic inspection."""

from __future__ import annotations

from enum import StrEnum
from pathlib import PurePosixPath

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AcquisitionPurpose(StrEnum):
    EVALUATION = "evaluation"
    INTEGRITY = "integrity"
    DUPLICATE_ANALYSIS = "duplicate_analysis"
    PRIVACY_INSPECTION = "privacy_inspection"
    LOCAL_RESEARCH = "local_research"
    LOCAL_PARSING = "local_parsing"
    INSPECTION = "inspection"
    TRAINING = "training"
    PUBLISHING = "publishing"
    PUBLIC_DISPLAY = "public_display"
    PUBLIC_DEMO = "public_demo"


class AcquisitionOperation(StrEnum):
    ACQUIRE = "acquire"
    IMPORT_LOCAL = "import_local"
    VERIFY = "verify"
    AUDIT = "audit"
    MANIFEST_BUILD = "manifest_build"
    MANIFEST_VALIDATE = "manifest_validate"
    STATUS = "status"
    REBUILD = "rebuild"
    PARSE = "parse"
    TRAIN = "train"
    PUBLISH = "publish"
    PUBLIC_DISPLAY = "public_display"
    PUBLIC_DEMO = "public_demo"


class FileExpectation(BaseModel):
    """Expected source file metadata without embedding source content."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    path: str = Field(min_length=1)
    format: str = Field(min_length=1)
    expected_columns: tuple[str, ...] = ()
    expected_records: int | None = Field(default=None, ge=0)
    split: str = ""

    @field_validator("path")
    @classmethod
    def require_relative_path(cls, value: str) -> str:
        candidate = PurePosixPath(value)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("file paths must be relative and cannot escape the specification")
        return value


class SourceSpecification(BaseModel):
    """Version-controlled acquisition contract for one permitted source."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    schema_version: int = Field(ge=1)
    source_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    identifier: str = Field(min_length=1)
    licence: str = Field(min_length=1)
    expected_records: int = Field(ge=1)
    expected_splits: dict[str, int] = Field(default_factory=dict)
    allowed_purposes: tuple[AcquisitionPurpose, ...] = ()
    files: tuple[FileExpectation, ...] = ()
    no_modeling_split: bool = False
    canonical_source: str = "unspecified"
    acquisition_method: str = "unspecified"
    notes: str = ""

    @field_validator("source_id")
    @classmethod
    def source_id_is_stable(cls, value: str) -> str:
        if any(character.isspace() for character in value) or "/" in value:
            raise ValueError("source_id must be a stable path-safe identifier")
        return value

    @field_validator("expected_splits")
    @classmethod
    def split_counts_are_positive(cls, value: dict[str, int]) -> dict[str, int]:
        if any(not name or count < 0 for name, count in value.items()):
            raise ValueError("split names must be non-empty and counts cannot be negative")
        return value


class Authorization(BaseModel):
    """The policy engine's explicit allow/deny result."""

    model_config = ConfigDict(frozen=True)

    allowed: bool
    source_id: str
    operation: AcquisitionOperation
    purpose: AcquisitionPurpose
    reason: str = Field(min_length=1)


class FileDigest(BaseModel):
    """Hash and size of one immutable local file."""

    model_config = ConfigDict(frozen=True)

    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size: int = Field(ge=1)


class IntegrityResult(BaseModel):
    """Sanitized integrity and duplicate counts."""

    schema_version: int = 1
    source_id: str
    files: tuple[FileDigest, ...] = ()
    row_counts: dict[str, int] = Field(default_factory=dict)
    schema_fingerprints: dict[str, str] = Field(default_factory=dict)
    physical_duplicate_count: int = Field(ge=0)
    duplicate_row_count: int = Field(ge=0)
    split_overlap_count: int = Field(ge=0)
    findings: tuple[str, ...] = ()


class PrivacyFinding(BaseModel):
    """Masked, non-legal privacy screening result."""

    model_config = ConfigDict(frozen=True)

    detector: str
    column: str
    file_path: str
    row_number: int = Field(ge=1)
    masked_value: str = Field(min_length=1)


class PrivacyResult(BaseModel):
    """Deterministic privacy-screen summary; never a legal clearance."""

    schema_version: int = 1
    source_id: str
    finding_count: int = Field(ge=0)
    findings: tuple[PrivacyFinding, ...] = ()
    legal_clearance: bool = False
    review_status: str = "pending_manual_review"


class PrivacySummary(BaseModel):
    """Sanitized, deterministic aggregate of a privacy screen."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    source_id: str
    findings_by_detector: dict[str, int] = Field(default_factory=dict)
    affected_record_count: int = Field(ge=0)
    findings_by_column: dict[str, int] = Field(default_factory=dict)
    deterministic_review_sample_size: int = Field(ge=0)
    confirmed_pii_count: int | None = Field(default=None, ge=0)
    likely_false_positive_count: int | None = Field(default=None, ge=0)
    unresolved_categories: tuple[str, ...] = ()
    manual_review_status: str = "pending_manual_review"


class StageEligibility(BaseModel):
    """Separate legal and use-stage decisions for one acquired snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    legal_clearance: bool = False
    authorized_for_local_parsing: bool = False
    authorized_for_evaluation: bool = False
    authorized_for_training: bool = False
    authorized_for_public_display: bool = False
