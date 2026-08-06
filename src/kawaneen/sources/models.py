"""Typed source records and fail-closed cross-field policy rules."""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class PermissionState(StrEnum):
    YES = "yes"
    NO = "no"
    CONDITIONAL = "conditional"
    UNKNOWN = "unknown"


class LicenceStatus(StrEnum):
    CONFIRMED = "confirmed"
    CONDITIONAL = "conditional"
    MISSING = "missing"
    PAPER_ONLY = "paper_only"
    CONFLICTING = "conflicting"
    NOT_APPLICABLE = "not_applicable"
    UNKNOWN = "unknown"


class Jurisdiction(StrEnum):
    SAUDI_ARABIA = "Saudi Arabia"
    EGYPT = "Egypt"
    UAE = "UAE"
    MULTI_JURISDICTION = "multi-jurisdiction"
    UNKNOWN = "unknown"


class SourceRole(StrEnum):
    PRIMARY_CORPUS = "primary_corpus"
    BENCHMARK = "benchmark"
    TRAINING = "training"
    TRANSFER = "transfer"
    REFERENCE = "reference"


class AuthorityLevel(StrEnum):
    OFFICIAL = "official"
    INSTITUTIONAL = "institutional"
    ACADEMIC = "academic"
    COMMUNITY = "community"
    UNKNOWN = "unknown"


class PrivacyRisk(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"


class Decision(StrEnum):
    APPROVED = "approved"
    CONDITIONAL = "conditional"
    EVALUATION_ONLY = "evaluation_only"
    LOCAL_RESEARCH_ONLY = "local_research_only"
    METADATA_ONLY = "metadata_only"
    BLOCKED_PENDING_REVIEW = "blocked_pending_review"
    EXCLUDED = "excluded"


class SourceType(StrEnum):
    DATASET = "dataset"
    OFFICIAL_OPEN_DATA = "official_open_data"
    OFFICIAL_PORTAL = "official_portal"
    PAPER = "paper"
    CODE = "code"
    OTHER = "other"


class AccessMethod(StrEnum):
    PUBLIC_DOWNLOAD = "public_download"
    PUBLIC_WEB = "public_web"
    REPOSITORY = "repository"
    PAPER_ONLY = "paper_only"
    REQUEST = "request"
    UNKNOWN = "unknown"


class AccessStatus(StrEnum):
    PUBLIC = "public"
    GATED = "gated"
    PAPER_ONLY = "paper_only"
    RESTRICTED = "restricted"
    UNKNOWN = "unknown"


class SourceRecord(BaseModel):
    """A source candidate with explicit provenance, rights, and decision data."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    source_id: str = Field(min_length=1)
    source_name: str = Field(min_length=1)
    publisher: str = Field(min_length=1)
    original_publisher: str = Field(min_length=1)
    jurisdiction: Jurisdiction
    source_type: SourceType
    description: str = Field(min_length=1)
    task: str = Field(min_length=1)
    language: str = Field(min_length=1)
    size: str = Field(min_length=1)
    size_unit: str = Field(min_length=1)
    file_format: str = Field(min_length=1)
    content_unit: str = Field(min_length=1)
    citation: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    evidence_url: str = Field(min_length=1)
    evidence_type: str = Field(min_length=1)
    evidence_summary: str = Field(min_length=1)
    provenance: str = Field(min_length=1)
    known_quality_issues: str = Field(min_length=1)
    contains_personal_data: PermissionState
    access_status: AccessStatus
    requires_auth: PermissionState
    licence_status: LicenceStatus
    licence_name: str = ""
    licence_evidence_url: str = ""
    permission_evidence_url: str = ""
    terms_url: str = ""
    access_method: AccessMethod
    automated_access_permission: PermissionState
    dataset_licence: PermissionState
    commercial_use: PermissionState
    derivatives: PermissionState
    paper_licence: PermissionState
    code_licence: PermissionState
    original_source_rights: PermissionState
    public_display_permission: PermissionState
    model_training_permission: PermissionState
    public_demo_permission: PermissionState
    attribution_required: PermissionState
    source_role: SourceRole
    authority_level: AuthorityLevel
    privacy_risk: PrivacyRisk
    decision: Decision
    verification_date: date
    conditions: str = ""
    required_rights: str = ""
    manual_action: str = ""
    notes: str = ""
    split_info: str = ""

    @field_validator("source_id")
    @classmethod
    def source_id_is_stable(cls, value: str) -> str:
        if any(character.isspace() for character in value):
            raise ValueError("source_id must not contain whitespace")
        return value

    @field_validator(
        "source_url",
        "evidence_url",
        "licence_evidence_url",
        "permission_evidence_url",
        "terms_url",
    )
    @classmethod
    def canonical_urls_only(cls, value: str) -> str:
        if any(marker in value.lower() for marker in ("/search?", "/search/", "?q=", "?query=")):
            raise ValueError("canonical source and evidence URLs cannot be discovery/search URLs")
        return value

    @model_validator(mode="after")
    def enforce_policy(self) -> SourceRecord:
        permission_fields = (
            self.dataset_licence,
            self.commercial_use,
            self.derivatives,
            self.automated_access_permission,
            self.original_source_rights,
            self.paper_licence,
            self.code_licence,
            self.public_display_permission,
            self.model_training_permission,
            self.public_demo_permission,
            self.attribution_required,
        )
        if PermissionState.YES in permission_fields and not self.permission_evidence_url:
            raise ValueError("permission_evidence_url is required for positive permissions")

        if self.attribution_required is PermissionState.YES and not (
            self.licence_evidence_url or self.terms_url
        ):
            raise ValueError("attribution requires explicit licence or terms evidence")

        if self.dataset_licence is PermissionState.YES and self.licence_status in {
            LicenceStatus.PAPER_ONLY,
            LicenceStatus.MISSING,
            LicenceStatus.UNKNOWN,
            LicenceStatus.CONFLICTING,
        }:
            raise ValueError("paper-only or unresolved licence cannot approve dataset licence")

        conditional_permissions = (
            self.dataset_licence,
            self.commercial_use,
            self.derivatives,
            self.automated_access_permission,
            self.public_display_permission,
            self.model_training_permission,
            self.public_demo_permission,
        )
        if PermissionState.CONDITIONAL in conditional_permissions and not self.conditions:
            raise ValueError("conditions are required for conditional permissions")

        if (
            self.privacy_risk is PrivacyRisk.HIGH
            and self.public_demo_permission is PermissionState.YES
        ):
            raise ValueError("high privacy risk requires documented mitigation before public demo")

        if self.decision is Decision.APPROVED:
            if self.automated_access_permission is PermissionState.NO:
                raise ValueError("approved sources cannot prohibit automated access")
            if self.licence_status not in {LicenceStatus.CONFIRMED, LicenceStatus.NOT_APPLICABLE}:
                raise ValueError("approved sources require resolved licence status")
            unresolved = (
                self.dataset_licence,
                self.original_source_rights,
                self.automated_access_permission,
                self.public_display_permission,
                self.model_training_permission,
                self.public_demo_permission,
            )
            if PermissionState.UNKNOWN in unresolved or self.privacy_risk is PrivacyRisk.UNKNOWN:
                raise ValueError("approved sources cannot have unresolved required rights")
            if PermissionState.NO in unresolved:
                raise ValueError("approved sources cannot prohibit a required use")
            if self.required_rights:
                raise ValueError("approved sources cannot have unresolved required rights")

        if self.decision is Decision.EVALUATION_ONLY and self.source_role not in {
            SourceRole.BENCHMARK,
            SourceRole.REFERENCE,
        }:
            raise ValueError("evaluation_only sources must be benchmarks or references")

        if self.decision is Decision.BLOCKED_PENDING_REVIEW and not self.manual_action:
            raise ValueError("blocked records require a manual_action")

        return self
