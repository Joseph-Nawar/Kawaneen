"""Typed canonical documents, units, fragments, and provenance."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class UnitType(StrEnum):
    FACTS = "facts"
    COURT_REASONING = "court_reasoning"
    APPLICABLE_LAWS = "applicable_laws"
    VERDICT = "verdict"
    CASE_TEXT = "case_text"
    EVENTS = "events"
    REASONING = "reasoning"
    RULING = "ruling"
    ARTICLE_FRAGMENT = "article_fragment"
    ARTICLE = "article"


class ReconstructionStatus(StrEnum):
    UNIQUE = "unique"
    EXPLICIT_FRAGMENT_SERIES = "explicit_fragment_series"
    CONTINUATION_CANDIDATE = "continuation_candidate"
    CONFLICTING_DUPLICATE = "conflicting_duplicate"
    UNRESOLVED = "unresolved"


class ArticleParseConfidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNRESOLVED = "unresolved"


class SourceProvenance(BaseModel):
    """Exact location of a canonical value in an immutable Phase 2 artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    source_id: str = Field(min_length=1)
    source_version: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    source_row: int = Field(ge=1)
    source_field: str = Field(min_length=1)
    split: str = ""


class CanonicalUnit(BaseModel):
    """A source-derived canonical unit with untouched text and exact provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    unit_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    unit_type: UnitType
    text: str
    provenance: SourceProvenance
    ordinal: int | None = Field(default=None, ge=1)


class CanonicalCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["case"]
    document_id: str = Field(min_length=1)
    title: str = ""
    provenance: SourceProvenance
    split: str = ""
    source_metadata: dict[str, str] = Field(default_factory=dict)


class CanonicalStatute(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["statute"]
    document_id: str = Field(min_length=1)
    title: str = ""
    provenance: SourceProvenance
    reconstruction_status: ReconstructionStatus
    raw_article_label: str = ""
    derived_article_ordinal: int | None = Field(default=None, ge=1)


CanonicalDocument = Annotated[CanonicalCase | CanonicalStatute, Field(discriminator="kind")]


class SourceFragment(BaseModel):
    """Immutable statutory source row retained before any reconstruction."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    fragment_id: str = Field(min_length=1)
    provenance: SourceProvenance
    raw_label: str
    law_name: str = ""
    law_type: str = ""
    derived_article_ordinal: int | None = Field(default=None, ge=1)
    explicit_part: int | None = Field(default=None, ge=1)
    article_label_structural_key: str | None = None
    article_parse_confidence: ArticleParseConfidence = ArticleParseConfidence.UNRESOLVED
    article_status_marker: str | None = None
    part_index: int | None = Field(default=None, ge=1)
    unit_type: Literal[UnitType.ARTICLE_FRAGMENT]
    text: str


class ReconstructionGroup(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    law_name: str
    raw_article_label: str
    status: ReconstructionStatus
    fragment_ids: tuple[str, ...]
    operations: tuple[str, ...] = ()


class RawAccounting(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str
    expected_records: int = Field(ge=0)
    accounted_records: int = Field(ge=0)
    canonical_documents: int = Field(ge=0)
    canonical_units: int = Field(ge=0)
    excluded_records: int = Field(ge=0)
    error_records: int = Field(ge=0)
