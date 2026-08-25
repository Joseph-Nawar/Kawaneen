"""Strict, source-grounded Phase 11A extraction contracts."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kawaneen.corpus.models import SourceProvenance


class ExtractionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=False)


class Modality(StrEnum):
    OBLIGATION = "obligation"
    PROHIBITION = "prohibition"
    PERMISSION = "permission"


class CandidateType(StrEnum):
    TEMPORAL = "temporal"
    MONETARY = "monetary"
    PERCENTAGE = "percentage"
    ARTICLE = "article"
    REGULATION = "regulation"


class NormalizationStatus(StrEnum):
    NORMALIZED = "normalized"
    PARTIAL = "partial"
    UNRESOLVED = "unresolved"


class Calendar(StrEnum):
    GREGORIAN = "gregorian"
    HIJRI = "hijri"
    NONE = "none"


class ProvenanceOrigin(StrEnum):
    METADATA = "metadata"
    DETERMINISTIC = "deterministic"
    LLM_SELECTED = "llm_selected"


class ExactSourceSpan(ExtractionModel):
    """A substring whose offsets are measured in canonical Python codepoints."""

    text: str = Field(min_length=1)
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    canonical_unit_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_range(self) -> ExactSourceSpan:
        if self.end_char <= self.start_char:
            raise ValueError("source span end must be greater than start")
        return self


class NormalizedRepresentation(ExtractionModel):
    calendar: Calendar = Calendar.NONE
    original_components: tuple[tuple[str, str], ...] = ()
    normalized_components: tuple[tuple[str, str], ...] = ()
    normalized_value: str | None = None


class Candidate(ExtractionModel):
    candidate_id: str = Field(pattern=r"^(T|M|P|A|R)[0-9]{3}$")
    candidate_type: CandidateType
    span: ExactSourceSpan
    raw_exact_text: str = Field(min_length=1)
    normalized: NormalizedRepresentation
    normalization_status: NormalizationStatus

    @model_validator(mode="after")
    def validate_candidate_text(self) -> Candidate:
        if self.raw_exact_text != self.span.text:
            raise ValueError("candidate raw text must equal its exact source span")
        expected = {
            CandidateType.TEMPORAL: "T",
            CandidateType.MONETARY: "M",
            CandidateType.PERCENTAGE: "P",
            CandidateType.ARTICLE: "A",
            CandidateType.REGULATION: "R",
        }[self.candidate_type]
        if not self.candidate_id.startswith(expected):
            raise ValueError("candidate ID prefix does not match candidate type")
        return self


class CandidateRegistry(ExtractionModel):
    canonical_text: str
    canonical_unit_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    candidates: tuple[Candidate, ...] = ()

    @model_validator(mode="after")
    def validate_order_and_uniqueness(self) -> CandidateRegistry:
        seen_ids: set[str] = set()
        seen_spans: set[tuple[int, int]] = set()
        for candidate in self.candidates:
            if candidate.candidate_id in seen_ids:
                raise ValueError("candidate IDs must be unique")
            key = (candidate.span.start_char, candidate.span.end_char)
            if key in seen_spans:
                raise ValueError("candidate spans must be deduplicated")
            seen_ids.add(candidate.candidate_id)
            seen_spans.add(key)
        if tuple(sorted(self.candidates, key=lambda item: item.span.start_char)) != self.candidates:
            raise ValueError("candidate ordering must follow source order")
        return self


class NormativeRule(ExtractionModel):
    modality: Modality
    actor_span: ExactSourceSpan | None = None
    action_span: ExactSourceSpan
    condition_spans: tuple[ExactSourceSpan, ...] = ()
    exception_spans: tuple[ExactSourceSpan, ...] = ()
    deadline_refs: tuple[str, ...] = ()
    monetary_threshold_refs: tuple[str, ...] = ()
    percentage_threshold_refs: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_candidate_prefixes(self) -> NormativeRule:
        for value in self.deadline_refs:
            if not re.fullmatch(r"T[0-9]{3}", value):
                raise ValueError("invalid deadline candidate reference")
        for value in self.monetary_threshold_refs:
            if not re.fullmatch(r"M[0-9]{3}", value):
                raise ValueError("invalid monetary candidate reference")
        for value in self.percentage_threshold_refs:
            if not re.fullmatch(r"P[0-9]{3}", value):
                raise ValueError("invalid percentage candidate reference")
        return self


class FieldProvenance(ExtractionModel):
    field_name: str = Field(min_length=1)
    origin: ProvenanceOrigin
    source_ids: tuple[str, ...] = ()
    detail: str = ""


class ValidationDiagnostic(ExtractionModel):
    code: str = Field(min_length=1)
    field_name: str = Field(min_length=1)
    message: str = Field(min_length=1)
    accepted: bool = False


class ValidationMetadata(ExtractionModel):
    raw_provider_schema_valid: bool = True
    proposal_valid: bool = True
    diagnostics: tuple[ValidationDiagnostic, ...] = ()


class ProposedSpan(ExtractionModel):
    """Provider-side exact text only; resolution adds canonical metadata later."""

    text: str = Field(min_length=1)
    occurrence: int | None = Field(default=None, ge=0)


class ProposedRule(ExtractionModel):
    modality: Modality
    actor: ProposedSpan | None = None
    action: ProposedSpan
    conditions: tuple[ProposedSpan, ...] = ()
    exceptions: tuple[ProposedSpan, ...] = ()
    deadline_refs: tuple[str, ...] = ()
    effective_date_refs: tuple[str, ...] = ()
    monetary_threshold_refs: tuple[str, ...] = ()
    percentage_threshold_refs: tuple[str, ...] = ()


class SemanticProposal(ExtractionModel):
    """The only data a semantic provider may contribute."""

    schema_version: Literal["phase11-proposal-v1"]
    regulated_entities: tuple[ProposedSpan, ...] = ()
    rules: tuple[ProposedRule, ...] = ()
    exceptions: tuple[ProposedSpan, ...] = ()
    penalties: tuple[ProposedSpan, ...] = ()
    deadline_refs: tuple[str, ...] = ()
    effective_date_refs: tuple[str, ...] = ()
    monetary_threshold_refs: tuple[str, ...] = ()
    percentage_threshold_refs: tuple[str, ...] = ()


class ExtractionResult(ExtractionModel):
    schema_version: Literal["phase11-extraction-v1"]
    extractor_version: str = Field(min_length=1)
    configuration: Literal["deterministic-v1", "hybrid-qwen-v1"]
    jurisdiction: Literal["SA"]
    source_provenance: SourceProvenance
    source_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    issuing_authority: str | None = None
    candidate_registry: CandidateRegistry | None = None
    regulated_entities: tuple[ExactSourceSpan, ...] = ()
    obligations: tuple[NormativeRule, ...] = ()
    prohibitions: tuple[NormativeRule, ...] = ()
    permissions: tuple[NormativeRule, ...] = ()
    rules: tuple[NormativeRule, ...] = ()
    deadlines: tuple[Candidate, ...] = ()
    effective_dates: tuple[Candidate, ...] = ()
    penalties: tuple[ExactSourceSpan, ...] = ()
    monetary_thresholds: tuple[Candidate, ...] = ()
    percentage_thresholds: tuple[Candidate, ...] = ()
    exceptions: tuple[ExactSourceSpan, ...] = ()
    referenced_articles: tuple[Candidate, ...] = ()
    referenced_regulations: tuple[Candidate, ...] = ()
    validation_metadata: ValidationMetadata = ValidationMetadata()
    field_provenance: tuple[FieldProvenance, ...] = ()

    @model_validator(mode="after")
    def validate_rule_groups(self) -> ExtractionResult:
        if self.rules != self.obligations + self.prohibitions + self.permissions:
            raise ValueError("rule groups must exactly partition rules by modality")
        return self
