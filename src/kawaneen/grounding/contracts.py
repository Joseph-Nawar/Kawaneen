"""Strict immutable contracts for deterministic grounding."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator


class GroundingModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceRecord(GroundingModel):
    """Metadata resolved from the canonical corpus, never from a hit."""

    document_id: str = Field(min_length=1)
    source_id: str | None = None
    document_title: str | None = None
    jurisdiction: str | None = None
    article: str | None = None
    page: str | None = None
    source_url: str | None = None


class CanonicalSourceSpan(GroundingModel):
    unit_id: str = Field(min_length=1)
    start: int = Field(ge=0)
    end: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_order(self) -> CanonicalSourceSpan:
        if self.end <= self.start:
            raise ValueError("canonical source span end must be greater than start")
        return self


class CanonicalEvidenceUnit(GroundingModel):
    unit_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    ordinal: int | None = Field(default=None, ge=1)
    display_text: str = Field(min_length=1)
    heading_path: tuple[str, ...] = ()
    source: SourceRecord


class ResolvedChunk(GroundingModel):
    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    source_unit_ids: tuple[str, ...] = Field(min_length=1)
    source_spans: tuple[CanonicalSourceSpan, ...] = ()
    units: tuple[CanonicalEvidenceUnit, ...] = Field(min_length=1)


class RetrievalInput(GroundingModel):
    query_id: str = Field(min_length=1)
    rank: int = Field(ge=1)
    chunk_id: str = Field(min_length=1)


class ContextUnit(GroundingModel):
    unit_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    ordinal: int | None = Field(default=None, ge=1)
    display_text: str = Field(min_length=1)
    heading_path: tuple[str, ...] = ()
    source: SourceRecord
    best_retrieval_rank: int = Field(ge=1)
    contributing_chunk_ids: tuple[str, ...] = Field(min_length=1)
    contributing_ranks: tuple[int, ...] = Field(min_length=1)


class ContextBlock(GroundingModel):
    block_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    source: SourceRecord
    heading_path: tuple[str, ...] = ()
    units: tuple[ContextUnit, ...] = Field(min_length=1)
    best_retrieval_rank: int = Field(ge=1)


class OmittedUnit(GroundingModel):
    unit_id: str = Field(min_length=1)
    contributing_chunk_ids: tuple[str, ...] = Field(min_length=1)
    best_retrieval_rank: int = Field(ge=1)
    reason: str = Field(min_length=1)


class EvidenceReference(GroundingModel):
    evidence_id: str = Field(pattern=r"^E[0-9]{3,}$")
    unit_id: str = Field(min_length=1)
    block_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    display_text: str = Field(min_length=1)
    heading_path: tuple[str, ...] = ()
    source: SourceRecord
    contributing_chunk_ids: tuple[str, ...] = Field(min_length=1)
    contributing_ranks: tuple[int, ...] = Field(min_length=1)


class ContextPack(GroundingModel):
    query_id: str = Field(min_length=1)
    phase8_selection_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    canonical_corpus_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    assembly_policy_version: str = Field(min_length=1)
    token_counter_identity: str = Field(min_length=1)
    max_context_tokens: int = Field(ge=0)
    token_count: int = Field(ge=0)
    units: tuple[ContextUnit, ...]
    blocks: tuple[ContextBlock, ...]
    evidence: tuple[EvidenceReference, ...]
    omissions: tuple[OmittedUnit, ...]
    input_chunk_ids: tuple[str, ...] = ()
    chunk_policy_hash: str = ""


class CitationRequest(GroundingModel):
    """The complete generator-facing citation contract."""

    evidence_id: str = Field(pattern=r"^E[0-9]{3,}$")
    quoted_text: str


class VerifiedCitation(GroundingModel):
    evidence_id: str = Field(pattern=r"^E[0-9]{3,}$")
    document_id: str = Field(min_length=1)
    document_title: str | None
    jurisdiction: str | None
    article: str | None
    page: str | None
    chunk_id: str = Field(min_length=1)
    source_url: str | None
    quoted_text: str = Field(min_length=1)


class ClaimDraft(GroundingModel):
    claim_id: str = Field(min_length=1)
    claim_text: str = Field(min_length=1)
    citations: tuple[CitationRequest, ...]


class GeneratedDraft(GroundingModel):
    answer_text: str
    claims: tuple[ClaimDraft, ...]


class CitationVerification(GroundingModel):
    request: CitationRequest
    valid: bool
    reason: str | None = None
    citation: VerifiedCitation | None = None


class VerificationResult(GroundingModel):
    valid_citations: tuple[VerifiedCitation, ...]
    invalid_citations: tuple[CitationVerification, ...]
    unsupported_claims: tuple[str, ...]
    structurally_valid: bool
    should_abstain: bool
    semantic_entailment_deferred: bool = True


class TokenCounter(Protocol):
    @property
    def identity(self) -> str: ...

    def count(self, text: str) -> int: ...
