"""Typed Phase 6 evaluation records and invariants."""

from __future__ import annotations

import hashlib
import json
from enum import IntEnum, StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kawaneen.chunking.models import CitationAnchor, SourceSpan


class QueryLanguage(StrEnum):
    ARABIC = "ar"
    ENGLISH = "en"
    CODE_SWITCHED = "ar-en"


class QueryRegister(StrEnum):
    FORMAL = "formal"
    SIMPLE = "simple"
    EGYPTIAN = "egyptian"
    PROFESSIONAL = "professional"


class QueryCategory(StrEnum):
    EXACT_PROVISION = "exact_provision"
    DEFINITION = "definition"
    DEADLINE = "deadline"
    AUTHORITY = "authority"
    CONDITIONS = "conditions"
    MULTI_EVIDENCE = "multi_evidence"
    CASE_HOLDING = "case_holding"
    UNANSWERABLE = "unanswerable"


class QueryType(StrEnum):
    REFERENCE_LOOKUP = "reference_lookup"
    LEGAL_CONCEPT = "legal_concept"
    PROCEDURE = "procedure"
    RESPONSIBILITY = "responsibility"
    CONDITIONS_EXCEPTIONS = "conditions_exceptions"
    REASONING = "reasoning"
    HOLDING_OUTCOME_REMEDY = "holding_outcome_remedy"
    ABSTENTION = "abstention"


class CreationMethod(StrEnum):
    BENCHMARK_DERIVED = "benchmark_derived"
    DOCUMENT_DERIVED = "document_derived"
    ROBUSTNESS_VARIANT = "robustness_variant"


class Answerability(StrEnum):
    ANSWERABLE = "answerable"
    UNANSWERABLE = "unanswerable"


class UnanswerableReason(StrEnum):
    AUTHORITATIVE_CURRENT_STATUTE_UNAVAILABLE = "authoritative_current_statute_unavailable"
    OUTSIDE_CORPUS_SCOPE = "outside_corpus_scope"
    INSUFFICIENT_SOURCE_EVIDENCE = "insufficient_source_evidence"
    TEMPORAL_AMBIGUITY = "temporal_ambiguity"


class Difficulty(StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ReviewState(StrEnum):
    DRAFT = "draft"
    PRIMARY_REVIEWED = "primary_reviewed"
    SECONDARY_REVIEWED = "secondary_reviewed"
    ADJUDICATED = "adjudicated"
    FROZEN = "frozen"


class DatasetSplit(StrEnum):
    DEV = "dev"
    HOLDOUT = "holdout"


class RelevanceGrade(IntEnum):
    IRRELEVANT = 0
    SUPPORTING = 1
    REQUIRED = 2


class EvidenceSpan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    unit_id: str = Field(min_length=1)
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    grade: RelevanceGrade

    @model_validator(mode="after")
    def validate_bounds(self) -> EvidenceSpan:
        if self.end <= self.start:
            raise ValueError("evidence span end must be greater than start")
        return self

    def as_source_span(self) -> SourceSpan:
        return SourceSpan(unit_id=self.unit_id, start=self.start, end=self.end)


class EvidenceGroup(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    group_id: str = Field(min_length=1)
    spans: tuple[EvidenceSpan, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_group(self) -> EvidenceGroup:
        if not any(span.grade > RelevanceGrade.IRRELEVANT for span in self.spans):
            raise ValueError("evidence group must contain positive relevance")
        return self


class ChunkQrel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str = Field(min_length=1)
    grade: RelevanceGrade


class SemanticTarget(BaseModel):
    """Evidence-derived proposition used to generate one query and answer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    category: QueryCategory
    proposition: str = ""
    context: str = ""
    provision_identifier: str = ""
    subject: str = ""
    effect: str = ""
    defined_term: str = ""
    definition: str = ""
    action: str = ""
    deadline: str = ""
    triggering_event: str = ""
    actor: str = ""
    power: str = ""
    object: str = ""
    condition: str = ""
    disposition: str = ""
    remedy: str = ""
    amount: str = ""
    premises: tuple[str, ...] = ()
    conclusion: str = ""

    @model_validator(mode="after")
    def validate_target(self) -> SemanticTarget:
        required = {
            QueryCategory.EXACT_PROVISION: (
                self.provision_identifier,
                self.effect,
            ),
            QueryCategory.DEFINITION: (self.defined_term, self.definition),
            QueryCategory.DEADLINE: (self.action, self.deadline),
            QueryCategory.AUTHORITY: (self.actor, self.power, self.object),
            QueryCategory.CONDITIONS: (self.condition, self.effect),
            QueryCategory.CASE_HOLDING: (self.disposition, self.object),
            QueryCategory.MULTI_EVIDENCE: (*self.premises, self.conclusion),
        }
        if self.category in required and any(
            not str(value).strip() for value in required[self.category]
        ):
            raise ValueError(f"semantic target is incomplete for {self.category.value}")
        return self


class ReviewMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    state: ReviewState = ReviewState.DRAFT
    human_verified: bool = False
    primary_reviewer: str | None = None
    secondary_reviewer: str | None = None
    adjudicator: str | None = None
    primary_decision: str | None = None
    secondary_decision: str | None = None
    disagreement: bool = False
    notes: str = ""
    review_provenance: str = ""


class DatasetItem(BaseModel):
    """One private query record whose gold is anchored in canonical source spans."""

    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())

    query_id: str = Field(min_length=1)
    intent_id: str = Field(min_length=1)
    variant_id: str | None = None
    base_intent_id: str | None = None
    query_text: str = Field(min_length=1)
    language: QueryLanguage
    register: QueryRegister
    category: QueryCategory
    query_type: QueryType
    jurisdiction: str = Field(min_length=1)
    temporal_scope: str = "source-relative"
    source_scope: tuple[str, ...] = ("alarb", "arabiccr")
    creation_method: CreationMethod
    benchmark_source: str | None = None
    answerability: Answerability
    unanswerable_reason: UnanswerableReason | None = None
    difficulty: Difficulty
    source_document_ids: tuple[str, ...] = ()
    evidence_groups: tuple[EvidenceGroup, ...] = ()
    verified_article_ids: tuple[str, ...] = ()
    semantic_target: SemanticTarget | None = None
    chunk_policy_id: str = "legal-structure-v1"
    chunk_policy_hash: str = ""
    chunk_qrels: tuple[ChunkQrel, ...] = ()
    gold_answer: str | None = None
    citation_anchors: tuple[CitationAnchor, ...] = ()
    review: ReviewMetadata = ReviewMetadata()
    dataset_version: str = "phase6-retrieval-eval-draft"
    split: DatasetSplit = DatasetSplit.DEV
    smoke: bool = False

    @property
    def human_verified(self) -> bool:
        return self.review.human_verified

    @model_validator(mode="after")
    def validate_semantics(self) -> DatasetItem:
        if self.creation_method is CreationMethod.ROBUSTNESS_VARIANT:
            if not self.variant_id or not self.base_intent_id:
                raise ValueError("robustness variants require variant_id and base_intent_id")
        elif self.variant_id is not None or self.base_intent_id is not None:
            raise ValueError("base items cannot carry variant identity")
        if self.answerability is Answerability.ANSWERABLE:
            if not self.gold_answer or not self.evidence_groups:
                raise ValueError("answerable items require gold answer and evidence")
            if self.unanswerable_reason is not None:
                raise ValueError("answerable items cannot have an unanswerable reason")
            if not any(
                span.grade > RelevanceGrade.IRRELEVANT
                for group in self.evidence_groups
                for span in group.spans
            ):
                raise ValueError("answerable items require positive evidence")
        else:
            if self.gold_answer or self.evidence_groups or self.chunk_qrels:
                raise ValueError("unanswerable items require zero gold evidence and qrels")
            if self.unanswerable_reason is None:
                raise ValueError("unanswerable items require a typed reason")
            if self.category is not QueryCategory.UNANSWERABLE:
                raise ValueError("unanswerable items must use the unanswerable category")
        if self.review.human_verified and self.review.state is ReviewState.DRAFT:
            raise ValueError("human verification requires a reviewed state")
        return self


def deterministic_intent_id(
    category: str, document_ids: tuple[str, ...], span_identity: tuple[object, ...]
) -> str:
    payload = {"category": category, "documents": sorted(document_ids), "span": span_identity}
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()[:24]
    return f"intent-{digest}"


def deterministic_query_id(intent_id: str, variant_id: str | None = None) -> str:
    seed = f"{intent_id}:{variant_id or 'base'}"
    return f"query-{hashlib.sha256(seed.encode()).hexdigest()[:24]}"


def citation_to_dict(anchor: CitationAnchor) -> dict[str, str | None]:
    return {"kind": anchor.kind, "label": anchor.label, "source_unit_id": anchor.source_unit_id}


def span_to_dict(span: EvidenceSpan) -> dict[str, object]:
    return {
        "unit_id": span.unit_id,
        "start": span.start,
        "end": span.end,
        "grade": int(span.grade),
    }
