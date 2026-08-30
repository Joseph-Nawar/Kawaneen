"""Strict, text-free contracts for Phase 15 governance and artifacts."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PHASE15_BASE_SHA = "03f58284426c84c6c813be2b1e1bbbbbfd1c9a2d"
PHASE15_SEED = 20260826
ARABIC_EMBEDDING_MODEL = "omarelshehy/Arabic-Retrieval-v1.0"
ALLAM_MODEL = "humain-ai/ALLaM-7B-Instruct-preview"


class ProvenanceLabel(StrEnum):
    HISTORICAL_FROZEN = "HISTORICAL_FROZEN"
    PHASE15_DEV = "PHASE15_DEV"
    HUMAN_REVIEWED_DIAGNOSTIC = "HUMAN_REVIEWED_DIAGNOSTIC"


class ResearchQuestionStatus(StrEnum):
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    INCONCLUSIVE = "INCONCLUSIVE"
    NOT_SUPPORTED = "NOT_SUPPORTED"


class ErrorCategory(StrEnum):
    OCR_FAILURE = "OCR failure"
    ARTICLE_SEGMENTATION_FAILURE = "article segmentation failure"
    NORMALIZATION_FAILURE = "normalization failure"
    LEXICAL_MISMATCH = "lexical mismatch"
    SEMANTIC_RETRIEVAL_FAILURE = "semantic retrieval failure"
    RERANKER_FAILURE = "reranker failure"
    MISSING_SOURCE = "missing source"
    WRONG_JURISDICTION = "wrong jurisdiction"
    INSUFFICIENT_CONTEXT = "insufficient context"
    GENERATOR_HALLUCINATION = "generator hallucination"
    CITATION_MISMATCH = "citation mismatch"
    AMBIGUOUS_QUESTION = "ambiguous question"


class Phase15Model(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        use_enum_values=False,
    )


class ExperimentPlan(Phase15Model):
    base_sha: str
    seed: int = PHASE15_SEED
    bootstrap_replicates: int = 2000
    confidence: float = 0.95
    research_questions: tuple[str, ...]
    hard_prohibitions: tuple[str, ...]
    arabic_embedding_model: str = ARABIC_EMBEDDING_MODEL
    provenance: ProvenanceLabel = ProvenanceLabel.PHASE15_DEV
    holdout_access_permitted: bool = False
    production_change_permitted: bool = False
    model_shopping_after_dev_results: bool = False

    @field_validator("base_sha")
    @classmethod
    def validate_base_sha(cls, value: str) -> str:
        if value != PHASE15_BASE_SHA:
            raise ValueError(f"Phase 15 requires exact base SHA {PHASE15_BASE_SHA}")
        return value

    @field_validator("seed")
    @classmethod
    def validate_seed(cls, value: int) -> int:
        if value != PHASE15_SEED:
            raise ValueError(f"Phase 15 seed is fixed at {PHASE15_SEED}")
        return value

    @field_validator("bootstrap_replicates")
    @classmethod
    def validate_replicates(cls, value: int) -> int:
        if value != 2000:
            raise ValueError("Phase 15 requires exactly 2000 bootstrap replicates")
        return value

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: float) -> float:
        if value != 0.95:
            raise ValueError("Phase 15 requires a 95% confidence interval")
        return value

    @field_validator("research_questions")
    @classmethod
    def validate_questions(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != 7 or any(not item for item in value):
            raise ValueError("Phase 15 freezes exactly seven non-empty research questions")
        return value

    @field_validator("hard_prohibitions")
    @classmethod
    def validate_prohibitions(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or any(not item for item in value):
            raise ValueError("Phase 15 requires frozen hard prohibitions")
        return value

    @field_validator("arabic_embedding_model")
    @classmethod
    def validate_embedding_model(cls, value: str) -> str:
        if value != ARABIC_EMBEDDING_MODEL:
            raise ValueError(f"Arabic embedding is fixed to {ARABIC_EMBEDDING_MODEL}")
        return value

    @model_validator(mode="after")
    def validate_prohibitions_are_active(self) -> Self:
        if self.holdout_access_permitted or self.production_change_permitted:
            raise ValueError("Phase 15 hard prohibitions cannot be disabled")
        if self.model_shopping_after_dev_results:
            raise ValueError("Phase 15 prohibits model shopping after DEV results")
        return self


class ModelLock(Phase15Model):
    model_id: str
    revision: str
    license: str | None = None
    config_sha256: str | None = None
    tokenizer_revision: str | None = None
    pooling: str | None = None
    dimension: int | None = Field(default=None, gt=0)
    normalize_embeddings: bool | None = None
    query_prefix: str | None = None
    passage_prefix: str | None = None
    dtype: str
    batch_size: int = Field(gt=0)
    runtime: str
    device: str
    quantization: dict[str, Any] | None = None
    preflight: dict[str, Any] | None = None

    @field_validator("revision")
    @classmethod
    def revision_must_be_resolved(cls, value: str) -> str:
        if not value or value in {"main", "latest", "unresolved"}:
            raise ValueError("model lock requires an exact immutable revision")
        return value


class ArtifactHash(Phase15Model):
    phase: str
    path: str
    sha256: str
    provenance: ProvenanceLabel = ProvenanceLabel.HISTORICAL_FROZEN
    read_only: bool = True

    @field_validator("path")
    @classmethod
    def reject_private_paths(cls, value: str) -> str:
        if "artifacts/private" in value or value.startswith("/"):
            raise ValueError("evidence registry paths must be tracked relative paths")
        return value


class EvidenceRegistry(Phase15Model):
    base_sha: str
    entries: tuple[ArtifactHash, ...]
    registry_read_only: bool = True
    holdout_private_paths: tuple[str, ...] = ()

    @field_validator("base_sha")
    @classmethod
    def registry_base_sha(cls, value: str) -> str:
        if value != PHASE15_BASE_SHA:
            raise ValueError("evidence registry must be anchored to the Phase 15 base")
        return value

    @model_validator(mode="after")
    def no_private_holdout(self) -> Self:
        if self.holdout_private_paths:
            raise ValueError("private HOLDOUT paths cannot enter the evidence registry")
        return self


class DialectManifest(Phase15Model):
    seed: int = PHASE15_SEED
    base_intent_ids: tuple[str, ...]
    accepted_variant_ids: tuple[str, ...]
    dialect_counts: dict[str, int]
    text_sha256_by_variant: dict[str, str] = {}
    provenance: ProvenanceLabel = ProvenanceLabel.PHASE15_DEV

    @model_validator(mode="after")
    def validate_exact_dialect_counts(self) -> Self:
        if len(self.base_intent_ids) != 20 or len(set(self.base_intent_ids)) != 20:
            raise ValueError("dialect manifest requires exactly 20 unique MSA base intents")
        if len(self.accepted_variant_ids) != 60 or len(set(self.accepted_variant_ids)) != 60:
            raise ValueError("dialect manifest requires exactly 60 unique accepted variants")
        if self.dialect_counts != {"egyptian": 20, "gulf_saudi": 20, "levantine": 20}:
            raise ValueError("dialect manifest requires 20 Egyptian, Gulf/Saudi, and Levantine")
        return self


class GeneratorSubsetManifest(Phase15Model):
    seed: int = PHASE15_SEED
    answerable_gold_present_ids: tuple[str, ...]
    answerable_gold_absent_ids: tuple[str, ...]
    unanswerable_ids: tuple[str, ...]
    provenance: ProvenanceLabel = ProvenanceLabel.PHASE15_DEV

    @model_validator(mode="after")
    def validate_exact_generator_counts(self) -> Self:
        groups = (
            self.answerable_gold_present_ids,
            self.answerable_gold_absent_ids,
            self.unanswerable_ids,
        )
        if tuple(map(len, groups)) != (31, 30, 19):
            raise ValueError("generator subset must contain exactly 31/30/19 IDs")
        all_ids = [item for group in groups for item in group]
        if len(set(all_ids)) != 80:
            raise ValueError("generator subset IDs must be unique")
        return self


class ReviewDecision(Phase15Model):
    case_id: str
    primary: ErrorCategory
    secondary: ErrorCategory | None = None
    confidence: int | None = Field(default=None, ge=1, le=5)
    note: str | None = None


class ReviewCase(Phase15Model):
    case_id: str
    language: str
    pipeline_stage: str
    legal_category: str
    answerability: str
    severity: str
    provenance: ProvenanceLabel = ProvenanceLabel.PHASE15_DEV
    holdout: bool = False
    query_text: str | None = None
    evidence_text: str | None = None
    diagnostics: dict[str, Any] = {}
    ai_suggestion: ErrorCategory | None = None
    ai_preclassification_attempted: bool = False

    @model_validator(mode="after")
    def dev_only(self) -> Self:
        if self.holdout or self.provenance is not ProvenanceLabel.PHASE15_DEV:
            raise ValueError("Phase 15 review cases must be DEV-only")
        return self


class MetricSummary(Phase15Model):
    metric: str
    system: str
    value: float | None = None
    paired_delta: float | None = None
    ci95: tuple[float, float] | None = None
    wins: int | None = Field(default=None, ge=0)
    ties: int | None = Field(default=None, ge=0)
    losses: int | None = Field(default=None, ge=0)
    provenance: ProvenanceLabel = ProvenanceLabel.PHASE15_DEV
