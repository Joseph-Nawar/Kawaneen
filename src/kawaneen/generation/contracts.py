"""Strict contracts at the untrusted local-generation boundary."""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Annotated, Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from kawaneen.grounding.contracts import ContextPack, VerifiedCitation


class GenerationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=False)


class GenerationDecision(StrEnum):
    ANSWER = "answer"
    ABSTAIN = "abstain"


class ClaimMode(StrEnum):
    DIRECT = "direct"
    INTERPRETATION = "interpretation"


class AbstentionReason(StrEnum):
    NO_CONTEXT = "NO_CONTEXT"
    LOW_RETRIEVAL_CONFIDENCE = "LOW_RETRIEVAL_CONFIDENCE"
    JURISDICTION_AMBIGUOUS = "JURISDICTION_AMBIGUOUS"
    JURISDICTION_MISMATCH = "JURISDICTION_MISMATCH"
    PERSONALIZED_LEGAL_ADVICE = "PERSONALIZED_LEGAL_ADVICE"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"
    SUPERSEDED_SOURCE = "SUPERSEDED_SOURCE"
    CURRENTNESS_UNVERIFIED = "CURRENTNESS_UNVERIFIED"
    REQUESTED_INFO_NOT_FOUND = "REQUESTED_INFO_NOT_FOUND"
    INVALID_GENERATION = "INVALID_GENERATION"
    SEMANTIC_SUPPORT_UNAVAILABLE = "SEMANTIC_SUPPORT_UNAVAILABLE"
    FUTURE_LAW_UNKNOWABLE = "FUTURE_LAW_UNKNOWABLE"
    AUTHORITATIVE_SOURCE_UNAVAILABLE = "AUTHORITATIVE_SOURCE_UNAVAILABLE"
    REQUIRED_CASE_SECTION_MISSING = "REQUIRED_CASE_SECTION_MISSING"
    CASE_FACTS_NOT_ESTABLISHED = "CASE_FACTS_NOT_ESTABLISHED"
    FORUM_OR_SOURCE_SCOPE_MISMATCH = "FORUM_OR_SOURCE_SCOPE_MISMATCH"


class GenerationSettings(GenerationModel):
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    do_sample: bool = False
    max_new_tokens: int = Field(default=384, gt=0)
    max_claims: int = Field(default=3, ge=1, le=3)
    total_input_tokens: int = Field(default=3584, gt=0)
    output_reservation: int = Field(default=384, gt=0)
    safety_margin: int = Field(default=128, ge=0)

    @model_validator(mode="after")
    def validate_budget(self) -> GenerationSettings:
        if self.output_reservation > self.max_new_tokens:
            raise ValueError("output reservation cannot exceed max_new_tokens")
        if self.output_reservation + self.safety_margin >= self.total_input_tokens:
            raise ValueError("input budget must leave room for output and safety margin")
        return self


STAGE_B_GENERATION_SETTINGS = GenerationSettings(
    max_new_tokens=512,
    output_reservation=512,
)

STAGE_C_GENERATION_SETTINGS = GenerationSettings(
    max_new_tokens=512,
    output_reservation=512,
)

STAGE_D_GENERATION_SETTINGS = GenerationSettings(
    max_new_tokens=512,
    output_reservation=512,
)


class ModelOutputCitation(GenerationModel):
    evidence_id: str = Field(pattern=r"^E[0-9]{3,}$")
    quoted_text: str = Field(min_length=1)

    @field_validator("quoted_text")
    @classmethod
    def quote_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("quoted_text must not be blank")
        return value


class ModelOutputClaim(GenerationModel):
    """Compatibility claim shape used by historical Stage-A artifacts."""

    mode: ClaimMode = ClaimMode.DIRECT
    text: str | None = Field(default=None, min_length=1)
    citations: tuple[ModelOutputCitation, ...] = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def claim_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("claim text must not be blank")
        return value


class DirectClaim(GenerationModel):
    mode: Literal[ClaimMode.DIRECT]
    citations: tuple[ModelOutputCitation, ...] = Field(min_length=1)


class InterpretationClaim(GenerationModel):
    mode: Literal[ClaimMode.INTERPRETATION]
    text: str = Field(min_length=1)
    citations: tuple[ModelOutputCitation, ...] = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def interpretation_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("interpretation text must not be blank")
        return value


GenerationClaim = Annotated[
    DirectClaim | InterpretationClaim,
    Field(discriminator="mode"),
]

QuoteReference = Annotated[str, Field(pattern=r"^Q[0-9]{3}$")]


class StageCDirectClaim(GenerationModel):
    mode: Literal[ClaimMode.DIRECT]
    quote_refs: tuple[QuoteReference, ...] = Field(min_length=1, max_length=3)


class StageCInterpretationClaim(GenerationModel):
    mode: Literal[ClaimMode.INTERPRETATION]
    text: str = Field(min_length=1)
    quote_refs: tuple[QuoteReference, ...] = Field(min_length=1, max_length=3)

    @field_validator("text")
    @classmethod
    def interpretation_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("interpretation text must not be blank")
        return value


StageCGenerationClaim = Annotated[
    StageCDirectClaim | StageCInterpretationClaim,
    Field(discriminator="mode"),
]


class GenerationPayload(GenerationModel):
    """Strict Stage-B provider payload; no untrusted metadata is accepted."""

    decision: GenerationDecision
    claims: tuple[GenerationClaim, ...] = Field(max_length=3)

    @model_validator(mode="after")
    def validate_claim_cardinality(self) -> GenerationPayload:
        if self.decision is GenerationDecision.ANSWER and not self.claims:
            raise ValueError("answer output must contain at least one claim")
        if self.decision is GenerationDecision.ABSTAIN and self.claims:
            raise ValueError("abstain output must contain zero claims")
        return self


class StageCGenerationPayload(GenerationModel):
    """Strict Stage-C payload containing only request-local quote references."""

    decision: GenerationDecision
    claims: tuple[StageCGenerationClaim, ...] = Field(max_length=3)

    @model_validator(mode="after")
    def validate_claim_cardinality(self) -> StageCGenerationPayload:
        if self.decision is GenerationDecision.ANSWER and not self.claims:
            raise ValueError("answer output must contain at least one claim")
        if self.decision is GenerationDecision.ABSTAIN and self.claims:
            raise ValueError("abstain output must contain zero claims")
        return self


class StageDDirectClaim(GenerationModel):
    mode: Literal[ClaimMode.DIRECT]
    quote_refs: tuple[QuoteReference, ...] = Field(min_length=1, max_length=3)


class StageDGenerationPayload(GenerationModel):
    """Stage-D provider payload: direct claims only, resolved server-side."""

    decision: GenerationDecision
    claims: tuple[StageDDirectClaim, ...] = Field(max_length=3)

    @model_validator(mode="after")
    def validate_claim_cardinality(self) -> StageDGenerationPayload:
        if self.decision is GenerationDecision.ANSWER and not self.claims:
            raise ValueError("answer output must contain at least one claim")
        if self.decision is GenerationDecision.ABSTAIN and self.claims:
            raise ValueError("abstain output must contain zero claims")
        return self


def generation_payload_schema() -> dict[str, object]:
    """Return the exact JSON Schema sent to Ollama for Stage B."""

    schema = cast(dict[str, object], GenerationPayload.model_json_schema())
    properties = cast(dict[str, object], schema["properties"])
    properties["decision"] = {
        "type": "string",
        "enum": [GenerationDecision.ANSWER.value, GenerationDecision.ABSTAIN.value],
    }
    claims = cast(dict[str, object], properties["claims"])
    # Pydantic's tuple schema expresses the bound through maxItems on current
    # versions, but keep it explicit because Ollama consumes this boundary.
    claims["maxItems"] = 3
    return schema


def stage_c_generation_payload_schema() -> dict[str, object]:
    """Return the exact compact quote-reference schema sent to Ollama."""

    schema = cast(dict[str, object], StageCGenerationPayload.model_json_schema())
    properties = cast(dict[str, object], schema["properties"])
    properties["decision"] = {
        "type": "string",
        "enum": [GenerationDecision.ANSWER.value, GenerationDecision.ABSTAIN.value],
    }
    claims = cast(dict[str, object], properties["claims"])
    claims["maxItems"] = 3
    return schema


def stage_d_generation_payload_schema() -> dict[str, object]:
    """Return the strict direct-only schema sent to Ollama for Stage D."""

    schema = cast(dict[str, object], StageDGenerationPayload.model_json_schema())
    properties = cast(dict[str, object], schema["properties"])
    properties["decision"] = {
        "type": "string",
        "enum": [GenerationDecision.ANSWER.value, GenerationDecision.ABSTAIN.value],
    }
    claims = cast(dict[str, object], properties["claims"])
    claims["maxItems"] = 3
    return schema


class ModelOutput(GenerationModel):
    decision: GenerationDecision
    claims: tuple[ModelOutputClaim, ...] = ()

    @model_validator(mode="after")
    def validate_claim_cardinality(self) -> ModelOutput:
        if len(self.claims) > 3:
            raise ValueError("model output cannot contain more than three claims")
        if self.decision is GenerationDecision.ANSWER and not self.claims:
            raise ValueError("answer output must contain at least one claim")
        if self.decision is GenerationDecision.ABSTAIN and self.claims:
            raise ValueError("abstain output must contain zero claims")
        return self


class GenerationRequest(GenerationModel):
    query: str = Field(min_length=1)
    context_pack: ContextPack
    settings: GenerationSettings = GenerationSettings()
    jurisdiction_text: str | None = None
    disclaimer_text: str = ""
    quote_registry: object | None = None

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value


class GenerationResult(GenerationModel):
    decision: GenerationDecision
    claims: tuple[ModelOutputClaim, ...] = ()
    abstention_reason: AbstentionReason | None = None
    detail: str | None = None

    @model_validator(mode="after")
    def validate_result(self) -> GenerationResult:
        if self.decision is GenerationDecision.ANSWER:
            if not self.claims:
                raise ValueError("answer result must contain claims")
            if self.abstention_reason is not None:
                raise ValueError("answer result cannot have an abstention reason")
        elif self.claims:
            raise ValueError("abstain result must contain zero claims")
        return self


class ModelCandidate(GenerationModel):
    name: str = Field(min_length=1)
    hf_identity: str = Field(min_length=1)
    hf_revision: str | None = None
    ollama_model: str | None = None
    role: str = Field(min_length=1)
    ollama_digest: str | None = None


class TokenizerFingerprint(GenerationModel):
    identity: str = Field(min_length=1)
    revision: str | None = None
    vocabulary_hash: str | None = None


class VerifiedClaim(GenerationModel):
    """A claim after Phase-9 structural citation verification."""

    mode: ClaimMode = ClaimMode.INTERPRETATION
    text: str = Field(min_length=1)
    citations: tuple[VerifiedCitation, ...] = Field(min_length=1)


def parse_model_output(payload: str | bytes) -> ModelOutput:
    """Parse only the strict JSON model-output contract."""

    try:
        value: Any = json.loads(payload)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("model output is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError("model output must be a JSON object")
    legacy_value = dict(cast(dict[str, object], value))
    raw_claims = legacy_value.get("claims")
    if isinstance(raw_claims, list):
        typed_claims = cast(list[object], raw_claims)
        legacy_claims: list[object] = []
        for raw_claim in typed_claims:
            if isinstance(raw_claim, dict) and "mode" not in raw_claim:
                normalized_claim = dict(cast(dict[str, object], raw_claim))
                normalized_claim["mode"] = ClaimMode.INTERPRETATION.value
                legacy_claims.append(normalized_claim)
            else:
                legacy_claims.append(cast(object, raw_claim))
        legacy_value["claims"] = legacy_claims
    return ModelOutput.model_validate(legacy_value)


def parse_generation_payload(payload: str | bytes) -> GenerationPayload:
    """Parse the Stage-B schema after provider-side validation."""

    try:
        value: Any = json.loads(payload)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("model output is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError("model output must be a JSON object")
    return GenerationPayload.model_validate(cast(dict[str, object], value))


def parse_stage_c_generation_payload(payload: str | bytes) -> StageCGenerationPayload:
    """Parse the compact Stage-C quote-reference contract."""

    try:
        value: Any = json.loads(payload)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("model output is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError("model output must be a JSON object")
    return StageCGenerationPayload.model_validate(cast(dict[str, object], value))


def parse_stage_d_generation_payload(payload: str | bytes) -> StageDGenerationPayload:
    """Parse the direct-only Stage-D provider contract."""

    try:
        value: Any = json.loads(payload)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("model output is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError("model output must be a JSON object")
    return StageDGenerationPayload.model_validate(cast(dict[str, object], value))
