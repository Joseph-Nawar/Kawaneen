"""Phase-10 Stage-A local generation and abstention infrastructure."""

from kawaneen.generation.contracts import (
    STAGE_B_GENERATION_SETTINGS,
    AbstentionReason,
    ClaimMode,
    DirectClaim,
    GenerationDecision,
    GenerationPayload,
    GenerationRequest,
    GenerationResult,
    GenerationSettings,
    InterpretationClaim,
    ModelCandidate,
    ModelOutput,
    ModelOutputCitation,
    ModelOutputClaim,
    generation_payload_schema,
    parse_generation_payload,
)

__all__ = [
    "STAGE_B_GENERATION_SETTINGS",
    "AbstentionReason",
    "ClaimMode",
    "DirectClaim",
    "GenerationDecision",
    "GenerationPayload",
    "GenerationRequest",
    "GenerationResult",
    "GenerationSettings",
    "InterpretationClaim",
    "ModelCandidate",
    "ModelOutput",
    "ModelOutputCitation",
    "ModelOutputClaim",
    "generation_payload_schema",
    "parse_generation_payload",
]
