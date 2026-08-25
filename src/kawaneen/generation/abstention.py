"""Fail-closed generation results and stable abstention vocabulary."""

from __future__ import annotations

from kawaneen.generation.contracts import (
    AbstentionReason,
    GenerationDecision,
    GenerationResult,
)


def invalid_generation_result(detail: str | None = None) -> GenerationResult:
    return GenerationResult(
        decision=GenerationDecision.ABSTAIN,
        abstention_reason=AbstentionReason.INVALID_GENERATION,
        detail=detail,
    )


def abstain(reason: AbstentionReason, detail: str | None = None) -> GenerationResult:
    return GenerationResult(
        decision=GenerationDecision.ABSTAIN,
        abstention_reason=reason,
        detail=detail,
    )
