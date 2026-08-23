"""Deterministic extractive generation baseline with no model dependencies."""

from __future__ import annotations

import re

from kawaneen.generation.abstention import abstain
from kawaneen.generation.contracts import (
    AbstentionReason,
    GenerationDecision,
    GenerationRequest,
    GenerationResult,
    ModelOutputCitation,
    ModelOutputClaim,
)

_LEXICAL_TOKEN = re.compile(r"(?:[^\W_]|[\u064b-\u065f])+", re.UNICODE)


def lexical_terms(text: str) -> frozenset[str]:
    return frozenset(match.group(0).casefold() for match in _LEXICAL_TOKEN.finditer(text))


class ExtractiveGenerator:
    """Select up to two complete Phase-9 evidence units by lexical overlap."""

    benchmark_only = True

    def generate(self, request: GenerationRequest) -> GenerationResult:
        if not request.context_pack.evidence:
            return abstain(AbstentionReason.NO_CONTEXT)
        query_terms = lexical_terms(request.query)
        candidates = [
            (
                len(query_terms & lexical_terms(evidence.display_text)),
                min(evidence.contributing_ranks),
                evidence.evidence_id,
                evidence,
            )
            for evidence in request.context_pack.evidence
        ]
        selected = sorted(
            (candidate for candidate in candidates if candidate[0] > 0),
            key=lambda candidate: (-candidate[0], candidate[1], candidate[2]),
        )[:2]
        if not selected:
            return abstain(AbstentionReason.REQUESTED_INFO_NOT_FOUND)
        claims = tuple(
            ModelOutputClaim(
                text=candidate[3].display_text,
                citations=(
                    ModelOutputCitation(
                        evidence_id=candidate[3].evidence_id,
                        quoted_text=candidate[3].display_text,
                    ),
                ),
            )
            for candidate in selected
        )
        return GenerationResult(decision=GenerationDecision.ANSWER, claims=claims)
