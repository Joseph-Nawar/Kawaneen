"""Semantic-support interface intentionally deferred beyond Stage A."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict


class SemanticAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    available: bool
    supported: bool | None
    reason: str


class SemanticSupport(Protocol):
    def assess(self, claim_text: str, evidence_text: str) -> SemanticAssessment: ...


class DeferredSemanticSupport:
    def assess(self, claim_text: str, evidence_text: str) -> SemanticAssessment:
        del claim_text, evidence_text
        return SemanticAssessment(
            available=False,
            supported=None,
            reason="semantic entailment is deferred beyond Phase 10 Stage A",
        )
