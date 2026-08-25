"""Request-local authoritative quote references for Phase-10 Stage C."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from pydantic import BaseModel, ConfigDict, Field

from kawaneen.generation.artifacts import artifact_fingerprint
from kawaneen.generation.contracts import (
    GenerationDecision,
    GenerationResult,
    ModelOutputCitation,
    ModelOutputClaim,
    StageCGenerationPayload,
    StageDGenerationPayload,
)
from kawaneen.grounding.contracts import ContextPack, EvidenceReference, SourceRecord

QUOTE_REGISTRY_POLICY_VERSION = "phase10-stage-c-quote-registry-v1"


class QuoteRegistryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    quote_id: str = Field(pattern=r"^Q[0-9]{3}$")
    canonical_unit_id: str = Field(min_length=1)
    evidence_id: str = Field(pattern=r"^E[0-9]{3,}$")
    block_id: str = Field(min_length=1)
    display_text: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    source: SourceRecord
    contributing_chunk_ids: tuple[str, ...] = Field(min_length=1)
    contributing_ranks: tuple[int, ...] = Field(min_length=1)


class QuoteRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    query_id: str = Field(min_length=1)
    policy_version: str = Field(min_length=1)
    entries: tuple[QuoteRegistryEntry, ...]

    @property
    def fingerprint(self) -> str:
        return artifact_fingerprint(self.model_dump(mode="json"))

    @property
    def evidence_to_quote_id(self) -> Mapping[str, str]:
        return {entry.evidence_id: entry.quote_id for entry in self.entries}

    def resolve(self, quote_id: str) -> QuoteRegistryEntry:
        for entry in self.entries:
            if entry.quote_id == quote_id:
                return entry
        raise ValueError(f"unknown quote reference: {quote_id}")


def build_quote_registry(
    context_pack: ContextPack,
    *,
    policy_version: str = QUOTE_REGISTRY_POLICY_VERSION,
) -> QuoteRegistry:
    """Assign local IDs in rendered block/unit order, deduplicated by unit ID."""

    evidence_by_unit: dict[str, EvidenceReference] = {}
    for evidence in context_pack.evidence:
        evidence_by_unit.setdefault(evidence.unit_id, evidence)
    entries: list[QuoteRegistryEntry] = []
    seen_units: set[str] = set()
    for block in context_pack.blocks:
        for unit in block.units:
            if unit.unit_id in seen_units:
                continue
            evidence = evidence_by_unit.get(unit.unit_id)
            if evidence is None:
                raise ValueError(f"context unit has no evidence reference: {unit.unit_id}")
            if evidence.display_text != unit.display_text:
                raise ValueError(f"evidence text mismatch for canonical unit: {unit.unit_id}")
            seen_units.add(unit.unit_id)
            entries.append(
                QuoteRegistryEntry(
                    quote_id=f"Q{len(entries) + 1:03d}",
                    canonical_unit_id=unit.unit_id,
                    evidence_id=evidence.evidence_id,
                    block_id=evidence.block_id,
                    display_text=evidence.display_text,
                    document_id=evidence.document_id,
                    source=evidence.source,
                    contributing_chunk_ids=evidence.contributing_chunk_ids,
                    contributing_ranks=evidence.contributing_ranks,
                )
            )
    return QuoteRegistry(
        query_id=context_pack.query_id,
        policy_version=policy_version,
        entries=tuple(entries),
    )


def stage_c_result_from_payload(
    payload: StageCGenerationPayload,
    registry: QuoteRegistry,
) -> GenerationResult:
    """Resolve model references into the existing Phase-9 citation contract."""

    if payload.decision is GenerationDecision.ABSTAIN:
        return GenerationResult(decision=GenerationDecision.ABSTAIN)
    claims: list[ModelOutputClaim] = []
    for claim in payload.claims:
        citations = tuple(
            ModelOutputCitation(
                evidence_id=entry.evidence_id,
                quoted_text=entry.display_text,
            )
            for quote_id in claim.quote_refs
            for entry in (registry.resolve(quote_id),)
        )
        claims.append(
            ModelOutputClaim(
                mode=claim.mode,
                text=cast(str | None, getattr(claim, "text", None)),
                citations=citations,
            )
        )
    return GenerationResult(
        decision=GenerationDecision.ANSWER,
        claims=tuple(claims),
    )


def stage_d_result_from_payload(
    payload: StageDGenerationPayload,
    registry: QuoteRegistry,
) -> GenerationResult:
    """Resolve Stage-D direct-only references through the same Phase-9 path."""

    if payload.decision is GenerationDecision.ABSTAIN:
        return GenerationResult(decision=GenerationDecision.ABSTAIN)
    claims: list[ModelOutputClaim] = []
    for claim in payload.claims:
        citations = tuple(
            ModelOutputCitation(
                evidence_id=entry.evidence_id,
                quoted_text=entry.display_text,
            )
            for quote_id in claim.quote_refs
            for entry in (registry.resolve(quote_id),)
        )
        claims.append(
            ModelOutputClaim(
                mode=claim.mode,
                text=None,
                citations=citations,
            )
        )
    return GenerationResult(decision=GenerationDecision.ANSWER, claims=tuple(claims))


def render_quote_registry_context(
    context_pack: ContextPack,
    registry: QuoteRegistry,
) -> str:
    """Render evidence with request-local labels and no model-controlled metadata."""

    from kawaneen.grounding.rendering import render_context

    return render_context(context_pack, evidence_labels=registry.evidence_to_quote_id)
