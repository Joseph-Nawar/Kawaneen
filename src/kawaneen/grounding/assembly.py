"""Deterministic canonical-unit context assembly."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from kawaneen.grounding.contracts import (
    CanonicalEvidenceUnit,
    ContextBlock,
    ContextPack,
    ContextUnit,
    EvidenceReference,
    OmittedUnit,
    RetrievalInput,
    TokenCounter,
)
from kawaneen.grounding.provenance import CanonicalCorpusResolver
from kawaneen.grounding.rendering import render_context

ASSEMBLY_POLICY_VERSION = "phase9-context-assembly-v1"


@dataclass(frozen=True, slots=True)
class _Candidate:
    unit: CanonicalEvidenceUnit
    best_rank: int
    chunk_ids: tuple[str, ...]
    ranks: tuple[int, ...]


class ContextAssembler:
    def __init__(
        self,
        resolver: CanonicalCorpusResolver,
        token_counter: TokenCounter,
        *,
        max_context_tokens: int,
        assembly_policy_version: str = ASSEMBLY_POLICY_VERSION,
    ) -> None:
        if max_context_tokens < 0:
            raise ValueError("max_context_tokens must be non-negative")
        self.resolver = resolver
        self.token_counter = token_counter
        self.max_context_tokens = max_context_tokens
        self.assembly_policy_version = assembly_policy_version

    def assemble(
        self,
        *,
        query_id: str,
        ranked_inputs: Sequence[RetrievalInput],
        phase8_selection_sha256: str,
        canonical_corpus_hash: str,
    ) -> ContextPack:
        if not query_id:
            raise ValueError("query_id must not be empty")
        if any(item.query_id != query_id for item in ranked_inputs):
            raise ValueError("ranked input query IDs do not match query_id")
        candidates: dict[str, _Candidate] = {}
        omissions: dict[str, OmittedUnit] = {}
        seen_chunks: set[str] = set()
        for item in sorted(ranked_inputs, key=lambda value: (value.rank, value.chunk_id)):
            if item.chunk_id in seen_chunks:
                raise ValueError(f"duplicate ranked chunk: {item.chunk_id}")
            seen_chunks.add(item.chunk_id)
            resolved = self.resolver.resolve_chunk(item.chunk_id)
            for unit in resolved.units:
                existing = candidates.get(unit.unit_id)
                if existing is not None:
                    candidates[unit.unit_id] = _Candidate(
                        unit=existing.unit,
                        best_rank=min(existing.best_rank, item.rank),
                        chunk_ids=_sorted_chunk_ids(
                            (*existing.chunk_ids, item.chunk_id),
                            (*existing.ranks, item.rank),
                        ),
                        ranks=tuple(sorted((*existing.ranks, item.rank))),
                    )
                    continue
                candidate = _Candidate(
                    unit=unit,
                    best_rank=item.rank,
                    chunk_ids=(item.chunk_id,),
                    ranks=(item.rank,),
                )
                trial = dict(candidates)
                trial[unit.unit_id] = candidate
                trial_pack = self._pack(
                    query_id=query_id,
                    candidates=trial,
                    omissions=tuple(omissions.values()),
                    phase8_selection_sha256=phase8_selection_sha256,
                    canonical_corpus_hash=canonical_corpus_hash,
                )
                if self._count(trial_pack) > self.max_context_tokens:
                    omissions[unit.unit_id] = OmittedUnit(
                        unit_id=unit.unit_id,
                        contributing_chunk_ids=candidate.chunk_ids,
                        best_retrieval_rank=candidate.best_rank,
                        reason="unit_exceeds_remaining_context_budget",
                    )
                else:
                    candidates[unit.unit_id] = candidate
        result = self._pack(
            query_id=query_id,
            candidates=candidates,
            omissions=tuple(omissions.values()),
            phase8_selection_sha256=phase8_selection_sha256,
            canonical_corpus_hash=canonical_corpus_hash,
            input_chunk_ids=tuple(
                item.chunk_id
                for item in sorted(ranked_inputs, key=lambda value: (value.rank, value.chunk_id))
            ),
        )
        if self._count(result) > self.max_context_tokens:
            raise AssertionError("context assembler exceeded token budget")
        return result

    def _pack(
        self,
        *,
        query_id: str,
        candidates: dict[str, _Candidate],
        omissions: tuple[OmittedUnit, ...],
        phase8_selection_sha256: str,
        canonical_corpus_hash: str,
        input_chunk_ids: tuple[str, ...] = (),
    ) -> ContextPack:
        units_by_document: dict[str, list[_Candidate]] = {}
        for candidate in candidates.values():
            units_by_document.setdefault(candidate.unit.document_id, []).append(candidate)
        document_ids = sorted(
            units_by_document,
            key=lambda document_id: (
                min(item.best_rank for item in units_by_document[document_id]),
                document_id,
            ),
        )
        context_units: list[ContextUnit] = []
        blocks: list[ContextBlock] = []
        for document_id in document_ids:
            ordered = sorted(
                units_by_document[document_id],
                key=lambda item: (item.unit.ordinal or 0, item.unit.unit_id),
            )
            current: list[ContextUnit] = []
            for candidate in ordered:
                context_unit = _context_unit(candidate)
                context_units.append(context_unit)
                if current and not _can_join(current[-1], context_unit):
                    blocks.append(_block(len(blocks) + 1, current))
                    current = []
                current.append(context_unit)
            if current:
                blocks.append(_block(len(blocks) + 1, current))
        evidence = tuple(
            EvidenceReference(
                evidence_id=f"E{index:03d}",
                unit_id=unit.unit_id,
                block_id=next(
                    block.block_id
                    for block in blocks
                    if unit.unit_id in {row.unit_id for row in block.units}
                ),
                document_id=unit.document_id,
                display_text=unit.display_text,
                heading_path=unit.heading_path,
                source=unit.source,
                contributing_chunk_ids=unit.contributing_chunk_ids,
                contributing_ranks=unit.contributing_ranks,
            )
            for index, unit in enumerate(context_units, start=1)
        )
        pack = ContextPack(
            query_id=query_id,
            phase8_selection_sha256=phase8_selection_sha256,
            canonical_corpus_hash=canonical_corpus_hash,
            assembly_policy_version=self.assembly_policy_version,
            token_counter_identity=str(self.token_counter.identity),
            max_context_tokens=self.max_context_tokens,
            token_count=0,
            units=tuple(context_units),
            blocks=tuple(blocks),
            evidence=evidence,
            omissions=tuple(
                sorted(omissions, key=lambda item: (item.best_retrieval_rank, item.unit_id))
            ),
            input_chunk_ids=input_chunk_ids,
            chunk_policy_hash=self.resolver.chunk_policy_hash or "",
        )
        return pack.model_copy(update={"token_count": self._count(pack)})

    def _count(self, pack: ContextPack) -> int:
        return int(self.token_counter.count(render_context(pack)))


def _context_unit(candidate: _Candidate) -> ContextUnit:
    return ContextUnit(
        unit_id=candidate.unit.unit_id,
        document_id=candidate.unit.document_id,
        ordinal=candidate.unit.ordinal,
        display_text=candidate.unit.display_text,
        heading_path=candidate.unit.heading_path,
        source=candidate.unit.source,
        best_retrieval_rank=candidate.best_rank,
        contributing_chunk_ids=candidate.chunk_ids,
        contributing_ranks=candidate.ranks,
    )


def _sorted_chunk_ids(chunk_ids: tuple[str, ...], ranks: tuple[int, ...]) -> tuple[str, ...]:
    return tuple(
        chunk_id
        for _, chunk_id in sorted(
            zip(ranks, chunk_ids, strict=True), key=lambda item: (item[0], item[1])
        )
    )


def _can_join(left: ContextUnit, right: ContextUnit) -> bool:
    return (
        left.document_id == right.document_id
        and left.heading_path == right.heading_path
        and left.ordinal is not None
        and right.ordinal is not None
        and right.ordinal == left.ordinal + 1
    )


def _block(index: int, units: list[ContextUnit]) -> ContextBlock:
    first = units[0]
    return ContextBlock(
        block_id=f"B{index:03d}",
        document_id=first.document_id,
        source=first.source,
        heading_path=first.heading_path,
        units=tuple(units),
        best_retrieval_rank=min(unit.best_retrieval_rank for unit in units),
    )
