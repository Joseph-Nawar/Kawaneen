"""Deterministic fixed, structural, neighbor, and parent-child chunk builders."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import replace
from typing import cast

from kawaneen.chunking.corpus import Phase5Corpus
from kawaneen.chunking.models import (
    ChunkPolicy,
    CitationAnchor,
    LegalChunk,
    SourceSpan,
    deterministic_chunk_id,
)
from kawaneen.chunking.structure import build_structural_leaf_chunks, section_units
from kawaneen.corpus.models import CanonicalUnit
from kawaneen.normalization.models import NormalizationPolicy
from kawaneen.normalization.policies import normalize_text

_TOKEN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def _document_node_id(document_id: str) -> str:
    return f"node:document:{hashlib.sha256(document_id.encode('utf-8')).hexdigest()[:32]}"


def _merge_spans(spans: list[SourceSpan]) -> tuple[SourceSpan, ...]:
    merged: list[SourceSpan] = []
    for span in sorted(spans, key=lambda value: (value.unit_id, value.start, value.end)):
        if merged and merged[-1].unit_id == span.unit_id and merged[-1].end >= span.start:
            prior = merged[-1]
            merged[-1] = SourceSpan(prior.unit_id, prior.start, max(prior.end, span.end))
        else:
            merged.append(span)
    return tuple(merged)


def _fixed_chunks(
    units: Iterable[CanonicalUnit],
    corpus: Phase5Corpus,
    chunk_policy: ChunkPolicy,
    normalization_policy: NormalizationPolicy,
) -> tuple[LegalChunk, ...]:
    selected = section_units(units)
    by_document: defaultdict[str, list[CanonicalUnit]] = defaultdict(list)
    for unit in selected:
        by_document[unit.document_id].append(unit)
    chunks: list[LegalChunk] = []
    maximum = cast(int, chunk_policy.config["token_maximum"])
    overlap = cast(int, chunk_policy.config["overlap"])
    for document_id in sorted(by_document):
        document_units = sorted(
            by_document[document_id], key=lambda unit: (unit.ordinal or 0, unit.unit_id)
        )
        stream_parts: list[tuple[CanonicalUnit, int, int]] = []
        cursor = 0
        for index, unit in enumerate(document_units):
            start = cursor
            end = start + len(unit.text)
            stream_parts.append((unit, start, end))
            cursor = end + (1 if index < len(document_units) - 1 else 0)
        stream = "\n".join(unit.text for unit in document_units)
        tokens = tuple((match.start(), match.end()) for match in _TOKEN.finditer(stream))
        step = max(maximum - overlap, 1)
        for first in range(0, len(tokens), step):
            last = min(first + maximum, len(tokens))
            stream_start, stream_end = tokens[first][0], tokens[last - 1][1]
            spans: list[SourceSpan] = []
            for unit, unit_start, unit_end in stream_parts:
                overlap_start = max(stream_start, unit_start)
                overlap_end = min(stream_end, unit_end)
                if overlap_end > overlap_start:
                    spans.append(
                        SourceSpan(
                            unit.unit_id, overlap_start - unit_start, overlap_end - unit_start
                        )
                    )
            merged = _merge_spans(spans)
            display = "\n".join(
                next(unit for unit in document_units if unit.unit_id == span.unit_id).text[
                    span.start : span.end
                ]
                for span in merged
            )
            normalized = normalize_text(display, normalization_policy)
            if not isinstance(normalized, str):
                raise TypeError("chunk normalization must return text")
            chunk_id = deterministic_chunk_id(
                chunk_policy.policy_id, document_id, merged, chunk_policy.policy_hash
            )
            chunks.append(
                LegalChunk(
                    chunk_id=chunk_id,
                    strategy_id=chunk_policy.policy_id,
                    chunk_policy_hash=chunk_policy.policy_hash,
                    source_unit_ids=tuple(span.unit_id for span in merged),
                    display_text=display,
                    search_text=normalized,
                    source_spans=merged,
                    parent_id=_document_node_id(document_id),
                    ancestor_ids=(_document_node_id(document_id),),
                    sibling_ids=(),
                    structure_path=("document", "fixed_window"),
                    citation_anchor=CitationAnchor(
                        kind="document", source_unit_id=merged[0].unit_id
                    ),
                    token_count=last - first,
                    normalization_policy_id=normalization_policy.policy_id,
                    normalization_policy_hash=normalization_policy.policy_hash,
                    provenance=next(
                        unit for unit in document_units if unit.unit_id == merged[0].unit_id
                    ).provenance.model_dump(),
                    context_source_spans=merged,
                )
            )
            if last == len(tokens):
                break
    return tuple(chunks)


def _with_strategy(
    base: LegalChunk, strategy_id: str, policy_hash: str, **changes: object
) -> LegalChunk:
    return replace(
        base,
        chunk_id=deterministic_chunk_id(strategy_id, base.chunk_id, base.source_spans, policy_hash),
        strategy_id=strategy_id,
        chunk_policy_hash=policy_hash,
        **changes,
    )


def _neighbor_chunks(
    units: Iterable[CanonicalUnit],
    corpus: Phase5Corpus,
    chunk_policy: ChunkPolicy,
    normalization_policy: NormalizationPolicy,
) -> tuple[LegalChunk, ...]:
    base = build_structural_leaf_chunks(units, corpus, normalization_policy)
    by_parent: defaultdict[str | None, list[LegalChunk]] = defaultdict(list)
    for chunk in base:
        by_parent[chunk.parent_id].append(chunk)
    unit_by_id = {unit.unit_id: unit for unit in units}
    result: list[LegalChunk] = []
    for siblings in by_parent.values():
        ordered = sorted(
            siblings,
            key=lambda chunk: (
                min(unit_by_id[span.unit_id].provenance.source_row for span in chunk.source_spans),
                min(span.start for span in chunk.source_spans),
                chunk.chunk_id,
            ),
        )
        for index, current in enumerate(ordered):
            neighbors = ordered[max(0, index - 1) : min(len(ordered), index + 2)]
            spans = tuple(span for chunk in neighbors for span in chunk.source_spans)
            context = " ".join(
                unit_by_id[span.unit_id].text[span.start : span.end] for span in spans
            )
            normalized = normalize_text(context, normalization_policy)
            if not isinstance(normalized, str):
                raise TypeError("neighbor normalization must return text")
            result.append(
                _with_strategy(
                    current,
                    chunk_policy.policy_id,
                    chunk_policy.policy_hash,
                    search_text=normalized,
                    context_source_spans=spans,
                    sibling_ids=tuple(
                        chunk.chunk_id for chunk in neighbors if chunk is not current
                    ),
                )
            )
    return tuple(sorted(result, key=lambda chunk: chunk.chunk_id))


def _parent_child_chunks(
    units: Iterable[CanonicalUnit],
    corpus: Phase5Corpus,
    chunk_policy: ChunkPolicy,
    normalization_policy: NormalizationPolicy,
) -> tuple[LegalChunk, ...]:
    base = build_structural_leaf_chunks(units, corpus, normalization_policy)
    unit_by_id = {unit.unit_id: unit for unit in units}
    result: list[LegalChunk] = []
    for chunk in base:
        parent_child = _with_strategy(chunk, chunk_policy.policy_id, chunk_policy.policy_hash)
        unit = unit_by_id[chunk.source_unit_ids[0]]
        result.append(
            replace(
                parent_child,
                indexed_child_ids=(parent_child.chunk_id,),
                context_source_spans=(SourceSpan(unit.unit_id, 0, len(unit.text)),),
            )
        )
    return tuple(result)


def build_chunks(
    units: Iterable[CanonicalUnit],
    corpus: Phase5Corpus,
    chunk_policy: ChunkPolicy,
    normalization_policy: NormalizationPolicy,
) -> tuple[LegalChunk, ...]:
    """Build one of the five chunk strategies with one frozen normalization policy."""

    if chunk_policy.policy_id in {"fixed-256-v1", "fixed-512-v1"}:
        return _fixed_chunks(units, corpus, chunk_policy, normalization_policy)
    if chunk_policy.policy_id == "legal-structure-v1":
        return tuple(
            _with_strategy(
                chunk,
                chunk_policy.policy_id,
                chunk_policy.policy_hash,
            )
            for chunk in build_structural_leaf_chunks(units, corpus, normalization_policy)
        )
    if chunk_policy.policy_id == "legal-structure-neighbor-v1":
        return _neighbor_chunks(units, corpus, chunk_policy, normalization_policy)
    if chunk_policy.policy_id == "legal-parent-child-v1":
        return _parent_child_chunks(units, corpus, chunk_policy, normalization_policy)
    raise ValueError(f"unknown chunk policy: {chunk_policy.policy_id}")
