"""Exact pre-normalization structural boundaries and integrity checks."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import NamedTuple

from kawaneen.chunking.corpus import Phase5Corpus
from kawaneen.chunking.models import (
    CitationAnchor,
    LegalChunk,
    SourceSpan,
    StructureNode,
    deterministic_chunk_id,
)
from kawaneen.corpus.models import CanonicalUnit
from kawaneen.normalization.models import NormalizationPolicy
from kawaneen.normalization.policies import normalize_text
from kawaneen.normalization.tokenization import tokenize

_TOKEN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
_PARAGRAPH_BREAK = re.compile(r"\n+")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?؟؛:])\s+")


class _Segment(NamedTuple):
    start: int
    end: int
    fallback: bool


@dataclass(frozen=True, slots=True)
class StructuralIntegrityReport:
    node_count: int
    chunk_count: int
    orphan_count: int
    cycle_count: int
    cross_parent_boundary_count: int
    invalid_span_count: int
    source_coverage_rate: float


def _token_spans(text: str) -> tuple[tuple[int, int], ...]:
    return tuple((match.start(), match.end()) for match in _TOKEN.finditer(text))


def _blocks(text: str) -> tuple[tuple[int, int], ...]:
    result: list[tuple[int, int]] = []
    start = 0
    for match in _PARAGRAPH_BREAK.finditer(text):
        if text[start : match.start()].strip():
            result.append((start, match.start()))
        start = match.end()
    if text[start:].strip():
        result.append((start, len(text)))
    return tuple(result)


def _window_segments(text: str, start: int, end: int, maximum: int, overlap: int) -> list[_Segment]:
    local = text[start:end]
    tokens = _token_spans(local)
    if len(tokens) <= maximum:
        return [_Segment(start, end, False)]
    step = max(maximum - overlap, 1)
    segments: list[_Segment] = []
    for first in range(0, len(tokens), step):
        last = min(first + maximum, len(tokens))
        segments.append(_Segment(start + tokens[first][0], start + tokens[last - 1][1], True))
        if last == len(tokens):
            break
    return segments


def _exact_segments(text: str, target: int, maximum: int) -> tuple[_Segment, ...]:
    segments: list[_Segment] = []
    for block_start, block_end in _blocks(text):
        block = text[block_start:block_end]
        if len(tokenize(block)) <= maximum:
            segments.append(_Segment(block_start, block_end, False))
            continue
        sentence_starts = [block_start]
        for match in _SENTENCE_BREAK.finditer(block):
            sentence_starts.append(block_start + match.end())
        sentence_ends = [start - 1 for start in sentence_starts[1:]] + [block_end]
        for sentence_start, sentence_end in zip(sentence_starts, sentence_ends, strict=True):
            if not text[sentence_start:sentence_end].strip():
                continue
            segments.extend(_window_segments(text, sentence_start, sentence_end, maximum, 64))
    if not segments:
        return ()
    packed: list[_Segment] = []
    for segment in segments:
        if packed and not packed[-1].fallback:
            current = packed[-1]
            combined = text[current.start : segment.end]
            if (
                not segment.fallback
                and len(tokenize(combined)) <= maximum
                and len(tokenize(text[current.start : current.end])) < target
            ):
                packed[-1] = _Segment(current.start, segment.end, False)
                continue
        packed.append(segment)
    return tuple(packed)


def split_exact_spans(
    text: str,
    unit_id: str = "__text__",
    *,
    target: int = 384,
    maximum: int = 512,
) -> tuple[SourceSpan, ...]:
    """Split text by source boundaries and return spans before normalization."""

    return tuple(
        span for span, _ in split_exact_segments(text, unit_id, target=target, maximum=maximum)
    )


def split_exact_segments(
    text: str,
    unit_id: str = "__text__",
    *,
    target: int = 384,
    maximum: int = 512,
) -> tuple[tuple[SourceSpan, bool], ...]:
    """Return exact spans and whether each span came from oversize fallback windows."""

    if target < 1 or maximum < target:
        raise ValueError("target and maximum token budgets are invalid")
    return tuple(
        (SourceSpan(unit_id, segment.start, segment.end), segment.fallback)
        for segment in _exact_segments(text, target, maximum)
    )


def _node_id(kind: str, identity: str) -> str:
    return f"node:{kind}:{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:32]}"


def section_units(units: Iterable[CanonicalUnit]) -> tuple[CanonicalUnit, ...]:
    selected = tuple(units)
    by_document: defaultdict[str, list[CanonicalUnit]] = defaultdict(list)
    for unit in selected:
        by_document[unit.document_id].append(unit)
    result: list[CanonicalUnit] = []
    structured_types = {"events", "reasoning", "ruling"}
    for document_id in sorted(by_document):
        document_units = sorted(
            by_document[document_id], key=lambda unit: (unit.ordinal or 0, unit.unit_id)
        )
        has_structured = any(
            unit.provenance.source_id == "arabiccr"
            and unit.unit_type.value in structured_types
            and unit.text.strip()
            for unit in document_units
        )
        for unit in document_units:
            if (
                unit.provenance.source_id == "arabiccr"
                and unit.unit_type.value == "case_text"
                and has_structured
            ):
                continue
            result.append(unit)
    return tuple(result)


def build_structure(
    units: Iterable[CanonicalUnit], corpus: Phase5Corpus
) -> tuple[StructureNode, ...]:
    if not {unit.document_id for unit in units}.issubset(corpus.document_ids):
        raise ValueError("structure units are outside the frozen Phase 5 corpus")
    sections = section_units(units)
    by_document: defaultdict[str, list[CanonicalUnit]] = defaultdict(list)
    for unit in sections:
        by_document[unit.document_id].append(unit)
    nodes: list[StructureNode] = []
    for document_id in sorted(by_document):
        document_node = _node_id("document", document_id)
        section_nodes: list[str] = []
        for unit in sorted(
            by_document[document_id], key=lambda item: (item.ordinal or 0, item.unit_id)
        ):
            section_node = _node_id("section", unit.unit_id)
            section_nodes.append(section_node)
            leaves = split_exact_spans(unit.text, unit.unit_id)
            child_ids = tuple(
                _node_id("paragraph", f"{unit.unit_id}:{span.start}:{span.end}") for span in leaves
            )
            nodes.append(
                StructureNode(
                    node_id=section_node,
                    kind="section",
                    document_id=document_id,
                    parent_id=document_node,
                    source_unit_id=unit.unit_id,
                    spans=(SourceSpan(unit.unit_id, 0, len(unit.text)),),
                    structure_path=("document", "section"),
                    children=child_ids,
                )
            )
            for child_id, span in zip(child_ids, leaves, strict=True):
                nodes.append(
                    StructureNode(
                        node_id=child_id,
                        kind="paragraph",
                        document_id=document_id,
                        parent_id=section_node,
                        source_unit_id=unit.unit_id,
                        spans=(span,),
                        structure_path=("document", "section", "paragraph"),
                        children=(),
                    )
                )
        nodes.append(
            StructureNode(
                node_id=document_node,
                kind="document",
                document_id=document_id,
                parent_id=None,
                source_unit_id=None,
                spans=(),
                structure_path=("document",),
                children=tuple(section_nodes),
            )
        )
    return tuple(sorted(nodes, key=lambda node: node.node_id))


def build_structural_leaf_chunks(
    units: Iterable[CanonicalUnit], corpus: Phase5Corpus, policy: NormalizationPolicy
) -> tuple[LegalChunk, ...]:
    selected = section_units(units)
    nodes = build_structure(selected, corpus)
    section_by_unit = {
        node.source_unit_id: node
        for node in nodes
        if node.kind == "section" and node.source_unit_id
    }
    spans_by_unit = {
        unit.unit_id: split_exact_segments(unit.text, unit.unit_id) for unit in selected
    }
    chunks: list[LegalChunk] = []
    for unit in selected:
        section = section_by_unit[unit.unit_id]
        for span, is_fallback in spans_by_unit[unit.unit_id]:
            display = unit.text[span.start : span.end]
            normalized = normalize_text(display, policy)
            if not isinstance(normalized, str):
                raise TypeError("chunk normalization must return text")
            node_id = _node_id("paragraph", f"{span.unit_id}:{span.start}:{span.end}")
            siblings = tuple(child for child in section.children if child != node_id)
            ancestors = (
                (section.parent_id, section.node_id) if section.parent_id else (section.node_id,)
            )
            chunks.append(
                LegalChunk(
                    chunk_id=deterministic_chunk_id(
                        "legal-structure-v1", unit.unit_id, (span,), policy.policy_hash
                    ),
                    strategy_id="legal-structure-v1",
                    chunk_policy_hash=policy.policy_hash,
                    source_unit_ids=(unit.unit_id,),
                    display_text=display,
                    search_text=normalized,
                    source_spans=(span,),
                    parent_id=section.node_id,
                    ancestor_ids=tuple(item for item in ancestors if item),
                    sibling_ids=siblings,
                    structure_path=("document", "section", "paragraph"),
                    citation_anchor=CitationAnchor(
                        kind="section", label=unit.unit_type.value, source_unit_id=unit.unit_id
                    ),
                    token_count=len(tokenize(display)),
                    normalization_policy_id=policy.policy_id,
                    normalization_policy_hash=policy.policy_hash,
                    provenance=unit.provenance.model_dump(),
                    fallback_reason="oversize_fallback" if is_fallback else None,
                )
            )
    return tuple(chunks)


def validate_structure(
    nodes: Iterable[StructureNode], chunks: Iterable[LegalChunk]
) -> StructuralIntegrityReport:
    node_list = tuple(nodes)
    chunk_list = tuple(chunks)
    by_id = {node.node_id: node for node in node_list}
    orphan_count = sum(
        node.parent_id is not None and node.parent_id not in by_id for node in node_list
    )
    cycle_count = 0
    for node in node_list:
        seen: set[str] = set()
        current: str | None = node.node_id
        while current is not None:
            if current in seen:
                cycle_count += 1
                break
            seen.add(current)
            node = by_id.get(current)
            current = node.parent_id if node is not None else None
    invalid_span_count = sum(
        span.start < 0 or span.end < span.start
        for chunk in chunk_list
        for span in chunk.source_spans
    )
    cross_parent = sum(
        len({node.parent_id for node in by_id.values() if node.node_id in chunk.sibling_ids}) > 1
        for chunk in chunk_list
    )
    covered: set[tuple[str, int]] = set()
    source_lengths: dict[str, int] = {}
    for chunk in chunk_list:
        for span in chunk.source_spans:
            covered.update((span.unit_id, index) for index in range(span.start, span.end) if True)
            source_lengths[span.unit_id] = max(source_lengths.get(span.unit_id, 0), span.end)
    total = sum(source_lengths.values())
    coverage = len(covered) / max(total, 1)
    return StructuralIntegrityReport(
        node_count=len(node_list),
        chunk_count=len(chunk_list),
        orphan_count=orphan_count,
        cycle_count=cycle_count,
        cross_parent_boundary_count=cross_parent,
        invalid_span_count=invalid_span_count,
        source_coverage_rate=coverage,
    )
