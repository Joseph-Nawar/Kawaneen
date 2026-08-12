"""Text-free chunk integrity and distribution diagnostics."""

from __future__ import annotations

import json
import statistics
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from kawaneen.chunking.models import LegalChunk, StructureNode
from kawaneen.corpus.models import CanonicalUnit
from kawaneen.normalization.tokenization import tokenize


@dataclass(frozen=True, slots=True)
class ChunkIntegrityReport:
    chunk_count: int
    orphan_count: int
    cycle_count: int
    invalid_span_count: int
    display_text_mismatch_count: int
    boundary_violation_count: int
    cross_parent_chunk_count: int
    source_coverage_rate: float

    def to_sanitized_dict(self) -> dict[str, object]:
        return {
            "chunk_count": self.chunk_count,
            "orphan_count": self.orphan_count,
            "cycle_count": self.cycle_count,
            "invalid_span_count": self.invalid_span_count,
            "display_text_mismatch_count": self.display_text_mismatch_count,
            "boundary_violation_count": self.boundary_violation_count,
            "cross_parent_chunk_count": self.cross_parent_chunk_count,
            "source_coverage_rate": self.source_coverage_rate,
        }


def validate_chunks(
    chunks: Iterable[LegalChunk], units: Iterable[CanonicalUnit], nodes: Iterable[StructureNode]
) -> ChunkIntegrityReport:
    chunk_list = tuple(chunks)
    unit_by_id = {unit.unit_id: unit for unit in units}
    node_list = tuple(nodes)
    node_by_id = {node.node_id: node for node in node_list}
    parent_by_unit = {
        node.source_unit_id: node.node_id
        for node in node_list
        if node.kind == "section" and node.source_unit_id is not None
    }
    orphan_count = sum(
        chunk.parent_id is not None and chunk.parent_id not in node_by_id for chunk in chunk_list
    )
    for node in node_list:
        if node.parent_id is not None and node.parent_id not in node_by_id:
            orphan_count += 1
    cycle_count = 0
    for node in node_list:
        seen: set[str] = set()
        current: str | None = node.node_id
        while current is not None:
            if current in seen:
                cycle_count += 1
                break
            seen.add(current)
            current = node_by_id[current].parent_id if current in node_by_id else None
    invalid_span_count = 0
    display_text_mismatch_count = 0
    source_nonspace: set[tuple[str, int]] = set()
    covered_nonspace: set[tuple[str, int]] = set()
    for unit in unit_by_id.values():
        source_nonspace.update(
            (unit.unit_id, offset) for offset, char in enumerate(unit.text) if not char.isspace()
        )
    cross_parent_count = 0
    boundary_violation_count = 0
    for chunk in chunk_list:
        parent_ids = {parent_by_unit.get(span.unit_id) for span in chunk.source_spans}
        parent_ids.discard(None)
        if len(parent_ids) > 1:
            cross_parent_count += 1
            if not chunk.strategy_id.startswith("fixed-"):
                boundary_violation_count += 1
        for span in chunk.source_spans:
            unit = unit_by_id.get(span.unit_id)
            if (
                unit is None
                or span.start < 0
                or span.end > len(unit.text)
                or span.end <= span.start
            ):
                invalid_span_count += 1
                continue
            covered_nonspace.update(
                (span.unit_id, offset)
                for offset in range(span.start, span.end)
                if not unit.text[offset].isspace()
            )
        expected_display = "\n".join(
            unit_by_id[span.unit_id].text[span.start : span.end]
            for span in chunk.source_spans
            if span.unit_id in unit_by_id
        )
        if chunk.display_text != expected_display:
            display_text_mismatch_count += 1
    return ChunkIntegrityReport(
        chunk_count=len(chunk_list),
        orphan_count=orphan_count,
        cycle_count=cycle_count,
        invalid_span_count=invalid_span_count,
        display_text_mismatch_count=display_text_mismatch_count,
        boundary_violation_count=boundary_violation_count,
        cross_parent_chunk_count=cross_parent_count,
        source_coverage_rate=len(covered_nonspace & source_nonspace) / max(len(source_nonspace), 1),
    )


def summarize_chunks(
    chunks: Iterable[LegalChunk], units: Iterable[CanonicalUnit]
) -> dict[str, object]:
    chunk_list = tuple(chunks)
    unit_list = tuple(units)
    lengths = sorted(chunk.token_count for chunk in chunk_list)
    source_tokens = sum(len(tokenize(unit.text)) for unit in unit_list)
    indexed_tokens = sum(lengths)
    p95_index = min(max(int(len(lengths) * 0.95) - 1, 0), max(len(lengths) - 1, 0))
    documents = {unit.document_id for unit in unit_list}
    return {
        "chunk_count": len(chunk_list),
        "chunks_per_document": len(chunk_list) / max(len(documents), 1),
        "token_mean": statistics.mean(lengths) if lengths else 0.0,
        "token_median": statistics.median(lengths) if lengths else 0.0,
        "token_p95": lengths[p95_index] if lengths else 0,
        "token_max": max(lengths, default=0),
        "indexed_token_total": indexed_tokens,
        "duplication_factor": indexed_tokens / max(source_tokens, 1),
        "fallback_count": sum(chunk.fallback_reason is not None for chunk in chunk_list),
        "source_unit_count": len(
            {span.unit_id for chunk in chunk_list for span in chunk.source_spans}
        ),
    }


def write_sanitized_chunk_manifest(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return path
