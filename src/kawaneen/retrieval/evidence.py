"""Evidence-group to chunk mapping derived only from frozen Phase 6 spans/qrels."""

from __future__ import annotations

from collections.abc import Mapping

from kawaneen.evaluation.models import DatasetItem
from kawaneen.retrieval.models import RetrievalChunk


def evidence_groups_to_chunks(
    item: DatasetItem,
    unit_to_chunks: Mapping[str, set[str]],
    chunks_by_id: Mapping[str, RetrievalChunk],
) -> dict[str, frozenset[str]]:
    positive_qrels = {qrel.chunk_id for qrel in item.chunk_qrels if int(qrel.grade) > 0}
    result: dict[str, frozenset[str]] = {}
    for group in item.evidence_groups:
        units = {span.unit_id for span in group.spans if int(span.grade) > 0}
        mapped = {
            chunk_id
            for unit_id in units
            for chunk_id in unit_to_chunks.get(unit_id, set())
            if chunk_id in positive_qrels and chunk_id in chunks_by_id
        }
        if not mapped and len(item.evidence_groups) == 1:
            mapped = positive_qrels
        result[group.group_id] = frozenset(mapped)
    return result
