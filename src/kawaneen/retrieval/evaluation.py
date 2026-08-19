"""Answerable-only evaluation over frozen qrels and evidence groups."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from kawaneen.evaluation.models import Answerability, DatasetItem
from kawaneen.retrieval.evidence import evidence_groups_to_chunks
from kawaneen.retrieval.metrics import (
    complete_evidence_recall_at_k,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from kawaneen.retrieval.models import RetrievalChunk
from kawaneen.retrieval.slices import QueryLengthBins, assign_slices

METRIC_NAMES = (
    "Recall@1",
    "Recall@5",
    "Recall@10",
    "MRR@10",
    "nDCG@10",
    "Precision@5",
    "CompleteEvidenceRecall@5",
    "CompleteEvidenceRecall@10",
)


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    metrics: Mapping[str, float]
    sample_count: int
    unanswerable_count: int
    per_query: Mapping[str, Mapping[str, float]]
    slices: Mapping[str, Mapping[str, Mapping[str, float | int]]]


def _mean(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    return {name: sum(row[name] for row in rows) / len(rows) for name in METRIC_NAMES}


def evaluate_rankings(
    items: Sequence[DatasetItem],
    rankings: Mapping[str, Sequence[str]],
    *,
    chunks: Sequence[RetrievalChunk],
    query_length_bins: QueryLengthBins,
    source_by_document: Mapping[str, str],
) -> EvaluationResult:
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    unit_to_chunks: defaultdict[str, set[str]] = defaultdict(set)
    unit_type_by_id: dict[str, str] = {}
    for chunk in chunks:
        for unit_id in chunk.source_unit_ids:
            unit_to_chunks[unit_id].add(chunk.chunk_id)
            unit_type_by_id[unit_id] = chunk.unit_type
    rows: dict[str, dict[str, float]] = {}
    slice_rows: defaultdict[str, defaultdict[str, list[dict[str, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for item in items:
        if item.answerability is not Answerability.ANSWERABLE:
            continue
        qrels = {qrel.chunk_id: int(qrel.grade) for qrel in item.chunk_qrels}
        retrieved = tuple(rankings.get(item.query_id, ()))
        groups = evidence_groups_to_chunks(item, unit_to_chunks, chunks_by_id)
        row = {
            "Recall@1": recall_at_k(retrieved, qrels, 1),
            "Recall@5": recall_at_k(retrieved, qrels, 5),
            "Recall@10": recall_at_k(retrieved, qrels, 10),
            "MRR@10": mrr_at_k(retrieved, qrels, 10),
            "nDCG@10": ndcg_at_k(retrieved, qrels, 10),
            "Precision@5": precision_at_k(retrieved, qrels, 5),
            "CompleteEvidenceRecall@5": complete_evidence_recall_at_k(retrieved, groups, 5),
            "CompleteEvidenceRecall@10": complete_evidence_recall_at_k(retrieved, groups, 10),
        }
        rows[item.query_id] = row
        for dimension, label in assign_slices(
            item, query_length_bins, unit_type_by_id, source_by_document
        ).items():
            slice_rows[dimension][label].append(row)
    aggregate = _mean(tuple(rows.values())) if rows else {name: 0.0 for name in METRIC_NAMES}
    slices = {
        dimension: {
            label: {"sample_count": len(values), **_mean(values)}
            for label, values in labels.items()
        }
        for dimension, labels in slice_rows.items()
    }
    return EvaluationResult(
        metrics=aggregate,
        sample_count=len(rows),
        unanswerable_count=sum(item.answerability is Answerability.UNANSWERABLE for item in items),
        per_query=rows,
        slices=slices,
    )


def robustness_degradation(
    parent_metrics: Mapping[str, float], variant_metrics: Mapping[str, float]
) -> dict[str, float]:
    return {
        metric: parent_metrics[metric] - variant_metrics[metric]
        for metric in parent_metrics.keys() & variant_metrics.keys()
    }
