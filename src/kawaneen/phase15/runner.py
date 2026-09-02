"""Pure DEV ranking evaluation used by Phase 15 experiment runners."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from kawaneen.retrieval.metrics import (
    complete_evidence_recall_at_k,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)

from .statistics import paired_bootstrap_delta

METRICS = (
    "Recall@1",
    "Recall@5",
    "Recall@10",
    "MRR@10",
    "nDCG@10",
    "CompleteEvidenceRecall@5",
    "CompleteEvidenceRecall@10",
)


@dataclass(frozen=True, slots=True)
class DevRankingResult:
    query_ids: tuple[str, ...]
    metrics: Mapping[str, tuple[float, ...]]


def summarize_ranking_runs(
    runs: Mapping[str, Mapping[str, Sequence[float]]],
    *,
    baseline: str | None = None,
    seed: int = 20260826,
    replicates: int = 2000,
) -> dict[str, object]:
    if not runs:
        raise ValueError("at least one ranking run is required")
    baseline_name = baseline or next(iter(runs))
    if baseline_name not in runs:
        raise ValueError(f"unknown baseline run: {baseline_name}")
    systems: dict[str, object] = {}
    for name, metrics in runs.items():
        systems[name] = {
            metric: {"mean": sum(values) / len(values), "n": len(values)}
            for metric, values in metrics.items()
            if values
        }
    deltas: dict[str, object] = {}
    for name, metrics in runs.items():
        if name == baseline_name:
            continue
        paired: dict[str, object] = {}
        for metric, values in metrics.items():
            if metric not in runs[baseline_name]:
                continue
            result = paired_bootstrap_delta(
                values,
                runs[baseline_name][metric],
                seed=seed,
                replicates=replicates,
            )
            paired[metric] = result.__dict__
        deltas[f"{name}-vs-{baseline_name}"] = paired
    return {
        "status": "RUN",
        "seed": seed,
        "bootstrap_replicates": replicates,
        "baseline": baseline_name,
        "systems": systems,
        "paired_deltas": deltas,
    }


def _qrels(record: Mapping[str, Any]) -> dict[str, int]:
    raw_qrels: Any = record.get("chunk_qrels", ())
    return {
        str(item["chunk_id"]): int(item.get("grade", 1))
        for item in cast(Sequence[dict[str, Any]], raw_qrels)
        if "chunk_id" in item
    }


def _evidence_groups(
    record: Mapping[str, Any], chunks_by_unit: Mapping[str, frozenset[str]]
) -> dict[str, frozenset[str]]:
    groups: dict[str, frozenset[str]] = {}
    raw_groups: Any = record.get("evidence_groups", ())
    for index, group in enumerate(cast(Sequence[dict[str, Any]], raw_groups)):
        group_id = str(group.get("group_id", index))
        raw_spans: Any = group.get("spans", ())
        chunk_ids = {
            chunk_id
            for span in cast(Sequence[dict[str, Any]], raw_spans)
            for chunk_id in chunks_by_unit.get(str(span.get("unit_id")), frozenset())
        }
        if chunk_ids:
            groups[group_id] = frozenset(chunk_ids)
    return groups


def evaluate_dev_rankings(
    records: Sequence[Mapping[str, Any]],
    rankings: Mapping[str, Sequence[str]],
    chunks: Sequence[Mapping[str, Any]],
) -> DevRankingResult:
    """Evaluate only answerable DEV records; no label lookup occurs for others."""

    chunks_by_unit_mutable: defaultdict[str, set[str]] = defaultdict(set)
    for chunk in chunks:
        chunk_id = str(chunk.get("chunk_id", ""))
        raw_unit_ids: Any = chunk.get("source_unit_ids", ())
        for unit_id in cast(Sequence[Any], raw_unit_ids):
            chunks_by_unit_mutable[str(unit_id)].add(chunk_id)
    chunks_by_unit = {
        unit_id: frozenset(chunk_ids) for unit_id, chunk_ids in chunks_by_unit_mutable.items()
    }
    rows: dict[str, dict[str, float]] = {}
    for record in records:
        if str(record.get("answerability", "")).lower() != "answerable":
            continue
        query_id = str(record["query_id"])
        retrieved = tuple(str(item) for item in rankings.get(query_id, ()))
        qrels = _qrels(record)
        groups = _evidence_groups(record, chunks_by_unit)
        rows[query_id] = {
            "Recall@1": recall_at_k(retrieved, qrels, 1),
            "Recall@5": recall_at_k(retrieved, qrels, 5),
            "Recall@10": recall_at_k(retrieved, qrels, 10),
            "MRR@10": mrr_at_k(retrieved, qrels, 10),
            "nDCG@10": ndcg_at_k(retrieved, qrels, 10),
            "CompleteEvidenceRecall@5": complete_evidence_recall_at_k(retrieved, groups, 5),
            "CompleteEvidenceRecall@10": complete_evidence_recall_at_k(retrieved, groups, 10),
        }
    query_ids = tuple(
        str(record["query_id"])
        for record in records
        if str(record.get("answerability", "")).lower() == "answerable"
    )
    # Keep metric columns aligned to the explicit identity order, including legitimate zero rows.
    return DevRankingResult(
        query_ids=query_ids,
        metrics={
            metric: tuple(rows[query_id][metric] for query_id in query_ids) for metric in METRICS
        },
    )
