"""Private lexical chunking ablation with citation and context diagnostics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import cast

from kawaneen.chunking.challenge import PrivateChunkChallenge
from kawaneen.chunking.models import LegalChunk, SourceSpan
from kawaneen.corpus.models import CanonicalUnit, SourceProvenance, UnitType
from kawaneen.normalization.metrics import (
    aggregate_query_metrics,
    compare_query_metric,
    paired_confidence_interval,
    query_metrics,
)
from kawaneen.normalization.models import NormalizationPolicy
from kawaneen.normalization.policies import get_policy, normalize_text
from kawaneen.normalization.retrieval import LexicalIndex, ScoredHit, build_index, score_query
from kawaneen.normalization.tokenization import tokenize


@dataclass(frozen=True, slots=True)
class ChunkingEvaluationReport:
    seed: int
    strategy_metrics: dict[str, dict[str, float]]
    slice_metrics: dict[str, dict[str, dict[str, float]]]
    citation_metrics: dict[str, dict[str, float]]
    context_metrics: dict[str, dict[str, float]]
    pairwise_wins_ties_losses: dict[str, dict[str, int]]
    paired_confidence_intervals: dict[str, dict[str, float | int]]
    private_results: dict[str, dict[str, tuple[str, ...]]]


@dataclass(frozen=True, slots=True)
class _Result:
    chunk_id: str
    score: float


def _overlap(left: SourceSpan, right: SourceSpan) -> int:
    return (
        max(0, min(left.end, right.end) - max(left.start, right.start))
        if left.unit_id == right.unit_id
        else 0
    )


def _span_overlap(spans: Sequence[SourceSpan], gold: Sequence[SourceSpan]) -> int:
    return sum(_overlap(source, target) for source in spans for target in gold)


def map_gold_spans_to_chunks(
    chunks: Iterable[LegalChunk], gold_spans: Sequence[SourceSpan]
) -> tuple[str, ...]:
    return tuple(
        chunk.chunk_id
        for chunk in sorted(chunks, key=lambda item: item.chunk_id)
        if _span_overlap(chunk.source_spans, gold_spans) > 0
    )


def _as_units(chunks: Sequence[LegalChunk]) -> tuple[CanonicalUnit, ...]:
    return tuple(
        CanonicalUnit(
            unit_id=chunk.chunk_id,
            document_id=chunk.parent_id or chunk.chunk_id,
            unit_type=UnitType.CASE_TEXT,
            text=chunk.search_text,
            provenance=SourceProvenance(
                source_id=str(chunk.provenance.get("source_id", "chunking")),
                source_version=str(chunk.provenance.get("source_version", "v1")),
                source_path=str(chunk.provenance.get("source_path", "private")),
                source_row=cast(int, chunk.provenance.get("source_row", index + 1)),
                source_field=str(chunk.provenance.get("source_field", "search_text")),
            ),
        )
        for index, chunk in enumerate(chunks)
    )


def _rank_chunks(
    chunks: Sequence[LegalChunk],
    index: LexicalIndex,
    query: str,
    normalization_policy: NormalizationPolicy,
) -> tuple[_Result, ...]:
    normalized_query = normalize_text(query, normalization_policy)
    if not isinstance(normalized_query, str):
        raise TypeError("query normalization must return text")
    scored = score_query(index, normalized_query)
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    if not chunks or not chunks[0].strategy_id.startswith("legal-parent-child"):
        return tuple(_Result(hit.unit_id, hit.score) for hit in scored[:10])
    best_by_parent: dict[str, ScoredHit] = {}
    for hit in scored:
        chunk = by_id[hit.unit_id]
        parent = chunk.parent_id or chunk.chunk_id
        current = best_by_parent.get(parent)
        if (
            current is None
            or hit.score > current.score
            or (hit.score == current.score and hit.unit_id < current.unit_id)
        ):
            best_by_parent[parent] = hit
    return tuple(
        _Result(hit.unit_id, hit.score)
        for hit in sorted(best_by_parent.values(), key=lambda item: (-item.score, item.unit_id))[
            :10
        ]
    )


def _metric_row(
    results: Sequence[_Result], chunks_by_id: Mapping[str, LegalChunk], gold: Sequence[SourceSpan]
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    result_ids = tuple(result.chunk_id for result in results)
    relevant_ids = map_gold_spans_to_chunks(chunks_by_id.values(), gold)
    retrieval = query_metrics(result_ids, relevant_ids)
    first = chunks_by_id[result_ids[0]] if result_ids else None
    first_overlap = _span_overlap(first.source_spans, gold) if first else 0
    first_context_overlap = (
        _span_overlap(first.context_source_spans or first.source_spans, gold) if first else 0
    )
    cited_length = sum(span.length for span in first.source_spans) if first else 0
    gold_length = sum(span.length for span in gold)
    citation = {
        "gold_span_coverage_at_1": float(first_overlap > 0),
        "gold_span_coverage_at_5": float(
            any(
                _span_overlap(chunks_by_id[result.chunk_id].source_spans, gold) > 0
                for result in results[:5]
            )
        ),
        "citation_precision_at_1": first_overlap / max(cited_length, 1),
        "citation_recall_at_1": first_overlap / max(gold_length, 1),
        "structural_anchor_accuracy_at_1": float(
            bool(first and first.citation_anchor and first_overlap > 0)
        ),
        "citation_overreach_at_1": 1.0 - first_overlap / max(cited_length, 1) if first else 0.0,
        "multi_structure_citation_rate": float(bool(first and len(first.source_unit_ids) > 1)),
    }
    context = {
        "context_coverage_at_1": float(first_context_overlap > 0),
        "context_coverage_at_5": float(
            any(
                _span_overlap(
                    chunks_by_id[result.chunk_id].context_source_spans
                    or chunks_by_id[result.chunk_id].source_spans,
                    gold,
                )
                > 0
                for result in results[:5]
            )
        ),
        "returned_context_token_expansion": (
            len(tokenize(first.search_text)) / max(first.token_count, 1) if first else 0.0
        ),
        "neighbor_attribution_leakage_rate": float(
            bool(
                first
                and first_context_overlap > 0
                and first_overlap == 0
                and "neighbor" in first.strategy_id
            )
        ),
        "parent_child_coverage": float(
            bool(first and first_overlap > 0 and "parent-child" in first.strategy_id)
        ),
    }
    return retrieval, citation, context


def run_chunking_ablation(
    strategy_chunks: Mapping[str, Iterable[LegalChunk]],
    challenge: PrivateChunkChallenge,
    *,
    seed: int = 20260812,
    normalization_policy: NormalizationPolicy | None = None,
) -> ChunkingEvaluationReport:
    """Evaluate all strategies with identical source-span challenge and BM25 settings."""

    policy = normalization_policy or get_policy("arabic-light-v1")
    strategy_list = tuple(sorted(strategy_chunks))
    if not strategy_list:
        raise ValueError("at least one chunk strategy is required")
    per_query: dict[str, list[dict[str, float]]] = {}
    per_slice: dict[str, dict[str, list[dict[str, float]]]] = {}
    citation_rows: dict[str, list[dict[str, float]]] = {}
    context_rows: dict[str, list[dict[str, float]]] = {}
    private_results: dict[str, dict[str, tuple[str, ...]]] = {}
    for strategy_id in strategy_list:
        chunks = tuple(strategy_chunks[strategy_id])
        index = build_index(_as_units(chunks), policy)
        by_id = {chunk.chunk_id: chunk for chunk in chunks}
        query_rows: list[dict[str, float]] = []
        citation_values: list[dict[str, float]] = []
        context_values: list[dict[str, float]] = []
        slices: defaultdict[str, list[dict[str, float]]] = defaultdict(list)
        results_for_strategy: dict[str, tuple[str, ...]] = {}
        for item in challenge.items:
            results = _rank_chunks(chunks, index, item.query_text, policy)
            result_row, citation_row, context_row = _metric_row(results, by_id, item.gold_spans)
            query_rows.append(result_row)
            citation_values.append(citation_row)
            context_values.append(context_row)
            slices[item.slice_name].append(result_row)
            results_for_strategy[item.query_id] = tuple(result.chunk_id for result in results)
        per_query[strategy_id] = query_rows
        per_slice[strategy_id] = dict(slices)
        citation_rows[strategy_id] = citation_values
        context_rows[strategy_id] = context_values
        private_results[strategy_id] = results_for_strategy

    pairwise: dict[str, dict[str, int]] = {}
    intervals: dict[str, dict[str, float | int]] = {}
    for left, right in combinations(strategy_list, 2):
        key = f"{left}__vs__{right}"
        pairwise[key] = compare_query_metric(per_query[left], per_query[right], "mrr_at_10")
        for metric in ("recall_at_10", "mrr_at_10", "ndcg_at_10"):
            intervals[f"{key}__{metric}"] = paired_confidence_interval(
                [row[metric] for row in per_query[left]],
                [row[metric] for row in per_query[right]],
                seed=seed,
            )
    return ChunkingEvaluationReport(
        seed=seed,
        strategy_metrics={
            strategy: aggregate_query_metrics(rows) for strategy, rows in per_query.items()
        },
        slice_metrics={
            strategy: {
                slice_name: aggregate_query_metrics(rows) for slice_name, rows in slices.items()
            }
            for strategy, slices in per_slice.items()
        },
        citation_metrics={
            strategy: aggregate_query_metrics(rows) for strategy, rows in citation_rows.items()
        },
        context_metrics={
            strategy: aggregate_query_metrics(rows) for strategy, rows in context_rows.items()
        },
        pairwise_wins_ties_losses=pairwise,
        paired_confidence_intervals=intervals,
        private_results=private_results,
    )
