"""Small experiment-only inverted-index BM25 harness; not production retrieval."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import combinations

from kawaneen.corpus.models import CanonicalUnit
from kawaneen.normalization.challenge import PrivateChallenge
from kawaneen.normalization.metrics import (
    aggregate_query_metrics,
    compare_query_metric,
    paired_confidence_interval,
    query_metrics,
)
from kawaneen.normalization.models import NormalizationPolicy
from kawaneen.normalization.policies import normalize_text
from kawaneen.normalization.tokenization import tokenize


@dataclass(frozen=True, slots=True)
class ScoredHit:
    unit_id: str
    score: float


@dataclass(frozen=True, slots=True)
class LexicalIndex:
    policy_id: str
    k1: float
    b: float
    unit_ids: tuple[str, ...]
    doc_lengths: dict[str, int]
    postings: dict[str, dict[str, int]]
    average_document_length: float


@dataclass(frozen=True, slots=True)
class AblationReport:
    seed: int
    k1: float
    b: float
    candidate_count: int
    challenge_query_ids: tuple[str, ...]
    policy_metrics: dict[str, dict[str, float]]
    slice_metrics: dict[str, dict[str, dict[str, float]]]
    pairwise_wins_ties_losses: dict[str, dict[str, int]]
    paired_confidence_intervals: dict[str, dict[str, float | int]]
    private_results: dict[str, dict[str, tuple[str, ...]]]


def build_index(
    units: Iterable[CanonicalUnit],
    policy: NormalizationPolicy,
    *,
    k1: float = 1.2,
    b: float = 0.75,
) -> LexicalIndex:
    selected = tuple(sorted(units, key=lambda unit: unit.unit_id))
    unit_ids = tuple(unit.unit_id for unit in selected)
    if len(set(unit_ids)) != len(unit_ids):
        raise ValueError("candidate unit IDs must be unique")
    postings: defaultdict[str, dict[str, int]] = defaultdict(dict)
    doc_lengths: dict[str, int] = {}
    for unit in selected:
        normalized = normalize_text(unit.text, policy)
        if not isinstance(normalized, str):
            raise TypeError("non-audit normalization must return str")
        counts = Counter(tokenize(normalized))
        doc_lengths[unit.unit_id] = sum(counts.values())
        for token, count in counts.items():
            postings[token][unit.unit_id] = count
    average = sum(doc_lengths.values()) / max(len(doc_lengths), 1)
    return LexicalIndex(
        policy_id=policy.policy_id,
        k1=k1,
        b=b,
        unit_ids=unit_ids,
        doc_lengths=doc_lengths,
        postings=dict(postings),
        average_document_length=average,
    )


def score_query(index: LexicalIndex, query: str) -> tuple[ScoredHit, ...]:
    """Return the complete deterministic BM25 ranking for one query."""

    query_counts = Counter(tokenize(query))
    scores: defaultdict[str, float] = defaultdict(float)
    document_count = len(index.unit_ids)
    for token, query_frequency in query_counts.items():
        posting = index.postings.get(token)
        if not posting:
            continue
        document_frequency = len(posting)
        idf = math.log(
            1.0 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
        )
        for unit_id, term_frequency in posting.items():
            length = index.doc_lengths[unit_id]
            denominator = term_frequency + index.k1 * (
                1.0 - index.b + index.b * length / max(index.average_document_length, 1.0)
            )
            scores[unit_id] += (
                idf * (term_frequency * (index.k1 + 1.0) / denominator) * query_frequency
            )
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return tuple(ScoredHit(unit_id=unit_id, score=score) for unit_id, score in ranked)


def retrieve(index: LexicalIndex, query: str, *, top_k: int = 10) -> tuple[ScoredHit, ...]:
    return score_query(index, query)[:top_k]


def run_ablation(
    units: Iterable[CanonicalUnit],
    challenge: PrivateChallenge,
    policies: Iterable[NormalizationPolicy],
    *,
    seed: int = 20260811,
    k1: float = 1.2,
    b: float = 0.75,
) -> AblationReport:
    """Run symmetric query/corpus normalization under one frozen harness."""

    candidates = tuple(units)
    policy_list = tuple(policies)
    if len({policy.policy_id for policy in policy_list}) != len(policy_list):
        raise ValueError("ablation policies must be unique")
    query_ids = tuple(item.query_id for item in challenge.items)
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("challenge query IDs must be unique")
    candidate_ids = {unit.unit_id for unit in candidates}
    if any(not set(relevant).issubset(candidate_ids) for relevant in challenge.qrels.values()):
        raise ValueError("qrels contain a candidate outside the frozen set")

    private_results: dict[str, dict[str, tuple[str, ...]]] = {}
    per_query: dict[str, list[dict[str, float]]] = {}
    per_slice: dict[str, dict[str, list[dict[str, float]]]] = {}
    policy_metrics: dict[str, dict[str, float]] = {}
    for policy in policy_list:
        index = build_index(candidates, policy, k1=k1, b=b)
        results: dict[str, tuple[str, ...]] = {}
        rows: list[dict[str, float]] = []
        slices: defaultdict[str, list[dict[str, float]]] = defaultdict(list)
        for item in challenge.items:
            query = normalize_text(item.query_text, policy)
            if not isinstance(query, str):
                raise TypeError("non-audit normalization must return str")
            hits = retrieve(index, query, top_k=10)
            result_ids = tuple(hit.unit_id for hit in hits)
            results[item.query_id] = result_ids
            metrics = query_metrics(result_ids, challenge.qrels[item.query_id])
            rows.append(metrics)
            slices[item.phenomenon].append(metrics)
        private_results[policy.policy_id] = results
        per_query[policy.policy_id] = rows
        per_slice[policy.policy_id] = dict(slices)
        policy_metrics[policy.policy_id] = aggregate_query_metrics(rows)

    pairwise: dict[str, dict[str, int]] = {}
    intervals: dict[str, dict[str, float | int]] = {}
    for left, right in combinations(policy_list, 2):
        key = f"{left.policy_id}__vs__{right.policy_id}"
        pairwise[key] = compare_query_metric(
            per_query[left.policy_id], per_query[right.policy_id], "mrr_at_10"
        )
        for metric in ("recall_at_10", "mrr_at_10", "ndcg_at_10"):
            intervals[f"{left.policy_id}__vs__{right.policy_id}__{metric}"] = (
                paired_confidence_interval(
                    [row[metric] for row in per_query[left.policy_id]],
                    [row[metric] for row in per_query[right.policy_id]],
                    seed=seed,
                )
            )

    slice_metrics = {
        policy_id: {
            phenomenon: aggregate_query_metrics(rows) for phenomenon, rows in slices.items()
        }
        for policy_id, slices in per_slice.items()
    }
    return AblationReport(
        seed=seed,
        k1=k1,
        b=b,
        candidate_count=len(candidates),
        challenge_query_ids=query_ids,
        policy_metrics=policy_metrics,
        slice_metrics=slice_metrics,
        pairwise_wins_ties_losses=pairwise,
        paired_confidence_intervals=intervals,
        private_results=private_results,
    )
