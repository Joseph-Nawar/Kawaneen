"""Phase-8 candidate diagnostics built on Phase-7 metric definitions."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

from kawaneen.retrieval.metrics import paired_bootstrap, recall_at_k, wins_ties_losses


def candidate_recall_at_k(retrieved: Sequence[str], qrels: Mapping[str, int], k: int = 20) -> float:
    return recall_at_k(retrieved, qrels, k)


def candidate_complete_evidence_recall_at_k(
    retrieved: Sequence[str], evidence_groups: Sequence[frozenset[str]], k: int = 20
) -> float:
    if not evidence_groups:
        return 0.0
    selected = set(retrieved[:k])
    return float(all(selected.intersection(group) for group in evidence_groups))


def rescue_damage_counts(
    bm25_rankings: Mapping[str, Sequence[str]],
    fusion_rankings: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Mapping[str, int]],
    *,
    k: int = 10,
) -> dict[str, int]:
    rescued = 0
    damaged = 0
    for query_id in sorted(set(bm25_rankings) & set(fusion_rankings) & set(qrels)):
        relevant = {chunk_id for chunk_id, grade in qrels[query_id].items() if grade > 0}
        bm25_hits = bool(set(bm25_rankings[query_id][:k]) & relevant)
        fusion_hits = bool(set(fusion_rankings[query_id][:k]) & relevant)
        rescued += int(not bm25_hits and fusion_hits)
        damaged += int(bm25_hits and not fusion_hits)
    return {"rescued": rescued, "damaged": damaged}


def provenance_fractions(provenances: Sequence[str]) -> dict[str, float]:
    counts = Counter(provenances)
    total = len(provenances)
    return {
        label: counts.get(label, 0) / max(total, 1)
        for label in ("sparse-only", "dense-only", "both")
    }


def paired_comparison(
    left: Sequence[float], right: Sequence[float], *, seed: int = 20260815
) -> dict[str, object]:
    interval = paired_bootstrap(left, right, seed=seed, replicates=2000, confidence=0.95)
    return {
        "estimate_left_minus_right": interval.estimate,
        "ci95_low": interval.low,
        "ci95_high": interval.high,
        **wins_ties_losses(left, right),
        "replicates": interval.replicates,
        "seed": interval.seed,
    }


def select_reranker_pipeline(
    rrf_metrics: Mapping[str, float], reranked_metrics: Mapping[str, float]
) -> dict[str, object]:
    """Apply the frozen Phase-8 reranker gate without tuning after observation."""
    ndcg_delta = float(reranked_metrics["nDCG@10"]) - float(rrf_metrics["nDCG@10"])
    recall_delta = float(reranked_metrics["Recall@10"]) - float(rrf_metrics["Recall@10"])
    cer_delta = float(reranked_metrics["CompleteEvidenceRecall@10"]) - float(
        rrf_metrics["CompleteEvidenceRecall@10"]
    )
    eligible = ndcg_delta >= 0.002 and recall_delta >= -0.01 and cer_delta >= -0.01
    return {
        "selected_pipeline": "rrf_reranked" if eligible else "rrf",
        "reranker_eligible": eligible,
        "nDCG@10_delta": ndcg_delta,
        "Recall@10_delta": recall_delta,
        "CompleteEvidenceRecall@10_delta": cer_delta,
        "thresholds": {
            "minimum_nDCG@10_delta": 0.002,
            "maximum_Recall@10_regression": 0.01,
            "maximum_CompleteEvidenceRecall@10_regression": 0.01,
        },
    }
