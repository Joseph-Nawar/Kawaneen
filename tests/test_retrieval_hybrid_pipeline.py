from kawaneen.retrieval.hybrid.contracts import FusionConfig, SourceHit
from kawaneen.retrieval.hybrid.evaluation import (
    candidate_complete_evidence_recall_at_k,
    candidate_recall_at_k,
    rescue_damage_counts,
)
from kawaneen.retrieval.hybrid.fusion import fuse_ranked_hits
from kawaneen.retrieval.hybrid.pipeline import rerank_for_serving


def test_candidate_metrics_and_rescue_damage_counts() -> None:
    qrels = {"q1": {"a": 2}, "q2": {"b": 2}}
    candidates = {"q1": ("a", "x"), "q2": ("x", "b")}
    bm25 = {"q1": ("a",), "q2": ("b",)}
    assert candidate_recall_at_k(candidates["q1"], qrels["q1"], 20) == 1.0
    assert candidate_complete_evidence_recall_at_k(candidates["q1"], (frozenset({"a"}),), 20)
    assert rescue_damage_counts(bm25, candidates, qrels, k=1) == {"rescued": 0, "damaged": 1}


def test_50_plus_50_fuses_to_20_and_serves_eight_after_rerank() -> None:
    sparse = tuple(SourceHit(f"s-{i}", float(i)) for i in range(50))
    dense = tuple(SourceHit(f"d-{i}", float(i)) for i in range(50))
    fused = fuse_ranked_hits(sparse=sparse, dense=dense, config=FusionConfig())
    scores = {candidate.chunk_id: float(20 - candidate.fused_rank) for candidate in fused}
    served = rerank_for_serving(fused, scores)
    assert len(fused) == 20
    assert len(served) == 8
    assert all(candidate.provenance for candidate in fused)
