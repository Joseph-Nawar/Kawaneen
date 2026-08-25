import pytest

from kawaneen.retrieval.hybrid.contracts import FusionConfig, SourceHit
from kawaneen.retrieval.hybrid.evaluation import (
    candidate_complete_evidence_recall_at_k,
    candidate_recall_at_k,
    rescue_damage_counts,
)
from kawaneen.retrieval.hybrid.fusion import fuse_ranked_hits
from kawaneen.retrieval.hybrid.pipeline import rerank_for_serving, retrieve_and_fuse


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


def test_retrieve_and_fuse_passes_query_and_configured_depths() -> None:
    calls: list[tuple[str, int]] = []

    def sparse(query: str, depth: int) -> tuple[SourceHit, ...]:
        calls.append((f"sparse:{query}", depth))
        return (SourceHit("s", 1.0),)

    def dense(query: str, depth: int) -> tuple[SourceHit, ...]:
        calls.append((f"dense:{query}", depth))
        return (SourceHit("d", 1.0),)

    fused = retrieve_and_fuse(
        "query", sparse_search=sparse, dense_search=dense, config=FusionConfig()
    )

    assert calls == [("sparse:query", 50), ("dense:query", 50)]
    assert tuple(candidate.chunk_id for candidate in fused) == ("d", "s")


def test_fusion_rejects_duplicate_source_ids() -> None:
    config = FusionConfig()
    with pytest.raises(ValueError, match="duplicate chunk ID in sparse"):
        fuse_ranked_hits(
            sparse=(SourceHit("same", 1.0), SourceHit("same", 0.5)),
            dense=(),
            config=config,
        )
    with pytest.raises(ValueError, match="duplicate chunk ID in dense"):
        fuse_ranked_hits(
            sparse=(),
            dense=(SourceHit("same", 1.0), SourceHit("same", 0.5)),
            config=config,
        )
