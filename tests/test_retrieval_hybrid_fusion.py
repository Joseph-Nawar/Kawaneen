import pytest

from kawaneen.retrieval.hybrid.contracts import FusionConfig, SourceHit
from kawaneen.retrieval.hybrid.fusion import fuse_ranked_hits


def test_rrf_arithmetic_and_provenance_are_exact() -> None:
    result = fuse_ranked_hits(
        sparse=(SourceHit("a", 10.0), SourceHit("shared", 8.0)),
        dense=(SourceHit("shared", 0.8), SourceHit("b", 0.7)),
        config=FusionConfig(sparse_weight=1.0, dense_weight=1.0),
    )

    assert [item.chunk_id for item in result] == ["shared", "a", "b"]
    assert result[0].provenance == "both"
    assert result[0].sparse_rank == 2
    assert result[0].dense_rank == 1
    assert result[0].sparse_score == 8.0
    assert result[0].dense_score == 0.8
    assert result[0].fused_score == pytest.approx(1 / 62 + 1 / 61)
    assert result[1].provenance == "sparse-only"
    assert result[2].provenance == "dense-only"


def test_weighted_rrf_changes_source_contribution() -> None:
    result = fuse_ranked_hits(
        sparse=(SourceHit("s", 1.0),),
        dense=(SourceHit("d", 1.0),),
        config=FusionConfig(sparse_weight=1.0, dense_weight=0.25),
    )

    assert result[0].chunk_id == "s"
    assert result[1].fused_score == pytest.approx(0.25 / 61)


def test_fusion_deduplicates_and_breaks_ties_by_best_rank_then_id() -> None:
    result = fuse_ranked_hits(
        sparse=(SourceHit("z", 1.0), SourceHit("a", 1.0)),
        dense=(SourceHit("a", 1.0), SourceHit("z", 1.0)),
        config=FusionConfig(sparse_weight=0.0, dense_weight=0.0),
    )

    assert [item.chunk_id for item in result] == ["a", "z"]
    assert all(item.provenance == "both" for item in result)


def test_fusion_truncates_to_twenty_candidates() -> None:
    hits = tuple(SourceHit(f"c-{index}", float(index)) for index in range(50))
    result = fuse_ranked_hits(sparse=hits, dense=(), config=FusionConfig())

    assert len(result) == 20
    assert result[-1].fused_rank == 20


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sparse_weight": -1.0},
        {"dense_weight": -1.0},
        {"rrf_k": 0},
        {"sparse_top_k": 49},
        {"dense_top_k": 51},
        {"candidate_k": 0},
    ],
)
def test_invalid_fusion_configuration_is_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        FusionConfig(**kwargs)
