from __future__ import annotations

from kawaneen.corpus.models import CanonicalUnit, SourceProvenance, UnitType
from kawaneen.normalization import get_policy
from kawaneen.normalization.challenge import ChallengeItem, PrivateChallenge
from kawaneen.normalization.retrieval import (
    build_index,
    retrieve,
    run_ablation,
    score_query,
    tokenize,
)


def _unit(unit_id: str, text: str, row: int) -> CanonicalUnit:
    return CanonicalUnit(
        unit_id=unit_id,
        document_id=unit_id,
        unit_type=UnitType.CASE_TEXT,
        text=text,
        provenance=SourceProvenance(
            source_id="synthetic",
            source_version="v1",
            source_path="fixture",
            source_row=row,
            source_field="text",
        ),
    )


def test_tokenizer_is_shared_and_preserves_legal_separators_as_tokens() -> None:
    assert tokenize("المادة ١٢/2024، A-B") == ("المادة", "١٢", "/", "2024", "،", "A", "-", "B")


def test_bm25_ranking_is_deterministic_and_uses_unit_id_tie_breaking() -> None:
    units = (_unit("b", "قانون مشترك", 2), _unit("a", "قانون مشترك", 1), _unit("c", "حكم مختلف", 3))
    index = build_index(units, get_policy("arabic-raw-v1"))
    first = retrieve(index, "قانون", top_k=3)
    second = retrieve(index, "قانون", top_k=3)
    assert [hit.unit_id for hit in first] == ["a", "b"]
    assert first == second


def test_score_query_exposes_full_scores_when_top_rank_is_unchanged() -> None:
    units = (
        _unit("a-target", "قانون مشترك مادة", 1),
        _unit("z-other", "قانون مشترك حكم", 2),
        _unit("third", "حكم مختلف", 3),
    )
    index = build_index(units, get_policy("arabic-raw-v1"))
    before = score_query(index, "قانون")
    after = score_query(index, "قانون مادة")
    assert before[0].unit_id == after[0].unit_id == "a-target"
    assert before[0].score != after[0].score
    assert [hit.unit_id for hit in before[:2]] == [hit.unit_id for hit in after[:2]]


def test_symmetric_policy_index_and_query_normalization_are_explicit() -> None:
    units = (_unit("target", "ألف", 1), _unit("other", "حكم", 2))
    raw_index = build_index(units, get_policy("arabic-raw-v1"))
    light_index = build_index(units, get_policy("arabic-light-v1"))
    assert retrieve(raw_index, "الف قانون", top_k=1) == ()
    assert retrieve(light_index, "الف قانون", top_k=1)[0].unit_id == "target"


def test_ablation_reuses_query_ids_and_candidate_set_for_all_policies() -> None:
    units = (_unit("target", "ألف", 1), _unit("other", "حكم", 2))
    item = ChallengeItem(
        query_id="q1",
        phenomenon="alef_forms",
        perturbation="alef_visual_variant",
        target_unit_ids=("target",),
        query_text="الف",
        source_display_text="ألف",
    )
    challenge = PrivateChallenge(
        seed=1,
        construction_version="fixture",
        items=(item,),
        qrels={"q1": ("target",)},
    )
    report = run_ablation(
        units, challenge, (get_policy("arabic-raw-v1"), get_policy("arabic-light-v1"))
    )
    assert report.challenge_query_ids == ("q1",)
    assert report.policy_metrics["arabic-raw-v1"]["recall_at_1"] == 0.0
    assert report.policy_metrics["arabic-light-v1"]["recall_at_1"] == 1.0
