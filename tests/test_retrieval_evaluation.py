import pytest

from kawaneen.evaluation.models import (
    Answerability,
    CreationMethod,
    DatasetItem,
    Difficulty,
    EvidenceGroup,
    EvidenceSpan,
    QueryCategory,
    QueryLanguage,
    QueryRegister,
    QueryType,
    RelevanceGrade,
)
from kawaneen.retrieval.evaluation import evaluate_rankings, robustness_degradation
from kawaneen.retrieval.models import RetrievalChunk
from kawaneen.retrieval.slices import QueryLengthBins


def _item(query_id: str, *, answerable: bool = True, variant: bool = False) -> DatasetItem:
    return DatasetItem(
        query_id=query_id,
        intent_id=f"intent-{query_id}",
        variant_id="variant-1" if variant else None,
        base_intent_id="base-1" if variant else None,
        query_text="ما الحكم؟",
        language=QueryLanguage.ARABIC,
        register=QueryRegister.FORMAL,
        category=QueryCategory.EXACT_PROVISION if answerable else QueryCategory.UNANSWERABLE,
        query_type=QueryType.REFERENCE_LOOKUP,
        jurisdiction="generic-jurisdiction",
        creation_method=CreationMethod.ROBUSTNESS_VARIANT
        if variant
        else CreationMethod.DOCUMENT_DERIVED,
        answerability=Answerability.ANSWERABLE if answerable else Answerability.UNANSWERABLE,
        unanswerable_reason=None if answerable else "outside_corpus_scope",
        difficulty=Difficulty.EASY,
        source_document_ids=("doc-1",),
        evidence_groups=(
            EvidenceGroup(
                group_id="g1",
                spans=(
                    EvidenceSpan(unit_id="unit-1", start=0, end=1, grade=RelevanceGrade.REQUIRED),
                ),
            ),
        )
        if answerable
        else (),
        gold_answer="answer" if answerable else None,
        chunk_qrels=() if not answerable else ({"chunk_id": "c1", "grade": 2},),
    )


def test_evaluation_excludes_unanswerables_from_ir_denominators() -> None:
    items = (_item("answerable"), _item("unanswerable", answerable=False))
    result = evaluate_rankings(
        items,
        {"answerable": ("c1",), "unanswerable": ()},
        chunks=(_chunk(),),
        query_length_bins=QueryLengthBins(3, 5),
        source_by_document={"doc-1": "alarb"},
    )

    assert result.sample_count == 1
    assert result.metrics["Recall@1"] == 1.0
    assert result.unanswerable_count == 1


def test_robustness_degradation_is_parent_minus_variant() -> None:
    assert robustness_degradation({"MRR@10": 0.8}, {"MRR@10": 0.5}) == {
        "MRR@10": pytest.approx(0.3)
    }


def _chunk() -> RetrievalChunk:
    return RetrievalChunk(
        chunk_id="c1",
        document_id="doc-1",
        source_id="alarb",
        unit_type="facts",
        display_text="evidence",
        search_text="evidence",
        source_unit_ids=("unit-1",),
        chunk_policy_hash="chunk",
        normalization_policy_id="arabic-raw-v1",
        normalization_policy_hash="norm",
        token_count=1,
    )
