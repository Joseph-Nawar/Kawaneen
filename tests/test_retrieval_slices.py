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
import pytest
from kawaneen.retrieval.slices import QueryLengthBins, assign_slices, build_query_length_bins


def item(*, variant: bool = False, text: str = "ما هو الحكم القانوني؟") -> DatasetItem:
    return DatasetItem(
        query_id="variant" if variant else "base",
        intent_id="intent",
        variant_id="v1" if variant else None,
        base_intent_id="base" if variant else None,
        query_text=text,
        language=QueryLanguage.ARABIC,
        register=QueryRegister.SIMPLE,
        category=QueryCategory.DEFINITION,
        query_type=QueryType.LEGAL_CONCEPT,
        jurisdiction="Saudi Arabia",
        creation_method=CreationMethod.ROBUSTNESS_VARIANT
        if variant
        else CreationMethod.DOCUMENT_DERIVED,
        answerability=Answerability.ANSWERABLE,
        difficulty=Difficulty.MEDIUM,
        source_document_ids=("doc-1",),
        evidence_groups=(
            EvidenceGroup(
                group_id="g1",
                spans=(EvidenceSpan(unit_id="u1", start=0, end=1, grade=RelevanceGrade.REQUIRED),),
            ),
        ),
        chunk_qrels=(),
        gold_answer="answer",
    )


def test_query_length_bins_are_freezable_and_reused() -> None:
    items = tuple(item(text=f"سؤال قانوني رقم {index}") for index in range(9))
    bins = build_query_length_bins(items)
    assert isinstance(bins, QueryLengthBins)
    assert bins.assign("سؤال قانوني") in {"short", "medium", "long"}
    assert bins == QueryLengthBins.from_dict(bins.to_dict())


def test_slice_assignment_covers_variant_and_evidence_type() -> None:
    bins = QueryLengthBins(short_max=3, medium_max=5)
    slices = assign_slices(item(variant=True), bins, {"u1": "definition"}, {"doc-1": "alarb"})

    assert slices["base_vs_variant"] == "variant"
    assert slices["gold_evidence_type"] == "mixed"
    assert slices["source"] == "alarb"


def test_slice_assignment_handles_mixed_sources_and_unknown_evidence() -> None:
    bins = QueryLengthBins(short_max=1, medium_max=2)
    unknown = assign_slices(
        item().model_copy(update={"source_document_ids": ("doc-1", "doc-2")}),
        bins,
        {"u1": "unknown-unit"},
        {"doc-1": "source-a", "doc-2": "source-b"},
    )
    assert unknown["gold_evidence_type"] == "mixed"
    assert unknown["document_type"] == "mixed"
    assert unknown["source"] == "mixed"
    assert bins.assign("واحد اثنان ثلاثة") == "long"


def test_query_length_bins_require_answerable_base_queries() -> None:
    with pytest.raises(ValueError, match="answerable base"):
        build_query_length_bins((item(variant=True),))
