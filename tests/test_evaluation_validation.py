from __future__ import annotations

import pytest

from kawaneen.evaluation.chunks import map_items_to_chunks
from kawaneen.evaluation.models import (
    Answerability,
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
from kawaneen.evaluation.validation import validate_items, validate_source_spans


def _item() -> DatasetItem:
    return DatasetItem(
        query_id="q",
        intent_id="i",
        query_text="ما القاعدة؟",
        language=QueryLanguage.ARABIC,
        register=QueryRegister.FORMAL,
        category=QueryCategory.DEFINITION,
        query_type=QueryType.LEGAL_CONCEPT,
        jurisdiction="Saudi Arabia",
        creation_method="document_derived",
        answerability=Answerability.ANSWERABLE,
        difficulty=Difficulty.EASY,
        source_document_ids=("d",),
        evidence_groups=(
            EvidenceGroup(
                group_id="g",
                spans=(EvidenceSpan(unit_id="u", start=0, end=5, grade=RelevanceGrade.REQUIRED),),
            ),
        ),
        gold_answer="قاعدة",
    )


def test_source_span_validation_rejects_unknown_unit_and_out_of_bounds() -> None:
    with pytest.raises(ValueError, match="unknown unit"):
        validate_source_spans((_item(),), {})
    with pytest.raises(ValueError, match="bounds"):
        validate_source_spans((_item(),), {"u": "نص"})


def test_evidence_maps_to_deterministic_legal_structure_qrel() -> None:
    from kawaneen.corpus.models import CanonicalUnit, SourceProvenance, UnitType
    from kawaneen.evaluation.corpus import freeze_evaluation_corpus

    unit = CanonicalUnit(
        unit_id="u",
        document_id="d",
        unit_type=UnitType.FACTS,
        text="هذه فقرة قانونية قصيرة.",
        provenance=SourceProvenance(
            source_id="alarb",
            source_version="v",
            source_path="x",
            source_row=1,
            source_field="facts",
        ),
    )
    corpus = freeze_evaluation_corpus(
        (unit,), canonical_root=__import__("pathlib").Path("data/interim/canonical")
    )
    item = _item().model_copy(update={"source_document_ids": ("d",)})
    mapped = map_items_to_chunks((item,), corpus)
    assert mapped[0].chunk_policy_hash
    assert mapped[0].chunk_qrels


def test_validation_summary_is_text_free() -> None:
    result = validate_items((_item(),), unit_texts={"u": "ن"})
    assert result.valid is False
    assert "query_text" not in result.model_dump_json()
