from __future__ import annotations

import pytest

from kawaneen.evaluation.models import (
    Answerability,
    DatasetItem,
    Difficulty,
    EvidenceGroup,
    EvidenceSpan,
    QueryCategory,
    QueryLanguage,
    RelevanceGrade,
    deterministic_intent_id,
)


def test_answerable_item_requires_positive_canonical_evidence_and_gold_answer() -> None:
    item = DatasetItem(
        query_id="q-1",
        intent_id="intent-1",
        query_text="ما الحكم؟",
        language=QueryLanguage.ARABIC,
        register="formal",
        category=QueryCategory.CASE_HOLDING,
        query_type="holding_outcome_remedy",
        jurisdiction="Saudi Arabia",
        creation_method="document_derived",
        answerability=Answerability.ANSWERABLE,
        difficulty=Difficulty.EASY,
        source_document_ids=("doc-1",),
        evidence_groups=(
            EvidenceGroup(
                group_id="g-1",
                spans=(
                    EvidenceSpan(unit_id="unit-1", start=0, end=4, grade=RelevanceGrade.REQUIRED),
                ),
            ),
        ),
        gold_answer="الحكم هو ...",
    )
    assert item.human_verified is False


def test_unanswerable_item_cannot_have_evidence_or_gold_answer() -> None:
    with pytest.raises(ValueError, match="unanswerable"):
        DatasetItem(
            query_id="q-1",
            intent_id="intent-1",
            query_text="ما النص الرسمي الحالي؟",
            language=QueryLanguage.ARABIC,
            register="formal",
            category=QueryCategory.UNANSWERABLE,
            query_type="unanswerable",
            jurisdiction="Saudi Arabia",
            creation_method="document_derived",
            answerability=Answerability.UNANSWERABLE,
            unanswerable_reason="authoritative_current_statute_unavailable",
            difficulty=Difficulty.MEDIUM,
            source_document_ids=(),
            gold_answer="لا أعلم",
        )


def test_deterministic_intent_id_is_stable_and_grade_is_typed() -> None:
    left = deterministic_intent_id("case_holding", ("doc-1",), ("unit-1", 0, 4))
    right = deterministic_intent_id("case_holding", ("doc-1",), ("unit-1", 0, 4))
    assert left == right
    assert RelevanceGrade.REQUIRED.value == 2
