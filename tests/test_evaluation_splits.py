from __future__ import annotations

import pytest

from kawaneen.evaluation.models import (
    Answerability,
    DatasetItem,
    DatasetSplit,
    Difficulty,
    EvidenceGroup,
    EvidenceSpan,
    QueryCategory,
    QueryLanguage,
    QueryRegister,
    QueryType,
    RelevanceGrade,
)
from kawaneen.evaluation.splits import (
    assign_provisional_splits,
    load_split,
    split_diagnostics,
)


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
                spans=(EvidenceSpan(unit_id="u", start=0, end=1, grade=RelevanceGrade.REQUIRED),),
            ),
        ),
        gold_answer="قاعدة",
    )


def test_connected_documents_and_variants_stay_together() -> None:
    first = _item()
    second = first.model_copy(
        update={
            "query_id": "q-2",
            "intent_id": "i-2",
            "source_document_ids": ("d", "d2"),
        }
    )
    variant = first.model_copy(
        update={
            "query_id": "q-3",
            "intent_id": "i",
            "variant_id": "simple-ar",
            "base_intent_id": "i",
            "creation_method": "robustness_variant",
        }
    )
    result = assign_provisional_splits((first, second, variant), holdout_fraction=0.34)
    assert len({item.split for item in result}) == 1
    assert split_diagnostics(result).cross_split_document_count == 0


def test_holdout_requires_explicit_access() -> None:
    item = _item().model_copy(update={"split": DatasetSplit.HOLDOUT})
    assert load_split((item,)) == ()
    with pytest.raises(PermissionError, match="allow_holdout"):
        load_split((item,), allow_holdout=False, split=DatasetSplit.HOLDOUT)
    assert load_split((item,), allow_holdout=True, split=DatasetSplit.HOLDOUT) == (item,)
