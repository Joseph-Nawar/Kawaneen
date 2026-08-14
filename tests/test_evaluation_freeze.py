from __future__ import annotations

from pathlib import Path

import pytest

from kawaneen.evaluation.freeze import FrozenMutationError, freeze_items
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
    ReviewMetadata,
    ReviewState,
)


def make_item() -> DatasetItem:
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


def _reviewed_item():
    item = make_item()
    return item.model_copy(
        update={
            "review": ReviewMetadata(
                state=ReviewState.ADJUDICATED,
                human_verified=True,
                primary_reviewer="a",
                secondary_reviewer="b",
                adjudicator="c",
            )
        }
    )


def test_freeze_emits_private_immutable_hash_manifest(tmp_path: Path) -> None:
    result = freeze_items((_reviewed_item(),), private_root=tmp_path, corpus_hash="c" * 64)
    assert result["dataset_version"] == "phase6-retrieval-eval-v1"
    assert (tmp_path / "frozen" / "phase6-retrieval-eval-v1" / "manifest.json").is_file()


def test_freeze_rejects_mutation_of_existing_v1(tmp_path: Path) -> None:
    freeze_items((_reviewed_item(),), private_root=tmp_path, corpus_hash="c" * 64)
    changed = _reviewed_item().model_copy(update={"query_text": "changed"})
    with pytest.raises(FrozenMutationError):
        freeze_items((changed,), private_root=tmp_path, corpus_hash="c" * 64)
