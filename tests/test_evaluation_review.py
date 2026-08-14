from __future__ import annotations

import json
from pathlib import Path

import pytest

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
from kawaneen.evaluation.review import (
    ReviewTransitionError,
    export_review_packet,
    import_review_packet,
    import_reviews,
    review_status,
)
from kawaneen.evaluation.serialization import write_items_jsonl


def test_review_packet_redacts_pii_in_source_excerpts(tmp_path: Path) -> None:
    source = tmp_path / "items.jsonl"
    packet = tmp_path / "packet.jsonl"
    source.write_text(
        json.dumps(
            {
                "query_id": "q-pii",
                "intent_id": "i-pii",
                "query_text": "ما الحكم؟",
                "language": "ar",
                "register": "formal",
                "category": "case_holding",
                "query_type": "holding_outcome_remedy",
                "jurisdiction": "Saudi Arabia",
                "creation_method": "document_derived",
                "answerability": "answerable",
                "difficulty": "medium",
                "source_document_ids": ["d-pii"],
                "evidence_groups": [
                    {
                        "group_id": "g-pii",
                        "spans": [{"unit_id": "u-pii", "start": 0, "end": 18, "grade": 2}],
                    }
                ],
                "gold_answer": "ثبت الحكم النتيجة.",
                "citation_anchors": [],
                "review": {"state": "draft", "human_verified": False},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    export_review_packet(source, packet, {"u-pii": "اتصل بالطرف على 0501234567."})
    record = json.loads(packet.read_text(encoding="utf-8"))
    excerpt = record["source_excerpts"][0]["excerpt"]
    assert "0501234567" not in excerpt
    assert "[REDACTED]" in excerpt


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


def test_review_export_is_private_and_import_preserves_unverified_default(tmp_path: Path) -> None:
    source = tmp_path / "items.jsonl"
    write_items_jsonl(source, (make_item(),))
    packet = export_review_packet(source, tmp_path / "review.jsonl")
    assert packet.parent == tmp_path
    imported = import_review_packet(packet)
    assert imported[0].review.human_verified is False
    assert imported[0].review.state.value == "draft"


def test_invalid_review_state_transition_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "items.jsonl"
    write_items_jsonl(source, (make_item(),))
    packet = export_review_packet(source, tmp_path / "review.jsonl")
    packet.write_text(
        packet.read_text(encoding="utf-8").replace('"state": "draft"', '"state": "frozen"', 1),
        encoding="utf-8",
    )
    with pytest.raises(ReviewTransitionError):
        import_review_packet(packet)


def test_review_status_reports_pending_human_gates() -> None:
    result = review_status((make_item(),))
    assert result["primary_reviewed"] == 0
    assert result["human_verified"] == 0


def test_import_reviews_applies_explicit_reviewer_fields_without_inference(tmp_path: Path) -> None:
    source = tmp_path / "items.jsonl"
    write_items_jsonl(source, (make_item(),))
    packet = export_review_packet(source, tmp_path / "review.jsonl")
    record = json.loads(packet.read_text(encoding="utf-8"))
    record["editable_review"].update(
        {
            "state": "primary_reviewed",
            "reviewer_id": "reviewer-a",
            "decision": "accept",
            "human_verified": False,
        }
    )
    packet.write_text(json.dumps(record) + "\n", encoding="utf-8")
    imported = import_reviews(source, packet)
    assert imported[0].review.primary_reviewer == "reviewer-a"
    assert imported[0].review.human_verified is False
