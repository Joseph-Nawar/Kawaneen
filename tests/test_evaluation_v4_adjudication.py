from __future__ import annotations

from pathlib import Path

import pytest
from private_support import external_review_path, private_repo_path

from kawaneen.evaluation.models import QueryCategory, SemanticTarget


@pytest.mark.private_artifact
def test_external_review_has_exact_bounded_dispositions() -> None:
    from kawaneen.evaluation.adjudication_v4 import load_v3_adjudication

    decisions = load_v3_adjudication(
        external_review_path("phase6_v3_external_ai_review_adjudication.jsonl")
    )
    counts = {decision.decision: 0 for decision in decisions}
    for decision in decisions:
        counts[decision.decision] += 1
    assert len(decisions) == 240
    assert counts == {
        "accept": 25,
        "correct": 118,
        "replace": 57,
        "regenerate_variant": 40,
    }


@pytest.mark.private_artifact
def test_v4_application_covers_bounded_record_transform() -> None:
    from kawaneen.evaluation.adjudication_v4 import apply_v3_adjudication, load_v3_adjudication
    from kawaneen.evaluation.corpus import freeze_evaluation_corpus, load_evaluation_units
    from kawaneen.evaluation.serialization import read_items_jsonl

    v3_items = read_items_jsonl(
        private_repo_path("phase6_evaluation", "draft-v3", "draft", "selected_and_variants.jsonl")
    )
    pool = read_items_jsonl(
        private_repo_path("phase6_evaluation", "draft-v3", "draft", "base_candidates.jsonl")
    )
    corpus = freeze_evaluation_corpus(
        load_evaluation_units(Path("data/interim/canonical")),
        canonical_root=Path("data/interim/canonical"),
    )
    result = apply_v3_adjudication(
        v3_items,
        corpus,
        load_v3_adjudication(
            external_review_path("phase6_v3_external_ai_review_adjudication.jsonl")
        ),
        pool,
    )
    assert len(result.base_pool) >= len(pool)
    assert len(result.bases) == 200
    assert len(result.variants) == 40
    assert len(result.mapping) == 240
    assert result.multi_source_audit["retrieval_scores_used"] is False


def test_holding_query_does_not_disclose_mixed_disposition() -> None:
    from kawaneen.evaluation.adjudication_v4 import validate_v4_semantic_contract

    target = SemanticTarget(
        category=QueryCategory.CASE_HOLDING,
        proposition="إلزام المدعى عليه بسداد 78,793 ريالاً ورفض ما زاد",
        disposition="إلزام المدعى عليه بسداد 78,793 ريالاً ورفض ما زاد",
        object="المبلغ المطالب به",
        amount="78,793 ريال",
        context="نزاع حول تنفيذ عقد توريد",
    )
    assert validate_v4_semantic_contract(
        QueryCategory.CASE_HOLDING,
        target,
        "في النزاع حول تنفيذ عقد توريد، ماذا قضت المحكمة؟",
        "قضى الحكم بإلزام المدعى عليه بسداد 78,793 ريالاً ورفض ما زاد.",
        ("إلزام المدعى عليه بسداد 78,793 ريالاً ورفض ما زاد",),
    )
    assert not validate_v4_semantic_contract(
        QueryCategory.CASE_HOLDING,
        target,
        "هل قضت المحكمة بإلزام المدعى عليه بسداد 78,793 ريالاً ورفض ما زاد؟",
        "قضى الحكم بإلزام المدعى عليه بسداد 78,793 ريالاً ورفض ما زاد.",
        ("إلزام المدعى عليه بسداد 78,793 ريالاً ورفض ما زاد",),
    )


def test_multi_query_does_not_contain_conclusion_and_requires_two_spans() -> None:
    from kawaneen.evaluation.adjudication_v4 import validate_v4_semantic_contract

    target = SemanticTarget(
        category=QueryCategory.MULTI_EVIDENCE,
        proposition="يستحق الطرف الأول التعويض عن الإخلال",
        premises=("أبرم الطرفان عقداً", "ثبت إخلال الطرف الثاني بالتنفيذ"),
        conclusion="يستحق الطرف الأول التعويض عن الإخلال",
        context="نزاع بشأن تنفيذ عقد",
    )
    assert validate_v4_semantic_contract(
        QueryCategory.MULTI_EVIDENCE,
        target,
        "كيف فصلت المحكمة النزاع بشأن تنفيذ عقد؟",
        "انتهت المحكمة إلى استحقاق الطرف الأول التعويض عن الإخلال.",
        (
            "أبرم الطرفان عقداً",
            "ثبت إخلال الطرف الثاني بالتنفيذ",
            "يستحق الطرف الأول التعويض عن الإخلال",
        ),
    )
    assert not validate_v4_semantic_contract(
        QueryCategory.MULTI_EVIDENCE,
        target,
        "هل يستحق الطرف الأول التعويض عن الإخلال؟",
        "انتهت المحكمة إلى استحقاق الطرف الأول التعويض عن الإخلال.",
        ("أبرم الطرفان عقداً", "ثبت إخلال الطرف الثاني بالتنفيذ"),
    )


@pytest.mark.parametrize(
    "text",
    [
        "بتاريخ 24/11/1444 عقدت الجلسة وحضر الطرفان.",
        "1. حصر وكيل المدعي طلبه.",
        "هل يحق للمدعي طلب التعويض؟",
        "الدائرة صالحة للفصل في القضية بعد اكتمال الطلبات.",
    ],
)
def test_v4_rejects_event_prefix_definition_or_authority_false_positive(text: str) -> None:
    from kawaneen.evaluation.adjudication_v4 import reject_v4_semantic_fragment

    assert reject_v4_semantic_fragment(text)


def test_condition_negation_is_preserved() -> None:
    from kawaneen.evaluation.adjudication_v4 import extract_v4_condition_target

    target = extract_v4_condition_target("إذا بني الاتفاق على خطأ أو خديعة فلا يكون ملزماً.")
    assert target is not None
    assert "فلا" in target.effect or "لا" in target.effect


def test_variant_language_gates_reject_blank_or_internal_metadata() -> None:
    from kawaneen.evaluation.adjudication_v4 import validate_v4_query_text

    assert not validate_v4_query_text("What did [Person Name] decide?", "english")
    assert not validate_v4_query_text("[intent abc] ما القاعدة؟", "ar")
    assert validate_v4_query_text("What deadline applied to filing the response?", "english")


def test_all_record_duplicate_gate_includes_variants() -> None:
    from kawaneen.evaluation.adjudication_v4 import duplicate_query_keys

    rows = [
        {"query_id": "a", "query_text": "ما القاعدة؟"},
        {"query_id": "b", "query_text": " ما   القاعدة؟ "},
    ]
    duplicates = duplicate_query_keys(rows)
    assert duplicates == {"ما القاعدة": ("a", "b")}


def test_v4_typed_extractors_cover_non_prefix_semantic_paths() -> None:
    from kawaneen.evaluation.adjudication_v4 import (
        _extract_target,
        _make_multi_target,
        validate_v4_semantic_contract,
    )

    exact = _extract_target(
        QueryCategory.EXACT_PROVISION,
        "المادة 29/2: يعتبر الإخطار صحيحاً ويثبت الأثر القانوني للطرف.",
        None,
    )
    assert exact is not None and exact.provision_identifier == "المادة 29/2"

    definition = _extract_target(
        QueryCategory.DEFINITION,
        "الترافع عن بعد: إجراء قانوني يتم عبر الوسائل الإلكترونية أمام المحكمة.",
        None,
    )
    assert definition is not None and definition.defined_term == "الترافع عن بعد"
    fallback_definition = _extract_target(
        QueryCategory.DEFINITION,
        "المقصود بفسخ هو فسخ العقد الذي ينهي الرابطة التعاقدية.",
        None,
    )
    assert fallback_definition is not None and fallback_definition.defined_term == "فسخ"
    assert _extract_target(QueryCategory.DEFINITION, "نص بلا علاقة تعريفية", None) is None
    assert _extract_target(QueryCategory.DEFINITION, "المقصود بفسخ هو تعريف قانوني", None) is None
    assert _extract_target(QueryCategory.DEFINITION, "المقصود بفسخ هو فسخ العقد اب", None) is None

    deadline = _extract_target(
        QueryCategory.DEADLINE,
        "يجب إيداع المذكرة خلال عشرة أيام بعد التبليغ.",
        None,
    )
    assert deadline is not None and deadline.triggering_event.startswith("بعد")
    assert _extract_target(QueryCategory.DEADLINE, "انعقدت الجلسة بتاريخ 24/11/1444.", None) is None
    assert _extract_target(QueryCategory.CONDITIONS, "لا توجد قاعدة هنا.", None) is None

    authority = _extract_target(
        QueryCategory.AUTHORITY,
        "المحكمة تختص بالنظر في الطلب.",
        None,
    )
    reverse_authority = _extract_target(
        QueryCategory.AUTHORITY,
        "من اختصاص المحكمة النظر في الطلب.",
        None,
    )
    assert authority is not None and reverse_authority is not None
    assert _extract_target(QueryCategory.AUTHORITY, "المحكمة تختص.", None) is None

    holding_seed = SemanticTarget(
        category=QueryCategory.CASE_HOLDING,
        disposition="إلزام جزئي",
        object="المطالبة المالية",
        context="نزاع تعاقدي",
    )
    holding = _extract_target(
        QueryCategory.CASE_HOLDING,
        "قضت المحكمة بإلزام الطرف بسداد 10,000 ريالاً ورفض ما زاد.",
        holding_seed,
    )
    assert holding is not None and "10,000" in holding.amount
    assert _extract_target(QueryCategory.CASE_HOLDING, "قضت المحكمة بالحكم.", None) is None

    assert _make_multi_target(((),), "نتيجة كافية") is None
    multi = _make_multi_target(
        ((None, "ثبت العقد بين الطرفين"), (None, "ثبت الإخلال بالتنفيذ")),
        "يستحق التعويض عن العقد، الإخلال بالتنفيذ",
    )
    assert multi is not None and len(multi.premises) == 2
    assert _make_multi_target(((None, "عقد"), (None, "إخلال")), "نتيجة") is None

    condition = SemanticTarget(
        category=QueryCategory.CONDITIONS,
        proposition="إذا تحقق الشرط؛ فلا يكون الأثر نافذاً",
        condition="إذا تحقق الشرط",
        effect="فلا يكون الأثر نافذاً",
    )
    assert not validate_v4_semantic_contract(
        QueryCategory.CONDITIONS,
        condition,
        "ما الشرط؟",
        ".",
        ("إذا تحقق الشرط فلا يكون الأثر نافذاً",),
    )


def test_v5_multi_evidence_rejects_redundant_single_group() -> None:
    from kawaneen.evaluation.adjudication_v5 import validate_v5_multi_necessity

    assert not validate_v5_multi_necessity(
        conclusion="ثبت العقد والإخلال فاستحق التعويض",
        premises=("ثبت العقد والإخلال فاستحق التعويض", "وردت مراسلات الطرفين"),
        group_texts=("ثبت العقد والإخلال فاستحق التعويض", "وردت مراسلات الطرفين"),
        query="ما النتيجة؟",
        answer="تثبت النتيجة: ثبت العقد والإخلال فاستحق التعويض؛ ووردت المراسلات.",
        grade2_group_count=2,
    )


@pytest.mark.private_artifact
def test_final_v5_adjudication_has_exact_bounded_dispositions_and_extensions() -> None:
    from kawaneen.evaluation.adjudication_v5 import load_v4_final_adjudication

    decisions = load_v4_final_adjudication(
        external_review_path("phase6_v4_final_external_ai_adjudication.jsonl")
    )
    counts = {decision: 0 for decision in ("accept", "correct", "replace", "regenerate_variant")}
    for decision in decisions:
        counts[decision.decision] += 1
    assert counts == {
        "accept": 25,
        "correct": 138,
        "replace": 37,
        "regenerate_variant": 40,
    }
    assert sum(item.evidence_action == "extend_same_unit" for item in decisions) == 3


@pytest.mark.private_artifact
def test_v5_build_writes_changed_review_candidate_without_verifying_records() -> None:
    from kawaneen.evaluation.orchestrator import run_build_draft_v5
    from kawaneen.evaluation.serialization import read_items_jsonl

    summary = run_build_draft_v5(
        review_file=external_review_path("phase6_v4_final_external_ai_adjudication.jsonl")
    )
    items = read_items_jsonl(
        private_repo_path("phase6_evaluation", "draft-v5", "draft", "selected_and_variants.jsonl")
    )
    assert summary["item_count"] == 240
    assert summary["validation"]["valid"] is True
    assert summary["exact_duplicate_groups_all_240"] == 0
    assert summary["review_gate"]["freeze_called"] is False
    assert sum(item.human_verified for item in items) == 0


def test_v5_english_variants_reject_transliteration_and_names() -> None:
    from kawaneen.evaluation.adjudication_v5 import validate_v5_variant_query

    assert not validate_v5_variant_query("What did aljdyh decide about rfa aljlsh?", "english")
    assert not validate_v5_variant_query(
        "What did the court decide about [Person Name]?", "english"
    )
    assert validate_v5_variant_query(
        "What deadline applied to filing the claimant's memorandum?", "english"
    )


def test_v5_english_variant_renderer_avoids_mechanical_fragments() -> None:
    from kawaneen.evaluation.adjudication_v5 import _english_case_issue, _english_identifier

    assert (
        _english_case_issue("أرفق وكيل المدعية أصل اتفاقية المخالصة") == "the settlement agreement"
    )
    assert _english_identifier("61") == "Article 61"


def test_v5_case_issue_generalizes_party_names() -> None:
    from kawaneen.evaluation.adjudication_v5 import neutralize_v5_case_issue

    issue = neutralize_v5_case_issue("رفعت شركة تم العربية للمقاولات دعوى ضد مصنع تكنولوجيا الحديد")
    assert "تم العربية" not in issue
    assert "تكنولوجيا الحديد" not in issue
    assert "الشركة" in issue or "المصنع" in issue


def test_v5_regression_gates_for_deadline_authority_polarity_and_ruling() -> None:
    from kawaneen.evaluation.adjudication_v5 import (
        validate_v5_authority_relation,
        validate_v5_condition_polarity,
        validate_v5_deadline_relation,
        validate_v5_ruling_span,
    )

    assert not validate_v5_deadline_relation("انعقدت الجلسة بتاريخ 24/11/1444", "إيداع المذكرة")
    assert not validate_v5_authority_relation("الدائرة صالحة للفصل في القضية")
    assert not validate_v5_condition_polarity(
        "إذا تحقق الشرط فلا يكون الأثر نافذاً", "يكون الأثر نافذاً"
    )
    assert not validate_v5_ruling_span("حكمت الدائرة بإلزام المدعى عليه")
    assert validate_v5_ruling_span("حكمت الدائرة بإلزام المدعى عليه ورفض ما زاد")
