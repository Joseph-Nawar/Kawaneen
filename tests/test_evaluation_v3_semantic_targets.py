from __future__ import annotations

from kawaneen.corpus.models import CanonicalUnit, SourceProvenance, UnitType
from kawaneen.evaluation.candidates import EvidenceDiscovery
from kawaneen.evaluation.models import QueryCategory, SemanticTarget
from kawaneen.evaluation.semantic_targets import (
    extract_semantic_target,
    render_semantic_answer,
    render_semantic_query,
    validate_semantic_target,
)


def _unit(text: str, unit_type: UnitType) -> CanonicalUnit:
    return CanonicalUnit(
        unit_id="fixture-unit",
        document_id="fixture-document",
        unit_type=unit_type,
        text=text,
        provenance=SourceProvenance(
            source_id="alarb",
            source_version="fixture",
            source_path="fixture",
            source_row=1,
            source_field=unit_type.value,
        ),
    )


def _discovery(unit: CanonicalUnit) -> EvidenceDiscovery:
    return EvidenceDiscovery(0, len(unit.text), unit.text, 1)


def test_exact_target_contains_provision_and_actual_settlement_effect() -> None:
    unit = _unit(
        "يُعتبر الصلح المنوه عنه سندًا تنفيذياً وتنقضى به الدعوى حسب المادة 29/2 "
        "من نظام المحاكم التجارية",
        UnitType.COURT_REASONING,
    )
    target = extract_semantic_target(QueryCategory.EXACT_PROVISION, unit, _discovery(unit))
    assert target is not None
    assert target.provision_identifier == "29/2"
    assert "الصلح" in target.subject
    assert "سند" in target.effect
    assert "الاختصاص التجاري" not in render_semantic_query(target)
    assert validate_semantic_target(QueryCategory.EXACT_PROVISION, target, (unit.text,))


def test_deadline_target_uses_the_actual_payment_action_not_a_template_topic() -> None:
    unit = _unit(
        "عرض وكيل المدعى عليها سداد المبلغ على ستة أشهر تبدأ من أغسطس 2023، أو "
        "السداد الكامل بتاريخ 30/10/2023",
        UnitType.FACTS,
    )
    target = extract_semantic_target(QueryCategory.DEADLINE, unit, _discovery(unit))
    assert target is not None
    assert "سداد" in target.action
    assert "ستة أشهر" in target.deadline
    assert "مذكرة الدفاع" not in render_semantic_query(target)


def test_authority_target_tracks_signature_authority_claim() -> None:
    unit = _unit(
        "أوضح وكيل المدعى عليه أن الموظف غير مخول بالتوقيع وأن الفواتير كان يجب "
        "أن تكون باسم المؤسسة",
        UnitType.FACTS,
    )
    target = extract_semantic_target(QueryCategory.AUTHORITY, unit, _discovery(unit))
    assert target is not None
    assert "الموظف" in target.actor
    assert "التوقيع" in target.object
    assert "نظر الدعوى" not in render_semantic_query(target)


def test_case_holding_preserves_partial_award_and_excess_rejection() -> None:
    unit = _unit(
        "ألزمت المحكمة المدعى عليه بسداد 78,793 ريالاً للمدعية ورفضت ما زاد عن هذا المبلغ.",
        UnitType.VERDICT,
    )
    target = extract_semantic_target(QueryCategory.CASE_HOLDING, unit, _discovery(unit))
    assert target is not None
    answer = render_semantic_answer(target)
    assert "78,793" in answer
    assert "رفضت ما زاد" in answer
    assert validate_semantic_target(QueryCategory.CASE_HOLDING, target, (unit.text,))


def test_definition_and_condition_answers_are_not_generic_placeholders() -> None:
    definition = _unit(
        "يقصد بالوكالة تفويض شخص غيره للقيام بتصرف معلوم.",
        UnitType.APPLICABLE_LAWS,
    )
    definition_target = extract_semantic_target(
        QueryCategory.DEFINITION, definition, _discovery(definition)
    )
    assert definition_target is not None
    assert "المفهوم القانوني" not in render_semantic_answer(definition_target)
    assert "الوكالة" in render_semantic_answer(definition_target)

    condition = _unit(
        "إذا اتفق الأطراف على الصلح أمام الدائرة أثبتته في المحضر وصار سنداً تنفيذياً.",
        UnitType.APPLICABLE_LAWS,
    )
    condition_target = extract_semantic_target(
        QueryCategory.CONDITIONS, condition, _discovery(condition)
    )
    assert condition_target is not None
    assert "إذا اتفق الأطراف" in condition_target.condition
    assert "سنداً تنفيذياً" in condition_target.effect
    assert "يتوقف تطبيق القاعدة" not in render_semantic_answer(condition_target)


def test_multi_target_rejects_unentailed_conclusion() -> None:
    first = "أبرم الطرفان عقد بيع سيارة بثمن قدره 108,880 ريال."
    second = "حصر المدعي طلبه في تسليم 56,400 ريال."
    target = extract_semantic_target(
        QueryCategory.MULTI_EVIDENCE,
        _unit(first + " " + second, UnitType.COURT_REASONING),
        EvidenceDiscovery(0, len(first + " " + second), first + " " + second, 1),
        evidence_texts=(first, second),
        conclusion="ثبت فسخ العقد ورد السيارة إلى البائع.",
    )
    assert target is None


def test_prefix_or_list_only_provision_evidence_fails_closed() -> None:
    unit = _unit("1. ح المادة 29/2 من النظام", UnitType.COURT_REASONING)
    target = extract_semantic_target(QueryCategory.EXACT_PROVISION, unit, _discovery(unit))
    assert target is None


def test_case_holding_cannot_be_sourced_from_facts() -> None:
    unit = _unit("رفض المدعى عليه السداد في الوقائع", UnitType.FACTS)
    target = extract_semantic_target(QueryCategory.CASE_HOLDING, unit, _discovery(unit))
    assert target is None


def test_variants_are_target_specific_and_not_internal_metadata() -> None:
    unit = _unit(
        "المحكمة ألزمت المدعى عليه بسداد 78,793 ريالاً للمدعية.",
        UnitType.VERDICT,
    )
    target = extract_semantic_target(QueryCategory.CASE_HOLDING, unit, _discovery(unit))
    assert target is not None
    for variant in ("simple-ar", "egyptian-ar", "english", "code-switch"):
        query = render_semantic_query(target, variant)
        assert "intent" not in query.casefold()
        assert "internal" not in query.casefold()
        assert query.strip()


def test_semantic_target_is_typed_and_serializable() -> None:
    target = SemanticTarget(
        category=QueryCategory.AUTHORITY,
        actor="المحكمة",
        power="تختص",
        object="نظر الدعوى التجارية",
    )
    assert target.category is QueryCategory.AUTHORITY
    assert target.model_dump(mode="json")["actor"] == "المحكمة"
