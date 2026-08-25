# Arabic literals are intentional source-language test data.
# ruff: noqa: RUF001

from kawaneen.extraction.candidates import build_candidate_registry
from kawaneen.extraction.contracts import Calendar, CandidateType, NormalizationStatus
from kawaneen.extraction.normalization import (
    normalize_date,
    normalize_duration,
    normalize_money,
    normalize_number,
    normalize_percentage,
    normalize_reference,
)

TEXT = (
    "يلتزم المرخص له خلال ٣٠ يوماً بدفع ١٬٢٥٠٫٥٠ ريال سعودي، ونسبة ١٥٪، "
    "وتاريخ 12/03/2024، وتاريخ ١/٨/١٤٤٥هـ، وفق المادة (7) من نظام الإفلاس."
)


def test_candidate_registry_normalizes_arabic_digits_money_percent_dates_and_refs() -> None:
    registry = build_candidate_registry(
        TEXT,
        canonical_unit_id="u1",
        document_id="d1",
    )
    types = [candidate.candidate_type for candidate in registry.candidates]
    assert CandidateType.TEMPORAL in types
    assert CandidateType.MONETARY in types
    assert CandidateType.PERCENTAGE in types
    assert CandidateType.ARTICLE in types
    assert CandidateType.REGULATION in types
    money = next(
        item for item in registry.candidates if item.candidate_type is CandidateType.MONETARY
    )
    assert money.raw_exact_text == "١٬٢٥٠٫٥٠ ريال سعودي"
    assert money.normalized.normalized_value == "1250.50 SAR"
    hijri = next(item for item in registry.candidates if item.normalized.calendar is Calendar.HIJRI)
    assert hijri.normalization_status is NormalizationStatus.NORMALIZED
    assert hijri.normalized.normalized_value == "1445-08-01"


def test_persian_digits_and_safe_word_duration_are_supported_without_text_mutation() -> None:
    text = "يجب الإخطار خلال ۱۲ روزاً ولمدة شهر واحد."
    registry = build_candidate_registry(text, canonical_unit_id="u2", document_id="d2")
    assert text == "يجب الإخطار خلال ۱۲ روزاً ولمدة شهر واحد."
    temporal = [
        item for item in registry.candidates if item.candidate_type is CandidateType.TEMPORAL
    ]
    assert any(item.normalized.normalized_value == "12 days" for item in temporal)
    assert any(item.normalized.normalized_value == "1 month" for item in temporal)


def test_duplicate_exact_spans_are_emitted_once_and_ids_are_stable() -> None:
    text = "المادة (7) تنص على المادة (7)."
    first = build_candidate_registry(text, canonical_unit_id="u3", document_id="d3")
    second = build_candidate_registry(text, canonical_unit_id="u3", document_id="d3")
    assert first.model_dump() == second.model_dump()
    articles = [item for item in first.candidates if item.candidate_type is CandidateType.ARTICLE]
    assert len(articles) == 2
    assert [item.candidate_id for item in articles] == ["A001", "A002"]
    assert [item.span.start_char for item in first.candidates] == sorted(
        item.span.start_char for item in first.candidates
    )


def test_v3_duration_spans_include_bracketed_and_multiword_quantities() -> None:
    examples = {
        "(ثلاثين) يوم": "30 days",
        "(ثلاثين) يوماً": "30 days",
        "(مائة وثمانين)يوماً": "180 days",
        "(خمسة عشر) يومًا": "15 days",
        "ثلاثين يومًا": "30 days",
        "سنتين": "2 year",
        "ثلاثة أشهر": "3 months",
        "أربع سنوات": "4 years",
        "ستة أشهر": "6 months",
    }
    for text, expected in examples.items():
        candidates = build_candidate_registry(
            text, canonical_unit_id="u", document_id="d"
        ).candidates
        temporal = [item for item in candidates if item.candidate_type is CandidateType.TEMPORAL]
        assert len(temporal) == 1
        assert temporal[0].raw_exact_text == text
        assert temporal[0].normalized.normalized_value == expected
        assert text[temporal[0].span.start_char : temporal[0].span.end_char] == text


def test_v3_money_variants_and_textual_percentages_are_bounded() -> None:
    text = (
        "خمسمائة ريال، عشرة آلافريال، خمسة آلافربإل، سبعة آلافربال، "
        "خمسة عشر ألف ريإل، مائةربال، اثنين في المائة، خمسة في المائة"
    )
    registry = build_candidate_registry(text, canonical_unit_id="u", document_id="d")
    money = [item for item in registry.candidates if item.candidate_type is CandidateType.MONETARY]
    percentages = [
        item for item in registry.candidates if item.candidate_type is CandidateType.PERCENTAGE
    ]
    assert [item.normalized.normalized_value for item in money] == [
        "500 SAR",
        "10000 SAR",
        "5000 SAR",
        "7000 SAR",
        "15000 SAR",
        "100 SAR",
    ]
    assert [item.normalized.normalized_value for item in percentages] == ["2%", "5%"]
    assert all(
        text[item.span.start_char : item.span.end_char] == item.raw_exact_text
        for item in money + percentages
    )
    corrupt = build_candidate_registry(
        "(0015) (9075) (5 96)", canonical_unit_id="u", document_id="d"
    )
    assert not any(item.candidate_type is CandidateType.PERCENTAGE for item in corrupt.candidates)


def test_v3_spelled_article_references_are_candidates() -> None:
    text = "المادة (الثانية) والمادة (الثالثة) والمادة (التاسعة والعشرين)."
    registry = build_candidate_registry(text, canonical_unit_id="u", document_id="d")
    articles = [
        item for item in registry.candidates if item.candidate_type is CandidateType.ARTICLE
    ]
    assert [item.raw_exact_text for item in articles] == [
        "المادة (الثانية)",
        "المادة (الثالثة)",
        "المادة (التاسعة والعشرين)",
    ]
    assert [item.normalized.normalized_value for item in articles] == ["2", "3", "29"]


def test_v3_regulation_boundaries_prefer_precision() -> None:
    text = (
        "نظام مراقبة البنوك. نظام السوق المالية. نظام المرافعات الشرعية. "
        "نظام مراقبة شركات التأمين التعاوني. اللائحة التنفيذية لإجراءات الاستئناف. "
        "نظام بعد تسعين يوم من تاريخ نشره. لائحة على الوثائق القضائية الصادرة بعد سريان. "
        "لائحة كيفيةذلك. نظام المالي وعدالةالتعاملات. نظام الجلسة تبدأ."
    )
    registry = build_candidate_registry(text, canonical_unit_id="u", document_id="d")
    regulations = [
        item.raw_exact_text
        for item in registry.candidates
        if item.candidate_type is CandidateType.REGULATION
    ]
    assert regulations == [
        "نظام مراقبة البنوك",
        "نظام السوق المالية",
        "نظام المرافعات الشرعية",
        "نظام مراقبة شركات التأمين التعاوني",
        "اللائحة التنفيذية لإجراءات الاستئناف",
    ]


def test_v3_preserves_temporal_id_when_only_span_normalization_is_repaired() -> None:
    old_identity = build_candidate_registry(
        "خلال ثلاثين يوم", canonical_unit_id="u", document_id="d"
    ).candidates[0]
    repaired = build_candidate_registry(
        "خلال (ثلاثين) يوم", canonical_unit_id="u", document_id="d"
    ).candidates[0]
    assert old_identity.candidate_id == repaired.candidate_id == "T001"
    assert repaired.raw_exact_text == "(ثلاثين) يوم"


def test_normalization_contracts_cover_conservative_boundary_cases() -> None:
    assert normalize_number("٠١٬٢٥٠٫٥٠") == "1250.50"
    assert normalize_number("not-a-number") is None

    money, money_status = normalize_money("مائةربال")
    assert money_status is NormalizationStatus.NORMALIZED
    assert money.normalized_value == "100 SAR"
    unresolved_money, unresolved_status = normalize_money("نص غامض ريال")
    assert unresolved_status is NormalizationStatus.UNRESOLVED
    assert unresolved_money.normalized_value is None

    percentage, percentage_status = normalize_percentage("اثنين في المائة")
    assert percentage_status is NormalizationStatus.NORMALIZED
    assert percentage.normalized_value == "2%"
    _, unresolved_percentage_status = normalize_percentage("نص غامض في المائة")
    assert unresolved_percentage_status is NormalizationStatus.UNRESOLVED

    invalid_date, invalid_date_status = normalize_date("31/13/2024")
    assert invalid_date_status is NormalizationStatus.PARTIAL
    assert invalid_date.calendar is Calendar.GREGORIAN
    arabic_date, arabic_date_status = normalize_date("١٥ رمضان ١٤٤٥هـ")
    assert arabic_date_status is NormalizationStatus.NORMALIZED
    assert arabic_date.normalized_value == "1445-09-15"
    _, unresolved_date_status = normalize_date("15 شهر غير معروف 2024")
    assert unresolved_date_status is NormalizationStatus.UNRESOLVED

    malformed_duration, duration_status = normalize_duration(") ٣٠ (يوم")
    assert duration_status is NormalizationStatus.NORMALIZED
    assert malformed_duration.normalized_value == "30 days"
    singular_duration, singular_status = normalize_duration("شهر")
    assert singular_status is NormalizationStatus.NORMALIZED
    assert singular_duration.normalized_value == "1 month"
    _, unresolved_duration_status = normalize_duration("مدة غير محددة")
    assert unresolved_duration_status is NormalizationStatus.UNRESOLVED

    reference, reference_status = normalize_reference("المادة (7)", "article")
    assert reference_status is NormalizationStatus.NORMALIZED
    assert reference.normalized_value == "7"
    regulation, regulation_status = normalize_reference("نظام السوق المالية", "regulation")
    assert regulation_status is NormalizationStatus.PARTIAL
    assert regulation.normalized_value == "نظام السوق المالية"
