from __future__ import annotations

# ruff: noqa: RUF001
from kawaneen.normalization.safety import validate_identifier_safety


def test_identifier_digit_and_punctuation_equivalents_are_safe() -> None:
    result = validate_identifier_safety("المادة (١٢/١٤٤٥) م/١٢٣", "الماده (12/1445) م/123")
    assert result.safe
    assert result.reasons == ()


def test_identifier_separator_deletion_is_unsafe() -> None:
    result = validate_identifier_safety("م/١٢٣-٤", "م/1234")
    assert not result.safe
    assert "separator_deleted" in result.reasons


def test_identifier_token_concatenation_is_unsafe() -> None:
    result = validate_identifier_safety("قرار ١٢٣ / ٢٠٢٤", "قرار123/2024")
    assert not result.safe
    assert "token_boundary_changed" in result.reasons


def test_mixed_digit_forms_preserve_identifier_structure() -> None:
    result = validate_identifier_safety("مرجع ١٢3-٤", "مرجع 123-4")
    assert result.safe


def test_date_decree_and_reference_separators_remain_safe() -> None:
    result = validate_identifier_safety(
        "مرسوم رقم ١٢٣-أ بتاريخ ١٤٤٥/٠١/٣١",
        "مرسوم رقم 123-أ بتاريخ 1445/01/31",
    )
    assert result.safe
