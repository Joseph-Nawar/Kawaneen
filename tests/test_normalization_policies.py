from __future__ import annotations

import unicodedata

# ruff: noqa: RUF001
import pytest

from kawaneen.normalization import (
    NormalizationResult,
    all_policies,
    get_policy,
    normalize_text,
)


def test_all_policies_are_versioned_and_have_stable_hashes() -> None:
    policies = all_policies()
    assert tuple(policy.policy_id for policy in policies) == (
        "arabic-raw-v1",
        "arabic-light-v1",
        "arabic-aggressive-v1",
    )
    assert len({policy.policy_hash for policy in policies}) == 3
    assert all(len(policy.policy_hash) == 64 for policy in policies)
    assert all(policy.version == 1 for policy in policies)
    assert policies == all_policies()


def test_unknown_policy_is_rejected() -> None:
    with pytest.raises(KeyError):
        get_policy("unknown-v1")


def test_raw_normalizes_nfc_bom_and_retrieval_whitespace() -> None:
    text = "\ufeffا\u0654\u00a0\n\tب  "
    assert normalize_text(text, get_policy("arabic-raw-v1")) == "أ ب"


def test_empty_and_non_allowlisted_format_control_inputs_are_preserved() -> None:
    for policy in all_policies():
        assert normalize_text("", policy) == ""
        result = normalize_text("\u200bأ", policy)
        assert result.startswith("\u200b")
        assert result != "\u200b"


def test_light_removes_only_explicit_arabic_variants() -> None:
    text = "أ إ آ ٱ ا ـ بَ بِ بُ بْ بّ بً بٍ بٌ"
    assert normalize_text(text, get_policy("arabic-light-v1")) == "ا ا ا ا ا ب ب ب ب ب ب ب ب"


def test_aggressive_adds_experimental_forms_digits_and_allowlisted_punctuation() -> None:
    text = "ى ة ٠١٢٣٤٥٦٧٨٩ ۰۱۲۳۴۵۶۷۸۹ ، ؛ ؟ ٫ ٬ – — −"
    assert normalize_text(text, get_policy("arabic-aggressive-v1")) == (
        "ي ه 0123456789 0123456789 , ; ? . , - - -"
    )


def test_disallowed_normalization_is_not_applied() -> None:
    text = "ﻻ café ١/٢ مادة-١"
    result = normalize_text(text, get_policy("arabic-aggressive-v1"))
    assert result == text.replace("١", "1").replace("٢", "2").replace("ة", "ه")
    assert "ﻻ" in result
    assert "é" in result


def test_normalization_is_idempotent_for_every_policy() -> None:
    text = "\ufeffأ\u0301\u00a0ـ بَ ى ة ١٢،"
    for policy in all_policies():
        once = normalize_text(text, policy)
        assert normalize_text(once, policy) == once


def test_audit_result_is_typed_and_has_transform_counts() -> None:
    result = normalize_text("أـبَ", get_policy("arabic-light-v1"), audit=True)
    assert isinstance(result, NormalizationResult)
    assert result.search_text == "اب"
    assert result.transform_counts["tatweel_removed"] == 1
    assert result.transform_counts["diacritics_removed"] == 1
    assert result.transform_counts["alef_folded"] == 1


def test_normalizer_does_not_apply_nfkc() -> None:
    text = "ﻻ"
    assert unicodedata.normalize("NFKC", text) != text
    assert normalize_text(text, get_policy("arabic-aggressive-v1")) == text
