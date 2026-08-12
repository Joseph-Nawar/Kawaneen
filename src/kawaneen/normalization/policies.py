"""Versioned Arabic normalization policy definitions and pure transforms."""

from __future__ import annotations

# ruff: noqa: RUF001
import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable
from types import MappingProxyType
from typing import cast

from kawaneen.normalization.models import NormalizationPolicy, NormalizationResult

_ARABIC_DIACRITICS = frozenset(
    chr(code)
    for start, end in ((0x0610, 0x061A), (0x064B, 0x065F), (0x0670, 0x0670), (0x06D6, 0x06ED))
    for code in range(start, end + 1)
)
_ALEF_FOLD = {"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا"}
_DIGIT_FOLD = {
    **{chr(0x0660 + index): str(index) for index in range(10)},
    **{chr(0x06F0 + index): str(index) for index in range(10)},
}
_PUNCTUATION_FOLD = {
    "،": ",",
    "؛": ";",
    "؟": "?",
    "٫": ".",
    "٬": ",",
    "–": "-",
    "—": "-",
    "−": "-",
}
_WHITESPACE = re.compile(r"\s+")

_POLICY_CONFIGS: tuple[dict[str, object], ...] = (
    {
        "policy_id": "arabic-raw-v1",
        "schema_version": 1,
        "version": 1,
        "transforms": ("nfc", "bom_remove", "retrieval_whitespace"),
        "safe_format_chars": ("U+FEFF",),
    },
    {
        "policy_id": "arabic-light-v1",
        "schema_version": 1,
        "version": 1,
        "transforms": (
            "nfc",
            "bom_remove",
            "retrieval_whitespace",
            "tatweel_remove",
            "arabic_diacritics_remove",
            "alef_fold",
        ),
        "safe_format_chars": ("U+FEFF",),
        "diacritic_ranges": ("U+0610-U+061A", "U+064B-U+065F", "U+0670", "U+06D6-U+06ED"),
        "alef_variants": ("أ", "إ", "آ", "ٱ"),
    },
    {
        "policy_id": "arabic-aggressive-v1",
        "schema_version": 1,
        "version": 1,
        "transforms": (
            "nfc",
            "bom_remove",
            "retrieval_whitespace",
            "tatweel_remove",
            "arabic_diacritics_remove",
            "alef_fold",
            "ya_maqsura_fold",
            "ta_marbuta_fold_experimental",
            "arabic_digit_fold",
            "allowlisted_punctuation_fold",
        ),
        "safe_format_chars": ("U+FEFF",),
        "diacritic_ranges": ("U+0610-U+061A", "U+064B-U+065F", "U+0670", "U+06D6-U+06ED"),
        "alef_variants": ("أ", "إ", "آ", "ٱ"),
        "punctuation_allowlist": tuple(
            f"U+{ord(source):04X}->U+{ord(target):04X}"
            for source, target in _PUNCTUATION_FOLD.items()
        ),
    },
)


def _policy_hash(config: dict[str, object]) -> str:
    encoded = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _build_policy(config: dict[str, object]) -> NormalizationPolicy:
    return NormalizationPolicy(
        policy_id=str(config["policy_id"]),
        version=cast(int, config["version"]),
        transforms=tuple(str(value) for value in cast(tuple[object, ...], config["transforms"])),
        config=MappingProxyType(dict(config)),
        policy_hash=_policy_hash(config),
    )


_POLICIES = tuple(_build_policy(config) for config in _POLICY_CONFIGS)
_POLICY_BY_ID = {policy.policy_id: policy for policy in _POLICIES}


def all_policies() -> tuple[NormalizationPolicy, ...]:
    return _POLICIES


def get_policy(policy_id: str) -> NormalizationPolicy:
    return _POLICY_BY_ID[policy_id]


def _increment(counts: dict[str, int], name: str, amount: int = 1) -> None:
    if amount:
        counts[name] = counts.get(name, 0) + amount


def _map_characters(
    text: str, mapping: dict[str, str], count_name: str, counts: dict[str, int]
) -> str:
    changed = sum(1 for char in text if char in mapping and mapping[char] != char)
    _increment(counts, count_name, changed)
    return text.translate(str.maketrans(mapping))


def _remove_characters(
    text: str, chars: Iterable[str], count_name: str, counts: dict[str, int]
) -> str:
    char_set = frozenset(chars)
    changed = sum(1 for char in text if char in char_set)
    _increment(counts, count_name, changed)
    return text.translate({ord(char): None for char in char_set})


def _normalize_whitespace(text: str, counts: dict[str, int]) -> str:
    collapsed = _WHITESPACE.sub(" ", text).strip()
    if collapsed != text:
        _increment(counts, "whitespace_mapped", sum(char.isspace() for char in text))
    if _WHITESPACE.search(text) is not None or text != text.strip():
        _increment(counts, "whitespace_collapsed", 1)
    return collapsed


def normalize_text(
    text: str, policy: NormalizationPolicy, *, audit: bool = False
) -> str | NormalizationResult:
    """Normalize text according to one explicit policy without side effects."""

    counts: dict[str, int] = {}
    normalized = unicodedata.normalize("NFC", text)
    if normalized != text:
        _increment(counts, "nfc_changed", 1)
    bom_count = normalized.count("\ufeff")
    normalized = normalized.replace("\ufeff", "")
    _increment(counts, "bom_removed", bom_count)
    normalized = _normalize_whitespace(normalized, counts)

    transforms = set(policy.transforms)
    if "tatweel_remove" in transforms:
        normalized = _remove_characters(normalized, ("ـ",), "tatweel_removed", counts)
    if "arabic_diacritics_remove" in transforms:
        normalized = _remove_characters(
            normalized, _ARABIC_DIACRITICS, "diacritics_removed", counts
        )
    if "alef_fold" in transforms:
        normalized = _map_characters(normalized, _ALEF_FOLD, "alef_folded", counts)
    if "ya_maqsura_fold" in transforms:
        normalized = _map_characters(normalized, {"ى": "ي"}, "ya_maqsura_folded", counts)
    if "ta_marbuta_fold_experimental" in transforms:
        normalized = _map_characters(
            normalized, {"ة": "ه"}, "ta_marbuta_folded_experimental", counts
        )
    if "arabic_digit_fold" in transforms:
        normalized = _map_characters(normalized, _DIGIT_FOLD, "arabic_digits_folded", counts)
    if "allowlisted_punctuation_fold" in transforms:
        normalized = _map_characters(
            normalized, _PUNCTUATION_FOLD, "allowlisted_punctuation_folded", counts
        )
    normalized = _normalize_whitespace(normalized, counts)

    result = NormalizationResult(search_text=normalized, transform_counts=counts)
    return result if audit else result.search_text


def policy_configurations() -> tuple[dict[str, object], ...]:
    """Return serializable policy configurations for manifests and tests."""

    return tuple(dict(config) for config in _POLICY_CONFIGS)
