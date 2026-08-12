"""Sanitized identifier and token-boundary safety diagnostics."""

from __future__ import annotations

# ruff: noqa: RUF001
import re
from dataclasses import dataclass

_SEPARATOR_EQUIVALENTS = {
    "،": ",",
    "؛": ";",
    "؟": "?",
    "٫": ".",
    "٬": ",",
    "–": "-",
    "—": "-",
    "−": "-",
}
_SEPARATORS = frozenset({"/", "-", ".", ",", ";", "?", ":", "(", ")"})
_DIGIT = re.compile(r"[0-9٠-٩۰-۹]")
_SAFE_IGNORABLES = frozenset(
    {"\ufeff", "\u0640"}
    | {chr(codepoint) for codepoint in range(0x0610, 0x061B)}
    | {chr(codepoint) for codepoint in range(0x064B, 0x0660)}
    | {"\u0670"}
    | {chr(codepoint) for codepoint in range(0x06D6, 0x06EE)}
)


@dataclass(frozen=True, slots=True)
class SafetyResult:
    safe: bool
    reasons: tuple[str, ...]


def _ignored_for_boundary(char: str) -> bool:
    return char.isspace() or char in _SAFE_IGNORABLES


def _separator_signature(text: str) -> tuple[tuple[str, bool, bool], ...]:
    signature: list[tuple[str, bool, bool]] = []
    for index, char in enumerate(text):
        separator = _SEPARATOR_EQUIVALENTS.get(char, char)
        if separator not in _SEPARATORS:
            continue
        left = index - 1
        while left >= 0 and _ignored_for_boundary(text[left]):
            left -= 1
        right = index + 1
        while right < len(text) and _ignored_for_boundary(text[right]):
            right += 1
        signature.append(
            (
                separator,
                left >= 0 and _DIGIT.fullmatch(text[left]) is not None,
                right < len(text) and _DIGIT.fullmatch(text[right]) is not None,
            )
        )
    return tuple(signature)


def _word_boundary_count(text: str) -> int:
    count = 0
    index = 0
    while index < len(text):
        if not text[index].isspace():
            index += 1
            continue
        left = index - 1
        while left >= 0 and _ignored_for_boundary(text[left]):
            left -= 1
        right = index
        while right < len(text) and text[right].isspace():
            right += 1
        while right < len(text) and text[right] in _SAFE_IGNORABLES:
            right += 1
        if left >= 0 and right < len(text) and text[left].isalnum() and text[right].isalnum():
            count += 1
        index = right
    return count


def validate_identifier_safety(text: str, normalized: str) -> SafetyResult:
    """Check that normalization preserves identifier separators and token boundaries."""

    reasons: set[str] = set()
    if _separator_signature(text) != _separator_signature(normalized):
        reasons.add("token_boundary_changed")
    if _word_boundary_count(normalized) < _word_boundary_count(text):
        reasons.add("token_boundary_changed")

    expected_separator_counts: dict[str, int] = {}
    actual_separator_counts: dict[str, int] = {}
    for char in text:
        canonical = _SEPARATOR_EQUIVALENTS.get(char, char)
        if canonical in _SEPARATORS:
            expected_separator_counts[canonical] = expected_separator_counts.get(canonical, 0) + 1
    for char in normalized:
        canonical = _SEPARATOR_EQUIVALENTS.get(char, char)
        if canonical in _SEPARATORS:
            actual_separator_counts[canonical] = actual_separator_counts.get(canonical, 0) + 1
    for separator, expected in expected_separator_counts.items():
        if actual_separator_counts.get(separator, 0) < expected:
            reasons.add("separator_deleted")
            break

    return SafetyResult(safe=not reasons, reasons=tuple(sorted(reasons)))
