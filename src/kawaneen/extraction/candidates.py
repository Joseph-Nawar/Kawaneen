"""Request-local deterministic candidate registry."""

# Arabic digit and separator literals are intentional source-language data.
# ruff: noqa: RUF001

from __future__ import annotations

import re
from dataclasses import dataclass

from kawaneen.extraction.contracts import (
    Candidate,
    CandidateRegistry,
    CandidateType,
    ExactSourceSpan,
    NormalizationStatus,
)
from kawaneen.extraction.normalization import (
    normalize_date,
    normalize_duration,
    normalize_money,
    normalize_percentage,
    normalize_reference,
)

CANDIDATE_REGISTRY_VERSION = "phase11-candidates-v3"


@dataclass(frozen=True)
class _Match:
    start: int
    end: int
    candidate_type: CandidateType
    raw: str


_DATE = re.compile(
    r"[\d٠-٩۰-۹]{1,2}\s*[/\-.]\s*[\d٠-٩۰-۹]{1,2}\s*[/\-.]\s*[\d٠-٩۰-۹]{4}\s*(?:هـ|ه)?"
)
_RIYAL_VARIANTS = r"(?:SAR|ريال(?:\s+سعودي)?|ر\.س|ربال|ريإل|ربإل)"
_MONEY_WORDS = (
    "خمسة عشر",
    "عشرة",
    "خمسة",
    "سبعة",
    "خمسمائة",
    "خمس مائة",
    "خمسمئة",
    "مائة",
    "مئة",
)
_MONEY = re.compile(
    rf"(?:[\d٠-٩۰-۹][\d٠-٩۰-۹\s.,٬٫]*|(?:{'|'.join(_MONEY_WORDS)})(?:\s*آلاف?|\s*الاف|\s*ألف)?)\s*{_RIYAL_VARIANTS}",
    re.I,
)
_PERCENTAGE_WORDS = "اثنين|اثنان|خمسة|خمس|ثلاثة|ثلاث|أربعة|أربع|واحد|واحدة|عشرة|عشرين|ثلاثين|خمسين"
_PERCENTAGE = re.compile(
    rf"(?:[\d٠-٩۰-۹][\d٠-٩۰-۹\s.,٬٫]*|(?:{_PERCENTAGE_WORDS}))\s*(?:%|٪|في\s+المائة)"
)
_DURATION_WORDS = (
    "مائة وثمانين",
    "مئة وثمانين",
    "خمسة عشر",
    "خمس عشرة",
    "أربعة عشر",
    "أربع عشرة",
    "ثلاثة عشر",
    "ثلاث عشر",
    "أحد عشر",
    "إحدى عشرة",
    "اثنا عشر",
    "اثني عشر",
    "تسعة عشر",
    "تسع عشرة",
    "ثمانية عشر",
    "ثماني عشرة",
    "سبعة عشر",
    "سبع عشرة",
    "ستة عشر",
    "ست عشرة",
    "ثلاثين",
    "عشرين",
    "أربعين",
    "خمسين",
    "ستين",
    "سبعين",
    "ثمانين",
    "تسعين",
    "ثلاثة",
    "ثلاث",
    "أربعة",
    "أربع",
    "خمسة",
    "خمس",
    "ستة",
    "ست",
    "سبعة",
    "سبع",
    "ثمانية",
    "ثمان",
    "تسعة",
    "تسع",
    "عشرة",
    "عشر",
)
_DURATION_UNITS = (
    "يومين",
    "يوماً",
    "يومًا",
    "يوما",
    "أياماً",
    "أيامًا",
    "أياما",
    "أيام",
    "ايام",
    "روزاً",
    "روزا",
    "روز",
    "شهرين",
    "شهراً",
    "شهرًا",
    "شهرا",
    "أشهر",
    "اشهر",
    "سنوات",
    "سنتين",
    "سنةً",
    "سنة",
    "عامين",
    "عام",
    "أعوام",
    "يوم",
    "شهر",
)
_DURATION_UNIT_PATTERN = "|".join(_DURATION_UNITS)
_DURATION_WORD_PATTERN = "|".join(_DURATION_WORDS)
_DURATION = re.compile(
    rf"(?<![\u0621-\u064A])(?:[\u200e\u200f]*\)\s*[\d٠-٩۰-۹]+\s*\([\u200e\u200f]*\s*(?:{_DURATION_UNIT_PATTERN})|"
    rf"\(?\s*(?:[\d٠-٩۰-۹]+|{_DURATION_WORD_PATTERN})\s*\)?\s*(?:{_DURATION_UNIT_PATTERN})|"
    r"شهر واحد|سنة واحدة|يوم واحد|سنتين|شهرين|يوم|شهر|سنة)(?![\u0621-\u064A])"
)
_ARTICLE_WORDS = (
    "الحادية عشرة",
    "الثانية عشرة",
    "الثالثة عشرة",
    "الرابعة عشرة",
    "الخامسة عشرة",
    "السادسة عشرة",
    "السابعة عشرة",
    "الثامنة عشرة",
    "التاسعة عشرة",
    "التاسعة والعشرين",
    "الأولى",
    "الاولى",
    "الثانية",
    "الثالثة",
    "الرابعة",
    "الخامسة",
    "السادسة",
    "السابعة",
    "الثامنة",
    "التاسعة",
    "العاشرة",
)
_ARTICLE = re.compile(
    rf"(?:(?<![\u0621-\u064A])|(?<=و))الم(?:ادة|اده)\s*(?:\(\s*(?:[\d٠-٩۰-۹]+|{'|'.join(_ARTICLE_WORDS)})\s*\)|[\d٠-٩۰-۹]+|{'|'.join(_ARTICLE_WORDS)})(?![\u0621-\u064A])"
)
_REGULATION_PREFIX = re.compile(
    r"(?<![\u0621-\u064A])(?:نظام|اللائحة\s+التنفيذية)(?![\u0621-\u064A])"
)
_REGULATION_STOP_WORDS = {
    "بعد",
    "دون",
    "دعوى",
    "جواب",
    "الجلسة",
    "على",
    "من",
    "في",
    "قبل",
    "المالي",
    "كيفية",
}
_REGULATION_TOKEN = re.compile(r"[\u0621-\u064A]+")


def _matches(text: str) -> list[_Match]:
    matches: list[_Match] = []
    patterns = (
        (_DATE, CandidateType.TEMPORAL),
        (_DURATION, CandidateType.TEMPORAL),
        (_MONEY, CandidateType.MONETARY),
        (_PERCENTAGE, CandidateType.PERCENTAGE),
        (_ARTICLE, CandidateType.ARTICLE),
    )
    for pattern, candidate_type in patterns:
        matches.extend(
            _Match(item.start(), item.end(), candidate_type, item.group(0))
            for item in pattern.finditer(text)
        )
    matches.extend(_regulation_matches(text))
    unique: dict[tuple[int, int], _Match] = {}
    for item in sorted(
        matches, key=lambda value: (value.start, value.end, value.candidate_type.value)
    ):
        unique.setdefault((item.start, item.end), item)
    return sorted(
        unique.values(), key=lambda value: (value.start, value.end, value.candidate_type.value)
    )


def _regulation_matches(text: str) -> list[_Match]:
    """Find bounded named-instrument shapes, never bare lexical continuations."""

    results: list[_Match] = []
    for prefix in _REGULATION_PREFIX.finditer(text):
        cursor = prefix.end()
        tokens: list[re.Match[str]] = []
        while len(tokens) < 6:
            whitespace = re.match(r"\s+", text[cursor:])
            if whitespace is None:
                break
            token_start = cursor + whitespace.end()
            token = _REGULATION_TOKEN.match(text, token_start)
            if token is None:
                break
            if token.group(0) in _REGULATION_STOP_WORDS:
                break
            tokens.append(token)
            cursor = token.end()
        if not tokens:
            continue
        first = tokens[0].group(0)
        if text[prefix.start() : prefix.end()].startswith("نظام"):
            if first in _REGULATION_STOP_WORDS:
                continue
            if not first.startswith("ال") and len(tokens) < 2:
                continue
        results.append(
            _Match(
                prefix.start(),
                tokens[-1].end(),
                CandidateType.REGULATION,
                text[prefix.start() : tokens[-1].end()],
            )
        )
    return results


def build_candidate_registry(
    canonical_text: str,
    *,
    canonical_unit_id: str,
    document_id: str,
) -> CandidateRegistry:
    counters = {candidate_type: 0 for candidate_type in CandidateType}
    candidates: list[Candidate] = []
    for item in _matches(canonical_text):
        counters[item.candidate_type] += 1
        prefix = {
            CandidateType.TEMPORAL: "T",
            CandidateType.MONETARY: "M",
            CandidateType.PERCENTAGE: "P",
            CandidateType.ARTICLE: "A",
            CandidateType.REGULATION: "R",
        }[item.candidate_type]
        span = ExactSourceSpan(
            text=item.raw,
            start_char=item.start,
            end_char=item.end,
            canonical_unit_id=canonical_unit_id,
            document_id=document_id,
        )
        if item.candidate_type is CandidateType.TEMPORAL:
            normalized, status = normalize_date(item.raw)
            if status is NormalizationStatus.UNRESOLVED:
                normalized, status = normalize_duration(item.raw)
        elif item.candidate_type is CandidateType.MONETARY:
            normalized, status = normalize_money(item.raw)
        elif item.candidate_type is CandidateType.PERCENTAGE:
            normalized, status = normalize_percentage(item.raw)
        elif item.candidate_type is CandidateType.ARTICLE:
            normalized, status = normalize_reference(item.raw, "article")
        else:
            normalized, status = normalize_reference(item.raw, "regulation")
        candidates.append(
            Candidate(
                candidate_id=f"{prefix}{counters[item.candidate_type]:03d}",
                candidate_type=item.candidate_type,
                span=span,
                raw_exact_text=item.raw,
                normalized=normalized,
                normalization_status=status,
            )
        )
    return CandidateRegistry(
        canonical_text=canonical_text,
        canonical_unit_id=canonical_unit_id,
        document_id=document_id,
        candidates=tuple(candidates),
    )
