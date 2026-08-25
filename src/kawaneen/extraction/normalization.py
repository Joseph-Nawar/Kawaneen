"""Conservative, additive normalization for regulatory candidate text."""

# Arabic digit and separator literals are intentional source-language data.
# ruff: noqa: RUF001

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from kawaneen.corpus.statutory import parse_article_label
from kawaneen.extraction.contracts import (
    Calendar,
    NormalizationStatus,
    NormalizedRepresentation,
)

_DIGIT_TRANSLATION = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_ARABIC_MONTHS = {
    "يناير": 1,
    "فبراير": 2,
    "مارس": 3,
    "أبريل": 4,
    "ابريل": 4,
    "مايو": 5,
    "يونيو": 6,
    "يوليو": 7,
    "أغسطس": 8,
    "اغسطس": 8,
    "سبتمبر": 9,
    "أكتوبر": 10,
    "اكتوبر": 10,
    "نوفمبر": 11,
    "ديسمبر": 12,
}
_HIJRI_MONTHS = {
    "محرم": 1,
    "صفر": 2,
    "ربيع الأول": 3,
    "ربيع الاول": 3,
    "ربيع الآخر": 4,
    "ربيع الاخر": 4,
    "جمادى الأولى": 5,
    "جمادى الاولى": 5,
    "جمادى الآخرة": 6,
    "جمادى الاخرة": 6,
    "رجب": 7,
    "شعبان": 8,
    "رمضان": 9,
    "شوال": 10,
    "ذو القعدة": 11,
    "ذو الحجة": 12,
}

# These are deliberately explicit, conservative forms.  They cover recurring
# Saudi legal/OCR spellings without attempting unrestricted Arabic number
# understanding.
_DURATION_WORD_VALUES = {
    "واحد": 1,
    "واحدة": 1,
    "اثنان": 2,
    "اثنين": 2,
    "اثنتان": 2,
    "اثنتين": 2,
    "ثلاثة": 3,
    "ثلاث": 3,
    "أربعة": 4,
    "أربع": 4,
    "خمسة": 5,
    "خمس": 5,
    "ستة": 6,
    "ست": 6,
    "سبعة": 7,
    "سبع": 7,
    "ثمانية": 8,
    "ثمان": 8,
    "تسعة": 9,
    "تسع": 9,
    "عشرة": 10,
    "عشر": 10,
    "أحد عشر": 11,
    "إحدى عشرة": 11,
    "اثنا عشر": 12,
    "اثني عشر": 12,
    "ثلاثة عشر": 13,
    "ثلاث عشر": 13,
    "أربعة عشر": 14,
    "أربع عشرة": 14,
    "خمسة عشر": 15,
    "خمس عشرة": 15,
    "ستة عشر": 16,
    "ست عشرة": 16,
    "سبعة عشر": 17,
    "سبع عشرة": 17,
    "ثمانية عشر": 18,
    "ثماني عشرة": 18,
    "تسعة عشر": 19,
    "تسع عشرة": 19,
    "عشرين": 20,
    "ثلاثين": 30,
    "أربعين": 40,
    "خمسين": 50,
    "ستين": 60,
    "سبعين": 70,
    "ثمانين": 80,
    "تسعين": 90,
    "مائة": 100,
    "مئة": 100,
    "مائة وثمانين": 180,
    "مئة وثمانين": 180,
}

_MONEY_WORD_VALUES = {
    "مائة": 100,
    "مئة": 100,
    "خمسمائة": 500,
    "خمس مائة": 500,
    "خمسمئة": 500,
    "عشرة آلاف": 10_000,
    "عشرة الاف": 10_000,
    "خمسة آلاف": 5_000,
    "خمسة الاف": 5_000,
    "سبعة آلاف": 7_000,
    "سبعة الاف": 7_000,
    "خمسة عشر ألف": 15_000,
    "خمسة عشر الف": 15_000,
}

_PERCENTAGE_WORD_VALUES = {
    "واحد": 1,
    "واحدة": 1,
    "اثنين": 2,
    "اثنان": 2,
    "ثلاثة": 3,
    "ثلاث": 3,
    "أربعة": 4,
    "أربع": 4,
    "خمسة": 5,
    "خمس": 5,
    "عشرة": 10,
    "عشرين": 20,
    "ثلاثين": 30,
    "خمسين": 50,
}
_ARTICLE_REFERENCE_WORD_VALUES = {
    "التاسعة والعشرين": 29,
}


def ascii_digits(value: str) -> str:
    return value.translate(_DIGIT_TRANSLATION)


def normalize_number(value: str) -> str | None:
    """Normalize a decimal without guessing ambiguous separators."""

    cleaned = ascii_digits(value).replace("٬", ",").replace("٫", ".")
    cleaned = re.sub(r"\s+", "", cleaned)
    if "," in cleaned and "." in cleaned:
        decimal_separator = "." if cleaned.rfind(".") > cleaned.rfind(",") else ","
        thousands_separator = "," if decimal_separator == "." else "."
        cleaned = cleaned.replace(thousands_separator, "").replace(decimal_separator, ".")
    elif "," in cleaned:
        parts = cleaned.split(",")
        cleaned = ".".join(parts) if len(parts) == 2 and len(parts[1]) != 3 else "".join(parts)
    try:
        Decimal(cleaned)
    except InvalidOperation:
        return None
    rendered = cleaned
    if rendered.startswith("+"):
        rendered = rendered[1:]
    if "." in rendered:
        whole, fraction = rendered.split(".", 1)
        whole = whole.lstrip("0") or "0"
        rendered = f"{whole}.{fraction}"
    else:
        rendered = rendered.lstrip("0") or "0"
    return rendered


def normalize_money(raw_text: str) -> tuple[NormalizedRepresentation, NormalizationStatus]:
    match = re.search(
        r"(?P<number>[\d٠-٩۰-۹][\d٠-٩۰-۹\s.,٬٫]*|[\u0621-\u064A]+(?:\s+[\u0621-\u064A]+){0,3})\s*"
        r"(?P<currency>SAR|ريال(?:\s+سعودي)?|ر\.س|ربال|ريإل|ربإل)",
        raw_text,
        re.I,
    )
    if not match:
        return NormalizedRepresentation(), NormalizationStatus.UNRESOLVED
    raw_number = match.group("number").strip()
    number = normalize_number(raw_number)
    if number is None:
        number = (
            str(_MONEY_WORD_VALUES.get(re.sub(r"\s+", " ", raw_number)))
            if re.sub(r"\s+", " ", raw_number) in _MONEY_WORD_VALUES
            else None
        )
    if number is None:
        return NormalizedRepresentation(), NormalizationStatus.UNRESOLVED
    return (
        NormalizedRepresentation(
            original_components=(
                ("number", match.group("number")),
                ("currency", match.group("currency")),
            ),
            normalized_components=(("amount", number), ("currency", "SAR")),
            normalized_value=f"{number} SAR",
        ),
        NormalizationStatus.NORMALIZED,
    )


def normalize_percentage(raw_text: str) -> tuple[NormalizedRepresentation, NormalizationStatus]:
    match = re.search(r"(?P<number>[\d٠-٩۰-۹][\d٠-٩۰-۹\s.,٬٫]*)\s*(?:%|٪|في\s+المائة)", raw_text)
    if match:
        raw_number = match.group("number")
        number = normalize_number(raw_number)
    else:
        match = re.search(
            r"(?P<number>[\u0621-\u064A]+(?:\s+[\u0621-\u064A]+)*)\s*في\s+المائة",
            raw_text,
        )
        if not match:
            return NormalizedRepresentation(), NormalizationStatus.UNRESOLVED
        raw_number = match.group("number").strip()
        number_value = _PERCENTAGE_WORD_VALUES.get(re.sub(r"\s+", " ", raw_number))
        number = str(number_value) if number_value is not None else None
    if number is None:
        return NormalizedRepresentation(), NormalizationStatus.UNRESOLVED
    return (
        NormalizedRepresentation(
            original_components=(("number", raw_number),),
            normalized_components=(("percentage", number),),
            normalized_value=f"{number}%",
        ),
        NormalizationStatus.NORMALIZED,
    )


def normalize_date(raw_text: str) -> tuple[NormalizedRepresentation, NormalizationStatus]:
    numeric = re.search(
        r"(?P<day>[\d٠-٩۰-۹]{1,2})\s*[/\-.]\s*(?P<month>[\d٠-٩۰-۹]{1,2})\s*[/\-.]\s*(?P<year>[\d٠-٩۰-۹]{4})\s*(?P<hijri>هـ|ه)?",
        raw_text,
    )
    if numeric:
        day, month, year = (
            int(ascii_digits(numeric.group(name))) for name in ("day", "month", "year")
        )
        calendar = Calendar.HIJRI if numeric.group("hijri") else Calendar.GREGORIAN
        if not (1 <= day <= 31 and 1 <= month <= 12):
            return NormalizedRepresentation(calendar=calendar), NormalizationStatus.PARTIAL
        return (
            NormalizedRepresentation(
                calendar=calendar,
                original_components=(
                    ("day", numeric.group("day")),
                    ("month", numeric.group("month")),
                    ("year", numeric.group("year")),
                ),
                normalized_components=(
                    ("day", f"{day:02d}"),
                    ("month", f"{month:02d}"),
                    ("year", str(year)),
                ),
                normalized_value=f"{year:04d}-{month:02d}-{day:02d}",
            ),
            NormalizationStatus.NORMALIZED,
        )
    month_match = re.search(
        r"(?P<day>[\d٠-٩۰-۹]{1,2})\s+"
        r"(?P<month>[\u0621-\u064A ]+)\s+"
        r"(?P<year>[\d٠-٩۰-۹]{4})(?P<hijri>\s*(?:هـ|ه))?",
        raw_text,
    )
    if month_match:
        month_name = month_match.group("month").strip()
        month_number = _ARABIC_MONTHS.get(month_name) or _HIJRI_MONTHS.get(month_name)
        if month_number is None:
            return NormalizedRepresentation(), NormalizationStatus.UNRESOLVED
        day = int(ascii_digits(month_match.group("day")))
        year = int(ascii_digits(month_match.group("year")))
        calendar = (
            Calendar.HIJRI if month_match.group("hijri") or year >= 1300 else Calendar.GREGORIAN
        )
        return (
            NormalizedRepresentation(
                calendar=calendar,
                original_components=(
                    ("day", month_match.group("day")),
                    ("month", month_name),
                    ("year", month_match.group("year")),
                ),
                normalized_components=(
                    ("day", f"{day:02d}"),
                    ("month", f"{month_number:02d}"),
                    ("year", str(year)),
                ),
                normalized_value=f"{year:04d}-{month_number:02d}-{day:02d}",
            ),
            NormalizationStatus.NORMALIZED,
        )
    return NormalizedRepresentation(), NormalizationStatus.UNRESOLVED


def normalize_duration(raw_text: str) -> tuple[NormalizedRepresentation, NormalizationStatus]:
    duration_words = "|".join(
        re.escape(item) for item in sorted(_DURATION_WORD_VALUES, key=len, reverse=True)
    )
    units = (
        r"يومين|يوماً|يومًا|يوما|يوم|أياماً|أيامًا|أياما|أيام|ايام|روزاً|روزا|روز|"
        r"شهرين|شهراً|شهرًا|شهرا|شهر|أشهر|اشهر|سنتين|سنةً|سنة|سنوات|عامين|عام|أعوام"
    )
    malformed = re.fullmatch(
        rf"[\u200e\u200f]*\)\s*(?P<number>[\d٠-٩۰-۹]+)\s*\([\u200e\u200f]*\s*(?P<unit>{units})",
        raw_text.strip(),
    )
    if malformed:
        number = int(ascii_digits(malformed.group("number")))
        unit = malformed.group("unit")
        canonical = (
            "day"
            if unit
            in {
                "يومين",
                "يوم",
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
            }
            else "month"
            if unit in {"شهرين", "شهراً", "شهرًا", "شهرا", "شهر", "أشهر", "اشهر"}
            else "year"
        )
        return NormalizedRepresentation(
            original_components=(("expression", raw_text),),
            normalized_components=(("amount", str(number)), ("unit", canonical)),
            normalized_value=f"{number} {canonical}{'s' if number != 1 else ''}",
        ), NormalizationStatus.NORMALIZED
    match = re.search(
        rf"\(?\s*(?P<number>[\d٠-٩۰-۹]+|{duration_words})\s*\)?\s*"
        r"(?P<unit>يومين|يوماً|يومًا|يوما|يوم|أياماً|أيامًا|أياما|أيام|ايام|روزاً|روزا|روز|شهرين|شهراً|شهرًا|شهرا|شهر|أشهر|اشهر|سنتين|سنةً|سنة|سنوات|عامين|عام|أعوام)",
        raw_text,
    )
    if match:
        raw_number = match.group("number").strip()
        numeric = ascii_digits(raw_number)
        number = (
            int(numeric)
            if numeric.isdigit()
            else _DURATION_WORD_VALUES.get(re.sub(r"\s+", " ", raw_number))
        )
        if number is None:
            return NormalizedRepresentation(), NormalizationStatus.UNRESOLVED
        unit = match.group("unit")
        canonical = (
            "day"
            if unit
            in {
                "يومين",
                "يوم",
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
            }
            else "month"
            if unit in {"شهرين", "شهراً", "شهرًا", "شهرا", "شهر", "أشهر", "اشهر"}
            else "year"
        )
        return (
            NormalizedRepresentation(
                original_components=(("number", match.group("number")), ("unit", unit)),
                normalized_components=(("amount", str(number)), ("unit", canonical)),
                normalized_value=f"{number} {canonical}{'s' if number != 1 else ''}",
            ),
            NormalizationStatus.NORMALIZED,
        )
    singular = {
        "يوم واحد": (1, "day"),
        "يوم": (1, "day"),
        "شهر واحد": (1, "month"),
        "شهر": (1, "month"),
        "شهرين": (2, "month"),
        "سنة واحدة": (1, "year"),
        "سنة": (1, "year"),
        "سنتين": (2, "year"),
    }
    stripped = raw_text.strip()
    if stripped in singular:
        number, unit = singular[stripped]
        normalized = f"{number} {unit}"
        return (
            NormalizedRepresentation(
                original_components=(("expression", raw_text),),
                normalized_components=(("amount", str(number)), ("unit", unit)),
                normalized_value=normalized,
            ),
            NormalizationStatus.NORMALIZED,
        )
    return NormalizedRepresentation(), NormalizationStatus.UNRESOLVED


def normalize_reference(
    raw_text: str, candidate_type: str
) -> tuple[NormalizedRepresentation, NormalizationStatus]:
    if candidate_type == "article":
        parseable = re.sub(r"\(\s*([^()]*)\s*\)", r"\1", raw_text)
        parsed = parse_article_label(parseable)
        if parsed.ordinal is None:
            body = re.sub(r"^الم(?:ادة|اده)\s*", "", parseable).strip()
            ordinal = _ARTICLE_REFERENCE_WORD_VALUES.get(body)
            if ordinal is not None:
                return (
                    NormalizedRepresentation(
                        original_components=(("article", raw_text),),
                        normalized_components=(("article", str(ordinal)),),
                        normalized_value=str(ordinal),
                    ),
                    NormalizationStatus.NORMALIZED,
                )
        if parsed.ordinal is not None:
            return (
                NormalizedRepresentation(
                    original_components=(("article", raw_text),),
                    normalized_components=(("article", str(parsed.ordinal)),),
                    normalized_value=str(parsed.ordinal),
                ),
                NormalizationStatus.NORMALIZED,
            )
    digits = re.search(r"[\d٠-٩۰-۹]+", raw_text)
    if candidate_type == "article" and digits:
        number = ascii_digits(digits.group(0))
        return (
            NormalizedRepresentation(
                original_components=(("article", digits.group(0)),),
                normalized_components=(("article", number),),
                normalized_value=number,
            ),
            NormalizationStatus.NORMALIZED,
        )
    if candidate_type == "regulation":
        return NormalizedRepresentation(normalized_value=raw_text), NormalizationStatus.PARTIAL
    return NormalizedRepresentation(), NormalizationStatus.UNRESOLVED
