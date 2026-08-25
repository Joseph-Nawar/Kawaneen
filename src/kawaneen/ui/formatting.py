"""Safe text presentation helpers for legal evidence."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Literal

from kawaneen.api.contracts import DocumentUnit


@dataclass(frozen=True)
class QuoteLocation:
    unit_id: str
    start_char: int
    end_char: int


def contains_arabic(text: str) -> bool:
    return any("\u0600" <= character <= "\u06ff" for character in text)


def text_direction(text: str) -> Literal["rtl", "ltr"]:
    return "rtl" if contains_arabic(text) else "ltr"


def highlight_literal(text: str, query: str) -> str:
    if not query.strip():
        return html.escape(text)
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    parts: list[str] = []
    cursor = 0
    for match in pattern.finditer(text):
        parts.append(html.escape(text[cursor : match.start()]))
        parts.append(f'<mark class="query-hit">{html.escape(match.group(0))}</mark>')
        cursor = match.end()
    parts.append(html.escape(text[cursor:]))
    return "".join(parts)


def locate_quote(units: tuple[DocumentUnit, ...], quote: str) -> QuoteLocation | None:
    if not quote:
        return None
    for unit in units:
        start = unit.text.find(quote)
        if start >= 0:
            return QuoteLocation(unit.unit_id, start, start + len(quote))
    return None
