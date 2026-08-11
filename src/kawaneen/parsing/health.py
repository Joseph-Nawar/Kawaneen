"""Lazy pypdf embedded-text health probe."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, cast

from kawaneen.parsing.models import PageHealth


def probe_pdf(path: Path) -> tuple[PageHealth, ...]:
    """Probe page text and image presence; requires the optional pypdf group."""

    try:
        pypdf: Any = cast(Any, importlib.import_module("pypdf"))
    except ImportError as exc:
        raise RuntimeError("PDF probing requires the optional parsing dependencies") from exc
    reader: Any = pypdf.PdfReader(str(path))
    pages: list[PageHealth] = []
    for number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        resources: Any = page.get("/Resources", {})
        xobjects: Any = resources.get("/XObject", {}) if resources else {}
        image_count = sum(
            1 for item in xobjects.values() if item.get_object().get("/Subtype") == "/Image"
        )
        pages.append(PageHealth(page_number=number, text_chars=len(text), image_count=image_count))
    return tuple(pages)
