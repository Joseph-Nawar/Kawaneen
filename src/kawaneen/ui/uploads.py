"""In-memory upload validation and bounded text segmentation."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import PurePath

from pypdf import PdfReader


@dataclass(frozen=True)
class UploadDecision:
    accepted: bool
    reason: str = ""


@dataclass(frozen=True)
class TextSegment:
    segment_id: str
    text: str
    start_char: int
    end_char: int


def validate_upload(name: str, size_bytes: int, allowed_bytes: int) -> UploadDecision:
    suffix = PurePath(name).suffix.lower()
    if suffix not in {".txt", ".md", ".pdf"}:
        return UploadDecision(False, "Only .txt, .md, and text-based .pdf files are supported.")
    if size_bytes > allowed_bytes:
        return UploadDecision(False, f"File exceeds the {allowed_bytes}-byte limit.")
    if size_bytes < 1:
        return UploadDecision(False, "The uploaded file is empty.")
    return UploadDecision(True)


def extract_text(name: str, payload: bytes) -> str:
    suffix = PurePath(name).suffix.lower()
    if suffix in {".txt", ".md"}:
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("Text files must be UTF-8 encoded.") from error
    elif suffix == ".pdf":
        try:
            reader = PdfReader(io.BytesIO(payload))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as error:
            raise ValueError("This PDF has no readable text; scanned-document OCR belongs to the ingestion pipeline.") from error
    else:
        raise ValueError("Only .txt, .md, and text-based .pdf files are supported.")
    if not text.strip():
        raise ValueError("This file has no readable text; scanned-document OCR belongs to the ingestion pipeline.")
    return text


def segment_text(text: str, max_chars: int = 18_000, max_segments: int = 5) -> tuple[TextSegment, ...]:
    if not text.strip():
        raise ValueError("Text must contain at least one non-whitespace character.")
    if max_chars < 1 or max_chars > 20_000:
        raise ValueError("max_chars must be between 1 and 20,000.")
    raw_paragraphs = [part for part in re.split(r"\n\s*\n", text) if part.strip()]
    pieces: list[tuple[int, int, str]] = []
    for paragraph in raw_paragraphs:
        paragraph_start = text.find(paragraph, pieces[-1][1] if pieces else 0)
        if len(paragraph) <= max_chars:
            pieces.append((paragraph_start, paragraph_start + len(paragraph), paragraph))
            continue
        cursor = 0
        while cursor < len(paragraph):
            limit = min(cursor + max_chars, len(paragraph))
            if limit < len(paragraph):
                boundary = paragraph.rfind(" ", cursor, limit)
                if boundary > cursor:
                    limit = boundary
            piece = paragraph[cursor:limit].rstrip()
            if not piece:
                limit = min(cursor + max_chars, len(paragraph))
                piece = paragraph[cursor:limit]
            absolute_start = paragraph_start + cursor
            pieces.append((absolute_start, absolute_start + len(piece), piece))
            cursor += len(piece)
            while cursor < len(paragraph) and paragraph[cursor].isspace():
                cursor += 1
    if len(pieces) > max_segments:
        count_label = "five" if max_segments == 5 else str(max_segments)
        raise ValueError(
            f"Input requires more than {count_label} segments; nothing was silently truncated."
        )
    return tuple(
        TextSegment(f"segment-{index:03d}", piece, start, end)
        for index, (start, end, piece) in enumerate(pieces, start=1)
    )
