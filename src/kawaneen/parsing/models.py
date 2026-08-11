"""Typed parser routes and page-level provenance."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ParseRoute(StrEnum):
    EMBEDDED_TEXT = "healthy_embedded_text"
    DAMAGED_MIXED = "damaged_mixed_text"
    FULL_PAGE_OCR = "image_only_scan"


class PageHealth(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    page_number: int = Field(ge=1)
    text_chars: int = Field(ge=0)
    image_count: int = Field(ge=0)
    suspicious_text: bool = False


class ParserProvenance(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    parser: str
    parser_version: str
    ocr_engine: str = "none"
    ocr_model: str = "none"
    page_number: int = Field(ge=1)
    extraction_method: str
    text: str = ""
    bounding_box: tuple[float, float, float, float] | None = None
    coordinate_origin: str = "bottom-left"
    block_type: str = "unknown"
    reading_order: int | None = Field(default=None, ge=1)
