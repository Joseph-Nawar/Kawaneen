"""Lazy Docling boundary and explicit RapidOCR provenance."""

from __future__ import annotations

import importlib
import re
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

from kawaneen.parsing.models import ParserProvenance

_STRUCTURAL_HEADING = re.compile(r"^(?:الباب|الفصل|القسم|المبحث)\b", re.UNICODE)


def classify_legal_block(text: str, docling_label: str) -> str:
    """Use Docling structure plus conservative Arabic legal-heading rules."""

    from kawaneen.corpus.statutory import parse_article_label

    stripped = text.strip()
    if parse_article_label(stripped).article_ordinal is not None:
        return "article_label"
    if _STRUCTURAL_HEADING.match(stripped):
        return "heading"
    if "section_header" in docling_label.lower():
        return "heading"
    return docling_label


class DoclingBackend:
    """Optional layout-aware parser; importing this class never imports Docling."""

    def __init__(
        self, *, ocr: bool = False, ocr_model: str = "PaddleOCR PP-OCRv5 Arabic candidate"
    ) -> None:
        self.ocr = ocr
        self.ocr_model = ocr_model if ocr else "none"

    def parse(self, path: Path) -> tuple[ParserProvenance, ...]:
        try:
            base_models: Any = cast(Any, importlib.import_module("docling.datamodel.base_models"))
            pipeline_options: Any = cast(
                Any, importlib.import_module("docling.datamodel.pipeline_options")
            )
            module: Any = cast(Any, importlib.import_module("docling.document_converter"))
        except ImportError as exc:
            raise RuntimeError(
                "Docling is required for layout parsing; install the parsing group"
            ) from exc
        options = pipeline_options.PdfPipelineOptions()
        options.do_ocr = self.ocr
        options.do_table_structure = False
        options.heading_hierarchy_options.enabled = True
        options.heading_hierarchy_options.use_numbering = True
        options.heading_hierarchy_options.use_style = True
        converter = module.DocumentConverter(
            format_options={
                base_models.InputFormat.PDF: module.PdfFormatOption(pipeline_options=options)
            }
        )
        result = converter.convert(str(path))
        document: Any = result.document
        if not document.export_to_text().strip():
            raise RuntimeError("Docling returned an empty structured document")
        items = getattr(document, "iterate_items", None)
        if not callable(items):
            return (
                ParserProvenance(
                    parser="docling",
                    parser_version=version("docling"),
                    ocr_engine="rapidocr" if self.ocr else "none",
                    ocr_model=self.ocr_model,
                    page_number=1,
                    extraction_method="docling_layout_pipeline",
                ),
            )
        blocks: list[ParserProvenance] = []
        for reading_order, item_and_level in enumerate(cast(Any, items()), start=1):
            item: Any = item_and_level[0]
            provenance = next(iter(getattr(item, "prov", ())), None)
            bbox = getattr(provenance, "bbox", None)
            blocks.append(
                ParserProvenance(
                    parser="docling",
                    parser_version=version("docling"),
                    ocr_engine="rapidocr" if self.ocr else "none",
                    ocr_model=self.ocr_model,
                    page_number=int(getattr(provenance, "page_no", 1)),
                    extraction_method="docling_layout_pipeline",
                    text=str(getattr(item, "text", "")),
                    bounding_box=(
                        float(bbox.l),
                        float(bbox.b),
                        float(bbox.r),
                        float(bbox.t),
                    )
                    if bbox is not None
                    else None,
                    block_type=classify_legal_block(
                        str(getattr(item, "text", "")), str(getattr(item, "label", "text"))
                    ),
                    reading_order=reading_order,
                )
            )
        return tuple(blocks) or (
            ParserProvenance(
                parser="docling",
                parser_version=version("docling"),
                ocr_engine="rapidocr" if self.ocr else "none",
                ocr_model=self.ocr_model,
                page_number=1,
                extraction_method="docling_layout_pipeline",
            ),
        )
