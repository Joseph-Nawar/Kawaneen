"""Offline text/layout quality metrics for the private parser benchmark."""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict

from kawaneen.parsing.anchored_gold import pdf_bottom_left_to_top_left
from kawaneen.parsing.health import probe_pdf


@dataclass(frozen=True)
class BenchmarkBlock:
    """One evaluated block, preserving its stable identity and page geometry."""

    block_id: str | None
    text: str
    block_type: str
    bounding_box: tuple[float, float, float, float] | None
    reading_order: int
    page_number: int = 1
    coordinate_origin: str = "top-left"


class BenchmarkMetrics(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    cer: float
    wer: float
    heading_precision: float
    heading_recall: float
    heading_f1: float
    exact_article_number_accuracy: float
    semantic_article_number_accuracy: float
    reading_order_accuracy: float
    page_reference_preservation: float
    critical_article_number_errors: int = 0


def qualification_status() -> dict[str, object]:
    """Return the committed benchmark gate without inventing unavailable metrics."""

    path = Path("data/manifests/parsing_benchmark.json")
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "status": "blocked_no_authorized_pages",
        "target_pages": 30,
        "available_pages": 0,
        "metrics": None,
    }


def preflight_pdfs(directory: Path) -> dict[str, Any]:
    """Inspect local PDFs without writing extracted legal text."""

    files = sorted(directory.glob("*.pdf"))
    if not files:
        return {"schema_version": 1, "sources": []}
    try:
        pypdf: Any = cast(Any, importlib.import_module("pypdf"))
    except ImportError as exc:
        raise RuntimeError("PDF preflight requires the optional parsing dependencies") from exc
    sources: list[dict[str, Any]] = []
    for path in files:
        reader: Any = pypdf.PdfReader(str(path))
        health = probe_pdf(path)
        pages: list[dict[str, Any]] = []
        for page, page_health in zip(reader.pages, health, strict=True):
            width = float(page.mediabox.width)
            height = float(page.mediabox.height)
            density = page_health.text_chars / max(1.0, width * height)
            complexity = (
                "image_only"
                if page_health.image_count and page_health.text_chars == 0
                else "mixed_image_text"
                if page_health.image_count
                else "dense_text"
                if density > 0.01
                else "sparse_text"
            )
            pages.append(
                {
                    "page": page_health.page_number,
                    "text_chars": page_health.text_chars,
                    "text_density": round(density, 6),
                    "image_count": page_health.image_count,
                    "dimensions_pt": [round(width, 1), round(height, 1)],
                    "likely_layout_complexity": complexity,
                }
            )
        sources.append(
            {
                "filename": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size": path.stat().st_size,
                "page_count": len(reader.pages),
                "embedded_text_pages": sum(page["text_chars"] > 0 for page in pages),
                "image_pages": sum(page["image_count"] > 0 for page in pages),
                "pages": pages,
            }
        )
    return {"schema_version": 1, "sources": sources}


def _f1(reference: tuple[str, ...], hypothesis: tuple[str, ...]) -> tuple[float, float, float]:
    expected, actual = set(reference), set(hypothesis)
    true_positive = len(expected & actual)
    precision = true_positive / len(actual) if actual else 1.0 if not expected else 0.0
    recall = true_positive / len(expected) if expected else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for index, left_char in enumerate(left, start=1):
        current = [index]
        for right_index, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def _token_edit_distance(left: Sequence[str], right: Sequence[str]) -> int:
    """Return Levenshtein distance over tokens for a true WER calculation."""

    previous = list(range(len(right) + 1))
    for index, token in enumerate(left, start=1):
        current = [index]
        for right_index, right_token in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (token != right_token),
                )
            )
        previous = current
    return previous[-1]


def calculate_metrics(
    *,
    reference: str,
    hypothesis: str,
    reference_headings: tuple[str, ...],
    hypothesis_headings: tuple[str, ...],
    reference_articles: tuple[str, ...],
    hypothesis_articles: tuple[str, ...],
    reference_pages: tuple[int, ...],
    hypothesis_pages: tuple[int, ...],
) -> BenchmarkMetrics:
    ref_words, hyp_words = reference.split(), hypothesis.split()
    heading_precision, heading_recall, heading_f1 = _f1(reference_headings, hypothesis_headings)
    exact = sum(
        a == b for a, b in zip(reference_articles, hypothesis_articles, strict=False)
    ) / max(1, len(reference_articles))
    semantic = sum(
        a.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
        == b.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
        for a, b in zip(reference_articles, hypothesis_articles, strict=False)
    ) / max(1, len(reference_articles))
    return BenchmarkMetrics(
        cer=_edit_distance(reference, hypothesis) / max(1, len(reference)),
        wer=_token_edit_distance(ref_words, hyp_words) / max(1, len(ref_words)),
        heading_precision=heading_precision,
        heading_recall=heading_recall,
        heading_f1=heading_f1,
        exact_article_number_accuracy=exact,
        semantic_article_number_accuracy=semantic,
        reading_order_accuracy=SequenceMatcher(None, ref_words, hyp_words).ratio(),
        page_reference_preservation=1.0 if reference_pages == hypothesis_pages else 0.0,
    )


def _article_number(text: str) -> str | None:
    """Extract only standalone article labels, never an inline article mention."""

    from kawaneen.corpus.statutory import parse_article_label

    parsed = parse_article_label(text.strip())
    return str(parsed.article_ordinal) if parsed.article_ordinal is not None else None


def _overlap(left: BenchmarkBlock, right: BenchmarkBlock) -> float:
    if left.bounding_box is None or right.bounding_box is None:
        return 0.0
    lx0, ly0, lx1, ly1 = left.bounding_box
    rx0, ry0, rx1, ry1 = right.bounding_box
    width = max(0.0, min(lx1, rx1) - max(lx0, rx0))
    height = max(0.0, min(ly1, ry1) - max(ly0, ry0))
    intersection = width * height
    union = (lx1 - lx0) * (ly1 - ly0) + (rx1 - rx0) * (ry1 - ry0) - intersection
    return intersection / union if union else 0.0


def assign_blocks_to_gold_region(
    gold_box: tuple[float, float, float, float],
    blocks: tuple[BenchmarkBlock, ...],
    *,
    overlap_threshold: float = 0.20,
    center_rule: bool = True,
) -> tuple[int, ...]:
    """Select prediction blocks by geometry only, then return declared order."""

    gx0, gy0, gx1, gy1 = gold_box
    selected: list[tuple[int, int]] = []
    for index, block in enumerate(blocks):
        if block.bounding_box is None:
            continue
        bx0, by0, bx1, by1 = block.bounding_box
        intersection_width = max(0.0, min(gx1, bx1) - max(gx0, bx0))
        intersection_height = max(0.0, min(gy1, by1) - max(gy0, by0))
        intersection = intersection_width * intersection_height
        block_area = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
        overlap = intersection / block_area if block_area else 0.0
        center_inside = gx0 <= (bx0 + bx1) / 2 <= gx1 and gy0 <= (by0 + by1) / 2 <= gy1
        if overlap >= overlap_threshold or (center_rule and center_inside):
            selected.append((index, block.reading_order))
    return tuple(index for index, _order in sorted(selected, key=lambda item: (item[1], item[0])))


def canonicalize_prediction_box(
    box: tuple[float, float, float, float], *, origin: str, page_height: float
) -> tuple[float, float, float, float]:
    """Convert a predicted block box to the benchmark's top-left system."""

    if origin == "top-left":
        return tuple(float(value) for value in box)  # type: ignore[return-value]
    if origin == "bottom-left":
        return pdf_bottom_left_to_top_left(box, page_height)
    raise ValueError(f"unsupported prediction coordinate origin: {origin}")


def _page_reference_tokens(text: str) -> tuple[str, ...]:
    import re

    return tuple(
        match.group(1)
        for match in re.finditer(
            r"(?:صفحة|ص)\s*([0-9٠-٩]+)",  # noqa: RUF001
            text,
            flags=re.UNICODE,
        )
    )


def _pairwise_order_accuracy(reference: Sequence[str], hypothesis: Sequence[str]) -> float:
    reference_pairs = {
        (left, right) for i, left in enumerate(reference) for right in reference[i + 1 :]
    }
    hypothesis_positions = {value: index for index, value in enumerate(hypothesis)}
    comparable = [
        pair
        for pair in reference_pairs
        if pair[0] in hypothesis_positions and pair[1] in hypothesis_positions
    ]
    if not comparable:
        return 1.0
    return sum(
        hypothesis_positions[left] < hypothesis_positions[right] for left, right in comparable
    ) / len(comparable)


def calculate_anchored_metrics(
    *, gold_records: Sequence[dict[str, object]], predicted_blocks: tuple[BenchmarkBlock, ...]
) -> BenchmarkMetrics:
    """Score regions only after spatial assignment to anchored gold boxes."""

    region_texts: list[tuple[str, str]] = []
    heading_gold = 0
    heading_true_positive = 0
    assigned_by_block: dict[int, str] = {}
    article_exact: list[bool] = []
    article_semantic: list[bool] = []
    critical_errors = 0
    page_reference_results: list[bool] = []
    gold_order: list[str] = []
    record_types: dict[str, str] = {}
    for record in gold_records:
        region_id = str(record["region_id"])
        gold_order.append(region_id)
        record_types[region_id] = str(record["region_type"])
        raw_gold_box = cast(Sequence[float], record["bounding_box"])
        gold_box = cast(
            tuple[float, float, float, float],
            tuple(float(value) for value in raw_gold_box),
        )
        assigned = assign_blocks_to_gold_region(gold_box, predicted_blocks)
        for index in assigned:
            assigned_by_block.setdefault(index, region_id)
        ordered_blocks = [predicted_blocks[index] for index in assigned]
        hypothesis = " ".join(block.text for block in ordered_blocks).strip()
        reference = str(record["gold_text"])
        region_texts.append((reference, hypothesis))
        region_type = str(record["region_type"])
        if region_type == "heading":
            heading_gold += 1
            if any(block.block_type == "heading" for block in ordered_blocks):
                heading_true_positive += 1
        if region_type == "article_label" and record.get("semantic_article_number") is not None:
            article_exact.append(reference.strip() == hypothesis.strip())
            expected = str(record["semantic_article_number"])
            actual = _article_number(hypothesis)
            article_semantic.append(actual == expected)
            if actual != expected:
                critical_errors += 1
        expected_refs = _page_reference_tokens(reference)
        actual_refs = _page_reference_tokens(hypothesis)
        page_reference_results.append(
            not expected_refs or all(token in actual_refs for token in expected_refs)
        )
    predicted_region_order: list[str] = []
    for index in sorted(
        range(len(predicted_blocks)), key=lambda item: (predicted_blocks[item].reading_order, item)
    ):
        region_id = assigned_by_block.get(index)
        if region_id is not None and region_id not in predicted_region_order:
            predicted_region_order.append(region_id)
    predicted_heading_indices = {
        index for index, block in enumerate(predicted_blocks) if block.block_type == "heading"
    }
    heading_false_positive = sum(
        index not in assigned_by_block
        or record_types.get(assigned_by_block[index], "") != "heading"
        for index in predicted_heading_indices
    )
    heading_precision = (
        heading_true_positive / (heading_true_positive + heading_false_positive)
        if heading_true_positive + heading_false_positive
        else 1.0
        if not heading_gold
        else 0.0
    )
    heading_recall = heading_true_positive / heading_gold if heading_gold else 1.0
    heading_f1 = (
        2 * heading_precision * heading_recall / (heading_precision + heading_recall)
        if heading_precision + heading_recall
        else 0.0
    )
    reference_text = " ".join(reference for reference, _ in region_texts)
    hypothesis_text = " ".join(hypothesis for _, hypothesis in region_texts)
    ref_words, hyp_words = reference_text.split(), hypothesis_text.split()
    return BenchmarkMetrics(
        cer=_edit_distance(reference_text, hypothesis_text) / max(1, len(reference_text)),
        wer=_token_edit_distance(ref_words, hyp_words) / max(1, len(ref_words)),
        heading_precision=heading_precision,
        heading_recall=heading_recall,
        heading_f1=heading_f1,
        exact_article_number_accuracy=sum(article_exact) / max(1, len(article_exact)),
        semantic_article_number_accuracy=sum(article_semantic) / max(1, len(article_semantic)),
        reading_order_accuracy=_pairwise_order_accuracy(gold_order, predicted_region_order),
        page_reference_preservation=sum(page_reference_results)
        / max(1, len(page_reference_results)),
        critical_article_number_errors=critical_errors,
    )


def _match_blocks(
    reference: tuple[BenchmarkBlock, ...], hypothesis: tuple[BenchmarkBlock, ...]
) -> tuple[tuple[tuple[int, int], ...], tuple[int, ...], tuple[int, ...]]:
    """Match stable IDs first, then only anonymous blocks by unambiguous geometry."""

    hypothesis_by_id = {
        block.block_id: index
        for index, block in enumerate(hypothesis)
        if block.block_id is not None
    }
    matched: list[tuple[int, int]] = []
    used_hypothesis: set[int] = set()
    unmatched_reference: list[int] = []
    for ref_index, block in enumerate(reference):
        if block.block_id is not None and block.block_id in hypothesis_by_id:
            hyp_index = hypothesis_by_id[block.block_id]
            matched.append((ref_index, hyp_index))
            used_hypothesis.add(hyp_index)
        else:
            unmatched_reference.append(ref_index)
    for ref_index in tuple(unmatched_reference):
        ref_block = reference[ref_index]
        if ref_block.block_id is not None:
            continue
        candidates = [
            (hyp_index, _overlap(ref_block, hyp_block))
            for hyp_index, hyp_block in enumerate(hypothesis)
            if hyp_index not in used_hypothesis and hyp_block.block_id is None
        ]
        if not candidates:
            continue
        hyp_index, score = max(candidates, key=lambda item: item[1])
        if score >= 0.5:
            matched.append((ref_index, hyp_index))
            used_hypothesis.add(hyp_index)
    matched_reference = {ref_index for ref_index, _ in matched}
    return (
        tuple(sorted(matched)),
        tuple(index for index in range(len(reference)) if index not in matched_reference),
        tuple(index for index in range(len(hypothesis)) if index not in used_hypothesis),
    )


def calculate_region_metrics(
    *, reference: tuple[BenchmarkBlock, ...], hypothesis: tuple[BenchmarkBlock, ...]
) -> BenchmarkMetrics:
    """Evaluate like-for-like regions rather than unrelated concatenated pages."""

    matched, unmatched_reference, unmatched_hypothesis = _match_blocks(reference, hypothesis)
    reference_text = " ".join(reference[index].text for index, _ in matched) + " ".join(
        reference[index].text for index in unmatched_reference
    )
    hypothesis_text = " ".join(hypothesis[index].text for _, index in matched) + " ".join(
        hypothesis[index].text for index in unmatched_hypothesis
    )
    reference_headings = tuple(
        reference[index].text for index, _ in matched if reference[index].block_type == "heading"
    )
    hypothesis_headings = tuple(
        hypothesis[index].text for _, index in matched if hypothesis[index].block_type == "heading"
    )
    reference_articles: list[str] = []
    hypothesis_articles: list[str] = []
    for reference_index, hypothesis_index in matched:
        reference_number = _article_number(reference[reference_index].text)
        hypothesis_number = _article_number(hypothesis[hypothesis_index].text)
        if reference_number is not None:
            reference_articles.append(reference_number)
        if hypothesis_number is not None:
            hypothesis_articles.append(hypothesis_number)
    matched_by_reference = dict(matched)
    reference_order = [
        index for index, _block in enumerate(reference) if index in matched_by_reference
    ]
    hypothesis_order = [
        index for index, _ in sorted(matched, key=lambda pair: hypothesis[pair[1]].reading_order)
    ]
    reading_order = (
        sum(left == right for left, right in zip(reference_order, hypothesis_order, strict=True))
        / len(reference_order)
        if reference_order
        else 1.0
    )
    metrics = calculate_metrics(
        reference=reference_text,
        hypothesis=hypothesis_text,
        reference_headings=reference_headings,
        hypothesis_headings=hypothesis_headings,
        reference_articles=tuple(reference_articles),
        hypothesis_articles=tuple(hypothesis_articles),
        reference_pages=(1,) if reference else (),
        hypothesis_pages=(1,) if hypothesis else (),
    )
    return metrics.model_copy(update={"reading_order_accuracy": reading_order})
