"""Independent, fail-closed anchors for the private parsing benchmark."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, cast

Box = tuple[float, float, float, float]


class GoldIntegrityError(ValueError):
    """Raised when anchored gold cannot be trusted for qualification."""


class AnchorError(GoldIntegrityError):
    """Raised when a verified string cannot be mapped to source geometry."""


class AmbiguousAnchorError(AnchorError):
    """Raised when a verified string has more than one source text match."""


@dataclass(frozen=True)
class GoldValidation:
    """Validated private gold summary."""

    page_count: int
    region_count: int
    region_ids: tuple[str, ...]
    source_hashes: tuple[tuple[str, str], ...]


_ARABIC_TATWEEL = "\u0640"
_WHITESPACE = re.compile(r"\s+", re.UNICODE)


def _locator_with_map(value: str) -> tuple[str, tuple[int, ...]]:
    chars: list[str] = []
    source_indexes: list[int] = []
    for source_index, char in enumerate(value):
        normalized = unicodedata.normalize("NFKC", char).replace(_ARABIC_TATWEEL, "")
        if _WHITESPACE.fullmatch(char):
            continue
        if not normalized:
            continue
        chars.append(normalized)
        source_indexes.extend([source_index] * len(normalized))
    return "".join(chars), tuple(source_indexes)


def _validate_box(box: Sequence[float], width: float, height: float) -> Box:
    if len(box) != 4:
        raise GoldIntegrityError("bounding box must have four coordinates")
    result = tuple(float(value) for value in box)
    x0, y0, x1, y1 = result
    if not all(value == value and abs(value) != float("inf") for value in result):
        raise GoldIntegrityError("bounding box coordinates must be finite")
    if not x0 < x1 or not y0 < y1:
        raise GoldIntegrityError("bounding box must be non-empty")
    if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
        raise GoldIntegrityError("bounding box is out of page bounds")
    return result  # type: ignore[return-value]


def pdf_bottom_left_to_top_left(box: Sequence[float], page_height: float) -> Box:
    """Convert PDFium's bottom-left geometry to canonical top-left geometry."""

    if page_height <= 0:
        raise GoldIntegrityError("page height must be positive")
    x0, y0, x1, y1 = (float(value) for value in box)
    return (x0, page_height - y1, x1, page_height - y0)


def scale_box(
    box: Sequence[float],
    source_width: float,
    source_height: float,
    target_width: float,
    target_height: float,
) -> Box:
    """Scale a box between page coordinate spaces without changing its origin."""

    if min(source_width, source_height, target_width, target_height) <= 0:
        raise GoldIntegrityError("page dimensions must be positive")
    x0, y0, x1, y1 = (float(value) for value in box)
    return (
        x0 * target_width / source_width,
        y0 * target_height / source_height,
        x1 * target_width / source_width,
        y1 * target_height / source_height,
    )


def union_boxes(boxes: Iterable[Sequence[float]]) -> Box:
    """Return the smallest box containing every non-empty input box."""

    values = [tuple(float(value) for value in box) for box in boxes]
    if not values:
        raise GoldIntegrityError("cannot union an empty box collection")
    if any(len(box) != 4 for box in values):
        raise GoldIntegrityError("every box must have four coordinates")
    return (
        min(box[0] for box in values),
        min(box[1] for box in values),
        max(box[2] for box in values),
        max(box[3] for box in values),
    )


def find_unique_text_anchor(
    page_text: str,
    verified_text: str,
    *,
    page_width: float,
    page_height: float,
    character_boxes: Sequence[Sequence[float]] | None = None,
) -> Box:
    """Find exactly one verified span and union its source character rectangles."""

    normalized_page, page_map = _locator_with_map(page_text)
    normalized_target, _ = _locator_with_map(verified_text)
    if not normalized_target:
        raise AnchorError("verified text is empty after locator normalization")
    starts: list[int] = []
    cursor = 0
    while True:
        match = normalized_page.find(normalized_target, cursor)
        if match < 0:
            break
        starts.append(match)
        cursor = match + 1
    if not starts:
        raise AnchorError("verified text has no unique PDF text-layer match")
    if len(starts) != 1:
        raise AmbiguousAnchorError(f"verified text has {len(starts)} PDF text-layer matches")
    if character_boxes is None:
        return (0.0, 0.0, float(page_width), float(page_height))
    if len(character_boxes) != len(page_text):
        raise AnchorError("PDFium character geometry length does not match page text")
    start = starts[0]
    end = start + len(normalized_target)
    source_indexes = page_map[start:end]
    if not source_indexes:
        raise AnchorError("verified text produced no source characters")
    boxes = [character_boxes[index] for index in source_indexes]
    return union_boxes(boxes)


def _require(record: Mapping[str, Any], key: str) -> Any:
    if key not in record:
        raise GoldIntegrityError(f"gold record missing {key}")
    return record[key]


def validate_anchored_gold(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_page_count: int,
    expected_region_count: int,
    source_hashes: Mapping[str, str] | None = None,
    expected_pages: set[str] | None = None,
) -> GoldValidation:
    """Validate all private gold invariants before allowing benchmark scoring."""

    page_ids = {_require(record, "page_id") for record in records}
    if len(page_ids) != expected_page_count:
        raise GoldIntegrityError(f"page count {len(page_ids)} != {expected_page_count}")
    if len(records) != expected_region_count:
        raise GoldIntegrityError(f"region count {len(records)} != {expected_region_count}")
    if expected_pages is not None and page_ids != expected_pages:
        raise GoldIntegrityError("gold page IDs do not match the frozen selection")
    ids: list[str] = []
    seen: set[str] = set()
    seen_hashes: dict[str, str] = {}
    for record in records:
        region_id = str(_require(record, "region_id"))
        if region_id in seen:
            raise GoldIntegrityError(f"duplicate region ID: {region_id}")
        seen.add(region_id)
        ids.append(region_id)
        text = _require(record, "gold_text")
        if not isinstance(text, str) or not text.strip():
            raise GoldIntegrityError(f"region {region_id} has non-empty text requirement")
        filename = str(_require(record, "source_pdf_filename"))
        digest = str(_require(record, "source_pdf_sha256"))
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise GoldIntegrityError(f"region {region_id} has invalid source hash")
        if source_hashes is not None and source_hashes.get(filename) != digest:
            raise GoldIntegrityError(f"source PDF hash mismatch for {filename}")
        seen_hashes[filename] = digest
        page_number = int(_require(record, "page_number"))
        if page_number < 1:
            raise GoldIntegrityError(f"region {region_id} has invalid page number")
        width = float(_require(record, "page_width"))
        height = float(_require(record, "page_height"))
        if width <= 0 or height <= 0:
            raise GoldIntegrityError(f"region {region_id} has invalid page dimensions")
        _validate_box(_require(record, "bounding_box"), width, height)
        if _require(record, "coordinate_origin") != "top-left":
            raise GoldIntegrityError("gold must use top-left coordinates")
        if _require(record, "coordinate_system") != "canonical_top_left_points":
            raise GoldIntegrityError("gold must use canonical top-left points")
        method = str(_require(record, "anchoring_method"))
        if method == "prediction_geometry" or "parser" in method.lower():
            raise GoldIntegrityError("prediction-derived gold geometry is prohibited")
        provenance = _require(record, "review_provenance")
        if not isinstance(provenance, Mapping):
            raise GoldIntegrityError("review provenance must be an object")
    return GoldValidation(
        page_count=len(page_ids),
        region_count=len(records),
        region_ids=tuple(ids),
        source_hashes=tuple(sorted(seen_hashes.items())),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _map_index(index: int, opcodes: Sequence[tuple[str, int, int, int, int]]) -> int:
    for tag, source_start, source_end, target_start, target_end in opcodes:
        if source_start <= index < source_end:
            if tag == "equal":
                return target_start + index - source_start
            return target_start + min(index - source_start, max(0, target_end - target_start - 1))
    return opcodes[-1][4] if opcodes else index


def _reference_span(
    page_text: str, verified_text: str, *, occurrence_index: int | None = None
) -> tuple[int, int, bool]:
    page_locator, _ = _locator_with_map(page_text)
    target_locator, _ = _locator_with_map(verified_text)
    occurrences: list[int] = []
    cursor = 0
    while True:
        found = page_locator.find(target_locator, cursor)
        if found < 0:
            break
        occurrences.append(found)
        cursor = found + 1
    if occurrences:
        if occurrence_index is None and len(occurrences) != 1:
            raise AmbiguousAnchorError(f"verified text has {len(occurrences)} source matches")
        index = occurrences[occurrence_index or 0]
        return index, index + len(target_locator), len(occurrences) != 1
    prefix = target_locator[: min(16, max(8, len(target_locator) // 3))]
    candidates: list[int] = []
    cursor = 0
    while True:
        found = page_locator.find(prefix, cursor)
        if found < 0:
            break
        candidates.append(found)
        cursor = found + 1
    if len(candidates) != 1:
        raise AnchorError("verified text has no unique sufficiently similar source span")
    start = candidates[0]
    window = page_locator[start : start + len(target_locator) + 24]
    score = SequenceMatcher(None, target_locator, window, autojunk=False).ratio()
    if score < 0.70:
        raise AnchorError("verified text has no sufficiently similar source text-layer match")
    local_alignment = SequenceMatcher(None, target_locator, window, autojunk=False).get_opcodes()
    local_end = _map_index(len(target_locator) - 1, local_alignment) + 1
    return start, min(len(page_locator), start + local_end), False


def _pdfium_anchor(
    path: Path,
    page_number: int,
    verified_text: str,
    *,
    occurrence_index: int | None = None,
    source_locator_text: str | None = None,
    source_occurrence_index: int | None = None,
) -> tuple[float, float, Box, bool]:
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise RuntimeError("pypdfium2 is required to anchor born-digital gold") from exc
    document = pdfium.PdfDocument(str(path))
    page = document[page_number - 1]
    width, height = (float(value) for value in page.get_size())
    text_page: Any = page.get_textpage()
    pdfium_text = text_page.get_text_range()
    character_boxes: list[Box] = [
        cast(
            Box,
            tuple(float(value) for value in text_page.get_charbox(index)),
        )
        for index in range(text_page.count_chars())
    ]
    if source_locator_text is not None:
        pdfium_locator, pdfium_map = _locator_with_map(pdfium_text)
        start, end, ambiguous = _reference_span(
            pdfium_text,
            source_locator_text,
            occurrence_index=source_occurrence_index,
        )
        pdf_box = union_boxes(character_boxes[pdfium_map[index]] for index in range(start, end))
        return width, height, pdf_bottom_left_to_top_left(pdf_box, height), ambiguous
    try:
        from pypdf import PdfReader
    except ImportError:
        reference_text = pdfium_text
    else:
        reference_text = PdfReader(str(path)).pages[page_number - 1].extract_text() or ""
    try:
        reference_start, reference_end, ambiguous = _reference_span(
            reference_text, verified_text, occurrence_index=occurrence_index
        )
        reference_locator, _ = _locator_with_map(reference_text)
        pdfium_locator, pdfium_map = _locator_with_map(pdfium_text)
        alignment = SequenceMatcher(
            None, reference_locator, pdfium_locator, autojunk=False
        ).get_opcodes()
        mapped_start = min(len(pdfium_map), _map_index(reference_start, alignment))
        mapped_end = min(
            len(pdfium_map), _map_index(max(reference_start, reference_end - 1), alignment) + 1
        )
        source_indexes = (pdfium_map[index] for index in range(mapped_start, mapped_end))
    except AnchorError:
        start, end, ambiguous = _reference_span(
            pdfium_text, verified_text, occurrence_index=occurrence_index
        )
        _pdfium_locator, pdfium_map = _locator_with_map(pdfium_text)
        source_indexes = (pdfium_map[index] for index in range(start, end))
    boxes = [
        character_boxes[index] for index in source_indexes if 0 <= index < len(character_boxes)
    ]
    if not boxes:
        raise AnchorError("verified text produced no PDFium character geometry")
    pdf_box = union_boxes(boxes)
    return width, height, pdf_bottom_left_to_top_left(pdf_box, height), ambiguous


def build_anchored_gold(
    selection_path: Path,
    external_gold_path: Path,
    source_dir: Path,
    output_path: Path,
    *,
    sama_annotations_path: Path | None = None,
    born_digital_overrides_path: Path | None = None,
) -> GoldValidation:
    """Build v2 gold from independent reviewed text and source-derived geometry."""

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection_by_id = {item["id"]: item for item in selection["selection"]}
    external_pages = _load_jsonl(external_gold_path)
    records: list[dict[str, Any]] = []
    sama_annotations: dict[tuple[str, str], Mapping[str, Any]] = {}
    if sama_annotations_path is not None and sama_annotations_path.is_file():
        for item in _load_jsonl(sama_annotations_path):
            sama_annotations[(str(item["page_id"]), str(item["region_id"]))] = item
    born_digital_overrides: dict[tuple[str, str], Mapping[str, Any]] = {}
    if born_digital_overrides_path is not None and born_digital_overrides_path.is_file():
        for item in _load_jsonl(born_digital_overrides_path):
            born_digital_overrides[(str(item["page_id"]), str(item["region_id"]))] = item
    for page in external_pages:
        page_id = str(page["page_id"])
        selected = selection_by_id[page_id]
        filename = str(selected["source_pdf"])
        source_path = source_dir / filename
        digest = _sha256(source_path)
        if digest != page["regions"][0]["provenance"]["source_pdf_sha256"]:
            raise GoldIntegrityError(f"source PDF hash mismatch for {filename}")
        for region in page["regions"]:
            adjudicated = region["adjudicated"]
            region_id = str(region["region_id"])
            if selected["category"] == "image_only_scan":
                annotation = sama_annotations.get((page_id, region_id))
                if annotation is None:
                    raise GoldIntegrityError(f"missing independent SAMA annotation for {region_id}")
                width = float(annotation["page_width"])
                height = float(annotation["page_height"])
                box = tuple(float(value) for value in annotation["bounding_box"])
                method = "independent_ai_visual_anchor"
                confidence = str(annotation.get("anchoring_confidence", "reviewed_visual"))
                provenance = {
                    "reviewer_type": "independent_ai_visual_review",
                    "human_verified": False,
                    "annotation_tool": annotation.get("annotation_tool", "private_manual_v1"),
                    "render_sha256": annotation.get("render_sha256"),
                }
            else:
                override = born_digital_overrides.get((page_id, region_id))
                if override is not None and override.get("manual_box") is not None:
                    try:
                        import pypdfium2 as pdfium
                    except ImportError as exc:
                        raise RuntimeError("pypdfium2 is required for page dimensions") from exc
                    width, height = (
                        float(value)
                        for value in pdfium.PdfDocument(str(source_path))[
                            int(selected["source_page"]) - 1
                        ].get_size()
                    )
                    box = tuple(float(value) for value in override["manual_box"])
                    was_ambiguous = True
                else:
                    width, height, box, was_ambiguous = _pdfium_anchor(
                        source_path,
                        int(selected["source_page"]),
                        str(adjudicated["text"]),
                        occurrence_index=(
                            int(override["occurrence_index"])
                            if override is not None and override.get("occurrence_index") is not None
                            else None
                        ),
                        source_locator_text=(
                            str(override["source_locator_text"])
                            if override is not None and override.get("source_locator_text")
                            else None
                        ),
                        source_occurrence_index=(
                            int(override["source_occurrence_index"])
                            if override is not None
                            and override.get("source_occurrence_index") is not None
                            else None
                        ),
                    )
                if was_ambiguous and override is None:
                    raise AmbiguousAnchorError(f"no independent disambiguation for {region_id}")
                method = "pdfium_text_geometry"
                if override is not None and override.get("manual_box") is not None:
                    method = "independent_source_visual_anchor"
                elif override is not None:
                    method = "pdfium_text_geometry_visual_disambiguation"
                confidence = "independent_visual_disambiguation" if override else "unique_text_span"
                provenance = {
                    "reviewer_type": region["provenance"]["reviewer_type"],
                    "human_verified": False,
                    "source_text_layer": True,
                    "gold_text_source": "independent_external_review",
                }
                if override is not None:
                    provenance.update(
                        {
                            "visual_disambiguation": True,
                            "visual_review_note": override.get("visual_review_note"),
                            "geometry_independent_of_predictions": True,
                        }
                    )
            record = {
                "page_id": page_id,
                "region_id": region_id,
                "source_pdf_filename": filename,
                "source_pdf_sha256": digest,
                "page_number": int(selected["source_page"]),
                "gold_text": str(adjudicated["text"]),
                "region_type": str(adjudicated["region_type"]),
                "semantic_article_number": adjudicated.get("expected_semantic_article_number"),
                "bounding_box": list(box),
                "page_width": width,
                "page_height": height,
                "coordinate_origin": "top-left",
                "coordinate_system": "canonical_top_left_points",
                "anchoring_method": method,
                "anchoring_confidence": confidence,
                "review_provenance": provenance,
            }
            records.append(record)
    expected_pages = set(selection_by_id)
    source_hashes = {
        filename: _sha256(source_dir / filename)
        for filename in {item["source_pdf"] for item in selection["selection"]}
    }
    validation = validate_anchored_gold(
        records,
        expected_page_count=30,
        expected_region_count=102,
        source_hashes=source_hashes,
        expected_pages=expected_pages,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records
        ),
        encoding="utf-8",
    )
    return validation
