"""Fail-closed Arabic statutory structure parsing and reconstruction."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

from kawaneen.corpus.models import (
    ArticleParseConfidence,
    ReconstructionGroup,
    ReconstructionStatus,
    SourceFragment,
)

_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_PART = re.compile(r"(?:part|section|جزء|قسم)\s*[/:-]?\s*([0-9]+)", re.IGNORECASE)
_STATUS = re.compile(r"(?:^|\s)(ملغاة|معدلة|مضافة)(?:\s|$)")
_ARTICLE_PREFIX = re.compile(r"^(?:المادة|الماده|article)\b", re.IGNORECASE)
_DIGIT_LABEL = re.compile(r"^(?:المادة|الماده|article)\s*([0-9]+)$", re.IGNORECASE)

_UNIT_VALUES = {
    "الأولى": 1,
    "الاولى": 1,
    "الأول": 1,
    "الاول": 1,
    "الحادية": 1,
    "الثانية": 2,
    "الثاني": 2,
    "الثالثة": 3,
    "الثالث": 3,
    "الرابعة": 4,
    "الرابع": 4,
    "الخامسة": 5,
    "الخامس": 5,
    "السادسة": 6,
    "السادس": 6,
    "السابعة": 7,
    "السابع": 7,
    "الثامنة": 8,
    "الثامن": 8,
    "التاسعة": 9,
    "التاسع": 9,
}

_TENS_VALUES = {
    "العشرون": 20,
    "الثلاثون": 30,
    "الأربعون": 40,
    "الاربعون": 40,
    "الخمسون": 50,
    "الستون": 60,
    "السبعون": 70,
    "الثمانون": 80,
    "التسعون": 90,
}

_HUNDRED_VALUES = {
    "المائة": 100,
    "المئة": 100,
    "مائة": 100,
    "مئة": 100,
    "المائتان": 200,
    "المائتين": 200,
    "مائتان": 200,
    "مائتين": 200,
    "الثلاثمائة": 300,
    "الثلاثمئة": 300,
    "الأربعمائة": 400,
    "الأربعمئة": 400,
    "الاربعمائة": 400,
    "الاربعمئة": 400,
    "الخمسمائة": 500,
    "الخمس مائة": 500,
    "الستمائة": 600,
    "الستمئة": 600,
    "السبعمائة": 700,
    "السبعمئة": 700,
    "الثمانمائة": 800,
    "الثمانمئة": 800,
    "التسعمائة": 900,
    "التسعمئة": 900,
}


def _compact(value: str) -> str:
    """Remove only layout separators used in a structural label."""

    return re.sub(r"[\s\u0640]+", "", value)


def _word_values() -> dict[str, int]:
    values: dict[str, int] = {}
    for word, value in _UNIT_VALUES.items():
        values[_compact(word)] = value
    for word, value in _TENS_VALUES.items():
        values[_compact(word)] = value
    for word, value in _HUNDRED_VALUES.items():
        values[_compact(word)] = value
    values["العاشرة"] = 10
    for unit_word, unit_value in _UNIT_VALUES.items():
        unit = _compact(unit_word)
        if unit_value == 1 and unit_word in {"الأول", "الاول"}:
            continue
        if unit_value <= 9:
            values[f"{unit}عشرة"] = unit_value + 10
            values[f"{unit}عشر"] = unit_value + 10
            for tens_word, tens_value in _TENS_VALUES.items():
                values[f"{unit}و{_compact(tens_word)}"] = unit_value + tens_value
    return values


_WORD_VALUES = _word_values()


@dataclass(frozen=True)
class ArticleLabel:
    """Typed structural metadata; `raw_label` is always untouched."""

    raw_label: str
    article_label_structural_key: str | None = None
    article_ordinal: int | None = None
    article_parse_confidence: ArticleParseConfidence = ArticleParseConfidence.UNRESOLVED
    part_index: int | None = None
    article_status_marker: str | None = None

    @property
    def article_label_raw(self) -> str:
        """Explicit field name used by the canonical structural contract."""

        return self.raw_label

    @property
    def ordinal(self) -> int | None:
        """Backward-compatible name used by existing adapters."""

        return self.article_ordinal

    @property
    def part(self) -> int | None:
        """Backward-compatible name used by existing adapters."""

        return self.part_index


def _unresolved(raw_label: str, part: int | None = None) -> ArticleLabel:
    return ArticleLabel(raw_label=raw_label, part_index=part)


def _parse_compact_words(compact: str) -> int | None:
    if compact in _WORD_VALUES:
        return _WORD_VALUES[compact]
    if "بعد" in compact:
        prefix, suffix = compact.split("بعد", 1)
        lower = _WORD_VALUES.get(prefix)
        base = _HUNDRED_VALUES.get(suffix)
        if lower is not None and base is not None and 1 <= lower <= 99:
            return base + lower
    return None


def parse_article_label(raw_label: str) -> ArticleLabel:
    """Parse an article heading without changing its source representation.

    The parser deliberately accepts only a complete article heading. It never
    searches arbitrary text for a number, which prevents part numbers and
    embedded references from becoming article ordinals.
    """

    if not raw_label or not _ARTICLE_PREFIX.match(raw_label.strip()):
        return _unresolved(raw_label)
    translated = raw_label.translate(_DIGITS)
    part_match = _PART.search(translated)
    part = int(part_match.group(1)) if part_match else None
    without_part = re.sub(r"\([^)]*\)", "", translated)
    status_match = _STATUS.search(without_part)
    status = status_match.group(1) if status_match else None
    body = without_part[: status_match.start()] if status_match else without_part
    numeric = _DIGIT_LABEL.fullmatch(body.strip())
    if numeric:
        ordinal = int(numeric.group(1))
        if not 1 <= ordinal <= 999:
            return _unresolved(raw_label, part)
        compact = numeric.group(1)
    else:
        prefix_match = _ARTICLE_PREFIX.match(body.strip())
        if prefix_match is None:
            return _unresolved(raw_label, part)
        phrase = body.strip()[prefix_match.end() :].strip()
        compact = _compact(phrase)
        ordinal = _parse_compact_words(compact)
        if ordinal is None or not 1 <= ordinal <= 999:
            return _unresolved(raw_label, part)
    key = f"article:{ordinal}|label:{compact}|status:{status or ''}"
    return ArticleLabel(
        raw_label=raw_label,
        article_label_structural_key=key,
        article_ordinal=ordinal,
        article_parse_confidence=ArticleParseConfidence.HIGH,
        part_index=part,
        article_status_marker=status,
    )


def classify_fragment_group(
    law_name: str, raw_article_label: str, fragments: Iterable[SourceFragment]
) -> ReconstructionGroup:
    """Classify a full structural group; only explicit parts are mergeable."""

    ordered = tuple(sorted(fragments, key=lambda item: item.provenance.source_row))
    if len(ordered) == 1:
        status = ReconstructionStatus.UNIQUE
    elif all(item.part_index is not None or item.explicit_part is not None for item in ordered):
        parts = {
            item.part_index if item.part_index is not None else item.explicit_part
            for item in ordered
        }
        status = (
            ReconstructionStatus.EXPLICIT_FRAGMENT_SERIES
            if len(parts) == len(ordered)
            else ReconstructionStatus.CONFLICTING_DUPLICATE
        )
    elif any(
        re.search(r"continued|continuation|استمرار|تابع", item.raw_label, re.I) for item in ordered
    ):
        status = ReconstructionStatus.CONTINUATION_CANDIDATE
    elif len({item.text for item in ordered}) > 1:
        status = ReconstructionStatus.CONFLICTING_DUPLICATE
    else:
        status = ReconstructionStatus.UNRESOLVED
    operations = (
        ("merge_explicit_parts",) if status is ReconstructionStatus.EXPLICIT_FRAGMENT_SERIES else ()
    )
    return ReconstructionGroup(
        law_name=law_name,
        raw_article_label=raw_article_label,
        status=status,
        fragment_ids=tuple(item.fragment_id for item in ordered),
        operations=operations,
    )


def classify_all(
    fragments: Iterable[SourceFragment], law_names: Iterable[str]
) -> tuple[ReconstructionGroup, ...]:
    groups: dict[tuple[str, str], list[SourceFragment]] = defaultdict(list)
    for fragment, law_name in zip(fragments, law_names, strict=True):
        key = fragment.article_label_structural_key
        if key is None:
            key = f"unresolved:{fragment.provenance.source_row}"
        groups[(law_name, key)].append(fragment)
    return tuple(
        classify_fragment_group(law_name, items[0].raw_label, items)
        for (law_name, _key), items in sorted(groups.items())
    )


def reconstruction_counts(groups: Iterable[ReconstructionGroup]) -> dict[str, int]:
    counts = Counter(group.status.value for group in groups)
    return {status.value: counts.get(status.value, 0) for status in ReconstructionStatus}


def duplicate_diagnostics(
    fragments: Iterable[SourceFragment], groups: Iterable[ReconstructionGroup]
) -> dict[str, Any]:
    """Return deterministic, text-free diagnostics for duplicate article keys."""

    groups = tuple(groups)
    fragment_by_id = {fragment.fragment_id: fragment for fragment in fragments}
    duplicate_groups = [group for group in groups if len(group.fragment_ids) > 1]
    size_distribution: Counter[int] = Counter(len(group.fragment_ids) for group in duplicate_groups)
    contiguity: Counter[str] = Counter()
    law_collisions: Counter[str] = Counter()
    label_styles: Counter[str] = Counter()
    part_markers = 0
    high_confidence = 0
    ambiguous = 0
    conflicts = 0
    sample: list[dict[str, object]] = []

    for group in duplicate_groups:
        members = [fragment_by_id[item] for item in group.fragment_ids]
        rows = sorted(item.provenance.source_row for item in members)
        consecutive = all(right - left == 1 for left, right in pairwise(rows))
        contiguity["consecutive" if consecutive else "non_consecutive"] += 1
        law_collisions[group.law_name] += 1
        parsed = parse_article_label(group.raw_article_label)
        if parsed.part_index is not None:
            label_styles["explicit_part"] += 1
            part_markers += 1
        elif parsed.article_ordinal is not None:
            label_styles["numeric_or_ordinal"] += 1
        else:
            label_styles["unparsed"] += 1
        marker = any(
            re.search(r"continued|continuation|استمرار|تابع", item.raw_label, re.IGNORECASE)
            for item in members
        )
        distinct_text = len({item.text for item in members}) > 1
        if marker and consecutive:
            high_confidence += 1
            review_status = "high_confidence_continuation_candidate"
        elif consecutive and distinct_text:
            ambiguous += 1
            review_status = "ambiguous_continuation_candidate"
        else:
            conflicts += 1
            review_status = "genuine_conflict_candidate"
        if len(sample) < 25:
            sample.append(
                {
                    "sample_rank": len(sample) + 1,
                    "law_name": group.law_name,
                    "group_size": len(members),
                    "consecutive_within_law": consecutive,
                    "review_status": review_status,
                    "reconstruction_status": group.status.value,
                }
            )

    return {
        "schema_version": 2,
        "total_group_count": len(groups),
        "duplicate_group_count": len(duplicate_groups),
        "rows_in_duplicate_groups": sum(len(group.fragment_ids) for group in duplicate_groups),
        "group_size_distribution": {
            str(size): count for size, count in sorted(size_distribution.items())
        },
        "source_row_contiguity_distribution": dict(sorted(contiguity.items())),
        "duplicate_groups_consecutive_within_law": contiguity.get("consecutive", 0),
        "law_level_collision_concentration": [
            {"law_name": law, "duplicate_group_count": count}
            for law, count in sorted(law_collisions.items(), key=lambda item: (-item[1], item[0]))
        ],
        "article_label_style_distribution": dict(sorted(label_styles.items())),
        "explicit_part_marker_frequency": part_markers,
        "high_confidence_continuation_candidates": high_confidence,
        "ambiguous_continuation_candidates": ambiguous,
        "genuine_conflict_candidates": conflicts,
        "review_sample": sample,
        "merge_policy": "Only explicit structurally proven part series may be auto-merged.",
    }


def build_statutory_review_samples(
    law_name: str,
    fragments: Iterable[SourceFragment],
    requested_targets: dict[str, int],
) -> tuple[dict[str, object], ...]:
    """Select deterministic review samples without partial-ordinal grouping.

    A sample is an exact article-ordinal target.  Status or explicit-part variants
    of that same ordinal are review context; a distinct ordinal is never context.
    When a requested ordinal is unavailable, the selected ordinal is reported as
    the target instead of silently retaining misleading target metadata.
    """

    by_ordinal: dict[int, list[tuple[SourceFragment, ArticleLabel]]] = defaultdict(list)
    for fragment in fragments:
        parsed = parse_article_label(fragment.raw_label)
        if parsed.article_ordinal is not None:
            by_ordinal[parsed.article_ordinal].append((fragment, parsed))
    if not by_ordinal:
        raise ValueError(f"no parseable statutory articles available for {law_name}")

    ordered_ordinals = tuple(sorted(by_ordinal))
    samples: list[dict[str, object]] = []
    for role, requested_ordinal in requested_targets.items():
        selected_ordinal = (
            requested_ordinal
            if requested_ordinal in by_ordinal
            else min(ordered_ordinals, key=lambda value: (abs(value - requested_ordinal), value))
        )
        members = sorted(
            by_ordinal[selected_ordinal], key=lambda item: item[0].provenance.source_row
        )
        samples.append(
            {
                "law_name": law_name,
                "sample_role": role,
                "requested_article_ordinal": selected_ordinal,
                "original_requested_article_ordinal": requested_ordinal,
                "selection_resolution": (
                    "exact_requested_ordinal"
                    if selected_ordinal == requested_ordinal
                    else "fallback_exact_available_ordinal"
                ),
                "target_present": any(
                    parsed.article_ordinal == selected_ordinal for _, parsed in members
                ),
                "members": [
                    {
                        "raw_article_label": fragment.raw_label,
                        "parsed_article_ordinal": parsed.article_ordinal,
                        "part_index": parsed.part_index,
                        "article_status_marker": parsed.article_status_marker,
                        "article_label_structural_key": parsed.article_label_structural_key,
                        "source_row": fragment.provenance.source_row,
                    }
                    for fragment, parsed in members
                ],
            }
        )
    return tuple(samples)
