"""Non-retrieval validation gates for private Phase 6 records."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from kawaneen.evaluation.models import Answerability, CreationMethod, DatasetItem, RelevanceGrade
from kawaneen.evaluation.semantic_targets import (
    render_semantic_answer,
    render_semantic_query,
    validate_semantic_target,
)

_EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")
_PHONE = re.compile(r"(?<!\d)(?:\+?\d[\d ()-]{7,}\d)(?!\d)")
_IMPLEMENTATION_TEXT = re.compile(
    r"(?:\[intent\s|internal reference|مرجع داخلي|corpus|chunk|qrel|benchmark|gold|evidence|"
    r"retriev|query[_ -]?id|intent[_ -]?id)",
    re.IGNORECASE,
)


class ValidationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    valid: bool
    item_count: int
    duplicate_query_count: int
    near_duplicate_query_count: int
    invalid_span_count: int
    invalid_evidence_group_count: int
    answerability_error_count: int
    missing_chunk_mapping_count: int
    privacy_finding_count: int
    lexical_overlap_count: int
    direct_leakage_count: int = 0
    implementation_text_count: int = 0
    semantic_target_error_count: int = 0

    def to_sanitized_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "item_count": self.item_count,
            "duplicate_query_count": self.duplicate_query_count,
            "near_duplicate_query_count": self.near_duplicate_query_count,
            "invalid_span_count": self.invalid_span_count,
            "invalid_evidence_group_count": self.invalid_evidence_group_count,
            "answerability_error_count": self.answerability_error_count,
            "missing_chunk_mapping_count": self.missing_chunk_mapping_count,
            "privacy_finding_count": self.privacy_finding_count,
            "lexical_overlap_count": self.lexical_overlap_count,
            "direct_leakage_count": self.direct_leakage_count,
            "implementation_text_count": self.implementation_text_count,
            "semantic_target_error_count": self.semantic_target_error_count,
        }


def benchmark_source_status(
    path: Path = Path("artifacts/private/phase6_evaluation/benchmark"),
) -> dict[str, object]:
    """Fail closed unless a permitted query/relevance schema is actually supplied."""

    if not path.exists():
        return {
            "status": "unavailable",
            "reason": "no permitted benchmark query/relevance instances are present",
            "fabricated_from_metadata": False,
        }
    if not path.is_file():
        return {
            "status": "blocked_schema_unverified",
            "reason": "benchmark source exists but its schema and governance must be reviewed",
            "fabricated_from_metadata": False,
        }
    return {
        "status": "blocked_schema_unverified",
        "reason": "benchmark source requires explicit schema and governance review",
        "fabricated_from_metadata": False,
    }


def validate_source_spans(items: tuple[DatasetItem, ...], unit_texts: dict[str, str]) -> None:
    for item in items:
        for group in item.evidence_groups:
            for span in group.spans:
                text = unit_texts.get(span.unit_id)
                if text is None:
                    raise ValueError(f"unknown unit in evidence: {span.unit_id}")
                if span.start < 0 or span.end > len(text) or span.end <= span.start:
                    raise ValueError(f"evidence span bounds exceed source text: {span.unit_id}")


def _privacy_findings(items: tuple[DatasetItem, ...]) -> int:
    return sum(
        len(pattern.findall(value))
        for item in items
        for value in (item.query_text, item.gold_answer or "", item.review.notes)
        for pattern in (_EMAIL, _IBAN, _PHONE)
    )


def _lexical_overlap(items: tuple[DatasetItem, ...], unit_texts: dict[str, str]) -> int:
    count = 0
    for item in items:
        query_tokens = set(re.findall(r"\w+", item.query_text.casefold()))
        source = " ".join(
            unit_texts.get(span.unit_id, "")
            for group in item.evidence_groups
            for span in group.spans
        )
        source_tokens = set(re.findall(r"\w+", source.casefold()))
        if query_tokens and len(query_tokens & source_tokens) / len(query_tokens) >= 0.5:
            count += 1
    return count


def _normalize(value: str) -> str:
    return re.sub(r"[^\wء-ي]+", " ", value.casefold()).strip()


def _direct_leakage(items: tuple[DatasetItem, ...], unit_texts: dict[str, str]) -> int:
    count = 0
    for item in items:
        if item.answerability is not Answerability.ANSWERABLE:
            continue
        evidence = " ".join(
            unit_texts.get(span.unit_id, "")[span.start : span.end]
            for group in item.evidence_groups
            for span in group.spans
        )
        query = _normalize(item.query_text)
        answer = _normalize(item.gold_answer or "")
        source = _normalize(evidence)
        direct = len(query) >= 12 and (query in answer or query in source)
        # N-gram overlap is retained as a diagnostic signal, but is not a
        # hard failure: legal questions necessarily share terminology with
        # their evidence.  Only direct embedding of the complete query is a
        # leakage gate.
        if direct:
            count += 1
    return count


def _implementation_text_count(items: tuple[DatasetItem, ...]) -> int:
    return sum(bool(_IMPLEMENTATION_TEXT.search(item.query_text)) for item in items)


def _semantic_target_errors(
    items: tuple[DatasetItem, ...], unit_texts: dict[str, str], *, required: bool
) -> int:
    if not required:
        return 0
    errors = 0
    base_by_intent = {
        item.intent_id: item
        for item in items
        if item.creation_method is not CreationMethod.ROBUSTNESS_VARIANT
    }
    for item in items:
        if item.answerability is Answerability.UNANSWERABLE:
            continue
        target = item.semantic_target
        if target is None:
            errors += 1
            continue
        evidence = tuple(
            unit_texts[span.unit_id][span.start : span.end]
            for group in item.evidence_groups
            for span in group.spans
        )
        variant = (
            item.variant_id if item.creation_method is CreationMethod.ROBUSTNESS_VARIANT else None
        )
        if item.dataset_version in {
            "phase6-retrieval-eval-draft-v4",
            "phase6-retrieval-eval-draft-v5",
        }:
            if item.dataset_version == "phase6-retrieval-eval-draft-v5":
                from kawaneen.evaluation.adjudication_v5 import validate_v5_item

                valid = validate_v5_item(item, evidence)
                errors += not valid
                continue
            from kawaneen.evaluation.adjudication_v4 import (
                clean_v4_text,
                validate_v4_semantic_contract,
                variant_query_v4,
            )

            valid = validate_v4_semantic_contract(
                item.category,
                target,
                item.query_text,
                item.gold_answer or "",
                tuple(clean_v4_text(value) for value in evidence),
            )
            if variant:
                valid = valid and item.query_text == variant_query_v4(target, variant)
        else:
            valid = validate_semantic_target(item.category, target, evidence)
            valid = valid and item.gold_answer == render_semantic_answer(target)
            valid = valid and item.query_text == render_semantic_query(target, variant)
        if item.creation_method is CreationMethod.ROBUSTNESS_VARIANT:
            parent = base_by_intent.get(item.base_intent_id or "")
            valid = valid and parent is not None and parent.semantic_target == target
            valid = valid and parent is not None and parent.evidence_groups == item.evidence_groups
            valid = valid and parent is not None and parent.gold_answer == item.gold_answer
        errors += not valid
    return errors


def validate_items(
    items: tuple[DatasetItem, ...],
    unit_texts: dict[str, str],
    *,
    require_semantic_targets: bool = False,
) -> ValidationSummary:
    base_items = tuple(item for item in items if item.creation_method.value != "robustness_variant")
    duplicate_query_count = len(base_items) - len(
        {_normalize(item.query_text) for item in base_items}
    )
    near_duplicate_query_count = sum(
        SequenceMatcher(None, left.query_text, right.query_text).ratio() >= 0.96
        for index, left in enumerate(base_items)
        for right in base_items[index + 1 :]
    )
    invalid_span_count = 0
    try:
        validate_source_spans(items, unit_texts)
    except ValueError:
        invalid_span_count = 1
    invalid_evidence_group_count = sum(
        not group.spans or not any(span.grade > RelevanceGrade.IRRELEVANT for span in group.spans)
        for item in items
        for group in item.evidence_groups
    )
    answerability_errors = sum(
        (
            item.answerability is Answerability.ANSWERABLE
            and (not item.gold_answer or not item.evidence_groups or not item.chunk_qrels)
        )
        or (
            item.answerability is Answerability.UNANSWERABLE
            and bool(item.gold_answer or item.evidence_groups or item.chunk_qrels)
        )
        for item in items
    )
    missing_chunk_mapping_count = sum(
        item.answerability is Answerability.ANSWERABLE and not item.chunk_qrels for item in items
    )
    privacy_count = _privacy_findings(items)
    direct_leakage_count = _direct_leakage(items, unit_texts)
    implementation_text_count = _implementation_text_count(items)
    semantic_target_error_count = _semantic_target_errors(
        items, unit_texts, required=require_semantic_targets
    )
    valid = not any(
        (
            duplicate_query_count,
            invalid_span_count,
            invalid_evidence_group_count,
            answerability_errors,
            missing_chunk_mapping_count,
            privacy_count,
            direct_leakage_count,
            implementation_text_count,
            semantic_target_error_count,
        )
    )
    return ValidationSummary(
        valid=valid,
        item_count=len(items),
        duplicate_query_count=duplicate_query_count,
        near_duplicate_query_count=near_duplicate_query_count,
        invalid_span_count=invalid_span_count,
        invalid_evidence_group_count=invalid_evidence_group_count,
        answerability_error_count=answerability_errors,
        missing_chunk_mapping_count=missing_chunk_mapping_count,
        privacy_finding_count=privacy_count,
        lexical_overlap_count=_lexical_overlap(items, unit_texts),
        direct_leakage_count=direct_leakage_count,
        implementation_text_count=implementation_text_count,
        semantic_target_error_count=semantic_target_error_count,
    )
