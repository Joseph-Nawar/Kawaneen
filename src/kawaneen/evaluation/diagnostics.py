"""Private machine-review diagnostics for Phase 6 draft-v2 records."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from kawaneen.corpus.models import CanonicalUnit
from kawaneen.evaluation.candidates import category_match
from kawaneen.evaluation.models import Answerability, CreationMethod, DatasetItem
from kawaneen.evaluation.semantic_targets import (
    render_semantic_answer,
    render_semantic_query,
    validate_semantic_target,
)

_PII = re.compile(
    r"(?:[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b|"
    r"(?<!\d)(?:\+?\d[\d ()-]{7,}\d)(?!\d))"
)
_IMPLEMENTATION = re.compile(
    r"(?:\[intent\s|internal reference|مرجع داخلي|corpus|chunk|qrel|benchmark|gold|evidence|"
    r"retriev|query[_ -]?id|intent[_ -]?id)",
    re.IGNORECASE,
)


def _normalize(value: str) -> str:
    return re.sub(r"[^\wء-ي]+", " ", value.casefold()).strip()


def _ngrams(value: str, size: int = 4) -> set[tuple[str, ...]]:
    tokens = _normalize(value).split()
    return {tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1)}


def _direct_overlap(left: str, right: str) -> bool:
    left_norm = _normalize(left)
    right_norm = _normalize(right)
    return len(left_norm) >= 12 and left_norm in right_norm


def _item_source(item: DatasetItem, unit_by_id: dict[str, CanonicalUnit]) -> str:
    if item.evidence_groups:
        return str(unit_by_id[item.evidence_groups[0].spans[0].unit_id].provenance.source_id)
    return "mixed-or-unanswerable"


def _record(
    item: DatasetItem,
    unit_by_id: dict[str, CanonicalUnit],
    normalized_queries: Counter[str],
    base_by_intent: dict[str, DatasetItem],
) -> dict[str, object]:
    evidence_text = " ".join(
        str(unit_by_id[span.unit_id].text)[span.start : span.end]
        for group in item.evidence_groups
        for span in group.spans
    )
    query_answer_direct = bool(
        item.gold_answer and _direct_overlap(item.query_text, item.gold_answer)
    )
    query_evidence_direct = _direct_overlap(item.query_text, evidence_text)
    query_answer_ngrams = len(_ngrams(item.query_text) & _ngrams(item.gold_answer or ""))
    query_evidence_ngrams = len(_ngrams(item.query_text) & _ngrams(evidence_text))
    legacy_category_valid = item.answerability is Answerability.UNANSWERABLE or all(
        category_match(
            item.category,
            str(unit_by_id[span.unit_id].text),
            str(unit_by_id[span.unit_id].unit_type.value),
        )
        for group in item.evidence_groups
        for span in group.spans
    )
    semantic_target_valid = False
    # Legacy v2 items have no typed semantic target; preserve their existing
    # diagnostic contract while v3 requires the explicit proposition gate.
    answer_entailment = (
        item.answerability is Answerability.UNANSWERABLE or item.semantic_target is None
    )
    query_from_target = True
    if item.answerability is Answerability.ANSWERABLE and item.semantic_target is not None:
        evidence_values = tuple(
            unit_by_id[span.unit_id].text[span.start : span.end]
            for group in item.evidence_groups
            for span in group.spans
        )
        if item.dataset_version == "phase6-retrieval-eval-final-candidate-v1":
            # Final literal review text is authoritative; conformance is checked
            # by the literal-patch validator rather than a semantic generator.
            semantic_target_valid = True
            query_from_target = True
        elif item.dataset_version == "phase6-retrieval-eval-draft-v5":
            from kawaneen.evaluation.adjudication_v5 import validate_v5_item

            semantic_target_valid = validate_v5_item(item, evidence_values)
            query_from_target = semantic_target_valid
        elif item.dataset_version == "phase6-retrieval-eval-draft-v4":
            from kawaneen.evaluation.adjudication_v4 import (
                clean_v4_text,
                validate_v4_semantic_contract,
                variant_query_v4,
            )

            semantic_target_valid = validate_v4_semantic_contract(
                item.category,
                item.semantic_target,
                item.query_text,
                item.gold_answer or "",
                tuple(clean_v4_text(value) for value in evidence_values),
            )
            variant = (
                item.variant_id
                if item.creation_method is CreationMethod.ROBUSTNESS_VARIANT
                else None
            )
            query_from_target = (
                item.query_text == variant_query_v4(item.semantic_target, variant)
                if variant
                else semantic_target_valid
            )
        else:
            semantic_target_valid = validate_semantic_target(
                item.category,
                item.semantic_target,
                evidence_values,
            )
            variant = (
                item.variant_id
                if item.creation_method is CreationMethod.ROBUSTNESS_VARIANT
                else None
            )
            query_from_target = item.query_text == render_semantic_query(
                item.semantic_target, variant
            )
        if item.dataset_version in {
            "phase6-retrieval-eval-final-candidate-v1",
            "phase6-retrieval-eval-draft-v5",
        }:
            answer_entailment = semantic_target_valid
        else:
            answer_entailment = (
                semantic_target_valid
                and item.gold_answer == render_semantic_answer(item.semantic_target)
            )
    category_valid = (
        semantic_target_valid if item.semantic_target is not None else legacy_category_valid
    )
    parent = base_by_intent.get(item.base_intent_id or "")
    variant_parent_valid = item.creation_method is not CreationMethod.ROBUSTNESS_VARIANT or (
        parent is not None
        and parent.semantic_target == item.semantic_target
        and parent.evidence_groups == item.evidence_groups
        and parent.gold_answer == item.gold_answer
        and bool(item.variant_id)
    )
    privacy_count = sum(
        len(_PII.findall(value)) for value in (item.query_text, item.gold_answer or "")
    )
    machine_quality = {
        "category_valid": category_valid,
        "evidence_sufficient": item.answerability is Answerability.UNANSWERABLE
        or (bool(item.evidence_groups) and (semantic_target_valid or item.semantic_target is None)),
        "answer_entailment": answer_entailment,
        "answer_entailment_proxy": answer_entailment,
        "natural_query": not bool(_IMPLEMENTATION.search(item.query_text)),
        "query_from_semantic_target": query_from_target,
        "query_answer_direct_overlap": query_answer_direct,
        "query_evidence_direct_overlap": query_evidence_direct,
        "query_answer_ngram_overlap_count": query_answer_ngrams,
        "query_evidence_ngram_overlap_count": query_evidence_ngrams,
        "privacy_resolved": privacy_count == 0,
        "duplicate_status": "duplicate"
        if normalized_queries[_normalize(item.query_text)] > 1
        else "unique",
        "variant_parent_valid": variant_parent_valid,
        "pass": all(
            (
                category_valid,
                answer_entailment,
                query_from_target,
                not bool(_IMPLEMENTATION.search(item.query_text)),
                not query_answer_direct,
                not query_evidence_direct,
                privacy_count == 0,
                variant_parent_valid,
            )
        ),
    }
    semantic_target_dump = (
        item.semantic_target.model_dump(mode="json") if item.semantic_target else None
    )
    if isinstance(semantic_target_dump, dict):
        semantic_target_dump = {
            key: _PII.sub("[REDACTED]", value) if isinstance(value, str) else value
            for key, value in semantic_target_dump.items()
        }
    return {
        "query_id": item.query_id,
        "intent_id": item.intent_id,
        "variant_id": item.variant_id,
        "base_intent_id": item.base_intent_id,
        "category": item.category.value,
        "source": _item_source(item, unit_by_id),
        "answerability": item.answerability.value,
        "unanswerable_reason": item.unanswerable_reason.value if item.unanswerable_reason else None,
        "evidence_spans": [
            {
                "unit_id": span.unit_id,
                "start": span.start,
                "end": span.end,
                "grade": int(span.grade),
            }
            for group in item.evidence_groups
            for span in group.spans
        ],
        "semantic_target": semantic_target_dump,
        "machine_quality": machine_quality,
        "privacy_finding_count": privacy_count,
    }


def build_review_diagnostics(
    items: tuple[DatasetItem, ...], units: tuple[CanonicalUnit, ...]
) -> tuple[dict[str, object], ...]:
    unit_by_id = {str(unit.unit_id): unit for unit in units}
    normalized_queries = Counter(_normalize(item.query_text) for item in items)
    base_by_intent = {
        item.intent_id: item
        for item in items
        if item.creation_method is not CreationMethod.ROBUSTNESS_VARIANT
    }
    return tuple(_record(item, unit_by_id, normalized_queries, base_by_intent) for item in items)


def write_review_diagnostics(path: Path, records: tuple[dict[str, object], ...]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records
        ),
        encoding="utf-8",
    )
    return path
