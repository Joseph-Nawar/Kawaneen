"""Literal application of the final external content-review patch."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, cast

from kawaneen.corpus.models import CanonicalUnit
from kawaneen.evaluation.chunks import map_items_to_chunks
from kawaneen.evaluation.models import (
    Answerability,
    CreationMethod,
    DatasetItem,
    EvidenceGroup,
    EvidenceSpan,
    RelevanceGrade,
    ReviewState,
    deterministic_query_id,
)
from kawaneen.evaluation.serialization import read_items_jsonl
from kawaneen.evaluation.splits import assign_provisional_splits, split_diagnostics

FINAL_VERSION = "phase6-retrieval-eval-final-candidate-v1"
FINAL_PRIVATE_ROOT = Path("artifacts/private/phase6_evaluation/final-candidate-v1")
V5_PRIVATE_ROOT = Path("artifacts/private/phase6_evaluation/draft-v5")
PATCH_PROVENANCE = "independent_ai_source_review"
EXPECTED_ACTIONS = {
    "accept_unchanged": 25,
    "edit_preserve_evidence": 167,
    "replace_with_candidate": 8,
    "variant_rewrite": 40,
}
_PII = re.compile(
    r"(?:[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b|"
    r"(?<!\d)(?:\+?\d[\d ()-]{7,}\d)(?!\d))"
)
_IMPLEMENTATION = re.compile(
    r"(?:\[intent\s|internal reference|مرجع داخلي|corpus|chunk|qrel|benchmark|gold|evidence|"
    r"retriev|query[_ -]?id|intent[_ -]?id)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class LiteralPatchRow:
    query_id: str
    intent_id: str
    category: str
    action: str
    old_query_text: str
    old_gold_answer: str | None
    new_query_text: str
    new_gold_answer: str | None
    variant_id: str | None
    base_intent_id: str | None
    split: str
    human_verified: bool
    review_provenance: str
    replacement_candidate_query_id: str | None = None
    use_candidate_span_indices: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class LiteralPatchSummary:
    patch_sha256: str
    applied_counts: dict[str, int]
    mismatches: int
    evidence_preservation_mismatches: int
    replacement_candidate_mismatches: int
    variant_parent_mismatches: int
    near_duplicate_pairs: tuple[dict[str, object], ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "patch_sha256": self.patch_sha256,
            "applied_counts": dict(sorted(self.applied_counts.items())),
            "mismatches": self.mismatches,
            "evidence_preservation_mismatches": self.evidence_preservation_mismatches,
            "replacement_candidate_mismatches": self.replacement_candidate_mismatches,
            "variant_parent_mismatches": self.variant_parent_mismatches,
            "near_duplicate_pairs": list(self.near_duplicate_pairs),
        }


@dataclass(frozen=True, slots=True)
class LiteralPatchResult:
    items: tuple[DatasetItem, ...]
    mapping: tuple[dict[str, object], ...]
    summary: LiteralPatchSummary


def _hash_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_row(record: dict[str, Any]) -> LiteralPatchRow:
    action = str(record.get("action", ""))
    if action not in EXPECTED_ACTIONS:
        raise ValueError(f"unknown literal patch action: {action}")
    raw_indices: object = record.get("use_candidate_span_indices")
    if raw_indices is None:
        indices: tuple[int, ...] = ()
    elif isinstance(raw_indices, list):
        values = cast(list[object], raw_indices)
        if not all(isinstance(value, int) for value in values):
            raise ValueError("candidate span indices must be an integer list")
        indices = tuple(value for value in values if isinstance(value, int))
    else:
        raise ValueError("candidate span indices must be an integer list")
    return LiteralPatchRow(
        query_id=str(record["query_id"]),
        intent_id=str(record["intent_id"]),
        category=str(record["category"]),
        action=action,
        old_query_text=str(record.get("old_query_text", "")),
        old_gold_answer=record.get("old_gold_answer"),
        new_query_text=str(record.get("new_query_text", "")),
        new_gold_answer=record.get("new_gold_answer"),
        variant_id=record.get("variant_id"),
        base_intent_id=record.get("base_intent_id"),
        split=str(record.get("split", "")),
        human_verified=bool(record.get("human_verified", False)),
        review_provenance=str(record.get("review_provenance", "")),
        replacement_candidate_query_id=record.get("replacement_candidate_query_id"),
        use_candidate_span_indices=indices,
    )


def load_literal_patch(
    path: Path,
    *,
    expected_query_ids: set[str] | None = None,
) -> tuple[LiteralPatchRow, ...]:
    rows = tuple(
        _parse_row(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if len(rows) != 240 or len({row.query_id for row in rows}) != 240:
        raise ValueError("literal patch must contain exactly 240 unique patch rows")
    if expected_query_ids is not None and {row.query_id for row in rows} != expected_query_ids:
        raise ValueError("literal patch IDs do not exactly match v5 IDs")
    counts = Counter(row.action for row in rows)
    if counts != Counter(EXPECTED_ACTIONS):
        raise ValueError(f"literal patch action counts mismatch: {dict(counts)}")
    if any(row.human_verified for row in rows):
        raise ValueError("literal patch cannot mark records human verified")
    if any(row.review_provenance != PATCH_PROVENANCE for row in rows):
        raise ValueError("literal patch provenance must be independent_ai_source_review")
    if any(
        row.action == "replace_with_candidate" and not row.replacement_candidate_query_id
        for row in rows
    ):
        raise ValueError("every replacement must name a candidate query ID")
    return rows


def _load_snapshot_units(path: Path) -> dict[str, CanonicalUnit]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        unit.unit_id: unit
        for unit in (CanonicalUnit.model_validate(row) for row in payload["units"])
    }


def _unit_text(unit: CanonicalUnit) -> str:
    return unit.text


def _validate_item_spans(item: DatasetItem, units: dict[str, CanonicalUnit]) -> None:
    for group in item.evidence_groups:
        if not group.spans or not any(
            span.grade > RelevanceGrade.IRRELEVANT for span in group.spans
        ):
            raise ValueError(f"invalid evidence group in {item.query_id}")
        for span in group.spans:
            unit = units.get(span.unit_id)
            if unit is None or span.end > len(_unit_text(unit)):
                raise ValueError(
                    f"candidate evidence span is outside frozen corpus: {item.query_id}"
                )


def validate_multi_candidate_spans(
    requested_indices: tuple[int, ...], actual_indices: tuple[int, ...]
) -> bool:
    return requested_indices == (0, 1) and actual_indices == (0, 1)


def _flatten_spans(item: DatasetItem) -> tuple[EvidenceSpan, ...]:
    return tuple(span for group in item.evidence_groups for span in group.spans)


def _tokens(value: str) -> set[str]:
    return {token for token in re.findall(r"[\wء-ي]+", value.casefold()) if len(token) > 2}


def _validate_multi_gold_components(
    gold: str, spans: tuple[EvidenceSpan, ...], units: dict[str, CanonicalUnit]
) -> None:
    components = tuple(part.strip() for part in re.split(r"،|؛", gold) if part.strip())
    if len(components) < 2:
        raise ValueError("multi-evidence gold must contain at least two requested components")
    for index, component in enumerate(components[:2]):
        source = _unit_text(units[spans[index].unit_id])[spans[index].start : spans[index].end]
        component_tokens = _tokens(component)
        source_tokens = _tokens(source)
        if (
            component_tokens
            and len(component_tokens & source_tokens) / len(component_tokens) < 0.35
        ):
            raise ValueError("multi-evidence gold component is unsupported by its requested span")


def _review_update(item: DatasetItem) -> object:
    return item.review.model_copy(
        update={
            "state": ReviewState.DRAFT,
            "human_verified": False,
            "review_provenance": PATCH_PROVENANCE,
        }
    )


def _literal_update(
    item: DatasetItem,
    *,
    query_text: str,
    gold_answer: str | None,
    query_id: str | None = None,
    intent_id: str | None = None,
    evidence_groups: tuple[EvidenceGroup, ...] | None = None,
) -> DatasetItem:
    return item.model_copy(
        update={
            "query_id": query_id or item.query_id,
            "intent_id": intent_id or item.intent_id,
            "query_text": query_text,
            "gold_answer": gold_answer,
            "evidence_groups": evidence_groups
            if evidence_groups is not None
            else item.evidence_groups,
            "dataset_version": FINAL_VERSION,
            "review": _review_update(item),
        }
    )


def _same_evidence(left: DatasetItem, right: DatasetItem) -> bool:
    return (
        left.source_document_ids == right.source_document_ids
        and left.evidence_groups == right.evidence_groups
        and left.citation_anchors == right.citation_anchors
        and left.verified_article_ids == right.verified_article_ids
    )


def _near_duplicate_pairs(items: tuple[DatasetItem, ...]) -> tuple[dict[str, object], ...]:
    result: list[dict[str, object]] = []
    for index, left in enumerate(items):
        for right in items[index + 1 :]:
            ratio = SequenceMatcher(None, left.query_text, right.query_text).ratio()
            if ratio >= 0.96:
                result.append(
                    {
                        "left_query_id": left.query_id,
                        "left_intent_id": left.intent_id,
                        "right_query_id": right.query_id,
                        "right_intent_id": right.intent_id,
                        "ratio": round(ratio, 6),
                        "same_intent": left.intent_id == right.intent_id,
                    }
                )
    return tuple(result)


def apply_literal_patch(
    v5_items_path: Path,
    patch_path: Path,
    *,
    candidate_pool_path: Path = V5_PRIVATE_ROOT / "draft" / "base_candidates.jsonl",
    corpus_units: dict[str, CanonicalUnit] | None = None,
) -> LiteralPatchResult:
    v5_items = read_items_jsonl(v5_items_path)
    expected_ids = {item.query_id for item in v5_items}
    patch = load_literal_patch(patch_path, expected_query_ids=expected_ids)
    by_id = {item.query_id: item for item in v5_items}
    candidates = {item.query_id: item for item in read_items_jsonl(candidate_pool_path)}
    if corpus_units is None:
        corpus_units = _load_snapshot_units(V5_PRIVATE_ROOT / "corpus" / "canonical_units.json")
    mapping: list[dict[str, object]] = []
    final_bases: dict[str, DatasetItem] = {}
    old_to_final_intent: dict[str, str] = {}
    evidence_preservation_mismatches = 0
    replacement_candidate_mismatches = 0
    variant_parent_mismatches = 0

    for row in patch:
        old = by_id[row.query_id]
        if old.query_text != row.old_query_text or old.gold_answer != row.old_gold_answer:
            raise ValueError(f"literal patch old text does not match v5: {row.query_id}")
        if row.action == "variant_rewrite":
            continue
        if row.action in {"accept_unchanged", "edit_preserve_evidence"}:
            if row.action == "accept_unchanged":
                updated = _literal_update(
                    old,
                    query_text=old.query_text,
                    gold_answer=old.gold_answer,
                )
                if not _same_evidence(old, updated):
                    evidence_preservation_mismatches += 1
            else:
                updated = _literal_update(
                    old,
                    query_text=row.new_query_text,
                    gold_answer=row.new_gold_answer,
                )
                if not _same_evidence(old, updated):
                    evidence_preservation_mismatches += 1
            final_bases[old.intent_id] = updated
            old_to_final_intent[old.intent_id] = updated.intent_id
            continue
        candidate_id = row.replacement_candidate_query_id or ""
        candidate = candidates.get(candidate_id)
        if candidate is None:
            replacement_candidate_mismatches += 1
            raise ValueError(f"replacement candidate is missing: {candidate_id}")
        _validate_item_spans(candidate, corpus_units)
        if candidate.category.value != row.category:
            replacement_candidate_mismatches += 1
            raise ValueError(f"replacement candidate category mismatch: {candidate_id}")
        groups = candidate.evidence_groups
        if row.use_candidate_span_indices:
            flat = _flatten_spans(candidate)
            actual = tuple(range(len(flat)))
            if not validate_multi_candidate_spans(row.use_candidate_span_indices, actual[:2]):
                replacement_candidate_mismatches += 1
                raise ValueError(
                    f"multi-evidence candidate span selection mismatch: {candidate_id}"
                )
            selected = (flat[0], flat[1])
            groups = tuple(
                EvidenceGroup(
                    group_id=f"literal-{candidate.intent_id}-{index}",
                    spans=(span.model_copy(update={"grade": RelevanceGrade.REQUIRED}),),
                )
                for index, span in enumerate(selected, start=1)
            )
            if row.new_gold_answer is None:
                raise ValueError("multi-evidence replacement requires a gold answer")
            _validate_multi_gold_components(row.new_gold_answer, selected, corpus_units)
            if len(groups) != 2 or any(len(group.spans) != 1 for group in groups):
                raise ValueError("multi-evidence replacement did not create two groups")
        updated = _literal_update(
            candidate,
            query_text=row.new_query_text,
            gold_answer=row.new_gold_answer,
            query_id=candidate.query_id,
            intent_id=candidate.intent_id,
            evidence_groups=groups,
        )
        evidence_matches = _same_evidence(candidate, updated)
        if row.use_candidate_span_indices:
            evidence_matches = (
                candidate.source_document_ids == updated.source_document_ids
                and candidate.citation_anchors == updated.citation_anchors
                and candidate.verified_article_ids == updated.verified_article_ids
                and _flatten_spans(updated) == _flatten_spans(candidate)[:2]
            )
        if not evidence_matches:
            replacement_candidate_mismatches += 1
            raise ValueError(f"replacement evidence was not preserved: {candidate_id}")
        final_bases[old.intent_id] = updated
        old_to_final_intent[old.intent_id] = updated.intent_id

    if len(final_bases) != 200:
        raise ValueError("literal patch did not produce exactly 200 base intents")
    final_variants: list[DatasetItem] = []
    for row in patch:
        if row.action != "variant_rewrite":
            continue
        old = by_id[row.query_id]
        old_parent = old.base_intent_id or old.intent_id
        parent = final_bases.get(old_parent)
        if parent is None:
            variant_parent_mismatches += 1
            raise ValueError(f"variant parent is missing: {old_parent}")
        updated = _literal_update(
            parent,
            query_text=row.new_query_text,
            gold_answer=parent.gold_answer,
            query_id=deterministic_query_id(parent.intent_id, old.variant_id),
            intent_id=parent.intent_id,
        ).model_copy(
            update={
                "variant_id": old.variant_id,
                "base_intent_id": parent.intent_id,
                "language": old.language,
                "register": old.register,
                "creation_method": CreationMethod.ROBUSTNESS_VARIANT,
                "split": parent.split,
                "smoke": parent.smoke,
            }
        )
        if not (
            updated.gold_answer == parent.gold_answer
            and _same_evidence(updated, parent)
            and updated.answerability is parent.answerability
        ):
            variant_parent_mismatches += 1
            raise ValueError(f"variant inheritance mismatch: {row.query_id}")
        final_variants.append(updated)

    bases = tuple(sorted(final_bases.values(), key=lambda item: item.intent_id))
    variants = tuple(sorted(final_variants, key=lambda item: item.query_id))
    all_items = assign_provisional_splits(
        map_items_to_chunks(bases + variants, _corpus_proxy(corpus_units))
    )
    final_bases_by_intent = {item.intent_id: item for item in all_items if item.variant_id is None}
    final_variants = [item for item in all_items if item.variant_id is not None]
    final_variants_by_key = {
        (item.base_intent_id, item.variant_id): item for item in final_variants
    }
    all_items = tuple(final_bases_by_intent.values()) + tuple(final_variants)
    all_items = tuple(sorted(all_items, key=lambda item: item.query_id))
    near_pairs = _near_duplicate_pairs(all_items)
    summary = LiteralPatchSummary(
        patch_sha256=_hash_bytes(patch_path),
        applied_counts=dict(sorted(Counter(row.action for row in patch).items())),
        mismatches=0,
        evidence_preservation_mismatches=evidence_preservation_mismatches,
        replacement_candidate_mismatches=replacement_candidate_mismatches,
        variant_parent_mismatches=variant_parent_mismatches,
        near_duplicate_pairs=near_pairs,
    )
    if any(
        (
            summary.mismatches,
            summary.evidence_preservation_mismatches,
            summary.replacement_candidate_mismatches,
            summary.variant_parent_mismatches,
        )
    ):
        raise ValueError("literal patch conformance failed")
    for old in v5_items:
        if old.variant_id is None:
            new = final_bases_by_intent[old_to_final_intent[old.intent_id]]
        else:
            final_parent = old_to_final_intent[old.base_intent_id or old.intent_id]
            new = final_variants_by_key[(final_parent, old.variant_id)]
        mapping.append(
            {
                "old_query_id": old.query_id,
                "old_intent_id": old.intent_id,
                "new_query_id": new.query_id,
                "new_intent_id": new.intent_id,
                "action": next(row.action for row in patch if row.query_id == old.query_id),
                "evidence_preserved": _same_evidence(old, new)
                if old.variant_id is None
                else _same_evidence(new, final_bases_by_intent[new.base_intent_id or ""]),
                "query_changed": old.query_text != new.query_text,
                "gold_changed": old.gold_answer != new.gold_answer,
                "replacement_candidate_query_id": next(
                    row.replacement_candidate_query_id
                    for row in patch
                    if row.query_id == old.query_id
                ),
            }
        )
    return LiteralPatchResult(all_items, tuple(mapping), summary)


def _corpus_proxy(units: dict[str, CanonicalUnit]) -> Any:
    """Provide the narrow corpus interface required by chunk mapping in tests."""

    from kawaneen.evaluation.corpus import EvaluationCorpus

    return EvaluationCorpus(
        units=tuple(units.values()),
        document_ids=frozenset(unit.document_id for unit in units.values()),
        unit_ids=frozenset(units),
        document_count_by_source={},
        unit_count_by_source={},
        unit_type_counts={},
        source_versions={},
        canonical_hashes={},
        content_policy_version="phase5-source-content-policy-v1",
        content_policy_hash="",
        document_ids_hash="",
        unit_ids_hash="",
        corpus_hash="",
    )


def validate_final_candidate(
    items: tuple[DatasetItem, ...],
    unit_texts: dict[str, str],
    *,
    corpus_hash: str,
    expected_corpus_hash: str,
    conformance: LiteralPatchSummary,
) -> dict[str, object]:
    base = tuple(item for item in items if item.variant_id is None)
    variants = tuple(item for item in items if item.variant_id is not None)
    expected_categories = Counter(
        {
            "exact_provision": 30,
            "definition": 25,
            "deadline": 20,
            "authority": 20,
            "conditions": 30,
            "multi_evidence": 25,
            "case_holding": 25,
            "unanswerable": 25,
        }
    )
    invalid_spans = 0
    invalid_groups = 0
    missing_qrels = 0
    answerability_errors = 0
    privacy = 0
    implementation = 0
    direct_leakage = 0
    for item in items:
        for group in item.evidence_groups:
            if not group.spans or not any(
                span.grade > RelevanceGrade.IRRELEVANT for span in group.spans
            ):
                invalid_groups += 1
            for span in group.spans:
                text = unit_texts.get(span.unit_id, "")
                invalid_spans += (
                    not text or span.start < 0 or span.end > len(text) or span.end <= span.start
                )
        privacy += sum(
            len(_PII.findall(value)) for value in (item.query_text, item.gold_answer or "")
        )
        implementation += bool(_IMPLEMENTATION.search(item.query_text))
        if item.answerability is Answerability.UNANSWERABLE:
            answerability_errors += bool(
                item.gold_answer or item.evidence_groups or item.chunk_qrels
            )
        else:
            answerability_errors += not bool(
                item.gold_answer and item.evidence_groups and item.chunk_qrels
            )
            missing_qrels += not bool(item.chunk_qrels)
            evidence = " ".join(
                unit_texts.get(span.unit_id, "")[span.start : span.end]
                for group in item.evidence_groups
                for span in group.spans
            )
            normalized_query = re.sub(r"[^\wء-ي]+", " ", item.query_text.casefold()).strip()
            normalized_answer = re.sub(
                r"[^\wء-ي]+", " ", (item.gold_answer or "").casefold()
            ).strip()
            normalized_evidence = re.sub(r"[^\wء-ي]+", " ", evidence.casefold()).strip()
            direct_leakage += bool(
                len(normalized_query) >= 12
                and (
                    normalized_query in normalized_answer or normalized_query in normalized_evidence
                )
            )
    normalized_queries = [
        re.sub(r"[^\wء-ي]+", " ", item.query_text.casefold()).strip() for item in items
    ]
    exact_duplicates = len(normalized_queries) - len(set(normalized_queries))
    near_pairs = _near_duplicate_pairs(items)
    cross_intent_near = sum(not bool(pair["same_intent"]) for pair in near_pairs)
    split = split_diagnostics(items)
    valid = all(
        (
            len(items) == 240,
            len(base) == 200,
            len(variants) == 40,
            Counter(item.category.value for item in base) == expected_categories,
            sum(item.answerability is Answerability.UNANSWERABLE for item in base) == 25,
            invalid_spans == 0,
            invalid_groups == 0,
            missing_qrels == 0,
            answerability_errors == 0,
            exact_duplicates == 0,
            privacy == 0,
            implementation == 0,
            direct_leakage == 0,
            cross_intent_near == 0,
            split.cross_split_document_count == 0,
            split.cross_split_intent_count == 0,
            corpus_hash == expected_corpus_hash,
            conformance.mismatches == 0,
            conformance.evidence_preservation_mismatches == 0,
            conformance.replacement_candidate_mismatches == 0,
            conformance.variant_parent_mismatches == 0,
            all(item.review.review_provenance == PATCH_PROVENANCE for item in items),
            all(not item.human_verified for item in items),
        )
    )
    return {
        "valid": valid,
        "item_count": len(items),
        "base_intent_count": len(base),
        "variant_count": len(variants),
        "base_category_counts": dict(sorted(Counter(item.category.value for item in base).items())),
        "unanswerable_count": sum(
            item.answerability is Answerability.UNANSWERABLE for item in base
        ),
        "invalid_span_count": invalid_spans,
        "invalid_evidence_group_count": invalid_groups,
        "missing_chunk_mapping_count": missing_qrels,
        "answerability_error_count": answerability_errors,
        "privacy_finding_count": privacy,
        "implementation_text_count": implementation,
        "direct_leakage_count": direct_leakage,
        "exact_duplicate_count": exact_duplicates,
        "near_duplicate_count": len(near_pairs),
        "cross_intent_near_duplicate_count": cross_intent_near,
        "near_duplicate_pairs": list(near_pairs),
        "split_diagnostics": split.model_dump(),
        "corpus_hash": corpus_hash,
        "corpus_hash_unchanged": corpus_hash == expected_corpus_hash,
    }
