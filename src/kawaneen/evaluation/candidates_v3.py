"""Phase 6 draft-v3 generation from evidence-derived semantic targets."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from kawaneen.chunking.models import CitationAnchor
from kawaneen.corpus.models import CanonicalUnit
from kawaneen.evaluation.candidates import (
    BASE_TARGETS,
    EvidenceDiscovery,
    clean_semantic_text,
    discover_evidence,
    proportional_source_order,
    semantic_segments,
    valid_semantic_text,
)
from kawaneen.evaluation.corpus import EvaluationCorpus
from kawaneen.evaluation.models import (
    Answerability,
    CreationMethod,
    DatasetItem,
    DatasetSplit,
    Difficulty,
    EvidenceGroup,
    EvidenceSpan,
    QueryCategory,
    QueryLanguage,
    QueryRegister,
    QueryType,
    RelevanceGrade,
    SemanticTarget,
    deterministic_intent_id,
    deterministic_query_id,
)
from kawaneen.evaluation.semantic_targets import (
    extract_semantic_target,
    render_semantic_answer,
    render_semantic_query,
    validate_semantic_target,
)
from kawaneen.evaluation.serialization import read_items_jsonl, write_items_jsonl

V3_VERSION = "phase6-retrieval-eval-draft-v3"
V3_PRIVATE_ROOT = Path("artifacts/private/phase6_evaluation/draft-v3")
V3_POOL_TARGETS = {
    QueryCategory.EXACT_PROVISION: 60,
    QueryCategory.DEFINITION: 50,
    QueryCategory.DEADLINE: 50,
    QueryCategory.AUTHORITY: 50,
    QueryCategory.CONDITIONS: 60,
    QueryCategory.MULTI_EVIDENCE: 50,
    QueryCategory.CASE_HOLDING: 50,
}
_QUERY_TYPES = {
    QueryCategory.EXACT_PROVISION: QueryType.REFERENCE_LOOKUP,
    QueryCategory.DEFINITION: QueryType.LEGAL_CONCEPT,
    QueryCategory.DEADLINE: QueryType.PROCEDURE,
    QueryCategory.AUTHORITY: QueryType.RESPONSIBILITY,
    QueryCategory.CONDITIONS: QueryType.CONDITIONS_EXCEPTIONS,
    QueryCategory.MULTI_EVIDENCE: QueryType.REASONING,
    QueryCategory.CASE_HOLDING: QueryType.HOLDING_OUTCOME_REMEDY,
}


def _query_key(value: str) -> str:
    return re.sub(r"[^\wء-ي]+", " ", value.casefold()).strip()


def _unit_text(unit: CanonicalUnit, discovery: EvidenceDiscovery) -> str:
    return clean_semantic_text(unit.text[discovery.start : discovery.end])


def _anchor(unit: CanonicalUnit) -> CitationAnchor:
    return CitationAnchor(kind="section", label=unit.unit_type.value, source_unit_id=unit.unit_id)


def _item_from_target(
    category: QueryCategory,
    selections: tuple[tuple[CanonicalUnit, EvidenceDiscovery], ...],
    target: SemanticTarget,
    ordinal: int,
) -> DatasetItem:
    documents = tuple(sorted({unit.document_id for unit, _span in selections}))
    identity = tuple(
        (unit.unit_id, span.start, span.end, target.proposition) for unit, span in selections
    )
    intent_id = deterministic_intent_id(category.value, documents, identity)
    evidence = tuple(
        EvidenceSpan(
            unit_id=unit.unit_id,
            start=span.start,
            end=span.end,
            grade=RelevanceGrade.REQUIRED,
        )
        for unit, span in selections
    )
    groups = (EvidenceGroup(group_id=f"group-{intent_id[7:]}", spans=evidence),)
    return DatasetItem(
        query_id=deterministic_query_id(intent_id),
        intent_id=intent_id,
        query_text=render_semantic_query(target),
        language=QueryLanguage.ARABIC,
        register=QueryRegister.FORMAL,
        category=category,
        query_type=_QUERY_TYPES[category],
        jurisdiction="Saudi Arabia",
        temporal_scope="source-relative",
        creation_method=CreationMethod.DOCUMENT_DERIVED,
        answerability=Answerability.ANSWERABLE,
        difficulty=Difficulty.HARD
        if category is QueryCategory.MULTI_EVIDENCE
        else Difficulty.MEDIUM,
        source_document_ids=documents,
        evidence_groups=groups,
        gold_answer=render_semantic_answer(target),
        semantic_target=target,
        citation_anchors=tuple(
            # Citation anchors remain Phase-5 typed anchors in the serialized item.
            _anchor(unit)
            for unit, _span in selections
        ),
        dataset_version=V3_VERSION,
    )


def _semantic_candidates(
    corpus: EvaluationCorpus,
) -> dict[QueryCategory, defaultdict[str, list[tuple[DatasetItem, SemanticTarget]]]]:
    result: dict[QueryCategory, defaultdict[str, list[tuple[DatasetItem, SemanticTarget]]]] = {
        category: defaultdict(list) for category in V3_POOL_TARGETS
    }
    for unit in corpus.units:
        for category in V3_POOL_TARGETS:
            for discovery in discover_evidence(category, unit):
                target = extract_semantic_target(category, unit, discovery)
                if target is None:
                    continue
                text = _unit_text(unit, discovery)
                if not validate_semantic_target(category, target, (text,)):
                    continue
                item = _item_from_target(category, ((unit, discovery),), target, 0)
                result[category][unit.provenance.source_id].append((item, target))
    return result


def _substantive_tokens(value: str) -> set[str]:
    return {
        token
        for token in clean_semantic_text(value).split()
        if len(token) >= 4 and token not in {"التي", "الذي", "على", "من", "في", "وهو"}
    }


def _multi_candidates(
    corpus: EvaluationCorpus,
) -> defaultdict[str, list[tuple[DatasetItem, SemanticTarget]]]:
    by_document: defaultdict[tuple[str, str], list[CanonicalUnit]] = defaultdict(list)
    for unit in corpus.units:
        by_document[(unit.provenance.source_id, unit.document_id)].append(unit)
    result: defaultdict[str, list[tuple[DatasetItem, SemanticTarget]]] = defaultdict(list)
    for (source, _document), units in sorted(by_document.items()):
        if len(result[source]) >= V3_POOL_TARGETS[QueryCategory.MULTI_EVIDENCE]:
            break
        premise_one: list[tuple[CanonicalUnit, EvidenceDiscovery]] = []
        premise_two: list[tuple[CanonicalUnit, EvidenceDiscovery]] = []
        conclusions: list[tuple[CanonicalUnit, EvidenceDiscovery, SemanticTarget]] = []
        for unit in sorted(units, key=lambda row: (row.ordinal or 0, row.unit_id)):
            if unit.unit_type.value not in {
                "facts",
                "court_reasoning",
                "events",
                "reasoning",
                "verdict",
                "ruling",
            }:
                continue
            segments = tuple(
                EvidenceDiscovery(segment.start, segment.end, clean_semantic_text(segment.text), 1)
                for segment in semantic_segments(unit.text)
                if valid_semantic_text(segment.text)
            )
            if unit.unit_type.value in {"facts", "events"}:
                premise_one.extend((unit, segment) for segment in segments[:12])
            elif unit.unit_type.value in {"court_reasoning", "reasoning"}:
                premise_two.extend((unit, segment) for segment in segments[:12])
            elif unit.unit_type.value in {"verdict", "ruling"}:
                for segment in segments[:8]:
                    target = extract_semantic_target(QueryCategory.CASE_HOLDING, unit, segment)
                    if target and validate_semantic_target(
                        QueryCategory.CASE_HOLDING, target, (_unit_text(unit, segment),)
                    ):
                        conclusions.append((unit, segment, target))
        found = False
        for first_unit, first_span in premise_one[:24]:
            first_text = _unit_text(first_unit, first_span)
            first_tokens = _substantive_tokens(first_text)
            if len(first_tokens) < 2:
                continue
            for second_unit, second_span in premise_two[:24]:
                second_text = _unit_text(second_unit, second_span)
                second_tokens = _substantive_tokens(second_text)
                if not first_tokens & second_tokens:
                    continue
                for conclusion_unit, conclusion_span, conclusion_target in conclusions:
                    conclusion_text = _unit_text(conclusion_unit, conclusion_span)
                    conclusion_tokens = _substantive_tokens(conclusion_text)
                    if (
                        not first_tokens & conclusion_tokens
                        or not second_tokens & conclusion_tokens
                    ):
                        continue
                    target = extract_semantic_target(
                        QueryCategory.MULTI_EVIDENCE,
                        second_unit,
                        second_span,
                        evidence_texts=(first_text, second_text, conclusion_text),
                        conclusion=conclusion_target.proposition,
                    )
                    if target is None or not validate_semantic_target(
                        QueryCategory.MULTI_EVIDENCE,
                        target,
                        (first_text, second_text, conclusion_text),
                    ):
                        continue
                    selections = (
                        (first_unit, first_span),
                        (second_unit, second_span),
                        (conclusion_unit, conclusion_span),
                    )
                    item = _item_from_target(
                        QueryCategory.MULTI_EVIDENCE, selections, target, len(result[source])
                    )
                    result[source].append((item, target))
                    found = True
                    break
                if found:
                    break
            if found:
                break
    return result


def _deduplicate(
    rows: defaultdict[str, list[tuple[DatasetItem, SemanticTarget]]],
) -> defaultdict[str, list[tuple[DatasetItem, SemanticTarget]]]:
    result: defaultdict[str, list[tuple[DatasetItem, SemanticTarget]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for source, candidates in rows.items():
        for item, target in candidates:
            key = (item.category.value, target.proposition.casefold())
            if key in seen:
                continue
            seen.add(key)
            result[source].append((item, target))
    return result


def _select_category(
    category: QueryCategory,
    rows: defaultdict[str, list[tuple[DatasetItem, SemanticTarget]]],
    target_count: int,
) -> list[tuple[DatasetItem, SemanticTarget]]:
    deduped = _deduplicate(rows)
    # A semantic target can be valid while its first rendering is still
    # indistinguishable from another target (for example two provisions with
    # the same generic subject).  Reject those rows before source-proportional
    # selection; this is a non-retrieval quality rule.
    unique: defaultdict[str, list[tuple[DatasetItem, SemanticTarget]]] = defaultdict(list)
    seen_queries: set[str] = set()
    for source, candidates in deduped.items():
        for item, semantic_target in candidates:
            normalized = _query_key(render_semantic_query(semantic_target))
            if normalized in seen_queries:
                continue
            seen_queries.add(normalized)
            unique[source].append((item, semantic_target))
    selected = proportional_source_order(unique, target_count)
    if len(selected) < target_count:
        raise ValueError(
            f"v3 semantic target pool cannot satisfy {category.value}: "
            f"{len(selected)} < {target_count}"
        )
    return selected


def _variant_parents(items: tuple[DatasetItem, ...]) -> tuple[DatasetItem, ...]:
    by_category: defaultdict[QueryCategory, list[DatasetItem]] = defaultdict(list)
    for item in items:
        if item.answerability is Answerability.ANSWERABLE:
            by_category[item.category].append(item)
    parents: list[DatasetItem] = []
    for round_index in range(10):
        for category in BASE_TARGETS:
            if category is QueryCategory.UNANSWERABLE:
                continue
            rows = sorted(by_category[category], key=lambda item: item.query_id)
            if round_index < len(rows) and rows[round_index] not in parents:
                parents.append(rows[round_index])
                if len(parents) == 10:
                    return tuple(parents)
    raise ValueError("v3 could not select ten answerable robustness parents")


def _variants(items: tuple[DatasetItem, ...]) -> tuple[DatasetItem, ...]:
    parents = _variant_parents(items)
    result: list[DatasetItem] = []
    for variant_id in ("simple-ar", "egyptian-ar", "english", "code-switch"):
        for base in parents:
            target = base.semantic_target
            if target is None:
                raise ValueError("v3 robustness parent is missing semantic target")
            query = render_semantic_query(target, variant_id)
            language = {
                "simple-ar": QueryLanguage.ARABIC,
                "egyptian-ar": QueryLanguage.ARABIC,
                "english": QueryLanguage.ENGLISH,
                "code-switch": QueryLanguage.CODE_SWITCHED,
            }[variant_id]
            register = {
                "simple-ar": QueryRegister.SIMPLE,
                "egyptian-ar": QueryRegister.EGYPTIAN,
                "english": QueryRegister.PROFESSIONAL,
                "code-switch": QueryRegister.PROFESSIONAL,
            }[variant_id]
            if (variant_id == "english" and not query) or not query.strip():
                raise ValueError("v3 English variant is empty")
            result.append(
                base.model_copy(
                    update={
                        "query_id": deterministic_query_id(base.intent_id, variant_id),
                        "variant_id": variant_id,
                        "base_intent_id": base.intent_id,
                        "query_text": query,
                        "language": language,
                        "register": register,
                        "creation_method": CreationMethod.ROBUSTNESS_VARIANT,
                        "dataset_version": V3_VERSION,
                    }
                )
            )
    return tuple(result)


def load_accepted_unanswerables(
    v2_items_path: Path, adjudication_path: Path
) -> tuple[DatasetItem, ...]:
    accepted = {
        str(record["query_id"])
        for record in (
            json.loads(line)
            for line in adjudication_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if record.get("adjudicated_ai_decision") == "accept"
        and record.get("category") == QueryCategory.UNANSWERABLE.value
        and record.get("creation_method") == CreationMethod.DOCUMENT_DERIVED.value
    }
    items = read_items_jsonl(v2_items_path)
    preserved = tuple(
        item.model_copy(
            update={"dataset_version": V3_VERSION, "split": DatasetSplit.DEV, "smoke": False}
        )
        for item in items
        if item.query_id in accepted
    )
    if len(preserved) != 25:
        raise ValueError(f"expected 25 accepted base unanswerables, found {len(preserved)}")
    return preserved


def build_v3_candidates(
    corpus: EvaluationCorpus,
    *,
    v2_items_path: Path,
    adjudication_path: Path,
    output_root: Path = V3_PRIVATE_ROOT,
) -> tuple[tuple[DatasetItem, ...], tuple[DatasetItem, ...], tuple[DatasetItem, ...]]:
    opportunities = _semantic_candidates(corpus)
    multi = _multi_candidates(corpus)
    opportunities[QueryCategory.MULTI_EVIDENCE] = multi
    pool: list[DatasetItem] = []
    selected_answerable: list[DatasetItem] = []
    for category, target_count in V3_POOL_TARGETS.items():
        selected = _select_category(category, opportunities[category], target_count)
        pool.extend(item for item, _target in selected)
        selected_answerable.extend(item for item, _target in selected[: BASE_TARGETS[category]])
    preserved = load_accepted_unanswerables(v2_items_path, adjudication_path)
    selected = tuple(selected_answerable) + preserved
    variants = _variants(selected)
    output_root.mkdir(parents=True, exist_ok=True)
    write_items_jsonl(output_root / "base_candidates.jsonl", tuple(pool) + preserved)
    write_items_jsonl(output_root / "selected_and_variants.jsonl", selected + variants)
    return tuple(pool) + preserved, selected, variants
