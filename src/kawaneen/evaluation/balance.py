"""Text-free, pre-review source-balance diagnostics for Phase 6."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from kawaneen.corpus.models import CanonicalUnit
from kawaneen.evaluation.candidates import (
    BASE_TARGETS,
    CATEGORY_TARGETS,
    discover_evidence,
)
from kawaneen.evaluation.models import DatasetItem, QueryCategory

if TYPE_CHECKING:
    from kawaneen.evaluation.corpus import EvaluationCorpus


SOURCE_NAMES = {"alarb": "ALARB", "arabiccr": "ArabiCCR"}
SECTION_PAIRS = {
    "facts_events": ("facts", "events"),
    "court_reasoning_reasoning": ("court_reasoning", "reasoning"),
    "verdict_ruling": ("verdict", "ruling"),
}


def _source_counts(
    items: tuple[DatasetItem, ...], corpus: EvaluationCorpus
) -> dict[str, Counter[str]]:
    by_unit = {unit.unit_id: unit for unit in corpus.units}
    by_document: defaultdict[str, set[str]] = defaultdict(set)
    for unit in corpus.units:
        by_document[unit.document_id].add(unit.provenance.source_id)
    result: dict[str, Counter[str]] = defaultdict(Counter)
    for item in items:
        if item.evidence_groups:
            source = by_unit[item.evidence_groups[0].spans[0].unit_id].provenance.source_id
        else:
            sources = by_document[item.source_document_ids[0]]
            source = sorted(sources)[0]
        result[item.category.value][source] += 1
    return result


def _opportunities(corpus: EvaluationCorpus) -> dict[str, Counter[str]]:
    result: dict[str, Counter[str]] = defaultdict(Counter)
    for unit in corpus.units:
        source = unit.provenance.source_id
        for category in BASE_TARGETS:
            if category is QueryCategory.MULTI_EVIDENCE or category is QueryCategory.UNANSWERABLE:
                continue
            if discover_evidence(category, unit):
                result[category.value][source] += 1

    documents: defaultdict[tuple[str, str], list[CanonicalUnit]] = defaultdict(list)
    for unit in corpus.units:
        documents[(unit.provenance.source_id, unit.document_id)].append(unit)
    for (source, _document_id), units in documents.items():
        if len(units) >= 2:
            result[QueryCategory.MULTI_EVIDENCE.value][source] += 1
    result[QueryCategory.UNANSWERABLE.value] = Counter(
        Counter(unit.provenance.source_id for unit in corpus.units)
    )
    # The unanswerable opportunity is one governed document, not one unit.
    result[QueryCategory.UNANSWERABLE.value] = Counter(
        {
            source: len(
                {unit.document_id for unit in corpus.units if unit.provenance.source_id == source}
            )
            for source in {unit.provenance.source_id for unit in corpus.units}
        }
    )
    return result


def _legacy_source_order_generated(corpus: EvaluationCorpus) -> dict[str, Counter[str]]:
    """Reproduce the pre-audit source-first cap for a sanitized before/after record."""

    result: dict[str, Counter[str]] = defaultdict(Counter)
    ordered = sorted(
        corpus.units,
        key=lambda unit: (unit.provenance.source_id, unit.document_id, unit.unit_id),
    )
    for category, target in CATEGORY_TARGETS.items():
        if category is QueryCategory.UNANSWERABLE:
            docs = sorted({(unit.document_id, unit.provenance.source_id) for unit in corpus.units})[
                :target
            ]
            result[category.value].update(source for _doc, source in docs)
            continue
        if category is QueryCategory.MULTI_EVIDENCE:
            by_document: defaultdict[str, list[CanonicalUnit]] = defaultdict(list)
            for unit in corpus.units:
                by_document[unit.document_id].append(unit)
            count = 0
            for document_id in sorted(by_document):
                rows = by_document[document_id]
                if len(rows) < 2:
                    continue
                result[category.value][rows[0].provenance.source_id] += 1
                count += 1
                if count == target:
                    break
            continue
        count = 0
        for unit in ordered:
            if discover_evidence(category, unit):
                result[category.value][unit.provenance.source_id] += 1
                count += 1
                if count == target:
                    break
    return result


def _section_parity(corpus: EvaluationCorpus) -> dict[str, object]:
    result: dict[str, object] = {}
    for pair_name, (left_type, right_type) in SECTION_PAIRS.items():
        sides: dict[str, object] = {}
        for label, unit_type in (("left", left_type), ("right", right_type)):
            rows = [unit for unit in corpus.units if unit.unit_type.value == unit_type]
            matches = {
                category.value: sum(bool(discover_evidence(category, unit)) for unit in rows)
                for category in BASE_TARGETS
            }
            sides[label] = {
                "unit_type": unit_type,
                "unit_count": len(rows),
                "opportunity_counts": matches,
                "opportunity_rates": {
                    category: round(count / len(rows), 6) if rows else 0.0
                    for category, count in matches.items()
                },
            }
        result[pair_name] = sides
    return result


def build_source_balance_audit(
    corpus: EvaluationCorpus,
    base_candidates: tuple[DatasetItem, ...],
    selected: tuple[DatasetItem, ...],
) -> dict[str, object]:
    opportunities = _opportunities(corpus)
    generated = _source_counts(base_candidates, corpus)
    selected_counts = _source_counts(selected, corpus)
    legacy = _legacy_source_order_generated(corpus)
    flow: dict[str, object] = {}
    rejection_counts: dict[str, object] = {}
    for category in CATEGORY_TARGETS:
        category_name = category.value
        sources = sorted(
            set(opportunities[category_name])
            | set(generated[category_name])
            | set(selected_counts[category_name])
        )
        flow[category_name] = {
            source: {
                "eligible_opportunities": opportunities[category_name][source],
                "generated_candidates": generated[category_name][source],
                "selected_base_intents": selected_counts[category_name][source],
            }
            for source in sources
        }
        rejection_counts[category_name] = {
            source: {
                "not_generated_after_opportunity_cap": max(
                    0, opportunities[category_name][source] - generated[category_name][source]
                ),
                "generated_not_selected_after_base_cap": max(
                    0, generated[category_name][source] - selected_counts[category_name][source]
                ),
                "sanitized_reasons": [
                    "category_target_capacity",
                    "base_intent_target_capacity",
                ],
            }
            for source in sources
        }
    heuristic: dict[str, dict[str, dict[str, object]]] = {}
    for category in BASE_TARGETS:
        category_name = category.value
        heuristic[category_name] = {}
        for source in sorted({unit.provenance.source_id for unit in corpus.units}):
            rows = [unit for unit in corpus.units if unit.provenance.source_id == source]
            count = opportunities[category_name][source]
            heuristic[category_name][source] = {
                "eligible_unit_count": len(rows),
                "opportunity_count": count,
                "opportunity_rate": round(count / len(rows), 6) if rows else 0.0,
                "selection_rule": (
                    "same category heuristic; opportunity-proportional source ordering"
                ),
            }
    return {
        "schema_version": 1,
        "audit_scope": "bounded pre-review source-balance audit",
        "retrieval_scores_used": False,
        "corpus_hash": corpus.corpus_hash,
        "base_candidate_count": len(base_candidates),
        "selected_base_intent_count": len(selected),
        "source_names": SOURCE_NAMES,
        "source_flow": flow,
        "rejection_counts": rejection_counts,
        "pre_audit_source_first_generated": {
            category: dict(sorted(counts.items())) for category, counts in sorted(legacy.items())
        },
        "post_audit_source_proportional_generated": {
            category: dict(sorted(generated[category].items())) for category in sorted(generated)
        },
        "heuristic_diagnostics": heuristic,
        "section_parity": _section_parity(corpus),
        "finding": {
            "cause": "source_sorted_candidate_cap",
            "corrected": True,
            "equal_split_forced": False,
            "selection_basis": "measured eligible opportunity share",
        },
    }


def source_balance_audit(
    corpus: EvaluationCorpus,
    base_candidates: tuple[DatasetItem, ...],
    selected: tuple[DatasetItem, ...],
    *,
    output_path: Path,
) -> dict[str, object]:
    import json

    report = build_source_balance_audit(corpus, base_candidates, selected)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return report
