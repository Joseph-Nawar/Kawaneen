"""Phase 6 corpus, draft, review, validation, and immutable-freeze orchestration."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import cast

from kawaneen.chunking.policies import get_chunk_policy
from kawaneen.evaluation.adjudication_v4 import (
    V4_PRIVATE_ROOT,
    V4_VERSION,
    apply_v3_adjudication,
    duplicate_query_keys,
    load_v3_adjudication,
    multi_opportunity_audit,
)
from kawaneen.evaluation.adjudication_v5 import (
    V5_PRIVATE_ROOT,
    V5_VERSION,
    apply_v4_adjudication,
    load_v4_final_adjudication,
)
from kawaneen.evaluation.balance import source_balance_audit
from kawaneen.evaluation.candidates import PRIVATE_ROOT, build_draft_candidates
from kawaneen.evaluation.candidates_v3 import (
    V3_PRIVATE_ROOT,
    V3_VERSION,
    build_v3_candidates,
)
from kawaneen.evaluation.chunks import map_items_to_chunks
from kawaneen.evaluation.corpus import (
    EvaluationCorpus,
    corpus_summary,
    freeze_evaluation_corpus,
    load_evaluation_units,
)
from kawaneen.evaluation.diagnostics import build_review_diagnostics, write_review_diagnostics
from kawaneen.evaluation.freeze import freeze_items
from kawaneen.evaluation.handoff import write_handoff_artifacts
from kawaneen.evaluation.literal_patch import (
    FINAL_PRIVATE_ROOT,
    FINAL_VERSION,
    LiteralPatchSummary,
    apply_literal_patch,
    validate_final_candidate,
)
from kawaneen.evaluation.models import DatasetItem, DatasetSplit, ReviewState
from kawaneen.evaluation.review import (
    _packet_record,  # pyright: ignore[reportPrivateUsage]
    export_review_packet,
    import_reviews,
    review_status,
)
from kawaneen.evaluation.serialization import (
    read_items_jsonl,
    write_items_jsonl,
    write_private_corpus_snapshot,
)
from kawaneen.evaluation.splits import assign_provisional_splits, split_diagnostics
from kawaneen.evaluation.validation import benchmark_source_status, validate_items

TRACKED_ROOT = Path("data/manifests/evaluation")
SUMMARY_PATH = TRACKED_ROOT / "phase6_corpus_summary.json"
DRAFT_SUMMARY_PATH = Path("data/evaluation/phase6_retrieval_eval_summary.json")
SOURCE_BALANCE_AUDIT_PATH = Path("data/evaluation/phase6_source_balance_audit.json")
PRIVATE_ITEMS = PRIVATE_ROOT / "draft" / "selected_and_variants.jsonl"
PRIVATE_BASE_CANDIDATES = PRIVATE_ROOT / "draft" / "base_candidates.jsonl"
PRIVATE_PACKET = PRIVATE_ROOT / "review" / "review_packet.jsonl"
PRIVATE_DIAGNOSTICS = PRIVATE_ROOT / "review" / "review_diagnostics.jsonl"
PRIVATE_HANDOFF = PRIVATE_ROOT / "handoff"
V3_ITEMS = V3_PRIVATE_ROOT / "draft" / "selected_and_variants.jsonl"
V3_BASE_CANDIDATES = V3_PRIVATE_ROOT / "draft" / "base_candidates.jsonl"
V3_PACKET = V3_PRIVATE_ROOT / "review" / "review_packet.jsonl"
V3_DIAGNOSTICS = V3_PRIVATE_ROOT / "review" / "review_diagnostics.jsonl"
V3_HANDOFF = V3_PRIVATE_ROOT / "handoff"
V3_SUMMARY_PATH = Path("data/evaluation/phase6_retrieval_eval_draft_v3_summary.json")
V3_TRACKED_SUMMARY_PATH = TRACKED_ROOT / "phase6_draft_v3_summary.json"
V3_BALANCE_PATH = Path("data/evaluation/phase6_v3_source_balance_audit.json")
V4_ITEMS = V4_PRIVATE_ROOT / "draft" / "selected_and_variants.jsonl"
V4_BASE_CANDIDATES = V4_PRIVATE_ROOT / "draft" / "base_candidates.jsonl"
V4_PACKET = V4_PRIVATE_ROOT / "review" / "review_packet.jsonl"
V4_DIAGNOSTICS = V4_PRIVATE_ROOT / "review" / "review_diagnostics.jsonl"
V4_MAPPING = V4_PRIVATE_ROOT / "review" / "v3_adjudication_application.jsonl"
V4_HANDOFF = V4_PRIVATE_ROOT / "handoff"
V4_SUMMARY_PATH = Path("data/evaluation/phase6_retrieval_eval_draft_v4_summary.json")
V4_TRACKED_SUMMARY_PATH = TRACKED_ROOT / "phase6_draft_v4_summary.json"
V5_ITEMS = V5_PRIVATE_ROOT / "draft" / "selected_and_variants.jsonl"
V5_BASE_CANDIDATES = V5_PRIVATE_ROOT / "draft" / "base_candidates.jsonl"
V5_PACKET = V5_PRIVATE_ROOT / "review" / "review_packet.jsonl"
V5_CHANGED_PACKET = V5_PRIVATE_ROOT / "review" / "changed_review_packet.jsonl"
V5_DIAGNOSTICS = V5_PRIVATE_ROOT / "review" / "review_diagnostics.jsonl"
V5_MAPPING = V5_PRIVATE_ROOT / "review" / "v4_adjudication_application.jsonl"
V5_HANDOFF = V5_PRIVATE_ROOT / "handoff"
V5_SUMMARY_PATH = Path("data/evaluation/phase6_retrieval_eval_draft_v5_summary.json")
V5_TRACKED_SUMMARY_PATH = TRACKED_ROOT / "phase6_draft_v5_summary.json"
FINAL_ITEMS = FINAL_PRIVATE_ROOT / "draft" / "selected_and_variants.jsonl"
FINAL_PACKET = FINAL_PRIVATE_ROOT / "review" / "review_packet.jsonl"
FINAL_DIAGNOSTICS = FINAL_PRIVATE_ROOT / "review" / "review_diagnostics.jsonl"
FINAL_CONFORMANCE = FINAL_PRIVATE_ROOT / "review" / "literal_patch_conformance.json"
FINAL_MAPPING = FINAL_PRIVATE_ROOT / "review" / "v5_literal_patch_application.jsonl"
FINAL_HANDOFF = FINAL_PRIVATE_ROOT / "handoff"
FINAL_SUMMARY_PATH = Path("data/evaluation/phase6_retrieval_eval_final_candidate_v1_summary.json")
FINAL_TRACKED_SUMMARY_PATH = TRACKED_ROOT / "phase6_final_candidate_v1_summary.json"
AI_REVIEWED_VERSION = "phase6-retrieval-eval-ai-reviewed-v1"
AI_REVIEWED_PRIVATE_ROOT = Path("artifacts/private/phase6_evaluation/ai-reviewed-v1")
AI_REVIEWED_MANIFEST_PATH = TRACKED_ROOT / "phase6_ai_reviewed_v1_manifest.json"
AI_REVIEWED_REPORT_PATH = Path("data/evaluation/phase6_retrieval_eval_ai_reviewed_v1_report.json")
EXPECTED_CORPUS_HASH = "290d7a91e5f435778e782b76284a9797fb7f5ae261380f0a923b56224e530daa"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


def _hash_items(items: tuple[DatasetItem, ...]) -> str:
    payload = [item.model_dump(mode="json") for item in items]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _hash_payload(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _v3_replacement_mapping(
    old_items: tuple[DatasetItem, ...],
    selected: tuple[DatasetItem, ...],
    variants: tuple[DatasetItem, ...],
    adjudication_path: Path,
    destination: Path,
) -> dict[str, object]:
    decisions = [
        json.loads(line)
        for line in adjudication_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rejected_base = [
        row
        for row in decisions
        if row.get("adjudicated_ai_decision") == "reject"
        and row.get("creation_method") == "document_derived"
        and row.get("category") != "unanswerable"
    ]
    rejected_variants = [
        row
        for row in decisions
        if row.get("adjudicated_ai_decision") == "reject"
        and row.get("creation_method") == "robustness_variant"
    ]
    new_base_by_category: dict[str, list[DatasetItem]] = {}
    for item in selected:
        if item.answerability.value == "answerable":
            new_base_by_category.setdefault(item.category.value, []).append(item)
    old_base_by_category: dict[str, list[dict[str, object]]] = {}
    for row in rejected_base:
        old_base_by_category.setdefault(str(row["category"]), []).append(row)
    rows: list[dict[str, object]] = []
    for category, old_rows in sorted(old_base_by_category.items()):
        new_rows = new_base_by_category.get(category, [])
        if len(old_rows) != len(new_rows):
            raise ValueError(f"v3 replacement count mismatch for {category}")
        for old, new in zip(
            sorted(old_rows, key=lambda row: str(row["query_id"])), new_rows, strict=True
        ):
            rows.append(
                {
                    "kind": "base_replacement",
                    "old_query_id": old["query_id"],
                    "old_intent_id": old["intent_id"],
                    "old_category": category,
                    "new_query_id": new.query_id,
                    "new_intent_id": new.intent_id,
                    "new_category": new.category.value,
                    "reason": "semantic_target_regeneration_after_external_source_review",
                }
            )
    old_variants_by_kind: dict[str, list[dict[str, object]]] = {}
    new_variants_by_kind: dict[str, list[DatasetItem]] = {}
    for row in rejected_variants:
        old_variants_by_kind.setdefault(str(row.get("variant_id")), []).append(row)
    for item in variants:
        new_variants_by_kind.setdefault(item.variant_id or "", []).append(item)
    for variant_id, old_rows in sorted(old_variants_by_kind.items()):
        new_rows = new_variants_by_kind.get(variant_id, [])
        if len(old_rows) != len(new_rows):
            raise ValueError(f"v3 variant replacement count mismatch for {variant_id}")
        for old, new in zip(
            sorted(old_rows, key=lambda row: str(row["query_id"])), new_rows, strict=True
        ):
            rows.append(
                {
                    "kind": "variant_replacement",
                    "old_query_id": old["query_id"],
                    "old_intent_id": old["intent_id"],
                    "old_variant_id": variant_id,
                    "new_query_id": new.query_id,
                    "new_intent_id": new.intent_id,
                    "new_variant_id": new.variant_id,
                    "new_base_intent_id": new.base_intent_id,
                    "reason": "semantic_target_regeneration_after_external_source_review",
                }
            )
    if len(rows) != len(rejected_base) + len(rejected_variants):
        raise ValueError("v3 replacement mapping does not cover every rejected record")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return {
        "path": destination.as_posix(),
        "rejected_base_replaced": len(rejected_base),
        "rejected_variants_replaced": len(rejected_variants),
        "accepted_base_unanswerables_preserved": sum(
            item.category.value == "unanswerable"
            and item.creation_method.value == "document_derived"
            for item in old_items
            if item.query_id
            in {
                row["query_id"]
                for row in decisions
                if row.get("adjudicated_ai_decision") == "accept"
            }
        ),
    }


def evaluation_plan() -> dict[str, object]:
    if AI_REVIEWED_MANIFEST_PATH.is_file():
        active_version, active_root = AI_REVIEWED_VERSION, AI_REVIEWED_PRIVATE_ROOT
    elif V5_ITEMS.is_file():
        active_version, active_root = V5_VERSION, V5_PRIVATE_ROOT
    elif V4_ITEMS.is_file():
        active_version, active_root = V4_VERSION, V4_PRIVATE_ROOT
    else:
        active_version, active_root = V3_VERSION, V3_PRIVATE_ROOT
    return {
        "schema_version": 1,
        "phase": "phase-06-retrieval-evaluation-dataset",
        "private_root": PRIVATE_ROOT.as_posix(),
        "active_draft_version": active_version,
        "previous_draft_version": (
            V4_VERSION
            if V5_ITEMS.is_file()
            else V3_VERSION
            if V4_ITEMS.is_file()
            else "phase6-retrieval-eval-draft-v2"
        ),
        "active_private_root": active_root.as_posix(),
        "release_classification": (
            "externally_ai_reviewed"
            if active_version == AI_REVIEWED_VERSION
            else "pre_review_draft"
        ),
        "corpus_scope": "full ALARB + ArabiCCR governed canonical retrieval corpus",
        "source_policy": (
            "ALARB facts/court_reasoning/applicable_laws/verdict; "
            "ArabiCCR EVENTS/REASONING/RULING with case_text fallback"
        ),
        "benchmark_source": "unavailable unless permitted query/relevance instances are supplied",
        "targets": {
            "base_candidates": 360,
            "base_intents": 200,
            "variants": 40,
            "dev": 160,
            "holdout": 80,
        },
        "chunk_policy_id": "legal-structure-v1",
        "scope_exclusions": [
            "OCR",
            "MOJ statute seed as retrieval gold",
            "embeddings",
            "dense retrieval",
            "reranking",
            "RAG",
            "APIs",
            "UI",
        ],
    }


def _full_corpus(private_root: Path, tracked_root: Path = TRACKED_ROOT):
    corpus = freeze_evaluation_corpus(
        load_evaluation_units(Path("data/interim/canonical")),
        canonical_root=Path("data/interim/canonical"),
    )
    write_private_corpus_snapshot(corpus, private_root / "corpus" / "canonical_units.json")
    _write_json(tracked_root / "phase6_corpus_summary.json", corpus_summary(corpus))
    return corpus


def run_build_draft(
    *, private_root: Path = PRIVATE_ROOT, tracked_root: Path = TRACKED_ROOT
) -> dict[str, object]:
    corpus = _full_corpus(private_root, tracked_root)
    draft = build_draft_candidates(corpus, output_root=private_root)
    mapped = map_items_to_chunks(draft.all_items, corpus)
    split_items = assign_provisional_splits(mapped)
    write_items_jsonl(private_root / "draft" / "selected_and_variants.jsonl", split_items)
    unit_texts = {unit.unit_id: unit.text for unit in corpus.units}
    export_review_packet(
        private_root / "draft" / "selected_and_variants.jsonl",
        private_root / "review" / "review_packet.jsonl",
        unit_texts,
    )
    diagnostics_path = private_root / "review" / "review_diagnostics.jsonl"
    handoff_root = private_root / "handoff"
    diagnostics = build_review_diagnostics(split_items, corpus.units)
    write_review_diagnostics(diagnostics_path, diagnostics)
    handoff = write_handoff_artifacts(corpus, split_items, handoff_root)
    source_by_item = Counter(
        source
        for item in split_items
        for source in {
            next(
                (unit.provenance.source_id for unit in corpus.units if unit.document_id == doc),
                "unknown",
            )
            for doc in item.source_document_ids
        }
    )
    category_counts = Counter(item.category.value for item in split_items)
    base_category_counts = Counter(item.category.value for item in draft.selected_base_candidates)
    variant_language_counts = Counter(item.language.value for item in draft.variants)
    unit_source_by_document = {unit.document_id: unit.provenance.source_id for unit in corpus.units}
    base_source_counts = Counter(
        unit_source_by_document[document_id]
        for item in draft.selected_base_candidates
        for document_id in item.source_document_ids
    )
    base_category_source_counts = Counter(
        f"{item.category.value}:{unit_source_by_document[document_id]}"
        for item in draft.selected_base_candidates
        for document_id in item.source_document_ids
    )
    language_counts = Counter(item.language.value for item in split_items)
    validation = validate_items(split_items, unit_texts)
    diagnostics = split_diagnostics(split_items)
    summary: dict[str, object] = {
        "schema_version": 1,
        "status": "phase6_draft_built_pending_human_review",
        "dataset_version": "phase6-retrieval-eval-draft-v2",
        "corpus_hash": corpus.corpus_hash,
        "base_candidate_count": len(draft.base_candidates),
        "base_intent_count": len(draft.selected_base_candidates),
        "variant_count": len(draft.variants),
        "item_count": len(split_items),
        "category_counts": dict(sorted(category_counts.items())),
        "base_category_counts": dict(sorted(base_category_counts.items())),
        "variant_language_counts": dict(sorted(variant_language_counts.items())),
        "language_counts": dict(sorted(language_counts.items())),
        "source_counts": dict(sorted(source_by_item.items())),
        "base_source_counts": dict(sorted(base_source_counts.items())),
        "base_category_source_counts": dict(sorted(base_category_source_counts.items())),
        "validation": validation.to_sanitized_dict(),
        "split_diagnostics": diagnostics.model_dump(),
        "review": review_status(split_items),
        "benchmark_source_status": "unavailable_document_derived_only",
        "private_artifacts": {
            "corpus_snapshot": (private_root / "corpus" / "canonical_units.json").as_posix(),
            "items": (private_root / "draft" / "selected_and_variants.jsonl").as_posix(),
            "review_packet": (private_root / "review" / "review_packet.jsonl").as_posix(),
            "review_diagnostics": diagnostics_path.as_posix(),
            "handoff_manifest": (handoff_root / "canonical_review_manifest.json").as_posix(),
            "handoff_context": (handoff_root / "phase6_review_source_context.jsonl").as_posix(),
        },
        "handoff": handoff,
    }
    _write_json(tracked_root / "phase6_draft_summary.json", summary)
    _write_json(DRAFT_SUMMARY_PATH, summary)
    return summary


def run_build_draft_v3(
    *,
    review_file: Path,
    private_root: Path = V3_PRIVATE_ROOT,
    tracked_root: Path = TRACKED_ROOT,
) -> dict[str, object]:
    """Regenerate draft-v3 from the bounded external adjudication.

    This path deliberately stops at pre-review artifacts.  It does not call
    freeze logic and does not alter the v2 private root.
    """

    v2_items_path = PRIVATE_ITEMS
    if not v2_items_path.is_file():
        raise ValueError("v2 draft is missing; cannot apply external adjudication")
    if not review_file.is_file():
        raise ValueError(f"external source review is missing: {review_file}")
    corpus = _full_corpus(private_root, tracked_root)
    base_pool, selected, variants = build_v3_candidates(
        corpus,
        v2_items_path=v2_items_path,
        adjudication_path=review_file,
        output_root=private_root / "draft",
    )
    all_unmapped = selected + variants
    mapped = map_items_to_chunks(all_unmapped, corpus)
    split_items = assign_provisional_splits(mapped)
    write_items_jsonl(V3_ITEMS, split_items)
    unit_texts = {unit.unit_id: unit.text for unit in corpus.units}
    export_review_packet(V3_ITEMS, V3_PACKET, unit_texts)
    diagnostics = build_review_diagnostics(split_items, corpus.units)
    write_review_diagnostics(V3_DIAGNOSTICS, diagnostics)
    handoff = write_handoff_artifacts(corpus, split_items, V3_HANDOFF)
    mapping = _v3_replacement_mapping(
        read_items_jsonl(v2_items_path),
        selected,
        variants,
        review_file,
        private_root / "review" / "v2_replacement_mapping.jsonl",
    )
    balance = source_balance_audit(
        corpus,
        base_pool,
        tuple(item for item in selected if item.creation_method.value == "document_derived"),
        output_path=V3_BALANCE_PATH,
    )
    validation = validate_items(
        split_items,
        unit_texts,
        require_semantic_targets=True,
    )
    split = split_diagnostics(split_items)
    by_document = {unit.document_id: unit.provenance.source_id for unit in corpus.units}
    category_source = Counter(
        f"{item.category.value}:{by_document[document_id]}"
        for item in selected
        for document_id in item.source_document_ids
    )
    source_counts = Counter(
        by_document[document_id] for item in split_items for document_id in item.source_document_ids
    )
    category_counts = Counter(item.category.value for item in split_items)
    pool_category_source = Counter(
        f"{item.category.value}:{by_document[document_id]}"
        for item in base_pool
        for document_id in item.source_document_ids
    )
    answerable = tuple(item for item in split_items if item.answerability.value == "answerable")
    evidence_rows = [
        {
            "query_id": item.query_id,
            "evidence_groups": [group.model_dump(mode="json") for group in item.evidence_groups],
            "chunk_qrels": [qrel.model_dump(mode="json") for qrel in item.chunk_qrels],
        }
        for item in split_items
    ]
    review_rows = [
        {"query_id": item.query_id, "review": item.review.model_dump(mode="json")}
        for item in split_items
    ]
    summary: dict[str, object] = {
        "schema_version": 1,
        "status": "phase6_draft_v3_built_pending_external_source_review",
        "dataset_version": V3_VERSION,
        "corpus_hash": corpus.corpus_hash,
        "corpus_inventory_hash": corpus.corpus_hash,
        "base_candidate_count": len(base_pool),
        "evidence_qualified_answerable_pool_count": len(base_pool) - 25,
        "base_intent_count": len(selected),
        "variant_count": len(variants),
        "item_count": len(split_items),
        "category_counts": dict(sorted(category_counts.items())),
        "selected_category_source_counts": dict(sorted(category_source.items())),
        "pool_category_source_counts": dict(sorted(pool_category_source.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "validation": validation.to_sanitized_dict(),
        "split_diagnostics": split.model_dump(),
        "review": review_status(split_items),
        "benchmark_source_status": benchmark_source_status(),
        "external_source_review": {
            "path": review_file.as_posix(),
            "accepted_base_unanswerables_preserved": 25,
            "rejected_records_replaced": 215,
            "mapping": mapping,
        },
        "semantic_target": {
            "required": True,
            "all_answerable_targets_present": all(
                item.semantic_target is not None for item in answerable
            ),
            "answerable_count": len(answerable),
            "start_zero_span_count": sum(
                span.start == 0
                for item in split_items
                for group in item.evidence_groups
                for span in group.spans
            ),
            "evidence_span_count": sum(
                len(group.spans) for item in split_items for group in item.evidence_groups
            ),
        },
        "hashes": {
            "item_set": _hash_items(split_items),
            "dev_ids": _hash_payload(
                sorted(item.query_id for item in split_items if item.split is DatasetSplit.DEV)
            ),
            "holdout_ids": _hash_payload(
                sorted(item.query_id for item in split_items if item.split is DatasetSplit.HOLDOUT)
            ),
            "evidence_and_qrels": _hash_payload(evidence_rows),
            "review_state": _hash_payload(review_rows),
            "chunk_policy": next(
                (item.chunk_policy_hash for item in split_items if item.chunk_qrels),
                None,
            ),
        },
        "privacy_and_text_policy": {
            "tracked_outputs_text_free": True,
            "generated_representations_private": True,
            "human_verified_count": sum(item.human_verified for item in split_items),
        },
        "private_artifacts": {
            "corpus_snapshot": (private_root / "corpus" / "canonical_units.json").as_posix(),
            "base_candidates": (private_root / "draft" / "base_candidates.jsonl").as_posix(),
            "items": V3_ITEMS.as_posix(),
            "review_packet": V3_PACKET.as_posix(),
            "review_diagnostics": V3_DIAGNOSTICS.as_posix(),
            "replacement_mapping": mapping["path"],
            "handoff_manifest": (V3_HANDOFF / "canonical_review_manifest.json").as_posix(),
            "handoff_context": (V3_HANDOFF / "phase6_review_source_context.jsonl").as_posix(),
        },
        "handoff": handoff,
        "source_balance_audit": {
            "path": V3_BALANCE_PATH.as_posix(),
            "retrieval_scores_used": balance["retrieval_scores_used"],
        },
    }
    _write_json(V3_TRACKED_SUMMARY_PATH, summary)
    _write_json(V3_SUMMARY_PATH, summary)
    return summary


def _write_v4_context(corpus: EvaluationCorpus, items: tuple[DatasetItem, ...], path: Path) -> Path:
    documents = {document_id for item in items for document_id in item.source_document_ids}
    rows: list[dict[str, object]] = []
    for unit in corpus.units:
        if unit.document_id in documents:
            rows.append(
                {
                    "source_id": unit.provenance.source_id,
                    "source_version": unit.provenance.source_version,
                    "document_id": unit.document_id,
                    "unit_id": unit.unit_id,
                    "unit_type": unit.unit_type.value,
                    "display_text": unit.text,
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def run_build_draft_v4(
    *,
    review_file: Path,
    private_root: Path = V4_PRIVATE_ROOT,
    tracked_root: Path = TRACKED_ROOT,
) -> dict[str, object]:
    """Apply v3 external decisions as a private, pre-review v4 candidate."""

    if not V3_ITEMS.is_file() or not V3_BASE_CANDIDATES.is_file():
        raise ValueError("v3 draft is missing; cannot apply v4 adjudication")
    if not review_file.is_file():
        raise ValueError(f"external v3 adjudication is missing: {review_file}")
    corpus = _full_corpus(private_root, tracked_root)
    v3_items = read_items_jsonl(V3_ITEMS)
    pool = read_items_jsonl(V3_BASE_CANDIDATES)
    decisions = load_v3_adjudication(review_file)
    result = apply_v3_adjudication(v3_items, corpus, decisions, pool)
    unchunked = result.bases + result.variants
    mapped = map_items_to_chunks(unchunked, corpus)
    split_items = assign_provisional_splits(mapped)
    write_items_jsonl(V4_BASE_CANDIDATES, result.base_pool)
    write_items_jsonl(V4_ITEMS, split_items)
    unit_texts = {unit.unit_id: unit.text for unit in corpus.units}
    export_review_packet(V4_ITEMS, V4_PACKET, unit_texts)
    raw_diagnostics = build_review_diagnostics(split_items, corpus.units)
    write_review_diagnostics(V4_DIAGNOSTICS, raw_diagnostics)
    mapping_path = V4_MAPPING
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in result.mapping
        ),
        encoding="utf-8",
    )
    context_path = _write_v4_context(
        corpus, split_items, V4_HANDOFF / "phase6_review_source_context.jsonl"
    )
    by_document = {unit.document_id: unit.provenance.source_id for unit in corpus.units}
    base_items = tuple(item for item in split_items if item.variant_id is None)
    source_counts = Counter(
        by_document[document_id] for item in base_items for document_id in item.source_document_ids
    )
    category_source_counts = Counter(
        f"{item.category.value}:{by_document[document_id]}"
        for item in base_items
        for document_id in item.source_document_ids
    )
    all_rows = [{"query_id": item.query_id, "query_text": item.query_text} for item in split_items]
    exact_duplicates = duplicate_query_keys(all_rows)
    answerable = tuple(item for item in split_items if item.answerability.value == "answerable")
    start_zero = sum(
        span.start == 0
        for item in answerable
        for group in item.evidence_groups
        for span in group.spans
    )
    validation = validate_items(split_items, unit_texts, require_semantic_targets=True)
    split = split_diagnostics(split_items)
    decision_counts = Counter(decision.decision for decision in decisions)
    accepted_ids = {decision.query_id for decision in decisions if decision.decision == "accept"}
    accepted_preserved = sum(
        bool(row["old_query_id"] == row["new_query_id"] and row["evidence_preserved"])
        for row in result.mapping
        if row["old_query_id"] in accepted_ids
    )
    summary: dict[str, object] = {
        "schema_version": 1,
        "status": "phase6_draft_v4_built_pending_final_external_source_review",
        "dataset_version": V4_VERSION,
        "corpus_hash": corpus.corpus_hash,
        "corpus_inventory_hash": corpus.corpus_hash,
        "adjudication_counts_applied": dict(sorted(decision_counts.items())),
        "accepted_base_unanswerables_preserved": accepted_preserved,
        "corrected_base_count": decision_counts["correct"],
        "replacement_base_count": decision_counts["replace"],
        "regenerated_variant_count": decision_counts["regenerate_variant"],
        "base_candidate_count": len(result.base_pool),
        "base_intent_count": len(base_items),
        "variant_count": len(split_items) - len(base_items),
        "item_count": len(split_items),
        "category_counts": dict(
            sorted(Counter(item.category.value for item in split_items).items())
        ),
        "base_category_counts": dict(
            sorted(Counter(item.category.value for item in base_items).items())
        ),
        "source_counts": dict(
            sorted(
                Counter(
                    by_document[doc] for item in split_items for doc in item.source_document_ids
                ).items()
            )
        ),
        "base_source_counts": dict(sorted(source_counts.items())),
        "base_category_source_counts": dict(sorted(category_source_counts.items())),
        "pool_category_counts": dict(
            sorted(Counter(item.category.value for item in result.base_pool).items())
        ),
        "exact_duplicate_groups_all_240": len(exact_duplicates),
        "validation": validation.to_sanitized_dict(),
        "split_diagnostics": split.model_dump(),
        "review": review_status(split_items),
        "benchmark_source_status": benchmark_source_status(),
        "multi_evidence_source_audit": {
            **multi_opportunity_audit(corpus),
            "selected_replacement_multi_by_source": dict(
                sorted(
                    cast(
                        dict[str, int],
                        result.multi_source_audit.get("qualified_by_source", {}),
                    ).items()
                )
            ),
        },
        "replacement_rejection_reasons": list(result.replacement_reasons),
        "evidence_statistics": {
            "answerable_count": len(answerable),
            "evidence_span_count": sum(
                len(group.spans) for item in answerable for group in item.evidence_groups
            ),
            "valid_span_count": sum(
                len(group.spans) for item in answerable for group in item.evidence_groups
            ),
            "legitimate_start_zero_span_count": start_zero,
        },
        "hashes": {
            "item_set": _hash_items(split_items),
            "dev_ids": _hash_payload(
                sorted(item.query_id for item in split_items if item.split is DatasetSplit.DEV)
            ),
            "holdout_ids": _hash_payload(
                sorted(item.query_id for item in split_items if item.split is DatasetSplit.HOLDOUT)
            ),
            "evidence_and_qrels": _hash_payload(
                [
                    {
                        "query_id": item.query_id,
                        "evidence": [
                            group.model_dump(mode="json") for group in item.evidence_groups
                        ],
                        "qrels": [qrel.model_dump(mode="json") for qrel in item.chunk_qrels],
                    }
                    for item in split_items
                ]
            ),
            "review_state": _hash_payload(
                [
                    {"query_id": item.query_id, "review": item.review.model_dump(mode="json")}
                    for item in split_items
                ]
            ),
            "policy_versions": _hash_payload(
                {
                    "content_policy_hash": corpus.content_policy_hash,
                    "chunk_policy_hashes": sorted({item.chunk_policy_hash for item in split_items}),
                }
            ),
        },
        "private_artifacts": {
            "corpus_snapshot": (private_root / "corpus" / "canonical_units.json").as_posix(),
            "base_candidates": V4_BASE_CANDIDATES.as_posix(),
            "items": V4_ITEMS.as_posix(),
            "review_packet": V4_PACKET.as_posix(),
            "review_diagnostics": V4_DIAGNOSTICS.as_posix(),
            "adjudication_application": mapping_path.as_posix(),
            "source_context": context_path.as_posix(),
        },
        "review_gate": {
            "human_verified_count": sum(item.human_verified for item in split_items),
            "freeze_called": False,
            "ready_for_final_external_source_review": validation.valid and not exact_duplicates,
        },
    }
    _write_json(V4_TRACKED_SUMMARY_PATH, summary)
    _write_json(V4_SUMMARY_PATH, summary)
    return summary


def _write_review_packet_for_items(
    items: tuple[DatasetItem, ...], destination: Path, unit_texts: dict[str, str]
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(
            json.dumps(_packet_record(item, unit_texts), ensure_ascii=False, sort_keys=True) + "\n"
            for item in items
        ),
        encoding="utf-8",
    )
    return destination


def _all_query_duplicate_groups(items: tuple[DatasetItem, ...]) -> dict[str, tuple[str, ...]]:
    return duplicate_query_keys(
        [{"query_id": item.query_id, "query_text": item.query_text} for item in items]
    )


def _near_duplicate_count_all(items: tuple[DatasetItem, ...]) -> int:
    return sum(
        SequenceMatcher(None, left.query_text, right.query_text).ratio() >= 0.96
        for index, left in enumerate(items)
        for right in items[index + 1 :]
    )


def run_build_draft_v5(
    *,
    review_file: Path,
    private_root: Path = V5_PRIVATE_ROOT,
    tracked_root: Path = TRACKED_ROOT,
) -> dict[str, object]:
    """Apply the final v4 external adjudication as a private draft-v5."""

    if not V4_ITEMS.is_file() or not V4_BASE_CANDIDATES.is_file():
        raise ValueError("v4 draft is missing; cannot apply v5 adjudication")
    if not review_file.is_file():
        raise ValueError(f"external v4 adjudication is missing: {review_file}")
    corpus = _full_corpus(private_root, tracked_root)
    v4_items = read_items_jsonl(V4_ITEMS)
    pool = read_items_jsonl(V4_BASE_CANDIDATES)
    decisions = load_v4_final_adjudication(review_file)
    result = apply_v4_adjudication(v4_items, corpus, decisions, pool)
    mapped = map_items_to_chunks(result.bases + result.variants, corpus)
    split_items = assign_provisional_splits(mapped)
    write_items_jsonl(V5_BASE_CANDIDATES, result.base_pool)
    write_items_jsonl(V5_ITEMS, split_items)
    unit_texts = {unit.unit_id: unit.text for unit in corpus.units}
    _write_review_packet_for_items(split_items, V5_PACKET, unit_texts)
    changed_ids = {
        str(row["old_query_id"])
        for row in result.mapping
        if any(
            bool(row[key])
            for key in (
                "query_changed",
                "answer_changed",
                "semantic_target_changed",
                "qrels_changed",
                "evidence_extended",
                "evidence_replaced",
            )
        )
    }
    changed_items = tuple(
        item
        for item in split_items
        if any(
            str(row["new_query_id"]) == item.query_id
            for row in result.mapping
            if str(row["old_query_id"]) in changed_ids
        )
    )
    _write_review_packet_for_items(changed_items, V5_CHANGED_PACKET, unit_texts)
    diagnostics = build_review_diagnostics(split_items, corpus.units)
    write_review_diagnostics(V5_DIAGNOSTICS, diagnostics)
    V5_MAPPING.parent.mkdir(parents=True, exist_ok=True)
    V5_MAPPING.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in result.mapping
        ),
        encoding="utf-8",
    )
    context_path = _write_v4_context(
        corpus, split_items, V5_HANDOFF / "phase6_review_source_context.jsonl"
    )
    by_document = {unit.document_id: str(unit.provenance.source_id) for unit in corpus.units}
    bases = tuple(item for item in split_items if item.variant_id is None)
    source_counts = Counter(
        by_document[document_id] for item in bases for document_id in item.source_document_ids
    )
    category_source_counts = Counter(
        f"{item.category.value}:{by_document[document_id]}"
        for item in bases
        for document_id in item.source_document_ids
    )
    exact_duplicates = _all_query_duplicate_groups(split_items)
    answerable = tuple(item for item in split_items if item.answerability.value == "answerable")
    validation = validate_items(split_items, unit_texts, require_semantic_targets=True)
    split = split_diagnostics(split_items)
    decision_counts = Counter(decision.decision for decision in decisions)
    replacement_counts = Counter(
        f"{row['new_category']}:{row['replacement_source']}"
        for row in result.mapping
        if row["evidence_replaced"]
    )
    variant_languages = Counter(item.language.value for item in result.variants)
    v4_corpus_hash = None
    if Path("data/evaluation/phase6_retrieval_eval_draft_v4_summary.json").is_file():
        v4_summary = json.loads(
            Path("data/evaluation/phase6_retrieval_eval_draft_v4_summary.json").read_text(
                encoding="utf-8"
            )
        )
        v4_corpus_hash = v4_summary.get("corpus_hash")
    evidence_rows = [
        {
            "query_id": item.query_id,
            "evidence": [group.model_dump(mode="json") for group in item.evidence_groups],
            "qrels": [qrel.model_dump(mode="json") for qrel in item.chunk_qrels],
        }
        for item in split_items
    ]
    summary: dict[str, object] = {
        "schema_version": 1,
        "status": "phase6_draft_v5_built_pending_changed_record_external_source_review",
        "dataset_version": V5_VERSION,
        "corpus_hash": corpus.corpus_hash,
        "corpus_hash_unchanged_from_v4": v4_corpus_hash == corpus.corpus_hash,
        "adjudication_counts_applied": dict(sorted(decision_counts.items())),
        "accepted_base_unanswerables_preserved": sum(
            row["decision"] == "accept" and row["old_query_id"] == row["new_query_id"]
            for row in result.mapping
        ),
        "corrected_base_count": decision_counts["correct"],
        "same_unit_extension_count": sum(1 for row in result.mapping if row["evidence_extended"]),
        "replacement_base_count": decision_counts["replace"],
        "replacement_by_category_source": dict(sorted(replacement_counts.items())),
        "regenerated_variant_count": decision_counts["regenerate_variant"],
        "base_candidate_count": len(result.base_pool),
        "base_intent_count": len(bases),
        "variant_count": len(result.variants),
        "item_count": len(split_items),
        "base_category_counts": dict(
            sorted(Counter(item.category.value for item in bases).items())
        ),
        "base_source_counts": dict(sorted(source_counts.items())),
        "base_category_source_counts": dict(sorted(category_source_counts.items())),
        "variant_language_counts": dict(sorted(variant_languages.items())),
        "pool_category_counts": dict(
            sorted(Counter(item.category.value for item in result.base_pool).items())
        ),
        "exact_duplicate_groups_all_240": len(exact_duplicates),
        "near_duplicate_pairs_all_240": _near_duplicate_count_all(split_items),
        "validation": validation.to_sanitized_dict(),
        "split_diagnostics": split.model_dump(),
        "review": review_status(split_items),
        "benchmark_source_status": benchmark_source_status(),
        "multi_evidence_source_audit": dict(result.multi_audit),
        "evidence_statistics": {
            "answerable_count": len(answerable),
            "evidence_span_count": sum(
                len(group.spans) for item in answerable for group in item.evidence_groups
            ),
            "legitimate_start_zero_span_count": sum(
                span.start == 0
                for item in answerable
                for group in item.evidence_groups
                for span in group.spans
            ),
            "invalid_span_count": validation.invalid_span_count,
            "missing_chunk_mapping_count": validation.missing_chunk_mapping_count,
        },
        "hashes": {
            "item_set": _hash_items(split_items),
            "dev_ids": _hash_payload(
                sorted(item.query_id for item in split_items if item.split is DatasetSplit.DEV)
            ),
            "holdout_ids": _hash_payload(
                sorted(item.query_id for item in split_items if item.split is DatasetSplit.HOLDOUT)
            ),
            "evidence_and_qrels": _hash_payload(evidence_rows),
            "review_state": _hash_payload(
                [
                    {"query_id": item.query_id, "review": item.review.model_dump(mode="json")}
                    for item in split_items
                ]
            ),
            "policy_versions": _hash_payload(
                {
                    "content_policy_hash": corpus.content_policy_hash,
                    "chunk_policy_hashes": sorted({item.chunk_policy_hash for item in split_items}),
                }
            ),
        },
        "private_artifacts": {
            "corpus_snapshot": (private_root / "corpus" / "canonical_units.json").as_posix(),
            "base_candidates": V5_BASE_CANDIDATES.as_posix(),
            "items": V5_ITEMS.as_posix(),
            "review_packet": V5_PACKET.as_posix(),
            "changed_review_packet": V5_CHANGED_PACKET.as_posix(),
            "review_diagnostics": V5_DIAGNOSTICS.as_posix(),
            "adjudication_application": V5_MAPPING.as_posix(),
            "source_context": context_path.as_posix(),
        },
        "review_gate": {
            "human_verified_count": sum(item.human_verified for item in split_items),
            "all_records_draft": all(
                item.review.state is ReviewState.DRAFT for item in split_items
            ),
            "freeze_called": False,
            "ready_for_changed_record_external_review": validation.valid and not exact_duplicates,
        },
    }
    _write_json(V5_TRACKED_SUMMARY_PATH, summary)
    _write_json(V5_SUMMARY_PATH, summary)
    return summary


def run_build_final_candidate(
    *,
    patch_file: Path,
    private_root: Path = FINAL_PRIVATE_ROOT,
    tracked_root: Path = TRACKED_ROOT,
) -> dict[str, object]:
    """Apply the final external patch literally and rebuild only derived data."""

    if not V5_ITEMS.is_file():
        raise ValueError("v5 draft is missing; cannot apply final literal patch")
    if not patch_file.is_file():
        raise ValueError(f"final literal patch is missing: {patch_file}")
    corpus = _full_corpus(V5_PRIVATE_ROOT)
    corpus_units = {unit.unit_id: unit for unit in corpus.units}
    result = apply_literal_patch(
        V5_ITEMS,
        patch_file,
        candidate_pool_path=V5_BASE_CANDIDATES,
        corpus_units=corpus_units,
    )
    unit_texts = {unit.unit_id: unit.text for unit in corpus.units}
    validation = validate_final_candidate(
        result.items,
        unit_texts,
        corpus_hash=corpus.corpus_hash,
        expected_corpus_hash=EXPECTED_CORPUS_HASH,
        conformance=result.summary,
    )
    if not validation["valid"]:
        raise ValueError("final literal candidate validation failed")
    private_root.mkdir(parents=True, exist_ok=True)
    write_items_jsonl(private_root / "draft" / "selected_and_variants.jsonl", result.items)
    _write_review_packet_for_items(
        result.items, private_root / "review" / "review_packet.jsonl", unit_texts
    )
    diagnostics = build_review_diagnostics(result.items, corpus.units)
    write_review_diagnostics(private_root / "review" / "review_diagnostics.jsonl", diagnostics)
    _write_json(
        private_root / "review" / "literal_patch_conformance.json", result.summary.to_dict()
    )
    (private_root / "review" / "v5_literal_patch_application.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in result.mapping
        ),
        encoding="utf-8",
    )
    context_path = _write_v4_context(
        corpus, result.items, private_root / "handoff" / "phase6_review_source_context.jsonl"
    )
    by_document = {unit.document_id: str(unit.provenance.source_id) for unit in corpus.units}
    bases = tuple(item for item in result.items if item.variant_id is None)
    category_counts = Counter(item.category.value for item in bases)
    source_counts = Counter(
        by_document[document_id] for item in bases for document_id in item.source_document_ids
    )
    category_source_counts = Counter(
        f"{item.category.value}:{by_document[document_id]}"
        for item in bases
        for document_id in item.source_document_ids
    )
    summary: dict[str, object] = {
        "schema_version": 1,
        "status": "phase6_final_candidate_v1_pending_formal_human_review",
        "dataset_version": FINAL_VERSION,
        "content_review_status": "independent_ai_source_review_closed",
        "corpus_hash": corpus.corpus_hash,
        "corpus_hash_unchanged": corpus.corpus_hash == EXPECTED_CORPUS_HASH,
        "patch_sha256": result.summary.patch_sha256,
        "patch_conformance": result.summary.to_dict(),
        "validation": validation,
        "base_intent_count": len(bases),
        "variant_count": len(result.items) - len(bases),
        "item_count": len(result.items),
        "base_category_counts": dict(sorted(category_counts.items())),
        "base_source_counts": dict(sorted(source_counts.items())),
        "base_category_source_counts": dict(sorted(category_source_counts.items())),
        "review": review_status(result.items),
        "split_diagnostics": split_diagnostics(result.items).model_dump(),
        "private_artifacts": {
            "items": (private_root / "draft" / "selected_and_variants.jsonl").as_posix(),
            "review_packet": (private_root / "review" / "review_packet.jsonl").as_posix(),
            "review_diagnostics": (private_root / "review" / "review_diagnostics.jsonl").as_posix(),
            "conformance": (private_root / "review" / "literal_patch_conformance.json").as_posix(),
            "mapping": (private_root / "review" / "v5_literal_patch_application.jsonl").as_posix(),
            "source_context": context_path.as_posix(),
        },
        "freeze_gate": {
            "human_verified_count": sum(item.human_verified for item in result.items),
            "all_records_draft": all(
                item.review.state is ReviewState.DRAFT for item in result.items
            ),
            "freeze_called": False,
            "formal_human_review_required": True,
        },
    }
    _write_json(tracked_root / FINAL_TRACKED_SUMMARY_PATH.name, summary)
    _write_json(FINAL_SUMMARY_PATH, summary)
    return summary


def _active_draft_paths() -> tuple[Path, Path, Path, Path, bool]:
    if FINAL_ITEMS.is_file():
        return FINAL_ITEMS, FINAL_PACKET, FINAL_PRIVATE_ROOT, FINAL_HANDOFF, True
    if V5_ITEMS.is_file():
        return V5_ITEMS, V5_PACKET, V5_PRIVATE_ROOT, V5_HANDOFF, True
    if V4_ITEMS.is_file():
        return V4_ITEMS, V4_PACKET, V4_PRIVATE_ROOT, V4_HANDOFF, True
    if V3_ITEMS.is_file():
        return V3_ITEMS, V3_PACKET, V3_PRIVATE_ROOT, V3_HANDOFF, True
    return PRIVATE_ITEMS, PRIVATE_PACKET, PRIVATE_ROOT, PRIVATE_HANDOFF, False


def export_review() -> dict[str, object]:
    items_path, packet_path, private_root, _handoff, _is_v3 = _active_draft_paths()
    if not items_path.is_file():
        raise ValueError("draft is missing; run evaluation build-draft first")
    corpus = _full_corpus(private_root)
    unit_texts = {unit.unit_id: unit.text for unit in corpus.units}
    path = export_review_packet(items_path, packet_path, unit_texts)
    return {
        "status": "review_exported",
        "path": path.as_posix(),
        "item_count": len(read_items_jsonl(items_path)),
    }


def import_review(packet: Path) -> dict[str, object]:
    items_path, _packet_path, _private_root, _handoff, _is_v3 = _active_draft_paths()
    if not items_path.is_file():
        raise ValueError("draft is missing; run evaluation build-draft first")
    items = import_reviews(items_path, packet)
    write_items_jsonl(items_path, items)
    return {
        "status": "review_imported",
        "review": review_status(items),
        "item_hash": _hash_items(items),
    }


def validate_evaluation() -> dict[str, object]:
    items_path, _packet_path, private_root, _handoff, is_v3 = _active_draft_paths()
    if not items_path.is_file():
        raise ValueError("draft is missing; run evaluation build-draft first")
    if items_path == FINAL_ITEMS:
        corpus = _full_corpus(FINAL_PRIVATE_ROOT)
        items = read_items_jsonl(items_path)
        raw_conformance = json.loads(FINAL_CONFORMANCE.read_text(encoding="utf-8"))
        conformance = LiteralPatchSummary(
            patch_sha256=str(raw_conformance["patch_sha256"]),
            applied_counts=dict(raw_conformance["applied_counts"]),
            mismatches=int(raw_conformance["mismatches"]),
            evidence_preservation_mismatches=int(
                raw_conformance["evidence_preservation_mismatches"]
            ),
            replacement_candidate_mismatches=int(
                raw_conformance["replacement_candidate_mismatches"]
            ),
            variant_parent_mismatches=int(raw_conformance["variant_parent_mismatches"]),
            near_duplicate_pairs=tuple(raw_conformance.get("near_duplicate_pairs", ())),
        )
        result = validate_final_candidate(
            items,
            {unit.unit_id: unit.text for unit in corpus.units},
            corpus_hash=corpus.corpus_hash,
            expected_corpus_hash=EXPECTED_CORPUS_HASH,
            conformance=conformance,
        )
        return {
            "valid": result["valid"],
            **result,
            "review": review_status(items),
        }
    corpus = _full_corpus(private_root)
    items = read_items_jsonl(items_path)
    result = validate_items(
        items,
        {unit.unit_id: unit.text for unit in corpus.units},
        require_semantic_targets=is_v3,
    )
    return {
        "valid": result.valid,
        **result.to_sanitized_dict(),
        "review": review_status(items),
        "split_diagnostics": split_diagnostics(items).model_dump(),
    }


def run_source_balance_audit() -> dict[str, object]:
    if not PRIVATE_ITEMS.is_file() or not PRIVATE_BASE_CANDIDATES.is_file():
        raise ValueError("draft is missing; run evaluation build-draft first")
    corpus = _full_corpus(PRIVATE_ROOT)
    base_candidates = read_items_jsonl(PRIVATE_BASE_CANDIDATES)
    selected = tuple(
        item
        for item in read_items_jsonl(PRIVATE_ITEMS)
        if item.creation_method.value == "document_derived"
    )
    return source_balance_audit(
        corpus,
        base_candidates,
        selected,
        output_path=SOURCE_BALANCE_AUDIT_PATH,
    )


def freeze_evaluation() -> dict[str, object]:
    items_path, _packet_path, private_root, _handoff, is_v3 = _active_draft_paths()
    if not items_path.is_file():
        return {"status": "blocked_missing_draft"}
    items = read_items_jsonl(items_path)
    status = review_status(items)
    diagnostics = split_diagnostics(items)
    corpus = freeze_evaluation_corpus(
        load_evaluation_units(Path("data/interim/canonical")),
        canonical_root=Path("data/interim/canonical"),
    )
    validation = validate_items(
        items,
        {unit.unit_id: unit.text for unit in corpus.units},
        require_semantic_targets=is_v3,
    )
    required = len(items)
    holdout = sum(item.split is DatasetSplit.HOLDOUT for item in items)
    dev = sum(item.split is DatasetSplit.DEV for item in items)
    blockers: list[str] = []
    if status["primary_reviewed"] != required:
        blockers.append("100% primary manual review required")
    if status["human_verified"] != required:
        blockers.append("all final records require explicit human verification")
    if any(
        item.review.state not in {ReviewState.ADJUDICATED, ReviewState.FROZEN} for item in items
    ):
        blockers.append("all final records require adjudicated review state")
    holdout_items = tuple(item for item in items if item.split is DatasetSplit.HOLDOUT)
    dev_items = tuple(item for item in items if item.split is DatasetSplit.DEV)
    recheck_required = max(1, (len(dev_items) + 3) // 4)
    rechecked_dev = sum(item.review.secondary_reviewer is not None for item in dev_items)
    rechecked_holdout = sum(item.review.secondary_reviewer is not None for item in holdout_items)
    hard_or_special = tuple(
        item
        for item in items
        if item.difficulty.value == "hard"
        or item.answerability.value == "unanswerable"
        or item.category.value == "multi_evidence"
    )
    if rechecked_holdout != len(holdout_items):
        blockers.append("100% independent holdout recheck required")
    if rechecked_dev < recheck_required:
        blockers.append("at least 25% independent dev recheck required")
    if any(item.review.secondary_reviewer is None for item in hard_or_special):
        blockers.append("all unanswerable, hard, and multi-evidence items require double review")
    if status["unresolved_disagreements"]:
        blockers.append("unresolved review disagreements remain")
    if not validation.valid:
        blockers.append("non-retrieval validation gates are not clear")
    if diagnostics.cross_split_document_count or diagnostics.cross_split_intent_count:
        blockers.append("cross-split leakage detected")
    if dev < 1 or holdout < 1:
        blockers.append("dev and holdout splits are required")
    if blockers:
        return {
            "status": "blocked_pending_human_review",
            "blockers": blockers,
            "review": status,
            "validation": validation.to_sanitized_dict(),
            "dev_count": dev,
            "holdout_count": holdout,
        }
    manifest = freeze_items(items, private_root=private_root, corpus_hash=corpus.corpus_hash)
    return {"status": "frozen", **manifest}


def _copy_private_release(source_root: Path, release_root: Path) -> None:
    if not source_root.is_dir():
        raise ValueError(f"source private release is missing: {source_root}")
    if not release_root.exists():
        shutil.copytree(source_root, release_root)
        return
    source_files = {
        path.relative_to(source_root): path for path in source_root.rglob("*") if path.is_file()
    }
    release_files = {
        path.relative_to(release_root): path for path in release_root.rglob("*") if path.is_file()
    }
    if set(source_files) != set(release_files):
        raise ValueError("immutable AI-reviewed release would be mutated")
    for relative, source_path in source_files.items():
        if source_path.read_bytes() != release_files[relative].read_bytes():
            raise ValueError("immutable AI-reviewed release would be mutated")


def _ai_reviewed_hashes(items: tuple[DatasetItem, ...]) -> dict[str, str]:
    evidence_qrels = [
        {
            "query_id": item.query_id,
            "evidence": [group.model_dump(mode="json") for group in item.evidence_groups],
            "qrels": [qrel.model_dump(mode="json") for qrel in item.chunk_qrels],
        }
        for item in items
    ]
    review_state = [
        {"query_id": item.query_id, "review": item.review.model_dump(mode="json")} for item in items
    ]
    return {
        "item_set": _hash_items(items),
        "dev_ids": _hash_payload(
            sorted(item.query_id for item in items if item.split is DatasetSplit.DEV)
        ),
        "holdout_ids": _hash_payload(
            sorted(item.query_id for item in items if item.split is DatasetSplit.HOLDOUT)
        ),
        "evidence_qrels": _hash_payload(evidence_qrels),
        "review_state": _hash_payload(review_state),
        "chunk_policy": get_chunk_policy("legal-structure-v1").policy_hash,
    }


def freeze_ai_reviewed_release(
    *,
    source_root: Path = FINAL_PRIVATE_ROOT,
    release_root: Path = AI_REVIEWED_PRIVATE_ROOT,
    manifest_path: Path = AI_REVIEWED_MANIFEST_PATH,
    report_path: Path = AI_REVIEWED_REPORT_PATH,
) -> dict[str, object]:
    """Materialize the externally AI-reviewed release without human attestations."""

    source_items_path = source_root / "draft" / "selected_and_variants.jsonl"
    source_conformance_path = source_root / "review" / "literal_patch_conformance.json"
    if not source_items_path.is_file() or not source_conformance_path.is_file():
        raise ValueError("final-candidate-v1 private artifacts are missing")
    items = read_items_jsonl(source_items_path)
    if len(items) != 240 or sum(item.variant_id is None for item in items) != 200:
        raise ValueError("AI-reviewed release requires exactly 200 bases and 40 variants")
    if any(item.human_verified for item in items):
        raise ValueError("AI-reviewed release cannot contain human-verified records")
    if any(item.review.review_provenance != "independent_ai_source_review" for item in items):
        raise ValueError("AI-reviewed release provenance is not uniform")

    corpus = freeze_evaluation_corpus(
        load_evaluation_units(Path("data/interim/canonical")),
        canonical_root=Path("data/interim/canonical"),
    )
    raw_conformance = json.loads(source_conformance_path.read_text(encoding="utf-8"))
    conformance = LiteralPatchSummary(
        patch_sha256=str(raw_conformance["patch_sha256"]),
        applied_counts=dict(raw_conformance["applied_counts"]),
        mismatches=int(raw_conformance["mismatches"]),
        evidence_preservation_mismatches=int(raw_conformance["evidence_preservation_mismatches"]),
        replacement_candidate_mismatches=int(raw_conformance["replacement_candidate_mismatches"]),
        variant_parent_mismatches=int(raw_conformance["variant_parent_mismatches"]),
        near_duplicate_pairs=tuple(raw_conformance.get("near_duplicate_pairs", ())),
    )
    validation = validate_final_candidate(
        items,
        {unit.unit_id: unit.text for unit in corpus.units},
        corpus_hash=corpus.corpus_hash,
        expected_corpus_hash=EXPECTED_CORPUS_HASH,
        conformance=conformance,
    )
    if not validation["valid"] or corpus.corpus_hash != EXPECTED_CORPUS_HASH:
        raise ValueError("final-candidate-v1 validation or corpus hash failed")

    hashes = _ai_reviewed_hashes(items)
    by_document = {unit.document_id: unit.provenance.source_id for unit in corpus.units}
    bases = tuple(item for item in items if item.variant_id is None)
    source_counts = Counter(
        by_document[document_id] for item in bases for document_id in item.source_document_ids
    )
    category_source_counts = Counter(
        f"{item.category.value}:{by_document[document_id]}"
        for item in bases
        for document_id in item.source_document_ids
    )
    language_register_counts = Counter(
        f"{item.language.value}:{item.register.value}" for item in items
    )
    limitation = (
        "This release underwent source-grounded generation, deterministic validation, "
        "external AI source review and adjudication, but not independent human legal-expert "
        "annotation. It is not a human-gold, expert-reviewed, or independently human-annotated "
        "benchmark."
    )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "frozen_ai_reviewed_engineering_release",
        "dataset_version": AI_REVIEWED_VERSION,
        "review_provenance": "independent_ai_source_review",
        "human_verified": False,
        "formal_human_review_required": False,
        "human_annotation_status": "not_independently_human_annotated",
        "limitation_statement": limitation,
        "corpus_hash": corpus.corpus_hash,
        "base_intent_count": len(bases),
        "variant_count": len(items) - len(bases),
        "item_count": len(items),
        "base_category_counts": dict(
            sorted(Counter(item.category.value for item in bases).items())
        ),
        "base_source_counts": dict(sorted(source_counts.items())),
        "base_category_source_counts": dict(sorted(category_source_counts.items())),
        "language_register_counts": dict(sorted(language_register_counts.items())),
        "split_counts": {
            "dev": sum(item.split is DatasetSplit.DEV for item in items),
            "holdout": sum(item.split is DatasetSplit.HOLDOUT for item in items),
            "smoke": sum(item.smoke for item in items),
        },
        "hashes": hashes,
        "policy": {
            "content_policy_version": "phase5-source-content-policy-v1",
            "content_policy_hash": corpus.content_policy_hash,
            "chunk_policy_id": "legal-structure-v1",
            "chunk_policy_hash": hashes["chunk_policy"],
        },
        "validation": validation,
        "literal_patch_conformance": conformance.to_dict(),
        "human_review_workflow": {
            "required_for_current_engineering_release": False,
            "optional_future_upgrade": "publication-grade human-reviewed-v1",
            "human_verified_count": 0,
        },
        "private_release_root": release_root.as_posix(),
    }
    report: dict[str, object] = {
        "schema_version": 1,
        "dataset_version": AI_REVIEWED_VERSION,
        "release_classification": "externally_ai_reviewed",
        "review_provenance": "independent_ai_source_review",
        "human_verified": False,
        "limitation_statement": limitation,
        "counts": {
            "base_intents": len(bases),
            "variants": len(items) - len(bases),
            "total_records": len(items),
        },
        "distributions": {
            "base_category": manifest["base_category_counts"],
            "base_source": manifest["base_source_counts"],
            "base_category_source": manifest["base_category_source_counts"],
            "language_register": manifest["language_register_counts"],
        },
        "corpus_hash": corpus.corpus_hash,
        "item_set_hash": hashes["item_set"],
        "dev_ids_hash": hashes["dev_ids"],
        "holdout_ids_hash": hashes["holdout_ids"],
        "evidence_qrel_hash": hashes["evidence_qrels"],
        "review_state_hash": hashes["review_state"],
        "chunk_policy_hash": hashes["chunk_policy"],
        "validation": validation,
        "literal_patch_conformance": conformance.to_dict(),
        "checks": {
            "private_records_copied_byte_identically": True,
            "retrieval_rankings_used": False,
            "semantic_generation_performed": False,
            "human_review_performed": False,
        },
        "human_review_workflow": manifest["human_review_workflow"],
    }
    _copy_private_release(source_root, release_root)
    _write_json(manifest_path, manifest)
    _write_json(report_path, report)
    return manifest


def evaluation_stats() -> dict[str, object]:
    if AI_REVIEWED_MANIFEST_PATH.is_file():
        manifest = json.loads(AI_REVIEWED_MANIFEST_PATH.read_text(encoding="utf-8"))
        return {
            "status": "frozen_ai_reviewed_engineering_release",
            "dataset_version": AI_REVIEWED_VERSION,
            "item_count": manifest["item_count"],
            "review_provenance": manifest["review_provenance"],
            "human_verified": False,
            "formal_human_review_required": False,
            "hashes": manifest["hashes"],
            "splits": manifest["split_counts"],
        }
    items_path, _packet_path, _private_root, _handoff, _is_v3 = _active_draft_paths()
    if not items_path.is_file():
        return {"status": "missing_draft"}
    items = read_items_jsonl(items_path)
    return {
        "status": "draft_pending_review",
        "item_count": len(items),
        "review": review_status(items),
        "splits": split_diagnostics(items).model_dump(),
        "item_set_hash": _hash_items(items),
    }
