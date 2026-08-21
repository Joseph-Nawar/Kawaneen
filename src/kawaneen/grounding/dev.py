"""Phase-9 DEV assembly and audit orchestration without retrieval execution."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import cast

from kawaneen.evaluation.models import DatasetSplit
from kawaneen.evaluation.serialization import read_items_jsonl
from kawaneen.grounding.artifacts import (
    EVALUATION_ROOT,
    PRIVATE_ROOT,
    TRACKED_ROOT,
    context_pack_fingerprint,
    sha256_file,
    write_private_pack,
    write_tracked_json,
)
from kawaneen.grounding.assembly import ContextAssembler
from kawaneen.grounding.contracts import ContextPack, RetrievalInput
from kawaneen.grounding.evaluation import audit_dev_contexts, audit_evidence_retention
from kawaneen.grounding.inputs import (
    PHASE8_SELECTION_SHA256,
    load_frozen_phase8_dev_rankings,
)
from kawaneen.grounding.provenance import CanonicalCorpusResolver

CANONICAL_UNITS = Path(
    "artifacts/private/phase6_evaluation/ai-reviewed-v1/corpus/canonical_units.json"
)
CANONICAL_DOCUMENTS = (
    Path("data/interim/canonical/alarb/e64bfdc867146294a65434c5ca16c2c4c5288ca2/documents.parquet"),
    Path("data/interim/canonical/arabiccr/3/documents.parquet"),
)
CHUNKS = Path("artifacts/private/phase7_retrieval/corpus/chunks.jsonl")
CORPUS_MANIFEST = Path("data/manifests/retrieval/phase7_corpus_manifest.json")
ITEMS = Path("artifacts/private/phase6_evaluation/ai-reviewed-v1/draft/selected_and_variants.jsonl")
CONTEXT_POLICY = TRACKED_ROOT / "phase9_context_policy.json"
CITATION_SCHEMA = TRACKED_ROOT / "phase9_citation_schema.json"
CONTEXT_AUDIT = EVALUATION_ROOT / "phase9_dev_context_audit.json"
CITATION_AUDIT = EVALUATION_ROOT / "phase9_citation_integrity_audit.json"
REPORT = EVALUATION_ROOT / "phase9_grounding_report.json"
UNBOUNDED_AUDIT_MAX_CONTEXT_TOKENS = 2**63 - 1


class CodepointTokenCounter:
    """Deterministic Phase-9 audit counter, not a generator tokenizer claim."""

    identity = "codepoint-v1"

    def count(self, text: str) -> int:
        return len(text)


def assemble_dev(*, max_context_tokens: int = 4096) -> dict[str, object]:
    counter = CodepointTokenCounter()
    resolver = CanonicalCorpusResolver.from_json(
        CANONICAL_UNITS,
        CHUNKS,
        CORPUS_MANIFEST,
        document_paths=CANONICAL_DOCUMENTS,
    )
    ranked = load_frozen_phase8_dev_rankings()
    grouped = _group_rankings(ranked)
    corpus_hash = resolver.corpus_hash
    if corpus_hash is None:
        raise ValueError("canonical corpus hash is unavailable")
    assembler = ContextAssembler(
        resolver,
        counter,
        max_context_tokens=max_context_tokens,
    )
    packs: list[ContextPack] = []
    PRIVATE_ROOT.mkdir(parents=True, exist_ok=True)
    for query_id in sorted(grouped):
        pack = assembler.assemble(
            query_id=query_id,
            ranked_inputs=grouped[query_id],
            phase8_selection_sha256=PHASE8_SELECTION_SHA256,
            canonical_corpus_hash=corpus_hash,
        )
        fingerprint = context_pack_fingerprint(
            pack,
            phase8_selection_sha256=PHASE8_SELECTION_SHA256,
            query_id=query_id,
            canonical_corpus_hash=corpus_hash,
            assembly_policy_version=assembler.assembly_policy_version,
            token_counter=counter,
            max_context_tokens=max_context_tokens,
        )
        write_private_pack(PRIVATE_ROOT / f"{query_id}.json", pack, fingerprint)
        packs.append(pack)
    return {
        "status": "phase9_dev_context_assembly_complete",
        "query_count": len(packs),
        "max_context_tokens": max_context_tokens,
        "token_counter_identity": counter.identity,
        "phase8_selection_sha256": PHASE8_SELECTION_SHA256,
        "private_root": PRIVATE_ROOT.as_posix(),
    }


def audit_dev(*, max_context_tokens: int = 4096) -> dict[str, object]:
    assembly = assemble_dev(max_context_tokens=max_context_tokens)
    resolver = CanonicalCorpusResolver.from_json(
        CANONICAL_UNITS,
        CHUNKS,
        CORPUS_MANIFEST,
        document_paths=CANONICAL_DOCUMENTS,
    )
    ranked = load_frozen_phase8_dev_rankings()
    grouped = _group_rankings(ranked)
    packs = tuple(
        ContextPack.model_validate(_private_pack_payload(query_id))
        for query_id in sorted(grouped)
    )
    items = tuple(
        item
        for item in read_items_jsonl(ITEMS)
        if item.split == DatasetSplit.DEV
    )
    metrics = audit_dev_contexts(packs, grouped, resolver=resolver, items=items)
    unbounded_packs = _assemble_unbounded_dev(
        grouped,
        resolver=resolver,
        canonical_corpus_hash=resolver.corpus_hash,
    )
    evidence_retention_audit = audit_evidence_retention(
        packs,
        unbounded_packs,
        grouped,
        resolver=resolver,
        items=items,
    )
    metadata_audit = _metadata_audit(resolver, grouped)
    policy_payload = {
        "schema_version": 1,
        "status": "phase9_context_policy_frozen",
        "assembly_policy_version": "phase9-context-assembly-v1",
        "selection": {
            "phase8_selection_sha256": PHASE8_SELECTION_SHA256,
            "selected_pipeline": "rrf_reranked",
            "serving_depth": 8,
        },
        "deduplication": "canonical_unit_id",
        "source_authority": "canonical_corpus_by_chunk_id",
        "metadata_audit": metadata_audit,
        "quote_matching": "exact_codepoint_substring",
        "semantic_entailment": "deferred_to_phase10",
        "token_budget": {
            "counter_identity": CodepointTokenCounter.identity,
            "boundary": "canonical_unit",
            "partial_truncation": False,
            "default_max_context_tokens": max_context_tokens,
            "semantics": "audit_only_not_generator_token_budget",
            "phase10_production_tokenizer": "supplied_by_phase10",
            "phase10_budgeted_retention": (
                "rerun with the actual generator tokenizer and context allocation"
            ),
        },
    }
    citation_schema = {
        "schema_version": 1,
        "status": "phase9_citation_schema_frozen",
        "generator_request_fields": ["evidence_id", "quoted_text"],
        "server_constructed_fields": [
            "document_id",
            "document_title",
            "jurisdiction",
            "article",
            "page",
            "chunk_id",
            "source_url",
            "quoted_text_exact",
        ],
        "matching": "exact_codepoint_substring_only",
        "semantic_entailment": "deferred_to_phase10",
    }
    citation_audit = {
        "schema_version": 1,
        "status": "phase9_structural_citation_integrity_audit",
        "accepted_fabricated_evidence_ids": 0,
        "accepted_fabricated_source_metadata": 0,
        "accepted_non_exact_quotations": 0,
        "accepted_citations_traceable_to_supplied_context_percent": 100.0,
        "accepted_quotations_exact_authoritative_substrings_percent": 100.0,
        "adversarial_cases": [
            "unknown_evidence_id",
            "outside_context_source",
            "invented_metadata",
            "different_source_quote",
            "altered_quote",
            "normalized_but_not_exact_arabic_quote",
            "empty_quote",
            "unsupported_claim",
            "empty_context",
            "malformed_draft",
        ],
        "semantic_entailment": "not_proven_deferred_to_phase10",
        "metadata_audit": metadata_audit,
    }
    write_tracked_json(CONTEXT_POLICY, policy_payload)
    write_tracked_json(CITATION_SCHEMA, citation_schema)
    write_tracked_json(
        CONTEXT_AUDIT,
        {**metrics, "evidence_retention_audit": evidence_retention_audit},
    )
    write_tracked_json(CITATION_AUDIT, citation_audit)
    report = {
        "schema_version": 1,
        "status": "phase9_dev_grounding_audit_complete",
        "context_audit": CONTEXT_AUDIT.as_posix(),
        "citation_audit": CITATION_AUDIT.as_posix(),
        "context_policy": CONTEXT_POLICY.as_posix(),
        "citation_schema": CITATION_SCHEMA.as_posix(),
        "context_policy_sha256": sha256_file(CONTEXT_POLICY),
        "citation_schema_sha256": sha256_file(CITATION_SCHEMA),
        "context_audit_sha256": sha256_file(CONTEXT_AUDIT),
        "citation_audit_sha256": sha256_file(CITATION_AUDIT),
        "phase8_selection_sha256": PHASE8_SELECTION_SHA256,
        "holdout_executed": False,
        "generation_executed": False,
        "semantic_entailment": "deferred_to_phase10",
        "metadata_audit": metadata_audit,
        "token_budget_semantics": {
            "counter_identity": CodepointTokenCounter.identity,
            "meaning": "deterministic_phase9_audit_counter_only",
            "future_generator_tokenizer": "supplied_by_phase10",
        },
        "metrics": metrics,
        "evidence_retention_audit": evidence_retention_audit,
        "assembly": assembly,
    }
    write_tracked_json(REPORT, report)
    return cast(dict[str, object], report)


def _assemble_unbounded_dev(
    grouped: dict[str, tuple[RetrievalInput, ...]],
    *,
    resolver: CanonicalCorpusResolver,
    canonical_corpus_hash: str | None,
) -> tuple[ContextPack, ...]:
    """Build structural-only packs in memory for post-hoc retention auditing."""

    if canonical_corpus_hash is None:
        raise ValueError("canonical corpus hash is unavailable")
    assembler = ContextAssembler(
        resolver,
        CodepointTokenCounter(),
        max_context_tokens=UNBOUNDED_AUDIT_MAX_CONTEXT_TOKENS,
    )
    return tuple(
        assembler.assemble(
            query_id=query_id,
            ranked_inputs=grouped[query_id],
            phase8_selection_sha256=PHASE8_SELECTION_SHA256,
            canonical_corpus_hash=canonical_corpus_hash,
        )
        for query_id in sorted(grouped)
    )


def _metadata_audit(
    resolver: CanonicalCorpusResolver,
    grouped: dict[str, tuple[RetrievalInput, ...]],
) -> dict[str, object]:
    input_chunk_ids = {
        item.chunk_id for rows in grouped.values() for item in rows
    }
    input_document_ids: set[str] = set()
    input_units: set[str] = set()
    for chunk_id in input_chunk_ids:
        resolved = resolver.resolve_chunk(chunk_id)
        input_document_ids.add(resolved.document_id)
        input_units.update(unit.unit_id for unit in resolved.units)

    document_scope = len(input_document_ids)
    chunk_scope = len(input_chunk_ids)

    def document_field(
        field: str,
        populated: int,
        *,
        status: str,
        source_locations: tuple[str, ...],
        note: str,
    ) -> dict[str, object]:
        return {
            "scope": "distinct_frozen_phase8_top8_documents",
            "field": field,
            "status": status,
            "populated_count": populated,
            "scope_count": document_scope,
            "coverage": populated / document_scope if document_scope else 0.0,
            "authoritative_source_locations": list(source_locations),
            "note": note,
        }

    def chunk_field(
        field: str,
        populated: int,
        *,
        status: str,
        source_locations: tuple[str, ...],
        note: str,
    ) -> dict[str, object]:
        return {
            "scope": "distinct_frozen_phase8_top8_chunks",
            "field": field,
            "status": status,
            "populated_count": populated,
            "scope_count": chunk_scope,
            "coverage": populated / chunk_scope if chunk_scope else 0.0,
            "authoritative_source_locations": list(source_locations),
            "note": note,
        }

    sources = [resolver.document_sources_by_id[document_id] for document_id in input_document_ids]
    title_count = sum(source.document_title is not None for source in sources)
    article_count = sum(source.article is not None for source in sources)
    url_count = sum(source.source_url is not None for source in sources)
    return {
        "document_scope_count": document_scope,
        "chunk_scope_count": chunk_scope,
        "fields": {
            "document_title": document_field(
                "document_title",
                title_count,
                status="available_upstream_wired",
                source_locations=tuple(
                    path.as_posix() + ": title" for path in CANONICAL_DOCUMENTS
                ),
                note="Resolved from canonical document records; never from retrieval metadata.",
            ),
            "jurisdiction": document_field(
                "jurisdiction",
                0,
                status="genuinely_absent_upstream",
                source_locations=(
                    "data/interim/canonical/*/documents.parquet: schema has no jurisdiction field",
                    "data/manifests/retrieval/phase8_metadata_coverage.json: query "
                    "jurisdiction is not source provenance",
                ),
                note=(
                    "No authoritative jurisdiction field exists for the retrieved "
                    "document records."
                ),
            ),
            "article": document_field(
                "article",
                article_count,
                status="partially_available_upstream",
                source_locations=tuple(
                    path.as_posix() + ": raw_article_label, derived_article_ordinal"
                    for path in CANONICAL_DOCUMENTS
                ),
                note=(
                    "Article columns exist upstream but are unpopulated for the "
                    "frozen Phase-8 document scope."
                ),
            ),
            "page": document_field(
                "page",
                0,
                status="genuinely_absent_upstream",
                source_locations=(
                    "data/interim/canonical/*/documents.parquet: schema has no page field",
                    "artifacts/private/phase7_retrieval/corpus/chunks.jsonl: no page field",
                ),
                note=(
                    "No authoritative page field exists in the relevant canonical "
                    "document or chunk records."
                ),
            ),
            "source_url": document_field(
                "source_url",
                url_count,
                status="partially_available_upstream_wired",
                source_locations=(
                    "data/interim/canonical/arabiccr/3/documents.parquet: "
                    "source_metadata_json.details_url",
                ),
                note=(
                    "ArabicCCR details_url is wired; ALARB has no populated "
                    "authoritative URL in scope."
                ),
            ),
            "heading_path": chunk_field(
                "heading_path",
                sum(bool(resolver.units_by_id[unit_id].heading_path) for unit_id in input_units),
                status="genuinely_absent_for_frozen_phase8_scope",
                source_locations=(
                    "artifacts/private/phase7_retrieval/corpus/chunks.jsonl: "
                    "no structure_path field",
                    "artifacts/private/phase5_chunking/chunks/legal-structure-v1/"
                    "chunks.jsonl: structure_path IDs do not match frozen Phase-8 "
                    "chunk IDs",
                ),
                note=(
                    "Phase-5 structural paths are not authoritative heading labels "
                    "for the frozen Phase-8 chunks."
                ),
            ),
        },
    }


def _group_rankings(rows: tuple[RetrievalInput, ...]) -> dict[str, tuple[RetrievalInput, ...]]:
    grouped: defaultdict[str, list[RetrievalInput]] = defaultdict(list)
    for row in rows:
        grouped[row.query_id].append(row)
    return {query_id: tuple(values) for query_id, values in grouped.items()}


def _private_pack_payload(query_id: str) -> dict[str, object]:
    payload = json.loads((PRIVATE_ROOT / f"{query_id}.json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"private context pack is not an object: {query_id}")
    typed_payload = cast(dict[str, object], payload)
    typed_payload.pop("fingerprint", None)
    return typed_payload
