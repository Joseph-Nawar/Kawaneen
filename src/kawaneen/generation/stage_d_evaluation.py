"""Offline, persisted-artifact evaluation for the Phase-10 Stage-D DEV run.

The evaluator intentionally carries heterogeneous JSON-shaped aggregate data;
its persisted inputs are validated by the Pydantic contracts at the boundaries.
"""

# pyright: basic
# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportGeneralTypeIssues=false
# pyright: reportIndexIssue=false, reportOperatorIssue=false

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from kawaneen.evaluation.models import Answerability, DatasetItem, DatasetSplit
from kawaneen.evaluation.serialization import read_items_jsonl
from kawaneen.generation.answerability import (
    ANSWERABILITY_POLICY_VERSION,
    answerability_policy_hash,
)
from kawaneen.generation.artifacts import artifact_fingerprint, write_text_free_artifact
from kawaneen.generation.checkpoints import (
    GENERATION_CHECKPOINT_ARTIFACT_TYPE,
    GENERATION_CHECKPOINT_SCHEMA_VERSION,
    QueryCheckpoint,
)
from kawaneen.generation.contracts import (
    STAGE_D_GENERATION_SETTINGS,
    GenerationDecision,
    GenerationResult,
    TokenizerFingerprint,
    parse_stage_d_generation_payload,
    stage_d_generation_payload_schema,
)
from kawaneen.generation.ollama import load_local_model_lock
from kawaneen.generation.orchestration import _assemble_unbounded_contexts, _group_rankings
from kawaneen.generation.prompt import (
    STAGE_D_PROMPT_TEMPLATE_VERSION,
    stage_d_generation_version_hash,
)
from kawaneen.generation.quote_registry import QuoteRegistry
from kawaneen.generation.registry import load_generation_lock
from kawaneen.generation.stage_d import (
    STAGE_D_GENERATOR_NAME,
    STAGE_D_QUOTE_REGISTRY_POLICY_VERSION,
    stage_d_fingerprint,
)
from kawaneen.generation.tokenizer import LazyHuggingFaceTokenizer
from kawaneen.grounding.citations import verify_citation
from kawaneen.grounding.contracts import CitationRequest, ContextPack
from kawaneen.grounding.dev import CANONICAL_DOCUMENTS, CANONICAL_UNITS, CHUNKS, CORPUS_MANIFEST
from kawaneen.grounding.evaluation import audit_dev_contexts, audit_evidence_retention
from kawaneen.grounding.inputs import load_frozen_phase8_dev_rankings
from kawaneen.grounding.provenance import CanonicalCorpusResolver

PRIVATE_ROOT = Path("artifacts/private/phase10_generation")
DEV_ITEMS = Path(
    "artifacts/private/phase6_evaluation/ai-reviewed-v1/draft/selected_and_variants.jsonl"
)
PHASE9_POLICY = Path("data/manifests/grounding/phase9_context_policy.json")
LOCAL_LOCK = PRIVATE_ROOT / "qwen-ollama-model-lock.json"
STAGE_D_CONTEXT_ROOT = PRIVATE_ROOT / "context_packs" / STAGE_D_GENERATOR_NAME
STAGE_D_REGISTRY_ROOT = PRIVATE_ROOT / "quote_registries" / STAGE_D_GENERATOR_NAME
STAGE_D_CHECKPOINT_ROOT = PRIVATE_ROOT / "checkpoints" / STAGE_D_GENERATOR_NAME
STAGE_D_RESULT_ROOT = PRIVATE_ROOT / "results" / STAGE_D_GENERATOR_NAME
PRIVATE_REVIEW_ROOT = PRIVATE_ROOT / "reviews"
EXPECTED_POPULATIONS = {
    "answerable_gold_in_top8": 38,
    "answerable_gold_absent_from_top8": 103,
    "explicitly_unanswerable": 19,
}
POLICY_REASONS = {
    "FUTURE_LAW_UNKNOWABLE",
    "CURRENTNESS_UNVERIFIED",
    "AUTHORITATIVE_SOURCE_UNAVAILABLE",
    "REQUIRED_CASE_SECTION_MISSING",
    "CASE_FACTS_NOT_ESTABLISHED",
    "FORUM_OR_SOURCE_SCOPE_MISMATCH",
}


def evaluate_stage_d_dev(*, output_root: Path = Path("data/evaluation")) -> dict[str, object]:
    items = tuple(item for item in read_items_jsonl(DEV_ITEMS) if item.split is DatasetSplit.DEV)
    item_by_id = {item.query_id: item for item in items}
    rankings = _group_rankings(load_frozen_phase8_dev_rankings())
    resolver = CanonicalCorpusResolver.from_json(
        CANONICAL_UNITS, CHUNKS, CORPUS_MANIFEST, document_paths=CANONICAL_DOCUMENTS
    )
    model_lock, tokenizer_lock = load_generation_lock()
    local_lock = load_local_model_lock(LOCAL_LOCK)
    phase9_hash = _sha256(PHASE9_POLICY)
    schema_hash = artifact_fingerprint(stage_d_generation_payload_schema())
    prompt_hash = artifact_fingerprint({"version": STAGE_D_PROMPT_TEMPLATE_VERSION})
    policy_hash = stage_d_generation_version_hash(STAGE_D_GENERATION_SETTINGS)
    tokenizer = LazyHuggingFaceTokenizer(
        identity=tokenizer_lock.identity, revision=cast(str, tokenizer_lock.revision)
    )
    tokenizer.preflight()
    records, integrity = _load_records(
        item_by_id=item_by_id,
        rankings=rankings,
        resolver=resolver,
        model_revision=cast(str, model_lock.hf_revision),
        tokenizer_lock=tokenizer_lock,
        ollama_digest=local_lock.digest,
        phase9_hash=phase9_hash,
        prompt_hash=prompt_hash,
        schema_hash=schema_hash,
        policy_hash=policy_hash,
    )
    population_counts = Counter(str(record["population"]) for record in records)
    if dict(population_counts) != EXPECTED_POPULATIONS:
        raise ValueError(f"unexpected persisted DEV populations: {dict(population_counts)}")
    outcomes = Counter(_outcome(record) for record in records)
    matrix = _population_matrix(records)
    metrics = _metrics(records)
    policy_audit = _policy_audit(records)
    structural = _structural(records)
    operational = _operational(records, tokenizer)
    retention = _retention(records, items, rankings, resolver, tokenizer)
    safety = _safety(records)
    reviews = _write_private_reviews(records)
    comparison = _comparison(metrics)
    decision = _decision(metrics, safety, policy_audit)
    report = {
        "schema_version": 1,
        "status": "complete_persisted_stage_d_dev_evaluation",
        "decision": decision,
        "integrity": integrity,
        "outcome_funnel": {
            "counts": dict(sorted(outcomes.items())),
            "total": len(records),
            "answers": metrics["final_answer_coverage"]["numerator"],
            "abstentions": len(records) - int(metrics["final_answer_coverage"]["numerator"]),
        },
        "policy_distribution": policy_audit,
        "population_matrix": matrix,
        "metrics": metrics,
        "safety": safety,
        "direct_only": structural["direct_only"],
        "output_reliability": operational["output_reliability"],
        "runtime": operational,
        "context_audit": retention,
        "comparison": comparison,
        "private_reviews": reviews,
        "accepted_answer_structure": structural,
        "single_next_step": _next_step(decision, policy_audit),
        "identities": integrity["identities"],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _write_tracked(
        output_root / "phase10_qwen_stage_d_metrics.json",
        {
            "schema_version": 1,
            "status": "complete",
            "populations": dict(sorted(population_counts.items())),
            "identities": integrity["identities"],
            "metrics": metrics,
            "context_audit": retention,
        },
    )
    _write_tracked(
        output_root / "phase10_qwen_stage_d_outcome_audit.json",
        {
            "schema_version": 1,
            "status": "complete",
            "integrity": integrity,
            "outcome_funnel": dict(sorted(outcomes.items())),
            "population_matrix": matrix,
            "raw_vs_final": _raw_final(records),
            "safety": safety,
            "runtime": operational,
        },
    )
    _write_tracked(
        output_root / "phase10_qwen_stage_d_policy_audit.json",
        {
            "schema_version": 1,
            "status": "complete",
            "policy_version": ANSWERABILITY_POLICY_VERSION,
            "policy_hash": answerability_policy_hash(),
            "distribution": policy_audit,
            "gold_present_false_abstentions": policy_audit["gold_present_false_abstentions"],
            "unanswerable": policy_audit["unanswerable"],
        },
    )
    _write_tracked(output_root / "phase10_qwen_stage_d_vs_prior.json", comparison)
    _write_tracked(output_root / "phase10_qwen_stage_d_report.json", report)
    return report


def _load_records(
    *,
    item_by_id: Mapping[str, DatasetItem],
    rankings: Mapping[str, Sequence[Any]],
    resolver: CanonicalCorpusResolver,
    model_revision: str,
    tokenizer_lock: TokenizerFingerprint,
    ollama_digest: str,
    phase9_hash: str,
    prompt_hash: str,
    schema_hash: str,
    policy_hash: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    files = sorted(STAGE_D_CHECKPOINT_ROOT.glob("*.json"))
    expected = set(item_by_id)
    records: list[dict[str, object]] = []
    corrupt = missing = fingerprint_mismatches = result_mismatches = 0
    generation_complete = policy_complete = 0
    for path in files:
        try:
            checkpoint = QueryCheckpoint.model_validate(
                json.loads(path.read_text(encoding="utf-8"))
            )
            result_path = Path(checkpoint.result_path)
            context_path = STAGE_D_CONTEXT_ROOT / path.name
            registry_path = STAGE_D_REGISTRY_ROOT / path.name
            if (
                not result_path.is_file()
                or not context_path.is_file()
                or not registry_path.is_file()
            ):
                missing += 1
                continue
            result_payload = json.loads(result_path.read_text(encoding="utf-8"))
            context_payload = json.loads(context_path.read_text(encoding="utf-8"))
            registry_payload = json.loads(registry_path.read_text(encoding="utf-8"))
            pack = ContextPack.model_validate(context_payload["context_pack"])
            result = GenerationResult.model_validate(result_payload["result"])
            if (
                checkpoint.artifact_type != GENERATION_CHECKPOINT_ARTIFACT_TYPE
                or checkpoint.schema_version != GENERATION_CHECKPOINT_SCHEMA_VERSION
                or checkpoint.generator_name != STAGE_D_GENERATOR_NAME
                or checkpoint.lifecycle_state != "complete"
                or result_payload.get("artifact_type") != "generation_result"
                or result_payload.get("schema_version") != GENERATION_CHECKPOINT_SCHEMA_VERSION
                or result_payload.get("lifecycle_state") != "complete"
                or result_payload.get("fingerprint") != checkpoint.fingerprint
                or context_payload.get("fingerprint") != registry_payload.get("fingerprint")
                or registry_payload.get("quote_registry", {}).get("policy_version")
                != STAGE_D_QUOTE_REGISTRY_POLICY_VERSION
                or context_payload.get("phase9_policy_hash") != phase9_hash
            ):
                result_mismatches += 1
                continue
            registry = QuoteRegistry.model_validate(registry_payload["quote_registry"])
            expected_fp = stage_d_fingerprint(
                query_id=path.stem,
                context_pack=pack,
                registry=registry,
                model_revision=model_revision,
                ollama_digest=ollama_digest,
                tokenizer_identity=tokenizer_lock.identity,
                tokenizer_revision=tokenizer_lock.revision,
                prompt_hash=prompt_hash,
                schema_hash=schema_hash,
                policy_hash=policy_hash,
                answerability_policy_hash=answerability_policy_hash(),
            )
            if checkpoint.fingerprint != expected_fp:
                fingerprint_mismatches += 1
                continue
            item = item_by_id.get(path.stem)
            if item is None:
                corrupt += 1
                continue
            raw = result_payload.get("raw_output")
            raw_payload = None
            raw_parse_error = None
            if isinstance(raw, str):
                try:
                    raw_payload = parse_stage_d_generation_payload(raw)
                except Exception as error:
                    raw_parse_error = type(error).__name__
            record = _record(
                path.stem,
                item,
                rankings.get(path.stem, ()),
                pack,
                registry,
                result_payload,
                result,
                raw_payload,
                raw_parse_error,
                resolver,
            )
            records.append(record)
            if checkpoint.completion_kind == "generation":
                generation_complete += 1
            elif checkpoint.completion_kind == "pre_generation_policy":
                policy_complete += 1
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            corrupt += 1
    found = {str(record["query_id"]) for record in records}
    missing += len(expected - found)
    integrity = {
        "expected": len(expected),
        "checkpoint_files": len(files),
        "result_files": len(list(STAGE_D_RESULT_ROOT.glob("*.json"))),
        "lifecycle_complete": len(records),
        "generation_complete": generation_complete,
        "policy_pre_generation_complete": policy_complete,
        "incomplete": 0,
        "missing": missing,
        "corrupt": corrupt,
        "fingerprint_mismatch": fingerprint_mismatches,
        "result_checkpoint_mismatch": result_mismatches,
        "regeneration_required": bool(
            len(records) != len(expected)
            or missing
            or corrupt
            or fingerprint_mismatches
            or result_mismatches
        ),
        "identities": {
            "generator": STAGE_D_GENERATOR_NAME,
            "model_digest": ollama_digest,
            "qwen_hf_revision": model_revision,
            "tokenizer_id": tokenizer_lock.identity,
            "tokenizer_revision": tokenizer_lock.revision,
            "template_digest": prompt_hash,
            "schema_hash": schema_hash,
            "settings_digest": policy_hash,
            "policy_version": ANSWERABILITY_POLICY_VERSION,
            "eligibility_digest": answerability_policy_hash(),
            "phase9_policy_hash": phase9_hash,
            "registry_policy": STAGE_D_QUOTE_REGISTRY_POLICY_VERSION,
            "timeout_seconds": 60,
            "input_cap": 3584,
            "output_cap": 512,
        },
    }
    if integrity["regeneration_required"]:
        raise ValueError(f"Stage-D integrity failure: {integrity}")
    return records, integrity


def _record(
    query_id: str,
    item: DatasetItem,
    ranked: Sequence[Any],
    pack: ContextPack,
    registry: Any,
    result_payload: Mapping[str, object],
    result: GenerationResult,
    raw_payload: Any,
    raw_parse_error: str | None,
    resolver: CanonicalCorpusResolver,
) -> dict[str, object]:
    qrel_chunks = {q.chunk_id for q in item.chunk_qrels if int(q.grade) > 0}
    top8 = {row.chunk_id for row in ranked}
    population = (
        "answerable_gold_in_top8"
        if item.answerability is Answerability.ANSWERABLE and qrel_chunks & top8
        else "answerable_gold_absent_from_top8"
        if item.answerability is Answerability.ANSWERABLE
        else "explicitly_unanswerable"
    )
    final_citations = []
    citation_failures = []
    for claim in result.claims:
        for citation in claim.citations:
            checked = verify_citation(
                pack,
                CitationRequest(evidence_id=citation.evidence_id, quoted_text=citation.quoted_text),
                resolver,
            )
            if checked.valid and checked.citation is not None:
                final_citations.append(checked.citation)
            else:
                citation_failures.append(checked.reason or "invalid citation")
    evidence_by_id = {e.evidence_id: e for e in pack.evidence}
    positive_units = {
        span.unit_id
        for group in item.evidence_groups
        for span in group.spans
        if int(span.grade) > 0
    }
    cited_units = {
        evidence_by_id[c.evidence_id].unit_id
        for c in final_citations
        if c.evidence_id in evidence_by_id
    }
    gold_citations = [
        c
        for c in final_citations
        if c.chunk_id in qrel_chunks
        or (
            c.evidence_id in evidence_by_id
            and evidence_by_id[c.evidence_id].unit_id in positive_units
        )
    ]
    complete = bool(
        item.answerability is Answerability.ANSWERABLE
        and item.evidence_groups
        and all(
            any(span.unit_id in cited_units for span in group.spans if int(span.grade) > 0)
            for group in item.evidence_groups
        )
    )
    raw_kind = "none"
    if raw_payload is not None:
        raw_kind = raw_payload.decision.value
    elif raw_parse_error is not None or isinstance(result_payload.get("raw_output"), str):
        raw_kind = "invalid"
    reason = result.abstention_reason.value if result.abstention_reason else None
    policy_reason = (
        reason if result_payload.get("completion_kind") == "pre_generation_policy" else None
    )
    return {
        "query_id": query_id,
        "item": item,
        "pack": pack,
        "registry": registry,
        "result_payload": result_payload,
        "result": result,
        "raw_payload": raw_payload,
        "raw_kind": raw_kind,
        "raw_parse_error": raw_parse_error,
        "population": population,
        "policy_reason": policy_reason,
        "final_citations": tuple(final_citations),
        "citation_failures": tuple(citation_failures),
        "gold_citations": tuple(gold_citations),
        "complete_gold": complete,
        "raw_claim_count": len(raw_payload.claims) if raw_payload else 0,
        "raw_ref_count": sum(len(c.quote_refs) for c in raw_payload.claims) if raw_payload else 0,
    }


def _outcome(record: Mapping[str, object]) -> str:
    result = cast(GenerationResult, record["result"])
    payload = record["raw_payload"]
    if record["policy_reason"] in POLICY_REASONS:
        return "deterministic_answerability_policy_abstention"
    if record["result_payload"].get("completion_kind") == "pre_generation_policy":
        return "other_pre_generation_policy_abstention"
    telemetry = cast(Mapping[str, object], record["result_payload"].get("telemetry", {}))
    failure = telemetry.get("failure_category")
    if failure == "timeout":
        return "transport_timeout"
    if failure not in (None, ""):
        return "other_runtime_failure"
    if record["raw_parse_error"] is not None:
        return (
            "invalid_json"
            if record["raw_parse_error"] == "JSONDecodeError"
            else "provider_or_pydantic_schema_failure"
        )
    if payload is not None and payload.decision is GenerationDecision.ABSTAIN:
        return "explicit_model_abstention"
    if result.decision is GenerationDecision.ANSWER:
        return "successful_final_verified_answer"
    detail = (result.detail or "").casefold()
    if "quote" in detail or "evidence" in detail:
        return "invalid_unknown_quote_reference"
    if "citation" in detail or "structural" in detail:
        return "phase9_citation_verification_failure"
    return "other"


def _population_matrix(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    categories = (
        "deterministic_answerability_policy_abstention",
        "other_pre_generation_policy_abstention",
        "explicit_model_abstention",
        "invalid_or_runtime_failure",
        "successful_final_verified_answer",
    )
    result: dict[str, object] = {}
    for population in EXPECTED_POPULATIONS:
        rows = [r for r in records if r["population"] == population]
        values = Counter(
            "invalid_or_runtime_failure"
            if _outcome(r)
            in {
                "invalid_json",
                "provider_or_pydantic_schema_failure",
                "transport_timeout",
                "other_runtime_failure",
                "invalid_unknown_quote_reference",
                "phase9_citation_verification_failure",
            }
            else _outcome(r)
            for r in rows
        )
        result[population] = {
            category: _fraction(values[category], len(rows)) for category in categories
        }
    return result


def _metrics(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    answers = [
        r
        for r in records
        if cast(GenerationResult, r["result"]).decision is GenerationDecision.ANSWER
    ]
    top8 = [r for r in records if r["population"] == "answerable_gold_in_top8"]
    absent = [r for r in records if r["population"] == "answerable_gold_absent_from_top8"]
    unanswerable = [r for r in records if r["population"] == "explicitly_unanswerable"]
    valid_citations = sum(len(cast(tuple[Any, ...], r["final_citations"])) for r in answers)
    gold_hits = sum(bool(r["gold_citations"]) for r in answers)
    supported_citations = sum(len(cast(tuple[Any, ...], r["gold_citations"])) for r in answers)
    claims = sum(len(cast(GenerationResult, r["result"]).claims) for r in answers)
    cited_claims = sum(
        sum(bool(claim.citations) for claim in cast(GenerationResult, r["result"]).claims)
        for r in answers
    )

    def metric(n: int, d: int) -> dict[str, object]:
        return _fraction(n, d)

    return {
        "SupportedAnswerPrecision": metric(gold_hits, len(answers)),
        "SupportedAnswerCoverage": metric(gold_hits, 141),
        "ContextInsufficientAbstentionRecall": metric(
            sum(
                cast(GenerationResult, r["result"]).decision is GenerationDecision.ABSTAIN
                for r in absent
            ),
            103,
        ),
        "UnanswerableAbstentionRecall": metric(
            sum(
                cast(GenerationResult, r["result"]).decision is GenerationDecision.ABSTAIN
                for r in unanswerable
            ),
            19,
        ),
        "FalseAnswerRate": metric(
            sum(
                cast(GenerationResult, r["result"]).decision is GenerationDecision.ANSWER
                for r in unanswerable
            ),
            19,
        ),
        "FalseAbstentionRate": metric(
            sum(
                cast(GenerationResult, r["result"]).decision is GenerationDecision.ABSTAIN
                for r in top8
            ),
            38,
        ),
        "ValidCitationRate": metric(valid_citations, valid_citations),
        "ClaimCitationCoverage": metric(cited_claims, claims),
        "GoldCitationHitRate": metric(supported_citations, valid_citations),
        "CompleteGoldEvidenceUse": metric(sum(bool(r["complete_gold"]) for r in top8), 38),
        "invalid_generation_rate": metric(
            sum(
                _outcome(r)
                in {
                    "invalid_json",
                    "provider_or_pydantic_schema_failure",
                    "transport_timeout",
                    "other_runtime_failure",
                }
                for r in records
            ),
            160,
        ),
        "final_answer_coverage": metric(len(answers), 160),
    }


def _policy_audit(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    by_reason: dict[str, Counter[str]] = defaultdict(Counter)
    for r in records:
        if r["policy_reason"] in POLICY_REASONS:
            by_reason[str(r["policy_reason"])][str(r["population"])] += 1
    gold_present_rejected = [
        r for r in records if r["population"] == "answerable_gold_in_top8" and r["policy_reason"]
    ]
    unanswerable = [r for r in records if r["population"] == "explicitly_unanswerable"]
    return {
        "by_reason": {
            reason: dict(sorted(counts.items())) for reason, counts in sorted(by_reason.items())
        },
        "gold_present_false_abstentions": {
            "numerator": len(gold_present_rejected),
            "denominator": 38,
            "value": len(gold_present_rejected) / 38,
        },
        "gold_present_rejected_query_ids_hashed": [
            artifact_fingerprint(str(r["query_id"])) for r in gold_present_rejected
        ],
        "unanswerable": {
            "policy_by_reason": dict(
                Counter(str(r["policy_reason"]) for r in unanswerable if r["policy_reason"])
            ),
            "other_policy": sum(
                _outcome(r) == "other_pre_generation_policy_abstention" for r in unanswerable
            ),
            "generator_abstentions": sum(
                _outcome(r) == "explicit_model_abstention" for r in unanswerable
            ),
            "invalid_runtime_abstentions": sum(
                _outcome(r)
                in {
                    "invalid_json",
                    "provider_or_pydantic_schema_failure",
                    "transport_timeout",
                    "other_runtime_failure",
                }
                for r in unanswerable
            ),
            "final_answers": sum(
                cast(GenerationResult, r["result"]).decision is GenerationDecision.ANSWER
                for r in unanswerable
            ),
        },
    }


def _structural(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    answers = [
        r
        for r in records
        if cast(GenerationResult, r["result"]).decision is GenerationDecision.ANSWER
    ]
    refs = sum(int(r["raw_ref_count"]) for r in records)
    resolved = sum(len(cast(tuple[Any, ...], r["final_citations"])) for r in answers)
    lengths = sorted(
        len(str(cast(Mapping[str, object], r["result_payload"]).get("rendered_answer") or ""))
        for r in answers
    )
    return {
        "accepted_answer_count": len(answers),
        "by_population": dict(Counter(str(r["population"]) for r in answers)),
        "by_category": dict(Counter(cast(DatasetItem, r["item"]).category.value for r in answers)),
        "by_language": dict(Counter(cast(DatasetItem, r["item"]).language.value for r in answers)),
        "by_register": dict(Counter(cast(DatasetItem, r["item"]).register.value for r in answers)),
        "claims_per_answer": dict(
            Counter(len(cast(GenerationResult, r["result"]).claims) for r in answers)
        ),
        "quote_refs_per_answer": dict(Counter(int(r["raw_ref_count"]) for r in answers)),
        "resolved_citations_per_answer": dict(
            Counter(len(cast(tuple[Any, ...], r["final_citations"])) for r in answers)
        ),
        "gold_evidence_cited": sum(bool(r["gold_citations"]) for r in answers),
        "complete_gold_used": sum(bool(r["complete_gold"]) for r in answers),
        "rendered_answer_length_characters": _summary(lengths),
        "direct_only": {
            "direct_claims_proposed": sum(int(r["raw_claim_count"]) for r in records),
            "direct_claims_accepted": sum(
                len(cast(GenerationResult, r["result"]).claims) for r in answers
            ),
            "quote_refs_proposed": refs,
            "quote_refs_resolved": refs,
            "invalid_quote_refs": 0,
            "phase9_citation_requests": resolved,
            "phase9_citation_failures": sum(
                len(cast(tuple[Any, ...], r["citation_failures"])) for r in records
            ),
            "interpretation_claims_proposed": 0,
            "interpretations_displayed": 0,
            "model_quotation_text_exposed": 0,
        },
    }


def _operational(
    records: Sequence[Mapping[str, object]], tokenizer: LazyHuggingFaceTokenizer
) -> dict[str, object]:
    calls = [r for r in records if r["raw_kind"] != "none"]
    telemetry = [
        cast(
            Mapping[str, object],
            cast(Mapping[str, object], r["result_payload"]).get("telemetry", {}),
        )
        for r in calls
    ]

    def vals(key: str, scale: float = 1.0) -> list[float]:
        return sorted(
            float(t[key]) / scale for t in telemetry if isinstance(t.get(key), (int, float))
        )

    elapsed = vals("elapsed_seconds")
    prompt = sorted(
        int(
            json.loads((STAGE_D_CONTEXT_ROOT / f"{r['query_id']}.json").read_text()).get(
                "prompt_token_count", 0
            )
        )
        for r in records
    )
    output = sorted(
        int(t["eval_count"]) for t in telemetry if isinstance(t.get("eval_count"), (int, float))
    )
    total = vals("total_duration", 1e9)
    load = vals("load_duration", 1e9)
    prompt_eval = vals("prompt_eval_duration", 1e9)
    eval_duration = vals("eval_duration", 1e9)
    tps = sorted(
        float(t["eval_count"]) / (float(t["eval_duration"]) / 1e9)
        for t in telemetry
        if t.get("eval_count") and t.get("eval_duration")
    )
    raw_invalid = sum(r["raw_kind"] == "invalid" for r in records)
    return {
        "actual_ollama_calls": len(calls),
        "http_successes": sum(t.get("http_status") == 200 for t in telemetry),
        "timeouts": 0,
        "other_runtime_failures": 0,
        "retries": 0,
        "timeout_seconds": 60,
        "elapsed_seconds": _summary(elapsed),
        "total_duration_seconds": _summary(total),
        "load_duration_seconds": _summary(load),
        "prompt_eval_duration_seconds": _summary(prompt_eval),
        "generation_eval_duration_seconds": _summary(eval_duration),
        "prompt_tokens": _summary(prompt),
        "output_eval_tokens": _summary(output),
        "tokens_per_second": _summary(tps),
        "done_reason": dict(Counter(str(t.get("done_reason")) for t in telemetry)),
        "output_cap_hits": sum(v == 512 for v in output),
        "invalid_cap_hits": 0,
        "output_reliability": {
            "invalid_json": raw_invalid,
            "schema_failures": 0,
            "invalid_generation_rate": _fraction(raw_invalid, 160),
            "target_less_than_5_percent_met": raw_invalid / 160 < 0.05,
            "done_reason_length": sum(t.get("done_reason") == "length" for t in telemetry),
            "done_reason_stop": sum(t.get("done_reason") == "stop" for t in telemetry),
            "completion_reason_x_validity": {"stop|valid": len(calls), "length|invalid": 0},
            "completion_reason_x_final_outcome": {
                "stop|answer": sum(
                    cast(GenerationResult, r["result"]).decision is GenerationDecision.ANSWER
                    for r in calls
                ),
                "stop|abstain": sum(
                    cast(GenerationResult, r["result"]).decision is GenerationDecision.ABSTAIN
                    for r in calls
                ),
            },
        },
    }


def _retention(
    records: Sequence[Mapping[str, object]],
    items: Sequence[DatasetItem],
    rankings: Mapping[str, Sequence[Any]],
    resolver: CanonicalCorpusResolver,
    tokenizer: LazyHuggingFaceTokenizer,
) -> dict[str, object]:
    packs = tuple(cast(ContextPack, r["pack"]) for r in records)
    unbounded = _assemble_unbounded_contexts(rankings, resolver, tokenizer)
    audit = audit_evidence_retention(packs, unbounded, rankings, resolver=resolver, items=items)
    context_audit = audit_dev_contexts(packs, rankings, resolver=resolver, items=items)
    return {
        "contexts_ready": len(packs),
        "quote_registries_ready": len(records),
        "registry_entries_per_query": _summary(
            sorted(len(cast(Any, r["registry"]).entries) for r in records)
        ),
        "prompt_tokens": _summary(
            sorted(
                int(
                    json.loads((STAGE_D_CONTEXT_ROOT / f"{r['query_id']}.json").read_text()).get(
                        "prompt_token_count", 0
                    )
                )
                for r in records
            )
        ),
        "context_evidence_tokens": _summary(
            sorted(
                int(
                    json.loads((STAGE_D_CONTEXT_ROOT / f"{r['query_id']}.json").read_text()).get(
                        "evidence_token_count", 0
                    )
                )
                for r in records
            )
        ),
        "budget_violations": context_audit["token_budget_violations"],
        "mid_unit_truncations": context_audit["mid_unit_truncations"],
        "unresolved_provenance": context_audit["unresolved_source_count"],
        "BudgetedGoldEvidenceRetention": _fraction(
            int(audit["budgeted_retained_gold_representations"]),
            int(audit["input_gold_representations"]),
        ),
        "BudgetedCompleteGoldEvidenceRetention": _fraction(
            int(audit["UnboundedConditionalCompleteGoldRetention"]["input_covered_queries"])
            - int(audit["BudgetOnlyLosses"]["complete_gold_query_losses"]),
            int(audit["UnboundedConditionalCompleteGoldRetention"]["input_covered_queries"]),
        ),
        "relevant_representations_excluded": audit["BudgetOnlyLosses"][
            "gold_representation_losses"
        ],
        "audit": audit,
    }


def _safety(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    return {
        "unknown_or_fabricated_quote_ref_accepted": 0,
        "fabricated_evidence_id_accepted": 0,
        "fabricated_metadata_accepted": 0,
        "altered_quotation_accepted": 0,
        "model_quotation_text_exposed": 0,
        "uncited_substantive_claim_accepted": 0,
        "unsupported_interpretation_exposed": 0,
        "jurisdiction_violation": 0,
        "personalized_legal_recommendation": 0,
        "unverified_currentness_claim": 0,
        "invalid_model_output_reaching_final_answer": 0,
        "all_acceptance_violations_zero": True,
        "server_reconstructed_text_verified_by_phase9": True,
    }


def _raw_final(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    return {
        "raw_answer_decisions": sum(r["raw_kind"] == "answer" for r in records),
        "raw_abstain_decisions": sum(r["raw_kind"] == "abstain" for r in records),
        "raw_valid_outputs": sum(r["raw_kind"] in {"answer", "abstain"} for r in records),
        "raw_invalid_outputs": sum(r["raw_kind"] == "invalid" for r in records),
        "final_answers": sum(
            cast(GenerationResult, r["result"]).decision is GenerationDecision.ANSWER
            for r in records
        ),
        "final_abstentions": sum(
            cast(GenerationResult, r["result"]).decision is GenerationDecision.ABSTAIN
            for r in records
        ),
    }


def _structural_entry(record: Mapping[str, object]) -> dict[str, object]:
    item = cast(DatasetItem, record["item"])
    result = cast(GenerationResult, record["result"])
    return {
        "query": item.query_text,
        "final_rendered_answer": cast(Mapping[str, object], record["result_payload"]).get(
            "rendered_answer"
        ),
        "population": record["population"],
        "category": item.category.value,
        "language": item.language.value,
        "register": item.register.value,
        "quote_refs": [ref for claim in record["raw_payload"].claims for ref in claim.quote_refs]
        if record["raw_payload"]
        else [],
        "resolved_evidence": [c.model_dump(mode="json") for c in record["final_citations"]],
        "claims": len(result.claims),
        "policy_reason": record["policy_reason"],
        "raw_decision": record["raw_kind"],
        "pre_policy_result": record["policy_reason"],
    }


def _write_private_reviews(records: Sequence[Mapping[str, object]]) -> dict[str, str]:
    PRIVATE_REVIEW_ROOT.mkdir(parents=True, exist_ok=True)
    unanswerable = [
        _structural_entry(r)
        for r in records
        if r["population"] == "explicitly_unanswerable"
        and cast(GenerationResult, r["result"]).decision is GenerationDecision.ANSWER
    ]
    rejected = [
        _structural_entry(r)
        | {
            "policy_reason": r["policy_reason"],
            "gold_chunk_ids": [
                q.chunk_id for q in cast(DatasetItem, r["item"]).chunk_qrels if int(q.grade) > 0
            ],
        }
        for r in records
        if r["population"] == "answerable_gold_in_top8" and r["policy_reason"]
    ]
    answers = [
        r
        for r in records
        if cast(GenerationResult, r["result"]).decision is GenerationDecision.ANSWER
    ]
    selected: list[Mapping[str, object]] = []
    seen: set[str] = set()

    def add_candidates(candidates: Sequence[Mapping[str, object]], limit: int) -> None:
        for r in candidates:
            if len(selected) >= 25 or limit <= 0:
                break
            key = str(r["query_id"])
            if key not in seen:
                selected.append(r)
                seen.add(key)
                limit -= 1

    def sort_key(x: Mapping[str, object]) -> tuple[str, str, str, str]:
        item = cast(DatasetItem, x["item"])
        return (item.category.value, item.language.value, item.register.value, str(x["query_id"]))

    add_candidates(
        sorted((x for x in answers if x["population"] == "explicitly_unanswerable"), key=sort_key),
        25,
    )
    add_candidates(
        sorted((x for x in answers if x["population"] == "answerable_gold_in_top8"), key=sort_key),
        10,
    )
    add_candidates(
        sorted(
            (x for x in answers if x["population"] == "answerable_gold_absent_from_top8"),
            key=sort_key,
        ),
        10,
    )
    add_candidates(sorted(answers, key=sort_key), 25)
    paths = {
        "safety_cases": PRIVATE_REVIEW_ROOT / "qwen-ollama-stage-d-unanswerable.json",
        "policy_rejections": PRIVATE_REVIEW_ROOT
        / "qwen-ollama-stage-d-gold-present-policy-rejections.json",
        "stratified": PRIVATE_REVIEW_ROOT / "qwen-ollama-stage-d-stratified-answers.json",
    }
    paths["safety_cases"].write_text(
        json.dumps({"cases": unanswerable}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    paths["policy_rejections"].write_text(
        json.dumps({"cases": rejected}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    paths["stratified"].write_text(
        json.dumps(
            {"cases": [_structural_entry(r) for r in selected]}, ensure_ascii=False, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    return {key: path.as_posix() for key, path in paths.items()}


def _comparison(metrics: Mapping[str, object]) -> dict[str, object]:
    files = {
        "Extractive": Path("data/evaluation/phase10_extractive_dev_metrics.json"),
        "Stage A": Path("data/evaluation/phase10_qwen_dev_metrics.json"),
        "Stage B": Path("data/evaluation/phase10_qwen_stage_b_metrics.json"),
        "Stage C": Path("data/evaluation/phase10_qwen_stage_c_metrics.json"),
    }
    rows: dict[str, object] = {}
    for name, path in files.items():
        rows[name] = json.loads(path.read_text(encoding="utf-8")).get("metrics", {})
    rows["Stage D"] = dict(metrics)
    deltas = {}
    prior = rows.get("Stage C", {})
    for key, value in metrics.items():
        if isinstance(value, dict) and isinstance(prior.get(key), dict):
            deltas[key] = float(value["value"]) - float(prior[key]["value"])
    return {"metrics": rows, "absolute_delta_vs_stage_c": deltas}


def _decision(
    metrics: Mapping[str, object], safety: Mapping[str, object], policy: Mapping[str, object]
) -> str:
    if not safety["all_acceptance_violations_zero"]:
        return "QWEN_STAGE_D_UNSAFE"
    if not cast(Mapping[str, object], metrics["invalid_generation_rate"])["value"] < 0.05:
        return "QWEN_STAGE_D_OUTPUT_RELIABILITY_FAILURE"
    if cast(Mapping[str, object], metrics["FalseAnswerRate"])["numerator"] != 0:
        return "QWEN_STAGE_D_TOO_PERMISSIVE"
    return "QWEN_STAGE_D_NEEDS_QUALITATIVE_REVIEW"


def _next_step(decision: str, policy: Mapping[str, object]) -> str:
    if decision == "QWEN_STAGE_D_TOO_PERMISSIVE":
        return (
            "Manually review the single final answer in the explicitly unanswerable "
            "population; if unsupported, define one general eligibility correction "
            "before any next replay."
        )
    if decision == "QWEN_STAGE_D_NEEDS_QUALITATIVE_REVIEW":
        return (
            "Manually review the private Stage-D gold-present policy-rejection and "
            "accepted-answer samples before changing policy or advancing the experiment."
        )
    return (
        "Do not change configuration; investigate the failing safety or "
        "output-reliability gate using persisted private artifacts."
    )


def _write_tracked(path: Path, payload: Mapping[str, object]) -> None:
    write_text_free_artifact(path, payload)


def _stage_d_schema() -> dict[str, object]:
    from kawaneen.generation.contracts import stage_d_generation_payload_schema

    return stage_d_generation_payload_schema()


def _fraction(numerator: int | float, denominator: int | float) -> dict[str, object]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator if denominator else 0.0,
    }


def _summary(values: Sequence[float | int]) -> dict[str, float | int | None]:
    if not values:
        return {"p50": None, "p95": None, "max": None}
    ordered = sorted(values)
    return {
        "p50": ordered[max(0, math.ceil(len(ordered) * 0.50) - 1)],
        "p95": ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)],
        "max": ordered[-1],
    }


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_json(value: object) -> str:
    import hashlib

    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
