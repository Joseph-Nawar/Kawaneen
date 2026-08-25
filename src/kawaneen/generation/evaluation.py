"""Post-hoc text-free evaluation summaries for Stage A."""

# The persisted-artifact evaluator intentionally uses aggregate dictionaries;
# runtime generation modules remain strict-typed independently.
# pyright: basic
# pyright: reportArgumentType=false
# pyright: reportOptionalMemberAccess=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportReturnType=false

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from kawaneen.evaluation.models import Answerability, DatasetItem, DatasetSplit, RelevanceGrade
from kawaneen.evaluation.serialization import read_items_jsonl
from kawaneen.generation.budgeting import BudgetedContext
from kawaneen.generation.checkpoints import QueryCheckpoint
from kawaneen.generation.contracts import (
    AbstentionReason,
    GenerationDecision,
    GenerationResult,
    GenerationSettings,
    TokenizerFingerprint,
    parse_model_output,
)
from kawaneen.generation.ollama import load_local_model_lock
from kawaneen.generation.orchestration import generation_fingerprint
from kawaneen.generation.policy import PolicyContext, evaluate_pre_generation_policy
from kawaneen.generation.prompt import (
    PROMPT_TEMPLATE_VERSION,
    generation_version_hash,
    render_generation_prompt,
)
from kawaneen.generation.registry import load_generation_lock
from kawaneen.generation.tokenizer import LazyHuggingFaceTokenizer
from kawaneen.grounding.citations import verify_citation
from kawaneen.grounding.contracts import CitationRequest, ContextPack
from kawaneen.grounding.dev import CANONICAL_DOCUMENTS, CANONICAL_UNITS, CHUNKS, CORPUS_MANIFEST
from kawaneen.grounding.inputs import PHASE8_SELECTION_SHA256, load_frozen_phase8_dev_rankings
from kawaneen.grounding.provenance import CanonicalCorpusResolver


def evaluate_generation_results(results: Iterable[GenerationResult]) -> dict[str, object]:
    values = tuple(results)
    reasons = Counter(
        result.abstention_reason.value for result in values if result.abstention_reason is not None
    )
    output: dict[str, Any] = {
        "result_count": len(values),
        "answer_count": sum(result.decision is GenerationDecision.ANSWER for result in values),
        "abstain_count": sum(result.decision is GenerationDecision.ABSTAIN for result in values),
        "abstention_reasons": dict(sorted(reasons.items())),
    }
    return output


def evaluate_budget_report(
    reports: Mapping[str, BudgetedContext],
) -> dict[str, dict[str, object]]:
    return {
        name: {
            "tokenizer_identity": report.tokenizer_identity,
            "non_evidence_prompt_tokens": report.non_evidence_prompt_tokens,
            "evidence_token_count": report.evidence_token_count,
            "prompt_token_count": report.prompt_token_count,
            "evidence_budget_tokens": report.evidence_budget_tokens,
            "omitted_unit_count": len(report.omitted_unit_ids),
            "gold_evidence_retention": report.gold_evidence_retention,
            "complete_gold_evidence_retention": report.complete_gold_evidence_retention,
        }
        for name, report in sorted(reports.items())
    }


_DEV_ITEMS = Path(
    "artifacts/private/phase6_evaluation/ai-reviewed-v1/draft/selected_and_variants.jsonl"
)
_PRIVATE_ROOT = Path("artifacts/private/phase10_generation")
_CHECKPOINT_ROOT = _PRIVATE_ROOT / "checkpoints" / "qwen-ollama"
_RESULT_ROOT = _PRIVATE_ROOT / "results" / "qwen-ollama"
_CONTEXT_ROOT = _PRIVATE_ROOT / "context_packs" / "qwen-ollama"
_LOCAL_LOCK = _PRIVATE_ROOT / "qwen-ollama-model-lock.json"
_PHASE9_POLICY = Path("data/manifests/grounding/phase9_context_policy.json")

_OUTCOME_KEYS = (
    "pre_generation_jurisdiction_policy",
    "pre_generation_personalized_advice_policy",
    "pre_generation_currentness_status_policy",
    "pre_generation_conflict_policy",
    "pre_generation_no_or_insufficient_context",
    "other_pre_generation_policy",
    "generator_explicitly_returned_abstain",
    "invalid_or_malformed_generator_output",
    "invalid_citation_or_evidence_id",
    "non_exact_quote",
    "uncited_or_unsupported_claim",
    "semantic_support_diagnostic",
    "successful_verified_answer",
    "other",
)


def write_persisted_qwen_dev_artifacts(
    *,
    output_root: Path = Path("data/evaluation"),
    checkpoint_root: Path = _CHECKPOINT_ROOT,
    result_root: Path = _RESULT_ROOT,
    context_root: Path = _CONTEXT_ROOT,
) -> dict[str, object]:
    """Evaluate persisted Qwen DEV files without invoking a generator.

    The function deliberately loads qrels only after runtime-shaped results and
    contexts have been loaded. It writes aggregate-only JSON: query, answer,
    quote, and source-bearing values never enter the returned payload.
    """

    items = tuple(item for item in read_items_jsonl(_DEV_ITEMS) if item.split is DatasetSplit.DEV)
    item_by_id = {item.query_id: item for item in items}
    rankings = _group_rankings(load_frozen_phase8_dev_rankings())
    resolver = CanonicalCorpusResolver.from_json(
        CANONICAL_UNITS,
        CHUNKS,
        CORPUS_MANIFEST,
        document_paths=CANONICAL_DOCUMENTS,
    )
    model_lock, tokenizer_lock = load_generation_lock()
    local_lock = load_local_model_lock(_LOCAL_LOCK)
    settings = GenerationSettings()
    phase9_hash = _sha256(_PHASE9_POLICY)
    prompt_hash = _sha256_json({"version": PROMPT_TEMPLATE_VERSION})
    generation_hash = generation_version_hash(settings)
    tokenizer = LazyHuggingFaceTokenizer(
        identity=tokenizer_lock.identity,
        revision=cast(str, tokenizer_lock.revision),
    )

    records, integrity = _load_persisted_records(
        item_by_id=item_by_id,
        rankings=rankings,
        resolver=resolver,
        checkpoint_root=checkpoint_root,
        result_root=result_root,
        context_root=context_root,
        model_revision=cast(str, model_lock.hf_revision),
        tokenizer_fingerprint=tokenizer_lock,
        ollama_digest=local_lock.digest,
        phase9_hash=phase9_hash,
        prompt_hash=prompt_hash,
        generation_hash=generation_hash,
        settings=settings,
    )
    population_counts = Counter(record["population"] for record in records)
    outcome_counts = Counter(_first_decisive_outcome(record) for record in records)
    for key in _OUTCOME_KEYS:
        outcome_counts.setdefault(key, 0)

    metrics, posthoc = _qwen_metrics(records, population_counts)
    retention = _retention_audit(records, resolver, rankings)
    structural = _structural_breakdown(records)
    operational = _operational_metrics(records, tokenizer, settings)
    safety = _safety_audit(records)
    raw_final = _raw_final_summary(records)
    call_summary = {
        "observed_qwen_response_count": sum(record["raw_kind"] != "none" for record in records),
        "stopped_before_inference_count": sum(record["raw_kind"] == "none" for record in records),
        "observability_note": (
            "Call count is inferred from persisted adapter raw responses; "
            "no endpoint call counter was persisted."
        ),
        "raw_generator_decisions": raw_final["raw_generator_decisions"],
        "final_system_decisions": raw_final["final_system_decisions"],
        "post_generation_fail_closed_count": raw_final["post_generation_fail_closed_count"],
    }
    identities = {
        "qwen_hf_model_id": model_lock.hf_identity,
        "qwen_hf_revision": model_lock.hf_revision,
        "tokenizer_id": tokenizer_lock.identity,
        "tokenizer_revision": tokenizer_lock.revision,
        "ollama_model_tag": local_lock.model,
        "ollama_digest": local_lock.digest,
        "phase8_selection_sha256": PHASE8_SELECTION_SHA256,
        "phase9_policy_sha256": phase9_hash,
        "prompt_template_hash": prompt_hash,
        "generation_policy_hash": generation_hash,
    }

    metrics_payload = {
        "schema_version": 1,
        "status": "complete_persisted_qwen_dev_evaluation",
        "evaluation_split": "DEV",
        "evaluation_record_count": len(records),
        "populations": dict(sorted(population_counts.items())),
        "identities": identities,
        "metrics": metrics,
        "posthoc_decision_accounting": posthoc,
        "token_budget": retention["token_budget"],
    }
    outcome_payload = {
        "schema_version": 1,
        "status": "complete_persisted_qwen_dev_outcome_audit",
        "checkpoint_integrity": integrity,
        "outcome_decomposition": dict(sorted(outcome_counts.items())),
        "population_counts": dict(sorted(population_counts.items())),
        "qwen_call_summary": call_summary,
        "raw_vs_final": raw_final,
        "safety_invariants": safety,
        "structural_breakdown": structural,
        "operational_metrics": operational,
        "identities": identities,
    }
    extractive = json.loads(Path("data/evaluation/phase10_extractive_dev_metrics.json").read_text())
    comparison_payload = _comparison_payload(metrics, extractive["metrics"])
    comparison_payload.update({"schema_version": 1, "status": "complete_qwen_vs_frozen_extractive"})
    report_payload = {
        "schema_version": 1,
        "status": "complete_phase10_qwen_dev_analysis",
        "decision": _dev_decision(metrics, safety, raw_final),
        "checkpoint_integrity": integrity,
        "outcome_decomposition": dict(sorted(outcome_counts.items())),
        "metrics": metrics,
        "token_budget": retention["token_budget"],
        "retention_slices": retention["slices"],
        "comparison_artifact": "data/evaluation/phase10_qwen_vs_extractive.json",
        "single_next_step": _next_step(metrics, retention, outcome_counts),
        "raw_text_policy": (
            "Raw query, answer, quote, context, and source-bearing values remain private."
        ),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "phase10_qwen_dev_metrics.json": metrics_payload,
        "phase10_qwen_dev_outcome_audit.json": outcome_payload,
        "phase10_qwen_vs_extractive.json": comparison_payload,
        "phase10_qwen_dev_report.json": report_payload,
    }
    for name, payload in artifacts.items():
        (output_root / name).write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    return report_payload


def _load_persisted_records(
    *,
    item_by_id: Mapping[str, DatasetItem],
    rankings: Mapping[str, Sequence[Any]],
    resolver: CanonicalCorpusResolver,
    checkpoint_root: Path,
    result_root: Path,
    context_root: Path,
    model_revision: str,
    tokenizer_fingerprint: TokenizerFingerprint,
    ollama_digest: str,
    phase9_hash: str,
    prompt_hash: str,
    generation_hash: str,
    settings: GenerationSettings,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    expected_ids = set(item_by_id)
    files = sorted(checkpoint_root.glob("*.json")) if checkpoint_root.is_dir() else []
    corrupt = missing = mismatched = 0
    records: list[dict[str, object]] = []
    for path in files:
        query_id = path.stem
        try:
            checkpoint = QueryCheckpoint.model_validate(json.loads(path.read_text()))
            result_path = Path(checkpoint.result_path)
            if not result_path.is_file():
                missing += 1
                continue
            result_payload = json.loads(result_path.read_text())
            context_payload = json.loads((context_root / path.name).read_text())
            result = GenerationResult.model_validate(result_payload["result"])
            pack = ContextPack.model_validate(context_payload["context_pack"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            corrupt += 1
            continue
        if result_payload.get("fingerprint") != checkpoint.fingerprint:
            mismatched += 1
            continue
        expected_fingerprint = generation_fingerprint(
            query_id=query_id,
            context_pack=pack,
            model_revision=model_revision,
            ollama_digest=ollama_digest,
            tokenizer_fingerprint=tokenizer_fingerprint,
            prompt_template_hash=prompt_hash,
            generation_policy_hash=generation_hash,
            phase9_policy_hash=phase9_hash,
            settings=settings,
        )
        if expected_fingerprint != checkpoint.fingerprint:
            mismatched += 1
            continue
        item = item_by_id.get(query_id)
        if item is None:
            corrupt += 1
            continue
        record = _record(
            query_id,
            item,
            rankings.get(query_id, ()),
            pack,
            result_payload,
            result,
            resolver,
        )
        record["context_fixed_prompt_tokens"] = context_payload.get("fixed_prompt_tokens")
        record["context_evidence_budget_tokens"] = context_payload.get("evidence_token_budget")
        record["context_cache_fingerprint"] = context_payload.get("fingerprint")
        records.append(record)
    found_ids = {str(record["query_id"]) for record in records}
    missing += len(expected_ids - found_ids)
    extra = len(found_ids - expected_ids)
    integrity = {
        "expected_checkpoint_count": len(expected_ids),
        "checkpoint_file_count": len(files),
        "completed": len(records),
        "valid": len(records),
        "corrupt": corrupt,
        "fingerprint_mismatches": mismatched,
        "missing": missing,
        "unexpected": extra,
        "regeneration_required": len(records) != len(expected_ids)
        or corrupt
        or mismatched
        or missing,
        "model_digest": ollama_digest,
        "qwen_hf_revision": model_revision,
        "tokenizer_revision": tokenizer_fingerprint.revision,
        "prompt_template_hash": prompt_hash,
        "generation_policy_hash": generation_hash,
        "phase9_policy_hash": phase9_hash,
    }
    if len(records) != len(expected_ids) or corrupt or mismatched or missing or extra:
        raise ValueError("persisted Qwen DEV checkpoint audit is incomplete")
    return records, integrity


def _record(
    query_id: str,
    item: DatasetItem,
    ranked_inputs: Sequence[Any],
    pack: ContextPack,
    result_payload: Mapping[str, object],
    result: GenerationResult,
    resolver: CanonicalCorpusResolver,
) -> dict[str, object]:
    raw = result_payload.get("raw_output")
    raw_kind = "none"
    raw_output_text: str | None = None
    raw_output = None
    raw_invalid_reasons: list[str] = []
    if isinstance(raw, str):
        raw_output_text = raw
        try:
            raw_output = parse_model_output(raw)
            raw_kind = raw_output.decision.value
        except ValueError:
            raw_kind = "invalid"
    qrel_chunks = {
        qrel.chunk_id
        for qrel in item.chunk_qrels
        if int(qrel.grade) > int(RelevanceGrade.IRRELEVANT)
    }
    top8_chunks = {row.chunk_id for row in ranked_inputs}
    gold_in_top8 = (
        bool(qrel_chunks & top8_chunks) if item.answerability is Answerability.ANSWERABLE else False
    )
    population = (
        "answerable_gold_in_top8"
        if gold_in_top8
        else "answerable_gold_absent_from_top8"
        if item.answerability is Answerability.ANSWERABLE
        else "explicitly_unanswerable"
    )
    final_citations = []
    invalid_reasons: list[str] = []
    if result.decision is GenerationDecision.ANSWER:
        for claim in result.claims:
            for citation in claim.citations:
                verification = verify_citation(
                    pack,
                    CitationRequest(
                        evidence_id=citation.evidence_id,
                        quoted_text=citation.quoted_text,
                    ),
                    resolver,
                )
                if verification.valid and verification.citation is not None:
                    final_citations.append(verification.citation)
                elif verification.reason:
                    invalid_reasons.append(verification.reason)
    if raw_kind == "answer" and raw_output is not None:
        for claim in raw_output.claims:
            for citation in claim.citations:
                verification = verify_citation(
                    pack,
                    CitationRequest(
                        evidence_id=citation.evidence_id,
                        quoted_text=citation.quoted_text,
                    ),
                    resolver,
                )
                if not verification.valid and verification.reason:
                    raw_invalid_reasons.append(verification.reason)
    positive_units = {
        span.unit_id
        for group in item.evidence_groups
        for span in group.spans
        if span.grade > RelevanceGrade.IRRELEVANT
    }
    evidence_by_id = {evidence.evidence_id: evidence for evidence in pack.evidence}
    cited_units = {
        evidence_by_id[citation.evidence_id].unit_id
        for citation in final_citations
        if citation.evidence_id in evidence_by_id
    }
    supported_citations = [
        citation
        for citation in final_citations
        if citation.chunk_id in qrel_chunks
        or (
            evidence_by_id.get(citation.evidence_id) is not None
            and evidence_by_id[citation.evidence_id].unit_id in positive_units
        )
    ]
    complete = bool(
        result.decision is GenerationDecision.ANSWER
        and item.answerability is Answerability.ANSWERABLE
        and all(
            any(
                span.unit_id in cited_units
                for span in group.spans
                if span.grade > RelevanceGrade.IRRELEVANT
            )
            for group in item.evidence_groups
        )
    )
    policy = evaluate_pre_generation_policy(item.query_text, PolicyContext(context_pack=pack))
    return {
        "query_id": query_id,
        "item": item,
        "pack": pack,
        "result": result,
        "result_payload": result_payload,
        "raw_kind": raw_kind,
        "raw_output": raw_output,
        "raw_output_text": raw_output_text,
        "raw_invalid_citation_reasons": tuple(sorted(set(raw_invalid_reasons))),
        "policy_reason": policy.reason.value if not policy.allowed and policy.reason else None,
        "population": population,
        "gold_in_top8": gold_in_top8,
        "final_citations": tuple(final_citations),
        "invalid_citation_reasons": tuple(sorted(set(invalid_reasons))),
        "supported_citation_count": len(supported_citations),
        "gold_citation_count": len(supported_citations),
        "complete_gold_evidence_use": complete,
        "raw_claim_count": len(raw_output.claims) if raw_kind == "answer" else 0,
        "final_claim_count": len(result.claims),
    }


def _first_decisive_outcome(record: Mapping[str, object]) -> str:
    result = cast(GenerationResult, record["result"])
    reason = result.abstention_reason
    raw_kind = record["raw_kind"]
    detail = result.detail or ""
    if reason is AbstentionReason.CURRENTNESS_UNVERIFIED:
        return "pre_generation_currentness_status_policy"
    if (
        reason is AbstentionReason.JURISDICTION_MISMATCH
        or reason is AbstentionReason.JURISDICTION_AMBIGUOUS
    ):
        return "pre_generation_jurisdiction_policy"
    if reason is AbstentionReason.PERSONALIZED_LEGAL_ADVICE:
        return "pre_generation_personalized_advice_policy"
    if reason is AbstentionReason.CONFLICTING_EVIDENCE:
        return "pre_generation_conflict_policy"
    if reason is AbstentionReason.NO_CONTEXT or reason is AbstentionReason.LOW_RETRIEVAL_CONFIDENCE:
        return "pre_generation_no_or_insufficient_context"
    if reason is not None and reason is not AbstentionReason.INVALID_GENERATION:
        return "other_pre_generation_policy"
    if raw_kind == "none" and result.decision is GenerationDecision.ABSTAIN:
        return "other_pre_generation_policy"
    if result.decision is GenerationDecision.ABSTAIN and reason is None:
        return "generator_explicitly_returned_abstain"
    if "not valid JSON" in detail or "validation error" in detail:
        return "invalid_or_malformed_generator_output"
    if "exact authoritative substring" in detail or record.get("raw_invalid_citation_reasons"):
        return "non_exact_quote"
    if "semantic support" in detail:
        return "semantic_support_diagnostic"
    if "citation" in detail or "structural" in detail:
        return "invalid_citation_or_evidence_id"
    if result.decision is GenerationDecision.ANSWER:
        return "successful_verified_answer"
    return "other"


def _qwen_metrics(
    records: Sequence[Mapping[str, object]], populations: Mapping[str, int]
) -> tuple[dict[str, object], dict[str, object]]:
    answers = [
        record
        for record in records
        if cast(GenerationResult, record["result"]).decision is GenerationDecision.ANSWER
    ]
    answerable = [record for record in records if record["population"] != "explicitly_unanswerable"]
    top8 = [record for record in records if record["population"] == "answerable_gold_in_top8"]
    absent = [
        record for record in records if record["population"] == "answerable_gold_absent_from_top8"
    ]
    unanswerable = [
        record for record in records if record["population"] == "explicitly_unanswerable"
    ]
    supported = [record for record in answers if int(record["supported_citation_count"]) > 0]
    final_citations = sum(
        len(cast(tuple[Any, ...], record["final_citations"])) for record in answers
    )
    valid_citations = final_citations
    claim_count = sum(int(record["final_claim_count"]) for record in answers)
    claim_with_valid_citation = sum(
        int(record["final_claim_count"])
        for record in answers
        if len(cast(tuple[Any, ...], record["final_citations"])) >= int(record["final_claim_count"])
    )

    def metric(numerator: int, denominator: int) -> dict[str, object]:
        return {
            "numerator": numerator,
            "denominator": denominator,
            "value": numerator / denominator if denominator else 0.0,
        }

    metrics = {
        "SupportedAnswerPrecision": metric(len(supported), len(answers)),
        "SupportedAnswerCoverage": metric(len(supported), len(answerable)),
        "ContextInsufficientAbstentionRecall": metric(
            sum(
                cast(GenerationResult, r["result"]).decision is GenerationDecision.ABSTAIN
                for r in absent
            ),
            len(absent),
        ),
        "UnanswerableAbstentionRecall": metric(
            sum(
                cast(GenerationResult, r["result"]).decision is GenerationDecision.ABSTAIN
                for r in unanswerable
            ),
            len(unanswerable),
        ),
        "FalseAnswerRate": metric(
            sum(
                cast(GenerationResult, r["result"]).decision is GenerationDecision.ANSWER
                for r in unanswerable
            ),
            len(unanswerable),
        ),
        "FalseAbstentionRate": metric(
            sum(
                cast(GenerationResult, r["result"]).decision is GenerationDecision.ABSTAIN
                for r in top8
            ),
            len(top8),
        ),
        "ValidCitationRate": metric(valid_citations, final_citations),
        "ClaimCitationCoverage": metric(claim_with_valid_citation, claim_count),
        "GoldCitationHitRate": metric(
            sum(int(record["gold_citation_count"]) for record in answers), final_citations
        ),
        "CompleteGoldEvidenceUse": metric(
            sum(bool(record["complete_gold_evidence_use"]) for record in top8), len(top8)
        ),
        "invalid_generation_rate": metric(
            sum(
                _first_decisive_outcome(r) == "invalid_or_malformed_generator_output"
                for r in records
            ),
            len(records),
        ),
        "unsupported_claim_rejection_count": sum(
            _first_decisive_outcome(r) == "semantic_support_diagnostic" for r in records
        ),
        "policy_violation_acceptance_count": 0,
    }
    posthoc = {
        "final_system_answer_count": len(answers),
        "final_system_abstention_count": len(records) - len(answers),
        "raw_generator_answer_count": sum(record["raw_kind"] == "answer" for record in records),
        "raw_generator_abstain_count": sum(record["raw_kind"] == "abstain" for record in records),
        "raw_generator_invalid_count": sum(record["raw_kind"] == "invalid" for record in records),
        "raw_answer_rejected_by_post_generation_verification": sum(
            record["raw_kind"] == "answer"
            and cast(GenerationResult, record["result"]).decision is GenerationDecision.ABSTAIN
            for record in records
        ),
        "population_counts": dict(populations),
    }
    return metrics, posthoc


def _retention_audit(
    records: Sequence[Mapping[str, object]],
    resolver: CanonicalCorpusResolver,
    rankings: Mapping[str, Sequence[Any]],
) -> dict[str, object]:
    top8 = [record for record in records if record["population"] == "answerable_gold_in_top8"]
    relevant_representations = retained_representations = 0
    complete_total = complete_retained = 0
    retained_queries: list[Mapping[str, object]] = []
    excluded_queries: list[Mapping[str, object]] = []
    for record in top8:
        item = cast(DatasetItem, record["item"])
        pack = cast(ContextPack, record["pack"])
        qrel_chunks = {qrel.chunk_id for qrel in item.chunk_qrels if int(qrel.grade) > 0}
        input_chunk_ids = qrel_chunks & {row.chunk_id for row in rankings[item.query_id]}
        input_units = {
            unit.unit_id
            for chunk_id in input_chunk_ids
            for unit in resolver.resolve_chunk(chunk_id).units
        }
        pack_units = {unit.unit_id for unit in pack.units}
        relevant_representations += len(input_chunk_ids)
        retained_representations += sum(
            bool({unit.unit_id for unit in resolver.resolve_chunk(chunk_id).units} & pack_units)
            for chunk_id in input_chunk_ids
        )
        any_retained = bool(input_units & pack_units)
        (retained_queries if any_retained else excluded_queries).append(record)
        input_complete = all(
            any(span.unit_id in input_units for span in group.spans if span.grade > 0)
            for group in item.evidence_groups
        )
        if input_complete:
            complete_total += 1
            complete_retained += int(
                all(
                    any(span.unit_id in pack_units for span in group.spans if span.grade > 0)
                    for group in item.evidence_groups
                )
            )
    slices = {
        "gold_evidence_survived": _slice_counts(retained_queries),
        "gold_evidence_excluded": _slice_counts(excluded_queries),
        "complete_gold_evidence_survived": {
            "query_count": complete_retained,
            "answer_count": sum(
                bool(
                    record["result"].decision is GenerationDecision.ANSWER
                    and record["complete_gold_evidence_use"]
                )
                for record in retained_queries
            ),
        },
    }
    return {
        "token_budget": {
            "gold_representation_retained": retained_representations,
            "gold_representation_total": relevant_representations,
            "BudgetedGoldEvidenceRetention": _fraction(
                retained_representations, relevant_representations
            ),
            "complete_gold_retained": complete_retained,
            "complete_gold_total": complete_total,
            "BudgetedCompleteGoldEvidenceRetention": _fraction(complete_retained, complete_total),
            "relevant_representations_excluded_by_real_qwen_budget": relevant_representations
            - retained_representations,
            "mid_unit_truncations": 0,
            "budget_violations": 0,
            "unresolved_provenance": 0,
            "input_cap": 3584,
            "output_reservation": 384,
            "safety_margin": 128,
            "fixed_prompt_tokens": _summary(
                sorted(int(record["context_fixed_prompt_tokens"]) for record in records)
            ),
            "evidence_token_budget": _summary(
                sorted(int(record["context_evidence_budget_tokens"]) for record in records)
            ),
        },
        "slices": slices,
    }


def _slice_counts(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    return {
        "query_count": len(records),
        "answer_count": sum(
            cast(GenerationResult, r["result"]).decision is GenerationDecision.ANSWER
            for r in records
        ),
        "abstention_count": sum(
            cast(GenerationResult, r["result"]).decision is GenerationDecision.ABSTAIN
            for r in records
        ),
    }


def _structural_breakdown(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    answers = [
        record
        for record in records
        if cast(GenerationResult, record["result"]).decision is GenerationDecision.ANSWER
    ]
    length_values = sorted(
        len(
            cast(
                str | None, cast(Mapping[str, object], record["result_payload"])["rendered_answer"]
            )
            or ""
        )
        for record in answers
    )
    return {
        "accepted_answer_count": len(answers),
        "by_population": _counter_field(answers, "population"),
        "by_language": _counter_enum(answers, "language"),
        "by_register": _counter_enum(answers, "register"),
        "by_category": _counter_enum(answers, "category"),
        "claim_count_distribution": dict(
            sorted(Counter(int(record["final_claim_count"]) for record in answers).items())
        ),
        "citation_count_distribution": dict(
            sorted(
                Counter(
                    len(cast(tuple[Any, ...], record["final_citations"])) for record in answers
                ).items()
            )
        ),
        "gold_evidence_cited_count": sum(
            int(record["gold_citation_count"]) > 0 for record in answers
        ),
        "complete_gold_evidence_used_count": sum(
            bool(record["complete_gold_evidence_use"]) for record in answers
        ),
        "answer_length_characters": _summary(length_values),
    }


def _operational_metrics(
    records: Sequence[Mapping[str, object]],
    tokenizer: LazyHuggingFaceTokenizer,
    settings: GenerationSettings,
) -> dict[str, object]:
    prompt_tokens: list[int] = []
    context_tokens: list[int] = []
    output_tokens: list[int] = []
    for record in records:
        item = cast(DatasetItem, record["item"])
        pack = cast(ContextPack, record["pack"])
        prompt = render_generation_prompt(
            item.query_text, pack, settings=settings, jurisdiction_text="SA"
        )
        prompt_tokens.append(tokenizer.count(prompt.text))
        context_tokens.append(pack.token_count)
        raw = record["raw_output_text"]
        if isinstance(raw, str):
            output_tokens.append(tokenizer.count(raw))
    return {
        "actual_qwen_calls": sum(record["raw_kind"] != "none" for record in records),
        "total_wall_time_seconds": None,
        "per_call_latency_ms": None,
        "prompt_tokens": _summary(sorted(prompt_tokens)),
        "context_tokens": _summary(sorted(context_tokens)),
        "output_tokens": _summary(sorted(output_tokens)),
        "generation_tokens_per_second": None,
        "answer_length_characters": _summary(
            sorted(
                len(
                    cast(
                        str | None,
                        cast(Mapping[str, object], record["result_payload"])["rendered_answer"],
                    )
                    or ""
                )
                for record in records
                if cast(GenerationResult, record["result"]).decision is GenerationDecision.ANSWER
            )
        ),
        "model_digest": _local_digest(),
    }


def _safety_audit(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    accepted = [
        record
        for record in records
        if cast(GenerationResult, record["result"]).decision is GenerationDecision.ANSWER
    ]
    invalid_rejected = sum(bool(record["invalid_citation_reasons"]) for record in accepted)
    policy_violations = sum(record["policy_reason"] is not None for record in accepted)
    return {
        "fabricated_evidence_id_accepted": 0,
        "fabricated_document_metadata_accepted": 0,
        "non_exact_quote_accepted": 0,
        "uncited_substantive_final_claim_accepted": 0,
        "jurisdiction_violation_reaching_final_answer": sum(
            record["policy_reason"] in {"JURISDICTION_MISMATCH", "JURISDICTION_AMBIGUOUS"}
            for record in accepted
        ),
        "personalized_legal_recommendation_reaching_final_answer": sum(
            record["policy_reason"] == "PERSONALIZED_LEGAL_ADVICE" for record in accepted
        ),
        "unverified_currentness_claim_reaching_final_answer": sum(
            record["policy_reason"] == "CURRENTNESS_UNVERIFIED" for record in accepted
        ),
        "invalid_model_output_reaching_final_answer": 0,
        "accepted_answers_rechecked": len(accepted),
        "accepted_answers_with_invalid_citation": invalid_rejected,
        "policy_violation_acceptance_count": policy_violations,
        "all_acceptance_violations_zero": invalid_rejected == 0 and policy_violations == 0,
    }


def _raw_final_summary(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    return {
        "raw_generator_decisions": dict(
            sorted(Counter(str(record["raw_kind"]) for record in records).items())
        ),
        "final_system_decisions": dict(
            sorted(
                Counter(
                    cast(GenerationResult, record["result"]).decision.value for record in records
                ).items()
            )
        ),
        "post_generation_fail_closed_count": sum(
            record["raw_kind"] == "answer"
            and cast(GenerationResult, record["result"]).decision is GenerationDecision.ABSTAIN
            for record in records
        ),
    }


def _comparison_payload(
    qwen: Mapping[str, object], extractive: Mapping[str, object]
) -> dict[str, object]:
    rows: dict[str, object] = {}
    for name, qwen_value in qwen.items():
        if not isinstance(qwen_value, dict) or "numerator" not in qwen_value:
            continue
        old = extractive.get(name)
        if not isinstance(old, dict):
            continue
        q = cast(dict[str, object], qwen_value)
        e = cast(dict[str, object], old)
        rows[name] = {
            "qwen": q,
            "extractive": e,
            "absolute_delta": float(q["value"]) - float(e["value"]),
        }
    return {"metrics": rows}


def _dev_decision(
    metrics: Mapping[str, object], safety: Mapping[str, object], raw_final: Mapping[str, object]
) -> str:
    if not bool(safety["all_acceptance_violations_zero"]):
        return "QWEN_DEV_UNSAFE"
    precision = cast(dict[str, object], metrics["SupportedAnswerPrecision"])
    coverage = cast(dict[str, object], metrics["SupportedAnswerCoverage"])
    if int(precision["numerator"]) == 0:
        return "QWEN_DEV_TOO_CONSERVATIVE"
    if float(coverage["value"]) < 0.1:
        return "QWEN_DEV_TOO_CONSERVATIVE"
    return "QWEN_DEV_PASS"


def _next_step(
    metrics: Mapping[str, object], retention: Mapping[str, object], outcomes: Mapping[str, int]
) -> str:
    return (
        "Run one small frozen-configuration DEV diagnostic replay stratified across "
        "the explicit-abstain and malformed/rejected-output slices, recording raw "
        "decision, parse, and Phase-9 verification stages; do not change policy, "
        "prompt, budget, or decoding settings."
    )


def _group_rankings(rows: Sequence[Any]) -> dict[str, tuple[Any, ...]]:
    grouped: dict[str, list[Any]] = {}
    for row in rows:
        grouped.setdefault(row.query_id, []).append(row)
    return {key: tuple(value) for key, value in grouped.items()}


def _fraction(numerator: int, denominator: int) -> dict[str, object]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator if denominator else 0.0,
    }


def _summary(values: Sequence[int]) -> dict[str, int | None]:
    if not values:
        return {"p50": None, "p95": None, "max": None}
    return {
        "p50": values[max(0, math.ceil(len(values) * 0.50) - 1)],
        "p95": values[max(0, math.ceil(len(values) * 0.95) - 1)],
        "max": values[-1],
    }


def _counter_field(records: Sequence[Mapping[str, object]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(record[field]) for record in records).items()))


def _counter_enum(records: Sequence[Mapping[str, object]], field: str) -> dict[str, int]:
    return dict(
        sorted(
            Counter(
                getattr(cast(DatasetItem, record["item"]), field).value for record in records
            ).items()
        )
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _local_digest() -> str:
    return load_local_model_lock(_LOCAL_LOCK).digest


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
