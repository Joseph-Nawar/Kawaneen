"""Bounded, resumable Phase 11B runtime shared by the DEV orchestration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from kawaneen.extraction.annotation import AnnotationRecord
from kawaneen.extraction.artifacts import write_private_json
from kawaneen.extraction.checkpoints import (
    ExtractionAttempt,
    ExtractionCheckpoint,
    ExtractionCheckpointStore,
    extraction_fingerprint,
)
from kawaneen.extraction.contracts import CandidateType, ExtractionResult
from kawaneen.extraction.deterministic import run_deterministic
from kawaneen.extraction.hybrid import assemble_hybrid_result
from kawaneen.extraction.provider import ExtractionProvider


def _result_path(result_root: Path, record_id: str) -> Path:
    return result_root / f"{hashlib.sha256(record_id.encode('utf-8')).hexdigest()}.json"


def _base_result(record: AnnotationRecord) -> ExtractionResult:
    result = run_deterministic(
        record.canonical_text,
        canonical_unit_id=record.canonical_unit_id,
        document_id=record.document_id,
        source_provenance=record.source_provenance,
    )
    registry = record.candidate_registry
    return result.model_copy(
        update={
            "candidate_registry": registry,
            "referenced_articles": tuple(
                candidate
                for candidate in registry.candidates
                if candidate.candidate_type is CandidateType.ARTICLE
            ),
            "referenced_regulations": tuple(
                candidate
                for candidate in registry.candidates
                if candidate.candidate_type is CandidateType.REGULATION
            ),
        }
    )


def _checkpoint(
    *,
    record: AnnotationRecord,
    result_path: Path,
    fingerprint: str,
    prompt_hash: str,
    schema_hash: str,
    qwen_model: str,
    qwen_digest: str,
    tokenizer_revision: str,
    lifecycle_state: str,
    context_prepared: bool,
    extraction_attempted: bool,
    extraction_completed: bool,
    final_validation_completed: bool,
    failure_reason: str | None = None,
    failure_type: str | None = None,
    attempt_count: int = 1,
    prior_failure_type: str | None = None,
    retry_reason: str | None = None,
    attempt_history: tuple[ExtractionAttempt, ...] = (),
) -> ExtractionCheckpoint:
    return ExtractionCheckpoint(
        checkpoint_id=record.canonical_unit_id,
        record_id=record.canonical_unit_id,
        result_path=result_path.as_posix(),
        fingerprint=fingerprint,
        source_unit_hash=record.source_fingerprint,
        extractor_configuration="hybrid-qwen-v1",
        candidate_version="phase11-candidates-v3",
        prompt_hash=prompt_hash,
        schema_hash=schema_hash,
        qwen_model=qwen_model,
        qwen_digest=qwen_digest,
        tokenizer_revision=tokenizer_revision,
        semantic_validation_policy="phase11-span-v1",
        lifecycle_state=lifecycle_state,  # type: ignore[arg-type]
        failure_reason=failure_reason,
        failure_type=failure_type,
        attempt_count=attempt_count,
        prior_failure_type=prior_failure_type,
        retry_reason=retry_reason,
        attempt_history=attempt_history,
        context_prepared=context_prepared,
        extraction_attempted=extraction_attempted,
        extraction_completed=extraction_completed,
        final_validation_completed=final_validation_completed,
    )


def _failure_type(*, failure_reason: str | None, failure_type: str | None) -> str:
    if failure_type:
        return failure_type
    if failure_reason is not None and failure_reason.startswith("TimeoutError:"):
        return "MODEL_TIMEOUT"
    return "RUNTIME_FAILURE"


def _existing_attempt_history(existing: ExtractionCheckpoint) -> tuple[ExtractionAttempt, ...]:
    if existing.attempt_history:
        return existing.attempt_history
    return (
        ExtractionAttempt(
            attempt_number=existing.attempt_count,
            lifecycle_state=existing.lifecycle_state,
            failure_type=(
                _failure_type(
                    failure_reason=existing.failure_reason,
                    failure_type=existing.failure_type,
                )
                if existing.lifecycle_state == "failed"
                else None
            ),
            retry_reason=existing.retry_reason,
        ),
    )


def _preserve_attempt_result(
    *, result_path: Path, record_id: str, attempt_number: int, failure_type: str
) -> None:
    if not result_path.is_file():
        return
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    write_private_json(
        result_path.parent / "attempts" / record_id / f"attempt-{attempt_number}.json",
        {
            "artifact_type": "phase11_hybrid_attempt_result",
            "record_id": record_id,
            "attempt_number": attempt_number,
            "lifecycle_state": "failed",
            "failure_type": failure_type,
            "result": payload,
        },
    )


def run_hybrid_records(
    records: list[AnnotationRecord],
    provider: ExtractionProvider,
    *,
    checkpoint_root: Path,
    result_root: Path,
    selection_fingerprint: str,
    semantic_release_fingerprint: str,
    candidate_compatible_release_fingerprint: str,
    prompt_hash: str,
    schema_hash: str,
    qwen_model: str,
    qwen_digest: str,
    tokenizer_revision: str,
    resume: bool = False,
    retry_timeouts: bool = False,
    accept_field_local_diagnostics: bool = False,
    experiment_stage: str = "hybrid-qwen-v1",
) -> dict[str, object]:
    """Run each supplied DEV record once, with no automatic provider retries."""

    if retry_timeouts and not resume:
        raise ValueError("--retry-timeouts requires --resume")

    checkpoint_store = ExtractionCheckpointStore(checkpoint_root)
    completed = failed = skipped = 0
    provider_calls_attempted = raw_responses_received = timeouts = 0
    statuses: list[dict[str, str]] = []
    ordered = sorted(
        records,
        key=lambda record: (
            record.source_provenance.source_id,
            record.source_provenance.source_version,
            record.source_provenance.source_path,
            record.source_provenance.source_row,
            record.document_id,
            record.canonical_unit_id,
        ),
    )
    for record in ordered:
        result_path = _result_path(result_root, record.canonical_unit_id)
        fingerprint = extraction_fingerprint(
            source_unit_hash=record.source_fingerprint,
            extractor_configuration="hybrid-qwen-v1",
            candidate_version="phase11-candidates-v3",
            prompt_hash=prompt_hash,
            schema_hash=schema_hash,
            qwen_model=qwen_model,
            qwen_digest=qwen_digest,
            tokenizer_revision=tokenizer_revision,
            semantic_validation_policy="phase11-span-v1",
            selection_fingerprint=selection_fingerprint,
            semantic_release_fingerprint=semantic_release_fingerprint,
            candidate_compatible_release_fingerprint=candidate_compatible_release_fingerprint,
        )
        existing: ExtractionCheckpoint | None = None
        try:
            existing = checkpoint_store.load(record.canonical_unit_id)
        except (OSError, ValueError):
            existing = None
        retrying = False
        attempt_number = 1
        prior_failure_type: str | None = None
        retry_reason: str | None = None
        attempt_history: tuple[ExtractionAttempt, ...] = ()
        if resume and existing is not None and existing.fingerprint == fingerprint:
            if existing.lifecycle_state == "complete" and checkpoint_store.valid(
                record.canonical_unit_id, fingerprint
            ):
                completed += 0
                skipped += 1
                statuses.append(
                    {"record_id": record.canonical_unit_id, "status": "skipped_complete"}
                )
                continue
            if existing.lifecycle_state == "failed":
                existing_failure_type = _failure_type(
                    failure_reason=existing.failure_reason,
                    failure_type=existing.failure_type,
                )
                if (
                    retry_timeouts
                    and existing_failure_type == "MODEL_TIMEOUT"
                    and existing.attempt_count == 1
                ):
                    retrying = True
                    attempt_number = 2
                    prior_failure_type = existing_failure_type
                    retry_reason = "authorized_timeout_retry"
                    attempt_history = _existing_attempt_history(existing)
                else:
                    skipped += 1
                    statuses.append(
                        {"record_id": record.canonical_unit_id, "status": "skipped_failed"}
                    )
                    continue

        result_root.mkdir(parents=True, exist_ok=True)
        if retrying:
            _preserve_attempt_result(
                result_path=result_path,
                record_id=record.canonical_unit_id,
                attempt_number=1,
                failure_type="MODEL_TIMEOUT",
            )
        checkpoint_store.write(
            _checkpoint(
                record=record,
                result_path=result_path,
                fingerprint=fingerprint,
                prompt_hash=prompt_hash,
                schema_hash=schema_hash,
                qwen_model=qwen_model,
                qwen_digest=qwen_digest,
                tokenizer_revision=tokenizer_revision,
                lifecycle_state="incomplete",
                context_prepared=True,
                extraction_attempted=False,
                extraction_completed=False,
                final_validation_completed=False,
                attempt_count=attempt_number,
                prior_failure_type=prior_failure_type,
                retry_reason=retry_reason,
                attempt_history=attempt_history,
            )
        )
        try:
            checkpoint_store.write(
                _checkpoint(
                    record=record,
                    result_path=result_path,
                    fingerprint=fingerprint,
                    prompt_hash=prompt_hash,
                    schema_hash=schema_hash,
                    qwen_model=qwen_model,
                    qwen_digest=qwen_digest,
                    tokenizer_revision=tokenizer_revision,
                    lifecycle_state="incomplete",
                    context_prepared=True,
                    extraction_attempted=True,
                    extraction_completed=False,
                    final_validation_completed=False,
                    attempt_count=attempt_number,
                    prior_failure_type=prior_failure_type,
                    retry_reason=retry_reason,
                    attempt_history=attempt_history,
                )
            )
            provider_calls_attempted += 1
            raw_response = provider.propose(record.canonical_text, record.candidate_registry)
            raw_responses_received += 1
            assembled = assemble_hybrid_result(
                record.canonical_text, _base_result(record), raw_response
            )
            has_diagnostics = bool(assembled.validation_metadata.diagnostics)
            if (
                (has_diagnostics and not accept_field_local_diagnostics)
                or not assembled.validation_metadata.raw_provider_schema_valid
            ):
                reason = "; ".join(
                    f"{item.code}: {item.message}"
                    for item in assembled.validation_metadata.diagnostics
                ) or "provider proposal failed validation"
                write_private_json(
                    result_path,
                    {
                        "artifact_type": "phase11_extraction_result",
                        "lifecycle_state": "failed",
                        "status": "invalid",
                        "record_id": record.canonical_unit_id,
                        "extractor": "hybrid-qwen-v1",
                        "stage": experiment_stage,
                        "selection_fingerprint": selection_fingerprint,
                        "semantic_release_fingerprint": semantic_release_fingerprint,
                        "candidate_compatible_release_fingerprint": (
                            candidate_compatible_release_fingerprint
                        ),
                        "result": assembled.model_dump(mode="json"),
                        "raw_provider_response": raw_response,
                        "attempt_count": attempt_number,
                        "prior_failure_type": prior_failure_type,
                        "retry_reason": retry_reason,
                    },
                )
                current_failure_type = "PROVIDER_VALIDATION_FAILURE"
                checkpoint_store.write(
                    _checkpoint(
                        record=record,
                        result_path=result_path,
                        fingerprint=fingerprint,
                        prompt_hash=prompt_hash,
                        schema_hash=schema_hash,
                        qwen_model=qwen_model,
                        qwen_digest=qwen_digest,
                        tokenizer_revision=tokenizer_revision,
                        lifecycle_state="failed",
                        context_prepared=True,
                        extraction_attempted=True,
                        extraction_completed=False,
                        final_validation_completed=True,
                        failure_reason=reason,
                        failure_type=current_failure_type,
                        attempt_count=attempt_number,
                        prior_failure_type=prior_failure_type,
                        retry_reason=retry_reason,
                        attempt_history=(
                            *attempt_history,
                            ExtractionAttempt(
                                attempt_number=attempt_number,
                                lifecycle_state="failed",
                                failure_type=current_failure_type,
                                retry_reason=retry_reason,
                            ),
                        ),
                    )
                )
                failed += 1
                statuses.append({"record_id": record.canonical_unit_id, "status": "failed"})
                continue
            ExtractionResult.model_validate(assembled.model_dump(mode="json"))
            write_private_json(
                result_path,
                {
                    "artifact_type": "phase11_extraction_result",
                    "lifecycle_state": "complete",
                    "status": "complete",
                    "record_id": record.canonical_unit_id,
                    "extractor": "hybrid-qwen-v1",
                    "stage": experiment_stage,
                    "selection_fingerprint": selection_fingerprint,
                    "semantic_release_fingerprint": semantic_release_fingerprint,
                    "candidate_compatible_release_fingerprint": (
                        candidate_compatible_release_fingerprint
                    ),
                    "result": assembled.model_dump(mode="json"),
                    "raw_provider_response": raw_response,
                    "attempt_count": attempt_number,
                    "prior_failure_type": prior_failure_type,
                    "retry_reason": retry_reason,
                },
            )
            checkpoint_store.write(
                _checkpoint(
                    record=record,
                    result_path=result_path,
                    fingerprint=fingerprint,
                    prompt_hash=prompt_hash,
                    schema_hash=schema_hash,
                    qwen_model=qwen_model,
                    qwen_digest=qwen_digest,
                    tokenizer_revision=tokenizer_revision,
                    lifecycle_state="complete",
                    context_prepared=True,
                    extraction_attempted=True,
                    extraction_completed=True,
                    final_validation_completed=True,
                    attempt_count=attempt_number,
                    prior_failure_type=prior_failure_type,
                    retry_reason=retry_reason,
                    attempt_history=(
                        *attempt_history,
                        ExtractionAttempt(
                            attempt_number=attempt_number,
                            lifecycle_state="complete",
                            retry_reason=retry_reason,
                        ),
                    ),
                )
            )
            completed += 1
            statuses.append({"record_id": record.canonical_unit_id, "status": "complete"})
        except Exception as error:
            reason = f"{type(error).__name__}: {error}"
            if isinstance(error, TimeoutError):
                timeouts += 1
            current_failure_type = (
                "MODEL_TIMEOUT" if isinstance(error, TimeoutError) else "RUNTIME_FAILURE"
            )
            write_private_json(
                result_path,
                {
                    "artifact_type": "phase11_extraction_result",
                    "lifecycle_state": "failed",
                    "status": "provider_or_runtime_error",
                    "record_id": record.canonical_unit_id,
                    "extractor": "hybrid-qwen-v1",
                    "stage": experiment_stage,
                    "selection_fingerprint": selection_fingerprint,
                    "semantic_release_fingerprint": semantic_release_fingerprint,
                    "candidate_compatible_release_fingerprint": (
                        candidate_compatible_release_fingerprint
                    ),
                    "failure": reason,
                    "attempt_count": attempt_number,
                    "prior_failure_type": prior_failure_type,
                    "retry_reason": retry_reason,
                },
            )
            checkpoint_store.write(
                _checkpoint(
                    record=record,
                    result_path=result_path,
                    fingerprint=fingerprint,
                    prompt_hash=prompt_hash,
                    schema_hash=schema_hash,
                    qwen_model=qwen_model,
                    qwen_digest=qwen_digest,
                    tokenizer_revision=tokenizer_revision,
                    lifecycle_state="failed",
                    context_prepared=True,
                    extraction_attempted=True,
                    extraction_completed=False,
                    final_validation_completed=True,
                    failure_reason=reason,
                    failure_type=current_failure_type,
                    attempt_count=attempt_number,
                    prior_failure_type=prior_failure_type,
                    retry_reason=retry_reason,
                    attempt_history=(
                        *attempt_history,
                        ExtractionAttempt(
                            attempt_number=attempt_number,
                            lifecycle_state="failed",
                            failure_type=current_failure_type,
                            retry_reason=retry_reason,
                        ),
                    ),
                )
            )
            failed += 1
            statuses.append({"record_id": record.canonical_unit_id, "status": "failed"})
    return {
        "record_count": len(ordered),
        "completed": completed,
        "failed": failed,
        "skipped": skipped,
        "model_calls": raw_responses_received,
        "provider_calls_attempted": provider_calls_attempted,
        "raw_responses_received": raw_responses_received,
        "timeouts": timeouts,
        "statuses": statuses,
    }


def replay_persisted_raw_responses(
    records: list[AnnotationRecord],
    *,
    source_result_root: Path,
    output_root: Path,
    selection_fingerprint: str,
    semantic_release_fingerprint: str,
    candidate_compatible_release_fingerprint: str,
    prompt_hash: str,
    schema_hash: str,
) -> dict[str, object]:
    """Replay persisted Stage-B0 responses without invoking a provider."""

    if any(record.split != "dev" for record in records):
        raise ValueError("Stage B1 offline replay is DEV-only")
    if len(records) != 80:
        raise ValueError("Stage B1 offline replay requires exactly 80 DEV records")
    output_root.mkdir(parents=True, exist_ok=True)
    replayed = completed = failed = pending = 0
    dropped_candidate_refs = dropped_spans = corrected_occurrences = 0
    rejected_occurrences = dropped_action_rules = empty_results = 0
    statuses: list[dict[str, object]] = []
    for record in sorted(records, key=lambda item: item.canonical_unit_id):
        source_path = _result_path(source_result_root, record.canonical_unit_id)
        if not source_path.is_file():
            raise ValueError(
                f"Stage B0 result is missing for DEV record: {record.canonical_unit_id}"
            )
        source_payload = json.loads(source_path.read_text(encoding="utf-8"))
        raw_response = source_payload.get("raw_provider_response")
        if not isinstance(raw_response, str):
            pending += 1
            statuses.append(
                {
                    "record_id": record.canonical_unit_id,
                    "status": "MODEL_TIMEOUT_PENDING_RERUN",
                }
            )
            continue
        replayed += 1
        assembled = assemble_hybrid_result(
            record.canonical_text, _base_result(record), raw_response
        )
        diagnostics = assembled.validation_metadata.diagnostics
        dropped_candidate_refs += sum(
            diagnostic.code == "INVALID_CANDIDATE_REFERENCE" for diagnostic in diagnostics
        )
        dropped_spans += sum(
            diagnostic.code == "UNSUPPORTED_MODEL_SPAN" for diagnostic in diagnostics
        )
        corrected_occurrences += sum(
            diagnostic.code == "INVALID_OCCURRENCE_CORRECTED" for diagnostic in diagnostics
        )
        rejected_occurrences += sum(
            diagnostic.code == "AMBIGUOUS_OR_INVALID_OCCURRENCE" for diagnostic in diagnostics
        )
        dropped_action_rules += sum(
            diagnostic.field_name.endswith(".action")
            and diagnostic.code
            in {
                "UNSUPPORTED_MODEL_SPAN",
                "AMBIGUOUS_OR_INVALID_OCCURRENCE",
            }
            for diagnostic in diagnostics
        )
        if not assembled.validation_metadata.raw_provider_schema_valid:
            failed += 1
            status = "OUTPUT_TRUNCATED_OR_INVALID_JSON"
            statuses.append({"record_id": record.canonical_unit_id, "status": status})
            write_private_json(
                _result_path(output_root, record.canonical_unit_id),
                {
                    "artifact_type": "phase11_extraction_result",
                    "lifecycle_state": "failed",
                    "status": status,
                    "stage": "hybrid-qwen-v1-stage-b1",
                    "source_stage": "hybrid-qwen-v1-stage-b0",
                    "record_id": record.canonical_unit_id,
                    "selection_fingerprint": selection_fingerprint,
                    "semantic_release_fingerprint": semantic_release_fingerprint,
                    "candidate_compatible_release_fingerprint": (
                        candidate_compatible_release_fingerprint
                    ),
                    "result": assembled.model_dump(mode="json"),
                    "raw_provider_response": raw_response,
                },
            )
            continue
        ExtractionResult.model_validate(assembled.model_dump(mode="json"))
        semantic_empty = all(
            not getattr(assembled, field_name)
            for field_name in (
                "regulated_entities",
                "rules",
                "deadlines",
                "effective_dates",
                "penalties",
                "monetary_thresholds",
                "percentage_thresholds",
                "exceptions",
            )
        )
        if semantic_empty:
            empty_results += 1
        completed += 1
        statuses.append(
            {
                "record_id": record.canonical_unit_id,
                "status": "complete",
                "diagnostic_count": len(diagnostics),
                "semantic_empty": semantic_empty,
            }
        )
        write_private_json(
            _result_path(output_root, record.canonical_unit_id),
            {
                "artifact_type": "phase11_extraction_result",
                "lifecycle_state": "complete",
                "status": "complete",
                "stage": "hybrid-qwen-v1-stage-b1",
                "source_stage": "hybrid-qwen-v1-stage-b0",
                "record_id": record.canonical_unit_id,
                "selection_fingerprint": selection_fingerprint,
                "semantic_release_fingerprint": semantic_release_fingerprint,
                "candidate_compatible_release_fingerprint": (
                    candidate_compatible_release_fingerprint
                ),
                "result": assembled.model_dump(mode="json"),
                "raw_provider_response": raw_response,
            },
        )
    summary: dict[str, object] = {
        "schema_version": "phase11-hybrid-qwen-stage-b1-offline-replay-v1",
        "artifact_type": "phase11_hybrid_stage_b1_offline_replay",
        "stage": "hybrid-qwen-v1-stage-b1",
        "source_stage": "hybrid-qwen-v1-stage-b0",
        "split": "dev",
        "record_count": len(records),
        "raw_responses_replayed": replayed,
        "records_now_complete": completed,
        "records_still_invalid": failed,
        "timeouts_pending_rerun": pending,
        "invalid_candidate_references_dropped": dropped_candidate_refs,
        "unsupported_spans_dropped": dropped_spans,
        "unique_occurrence_corrections": corrected_occurrences,
        "ambiguous_occurrences_rejected": rejected_occurrences,
        "rules_dropped_action_invalid": dropped_action_rules,
        "valid_empty_results": empty_results,
        "selection_fingerprint": selection_fingerprint,
        "semantic_release_fingerprint": semantic_release_fingerprint,
        "candidate_compatible_release_fingerprint": candidate_compatible_release_fingerprint,
        "prompt_template_sha256": prompt_hash,
        "schema_sha256": schema_hash,
        "provider_calls_attempted": 0,
        "raw_responses_received": 0,
        "timeouts": 0,
        "statuses": statuses,
    }
    write_private_json(output_root / "replay_manifest.json", summary)
    return summary


__all__ = ["replay_persisted_raw_responses", "run_hybrid_records"]
