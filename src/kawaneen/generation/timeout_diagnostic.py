"""Bounded, private telemetry for the frozen Stage-B timeout cohort."""

from __future__ import annotations

import hashlib
import json
import socket
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from urllib.error import HTTPError, URLError

from pydantic import BaseModel, ConfigDict, Field

from kawaneen.generation.artifacts import artifact_fingerprint, write_text_free_artifact
from kawaneen.generation.contracts import (
    STAGE_B_GENERATION_SETTINGS,
    AbstentionReason,
    GenerationDecision,
    GenerationPayload,
    GenerationRequest,
    GenerationResult,
    ModelOutputCitation,
    ModelOutputClaim,
    TokenizerFingerprint,
    generation_payload_schema,
    parse_generation_payload,
)
from kawaneen.generation.ollama import (
    LOCAL_OLLAMA_LOCK_PATH,
    OllamaDiagnosticHTTPResponse,
    OllamaDiagnosticTransportError,
    OllamaGenerator,
    UrllibOllamaTransport,
    extract_ollama_response_text,
    load_local_model_lock,
)
from kawaneen.generation.orchestration import (
    PHASE9_POLICY,
    RUNTIME_ITEMS,
    STAGE_B_CHECKPOINT_ROOT,
    STAGE_B_CONTEXT_CACHE_ROOT,
    STAGE_B_GENERATOR_NAME,
    STAGE_B_RESULTS_ROOT,
    RuntimeQuery,
    generation_fingerprint,
    load_runtime_dev_queries,
)
from kawaneen.generation.postprocessing import finalize_generation
from kawaneen.generation.prompt import (
    STAGE_B_PROMPT_TEMPLATE_VERSION,
    render_stage_b_generation_prompt,
    stage_b_generation_version_hash,
)
from kawaneen.generation.registry import load_generation_lock
from kawaneen.generation.tokenizer import LazyHuggingFaceTokenizer
from kawaneen.grounding.contracts import ContextPack
from kawaneen.grounding.dev import CANONICAL_DOCUMENTS, CANONICAL_UNITS, CHUNKS, CORPUS_MANIFEST
from kawaneen.grounding.provenance import CanonicalCorpusResolver

TIMEOUT_DIAGNOSTIC_EXPECTED_COUNT = 27
TIMEOUT_DIAGNOSTIC_ROOT = (
    Path("artifacts/private/phase10_generation/diagnostics/qwen-stage-b-timeouts")
)
TIMEOUT_DIAGNOSTIC_RECORD_ROOT = TIMEOUT_DIAGNOSTIC_ROOT / "records"
TIMEOUT_DIAGNOSTIC_ENVELOPE_ROOT = TIMEOUT_DIAGNOSTIC_ROOT / "envelopes"
TIMEOUT_DIAGNOSTIC_MANIFEST = Path(
    "data/evaluation/phase10_qwen_stage_b_timeout_diagnostic.json"
)
TIMEOUT_DIAGNOSTIC_V2_ROOT = Path(
    "artifacts/private/phase10_generation/diagnostics/qwen-stage-b-timeouts-v2"
)
TIMEOUT_DIAGNOSTIC_V2_MANIFEST = Path(
    "data/evaluation/phase10_qwen_stage_b_timeout_diagnostic_v2.json"
)
TIMEOUT_DIAGNOSTIC_V2_COHORT_HASH = (
    "c692ab9ca2790dcabfb5e082e52e5d324555f56bc7c47b1cc6a69d0c1878c554"
)
TIMEOUT_DIAGNOSTIC_V2_PARSER_VERSION = "phase10-timeout-diagnostic-v2"
TIMEOUT_DIAGNOSTIC_V2_EXTRACTION_VERSION = "ollama-response-envelope-v1"
TIMEOUT_DIAGNOSTIC_OFFLINE_PARSER_VERSION = "phase10-timeout-offline-evaluator-v1"
_NATIVE_KEYS = (
    "done",
    "done_reason",
    "total_duration",
    "load_duration",
    "prompt_eval_count",
    "prompt_eval_duration",
    "eval_count",
    "eval_duration",
)


class TimeoutConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    http_client: str = "urllib.request"
    socket_timeout_seconds: float = Field(gt=0)
    connect_timeout_seconds: float | None = None
    read_timeout_seconds: float | None = None
    write_timeout_seconds: float | None = None
    pool_timeout_seconds: float | None = None
    overall_timeout_seconds: float | None = None
    retry_count: int = Field(ge=0)
    streaming: bool


class DiagnosticTimeoutError(TimeoutError):
    """A testable/future-proof phase marker for clients that expose timeout phases."""

    def __init__(self, phase: str, message: str = "timed out") -> None:
        if phase not in {"connect_timeout", "read_timeout", "write_timeout", "pool_timeout"}:
            raise ValueError("unsupported diagnostic timeout phase")
        super().__init__(message)
        self.phase = phase


def current_timeout_configuration() -> TimeoutConfiguration:
    """Describe the existing adapter without changing its timeout behavior."""

    return TimeoutConfiguration(
        socket_timeout_seconds=UrllibOllamaTransport().timeout_seconds,
        retry_count=0,
        streaming=False,
    )


class TimeoutDiagnosticResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    http_status: int | None = None
    raw_text: str = ""
    envelope_text: str | None = None
    native_metadata: dict[str, object] = {}


class TimeoutDiagnosticRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    query_id_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_tag: str = ""
    model_digest: str = ""
    hf_revision: str = ""
    tokenizer_id: str = ""
    tokenizer_revision: str = ""
    stage_b_prompt_hash: str = ""
    stage_b_schema_hash: str = ""
    stage_b_policy_hash: str = ""
    status: str
    failure_category: str | None = None
    exception_class: str | None = None
    exception_type: str | None = None
    elapsed_seconds: float = Field(ge=0)
    monotonic_request_start: float = Field(ge=0)
    response_received: bool
    http_status: int | None = None
    raw_output_present: bool
    response_byte_count: int = Field(ge=0)
    response_character_count: int = Field(ge=0)
    parsed_output_token_count: int | None = Field(default=None, ge=0)
    parse_outcome: str
    generator_decision: str | None = None
    final_verification_stage: str | None = None
    prompt_tokens: int = Field(ge=0)
    evidence_tokens: int = Field(ge=0)
    output_cap: int = Field(gt=0)
    timeout_configuration: TimeoutConfiguration
    native_metadata: dict[str, object] = {}


@dataclass(frozen=True, slots=True)
class FrozenTimeoutCohort:
    query_ids: tuple[str, ...]
    query_ids_hash: str
    non_timeout_records: int
    invalid_fingerprints: int
    raw_outputs_present: int


def classify_ollama_exception(error: BaseException, *, response_received: bool) -> str:
    """Classify transport failures without pretending urllib exposes phases."""

    if isinstance(error, OllamaDiagnosticTransportError):
        return classify_ollama_exception(
            error.original_error,
            response_received=error.response_received,
        )
    if isinstance(error, DiagnosticTimeoutError):
        return error.phase
    if isinstance(error, (TimeoutError, socket.timeout)):
        return "read_timeout" if response_received else "overall_timeout"
    if isinstance(error, HTTPError):
        return "http_error"
    if isinstance(error, URLError):
        reason = error.reason
        if isinstance(reason, BaseException) and reason is not error:
            nested = classify_ollama_exception(reason, response_received=response_received)
            if nested != "other":
                return nested
        return "connection_error"
    if isinstance(error, (ConnectionError, ConnectionResetError, ConnectionRefusedError)):
        return "connection_error"
    if isinstance(error, (json.JSONDecodeError, ValueError)):
        return "malformed_response"
    return "other"


def _query_hash(query_id: str) -> str:
    return artifact_fingerprint(query_id)


def _ids_hash(query_ids: tuple[str, ...]) -> str:
    return hashlib.sha256(("\n".join(query_ids) + "\n").encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stage_b_fingerprint(query_id: str, context_path: Path) -> str:
    payload = json.loads(context_path.read_text(encoding="utf-8"))
    context_pack = ContextPack.model_validate(payload["context_pack"])
    candidate, tokenizer = load_generation_lock()
    lock = load_local_model_lock(LOCAL_OLLAMA_LOCK_PATH)
    phase9_hash = _sha256_file(PHASE9_POLICY) if PHASE9_POLICY.is_file() else ""
    prompt_hash = artifact_fingerprint({"version": STAGE_B_PROMPT_TEMPLATE_VERSION})
    policy_hash = stage_b_generation_version_hash(STAGE_B_GENERATION_SETTINGS)
    return generation_fingerprint(
        query_id=query_id,
        context_pack=context_pack,
        model_revision=cast(str, candidate.hf_revision),
        ollama_digest=lock.digest,
        tokenizer_fingerprint=TokenizerFingerprint(
            identity=tokenizer.identity,
            revision=tokenizer.revision,
            vocabulary_hash=tokenizer.vocabulary_hash,
        ),
        prompt_template_hash=prompt_hash,
        generation_policy_hash=policy_hash,
        phase9_policy_hash=phase9_hash,
        settings=STAGE_B_GENERATION_SETTINGS,
        generator_name=STAGE_B_GENERATOR_NAME,
    )


def select_frozen_stage_b_timeout_cohort(
    *,
    results_root: Path = STAGE_B_RESULTS_ROOT,
    checkpoint_root: Path = STAGE_B_CHECKPOINT_ROOT,
    context_root: Path = STAGE_B_CONTEXT_CACHE_ROOT,
    runtime_items: Path = RUNTIME_ITEMS,
) -> FrozenTimeoutCohort:
    """Select only persisted Stage-B timeout/no-raw records and validate identity."""

    runtime_ids = {item.query_id for item in load_runtime_dev_queries(runtime_items)}
    result_paths = sorted(results_root.glob("*.json"))
    if len(result_paths) != 160:
        raise ValueError(
            f"frozen Stage-B results must contain 160 records, got {len(result_paths)}"
        )
    selected: list[str] = []
    non_timeout = 0
    invalid_fingerprints = 0
    raw_outputs_present = 0
    for path in result_paths:
        raw_record = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw_record, dict):
            raise ValueError("Stage-B result is not an object")
        record = cast(dict[str, object], raw_record)
        query_id = record.get("query_id")
        if not isinstance(query_id, str) or query_id not in runtime_ids:
            raise ValueError("Stage-B result contains a non-DEV query")
        expected = _stage_b_fingerprint(query_id, context_root / path.name)
        raw_checkpoint = json.loads((checkpoint_root / path.name).read_text(encoding="utf-8"))
        if not isinstance(raw_checkpoint, dict):
            raise ValueError("Stage-B checkpoint is not an object")
        checkpoint = cast(dict[str, object], raw_checkpoint)
        if record.get("fingerprint") != expected or checkpoint.get("fingerprint") != expected:
            invalid_fingerprints += 1
        result = record.get("result")
        result_mapping = cast(dict[str, object], result) if isinstance(result, dict) else None
        is_timeout = (
            result_mapping is not None
            and result_mapping.get("detail") == "timed out"
            and record.get("raw_output") is None
        )
        if is_timeout:
            selected.append(query_id)
            if record.get("raw_output") is not None:
                raw_outputs_present += 1
        else:
            non_timeout += 1
    ordered = tuple(sorted(selected))
    if len(ordered) != TIMEOUT_DIAGNOSTIC_EXPECTED_COUNT:
        raise ValueError(
            f"frozen Stage-B timeout cohort must contain 27 records, got {len(ordered)}"
        )
    if invalid_fingerprints:
        raise ValueError("frozen Stage-B timeout cohort contains fingerprint mismatches")
    return FrozenTimeoutCohort(
        query_ids=ordered,
        query_ids_hash=_ids_hash(ordered),
        non_timeout_records=non_timeout,
        invalid_fingerprints=invalid_fingerprints,
        raw_outputs_present=raw_outputs_present,
    )


def _parse_response(response: TimeoutDiagnosticResponse) -> tuple[str, str | None]:
    try:
        payload = parse_generation_payload(response.raw_text)
    except (TypeError, ValueError):
        return "invalid_pydantic_or_json", None
    return "valid_pydantic", payload.decision.value


def write_diagnostic_record(
    *,
    query_id: str,
    request_fingerprint: str,
    timeout_configuration: TimeoutConfiguration,
    elapsed_seconds: float,
    prompt_tokens: int,
    evidence_tokens: int,
    response: TimeoutDiagnosticResponse | None = None,
    error: BaseException | None = None,
    response_received: bool = False,
    monotonic_request_start: float | None = None,
    final_verification_stage: str | None = None,
    model_tag: str = "",
    model_digest: str = "",
    hf_revision: str = "",
    tokenizer_id: str = "",
    tokenizer_revision: str = "",
    stage_b_prompt_hash: str = "",
    stage_b_schema_hash: str = "",
    stage_b_policy_hash: str = "",
) -> TimeoutDiagnosticRecord:
    if response is not None and error is not None:
        raise ValueError("diagnostic record cannot contain both response and error")
    start = time.monotonic() if monotonic_request_start is None else monotonic_request_start
    native = (
        {
            key: response.native_metadata[key]
            for key in _NATIVE_KEYS
            if key in response.native_metadata
        }
        if response is not None
        else {}
    )
    if response is not None:
        parse_outcome, decision = _parse_response(response)
        envelope_text = response.envelope_text or response.raw_text
        eval_count = response.native_metadata.get("eval_count")
        parsed_output_tokens = eval_count if isinstance(eval_count, int) else None
        return TimeoutDiagnosticRecord(
            query_id_hash=_query_hash(query_id),
            request_fingerprint=request_fingerprint,
            model_tag=model_tag,
            model_digest=model_digest,
            hf_revision=hf_revision,
            tokenizer_id=tokenizer_id,
            tokenizer_revision=tokenizer_revision,
            stage_b_prompt_hash=stage_b_prompt_hash,
            stage_b_schema_hash=stage_b_schema_hash,
            stage_b_policy_hash=stage_b_policy_hash,
            status="success" if parse_outcome == "valid_pydantic" else "failure",
            failure_category=None if parse_outcome == "valid_pydantic" else "malformed_response",
            elapsed_seconds=elapsed_seconds,
            monotonic_request_start=start,
            response_received=True,
            http_status=response.http_status,
            raw_output_present=bool(response.raw_text),
            response_byte_count=len(envelope_text.encode("utf-8")),
            response_character_count=len(envelope_text),
            parsed_output_token_count=parsed_output_tokens,
            parse_outcome=parse_outcome,
            generator_decision=decision,
            final_verification_stage=final_verification_stage,
            prompt_tokens=prompt_tokens,
            evidence_tokens=evidence_tokens,
            output_cap=STAGE_B_GENERATION_SETTINGS.max_new_tokens,
            timeout_configuration=timeout_configuration,
            native_metadata=native,
        )
    if error is None:
        raise ValueError("diagnostic record requires a response or error")
    category = classify_ollama_exception(error, response_received=response_received)
    original = (
        error.original_error if isinstance(error, OllamaDiagnosticTransportError) else error
    )
    raw_text = error.raw_text if isinstance(error, OllamaDiagnosticTransportError) else None
    return TimeoutDiagnosticRecord(
        query_id_hash=_query_hash(query_id),
        request_fingerprint=request_fingerprint,
        model_tag=model_tag,
        model_digest=model_digest,
        hf_revision=hf_revision,
        tokenizer_id=tokenizer_id,
        tokenizer_revision=tokenizer_revision,
        stage_b_prompt_hash=stage_b_prompt_hash,
        stage_b_schema_hash=stage_b_schema_hash,
        stage_b_policy_hash=stage_b_policy_hash,
        status="failure",
        failure_category=category,
        exception_class=type(original).__name__,
        exception_type=f"{type(original).__module__}.{type(original).__name__}",
        elapsed_seconds=elapsed_seconds,
        monotonic_request_start=start,
        response_received=response_received,
        http_status=(
            error.http_status if isinstance(error, OllamaDiagnosticTransportError) else None
        ),
        raw_output_present=bool(raw_text),
        response_byte_count=len(raw_text.encode("utf-8")) if raw_text else 0,
        response_character_count=len(raw_text) if raw_text else 0,
        parse_outcome="not_parsed",
        prompt_tokens=prompt_tokens,
        evidence_tokens=evidence_tokens,
        output_cap=STAGE_B_GENERATION_SETTINGS.max_new_tokens,
        timeout_configuration=timeout_configuration,
        native_metadata={},
    )


def _record_path(root: Path, query_id: str) -> Path:
    if not query_id or Path(query_id).name != query_id:
        raise ValueError("unsafe diagnostic query ID")
    return root / "records" / f"{query_id}.json"


def _write_private_record(path: Path, record: TimeoutDiagnosticRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(record.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_private_envelope(root: Path, query_id: str, envelope_text: str) -> None:
    path = root / "envelopes" / f"{query_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if path.read_text(encoding="utf-8") != envelope_text:
            raise ValueError("immutable diagnostic envelope differs")
        return
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(envelope_text, encoding="utf-8")
    temporary.replace(path)


def run_diagnostic_cases(
    *,
    query_ids: tuple[str, ...],
    runner: Any,
    root: Path,
    resume: bool,
) -> int:
    """Atomically persist injected diagnostic cases; production supplies the Ollama runner."""

    completed = 0
    for query_id in query_ids:
        path = _record_path(root, query_id)
        if resume and path.is_file():
            try:
                TimeoutDiagnosticRecord.model_validate(
                    json.loads(path.read_text(encoding="utf-8"))
                )
                completed += 1
                continue
            except (OSError, json.JSONDecodeError, ValueError):
                pass
        record = runner(query_id)
        if not isinstance(record, TimeoutDiagnosticRecord):
            raise TypeError("diagnostic runner returned an invalid record")
        _write_private_record(path, record)
        completed += 1
    return completed


def timeout_diagnostic_status(*, root: Path = TIMEOUT_DIAGNOSTIC_ROOT) -> dict[str, object]:
    """Read only diagnostic JSON; never loads model, source, or corpus."""

    completed = 0
    corrupt = 0
    success = 0
    other_failure = 0
    by_subtype: dict[str, int] = {}
    records_root = root / "records"
    for path in sorted(records_root.glob("*.json")) if records_root.is_dir() else ():
        try:
            record = TimeoutDiagnosticRecord.model_validate(
                json.loads(path.read_text(encoding="utf-8"))
            )
        except (OSError, json.JSONDecodeError, ValueError):
            corrupt += 1
            continue
        completed += 1
        if record.status == "success":
            success += 1
        elif record.failure_category in {
            "connect_timeout",
            "read_timeout",
            "write_timeout",
            "pool_timeout",
            "overall_timeout",
        }:
            category = record.failure_category
            by_subtype[category] = by_subtype.get(category, 0) + 1
        else:
            other_failure += 1
    return {
        "expected": TIMEOUT_DIAGNOSTIC_EXPECTED_COUNT,
        "completed": completed,
        "success": success,
        "timeout_by_subtype": dict(sorted(by_subtype.items())),
        "other_failure": other_failure,
        "corrupt": corrupt,
        "missing": max(TIMEOUT_DIAGNOSTIC_EXPECTED_COUNT - completed - corrupt, 0),
    }


def timeout_diagnostic_v2_status(
    *, root: Path = TIMEOUT_DIAGNOSTIC_V2_ROOT
) -> dict[str, object]:
    """Read only the versioned v2 namespace; never inspect v1 records."""

    return timeout_diagnostic_status(root=root)


def _generation_result_from_payload(payload: object) -> GenerationResult:
    parsed = cast(GenerationPayload, payload)
    return GenerationResult(
        decision=parsed.decision,
        claims=tuple(
            ModelOutputClaim(
                mode=claim.mode,
                text=getattr(claim, "text", None),
                citations=tuple(
                    ModelOutputCitation(
                        evidence_id=citation.evidence_id,
                        quoted_text=citation.quoted_text,
                    )
                    for citation in claim.citations
                ),
            )
            for claim in parsed.claims
        ),
    )


def _write_corrected_record(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def evaluate_persisted_timeout_diagnostic(
    *,
    root: Path = TIMEOUT_DIAGNOSTIC_ROOT,
    expected_query_ids: tuple[str, ...] | None = None,
    parser_version: str = TIMEOUT_DIAGNOSTIC_OFFLINE_PARSER_VERSION,
    response_extraction_version: str = TIMEOUT_DIAGNOSTIC_V2_EXTRACTION_VERSION,
) -> dict[str, object]:
    """Re-evaluate saved HTTP envelopes without network, model, or tokenizer access."""

    records_root = root / "records"
    envelopes_root = root / "envelopes"
    record_paths = {path.stem: path for path in records_root.glob("*.json")}
    if expected_query_ids is None:
        expected = set(select_frozen_stage_b_timeout_cohort().query_ids)
    else:
        expected = set(expected_query_ids)
    envelope_paths = {path.stem: path for path in envelopes_root.glob("*.json")}
    missing = sorted(expected - set(envelope_paths))
    if missing:
        return {
            "status": "insufficient_evidence",
            "expected": len(expected),
            "envelopes_available": len(expected) - len(missing),
            "missing_envelopes": len(missing),
            "http_requests": 0,
            "model_calls": 0,
            "reason": "raw HTTP envelopes were not persisted",
        }

    counts = {
        "valid_generated_responses": 0,
        "explicit_abstentions": 0,
        "raw_answer_decisions": 0,
        "syntactically_invalid_json": 0,
        "schema_pydantic_invalid": 0,
        "citation_quotation_failures": 0,
        "semantic_support_removals": 0,
        "final_verified_answers": 0,
        "other_failures": 0,
    }
    termination: dict[str, dict[str, int]] = {}
    final_termination: dict[str, dict[str, int]] = {}
    provenance: list[dict[str, object]] = []
    telemetry_records: list[TimeoutDiagnosticRecord] = []
    query_map: dict[str, RuntimeQuery] = {}
    resolver: CanonicalCorpusResolver | None = None
    try:
        query_map = {item.query_id: item for item in load_runtime_dev_queries()}
    except (OSError, ValueError):
        query_map = {}

    for query_id in sorted(expected):
        envelope_path = envelope_paths[query_id]
        record_path = record_paths.get(query_id)
        if record_path is None:
            counts["other_failures"] += 1
            continue
        envelope_bytes = envelope_path.read_bytes()
        original_record_bytes = record_path.read_bytes()
        record = TimeoutDiagnosticRecord.model_validate(json.loads(original_record_bytes))
        telemetry_records.append(record)
        envelope = json.loads(envelope_bytes.decode("utf-8"))
        final_outcome = "not_evaluated"
        if not isinstance(envelope, dict):
            counts["schema_pydantic_invalid"] += 1
            outcome = "schema_pydantic_invalid"
            done_reason = None
        else:
            envelope_mapping = cast(dict[str, object], envelope)
            done_reason_value = envelope_mapping.get("done_reason")
            done_reason = done_reason_value if isinstance(done_reason_value, str) else None
            try:
                nested = extract_ollama_response_text(envelope_mapping)
                try:
                    json.loads(nested)
                except json.JSONDecodeError:
                    counts["syntactically_invalid_json"] += 1
                    outcome = "syntactically_invalid_json"
                else:
                    try:
                        payload = parse_generation_payload(nested)
                    except (TypeError, ValueError):
                        counts["schema_pydantic_invalid"] += 1
                        outcome = "schema_pydantic_invalid"
                    else:
                        counts["valid_generated_responses"] += 1
                        if payload.decision is GenerationDecision.ABSTAIN:
                            counts["explicit_abstentions"] += 1
                            outcome = "explicit_abstention"
                            final_outcome = "final_abstain"
                        else:
                            counts["raw_answer_decisions"] += 1
                            outcome = "raw_answer"
                            context_path = (
                                STAGE_B_CONTEXT_CACHE_ROOT / f"{query_id}.json"
                            )
                            if query_id in query_map and context_path.is_file():
                                if resolver is None:
                                    resolver = CanonicalCorpusResolver.from_json(
                                        CANONICAL_UNITS,
                                        CHUNKS,
                                        CORPUS_MANIFEST,
                                        document_paths=CANONICAL_DOCUMENTS,
                                    )
                                context = json.loads(context_path.read_text(encoding="utf-8"))
                                pack = ContextPack.model_validate(context["context_pack"])
                                finalized = finalize_generation(
                                    pack,
                                    _generation_result_from_payload(payload),
                                    resolver,
                                    jurisdiction_text="SA",
                                )
                                if finalized.result.decision is GenerationDecision.ANSWER:
                                    counts["final_verified_answers"] += 1
                                    outcome = "final_verified_answer"
                                    final_outcome = "final_verified_answer"
                                elif (
                                    finalized.result.abstention_reason
                                    is AbstentionReason.SEMANTIC_SUPPORT_UNAVAILABLE
                                ):
                                    counts["semantic_support_removals"] += 1
                                    outcome = "semantic_support_removal"
                                    final_outcome = "final_abstain"
                                elif (
                                    finalized.verification is not None
                                    and not finalized.verification.structurally_valid
                                ):
                                    counts["citation_quotation_failures"] += 1
                                    outcome = "citation_quotation_failure"
                                    final_outcome = "final_abstain"
                                else:
                                    counts["other_failures"] += 1
                                    outcome = "other_failure"
                                    final_outcome = "final_abstain"
            except (TypeError, ValueError, json.JSONDecodeError):
                counts["schema_pydantic_invalid"] += 1
                outcome = "schema_pydantic_invalid"

        if done_reason is not None:
            by_reason = termination.setdefault(done_reason, {})
            by_reason[outcome] = by_reason.get(outcome, 0) + 1
            final_by_reason = final_termination.setdefault(done_reason, {})
            final_by_reason[final_outcome] = final_by_reason.get(final_outcome, 0) + 1
        provenance.append(
            {
                "query_id_hash": record.query_id_hash,
                "original_record_sha256": hashlib.sha256(original_record_bytes).hexdigest(),
                "original_envelope_sha256": hashlib.sha256(envelope_bytes).hexdigest(),
                "parser_version": parser_version,
                "response_extraction_version": response_extraction_version,
                "done_reason": done_reason,
                "outcome": outcome,
                "final_outcome": final_outcome,
            }
        )

    corrected_root = root / "corrected-records"
    for item in provenance:
        query_hash = cast(str, item["query_id_hash"])
        _write_corrected_record(corrected_root / f"{query_hash}.json", item)
    elapsed = [item.elapsed_seconds for item in telemetry_records]
    load_duration = [
        value
        for item in telemetry_records
        for value in [item.native_metadata.get("load_duration")]
        if isinstance(value, (int, float))
    ]
    eval_duration = [
        value
        for item in telemetry_records
        for value in [item.native_metadata.get("eval_duration")]
        if isinstance(value, (int, float))
    ]
    output_cap_hits = sum(
        item.parsed_output_token_count == item.output_cap for item in telemetry_records
    )
    transport_successes = sum(
        item.http_status is not None
        and 200 <= item.http_status < 300
        and item.exception_class is None
        for item in telemetry_records
    )
    http_errors = sum(
        item.failure_category == "http_error"
        or (item.http_status is not None and item.http_status >= 400)
        for item in telemetry_records
    )
    transport_failures = sum(
        item.exception_class is not None
        or item.failure_category
        in {
            "connect_timeout",
            "read_timeout",
            "write_timeout",
            "pool_timeout",
            "overall_timeout",
            "connection_error",
            "http_error",
            "other",
        }
        for item in telemetry_records
    )
    return {
        "status": "complete",
        "expected": len(expected),
        "envelopes_available": len(expected),
        "missing_envelopes": 0,
        **counts,
        "done_reason": termination,
        "done_reason_final_verification": final_termination,
        "transport": {
            "success": transport_successes,
            "timeout_by_subtype": {
                category: sum(item.failure_category == category for item in telemetry_records)
                for category in (
                    "connect_timeout",
                    "read_timeout",
                    "write_timeout",
                    "pool_timeout",
                    "overall_timeout",
                )
                if any(item.failure_category == category for item in telemetry_records)
            },
            "other_failures": transport_failures - http_errors - sum(
                item.failure_category
                in {
                    "connect_timeout",
                    "read_timeout",
                    "write_timeout",
                    "pool_timeout",
                    "overall_timeout",
                }
                for item in telemetry_records
            ),
            "http_errors": http_errors,
            "elapsed_seconds": _distribution(elapsed),
            "crossed_30_second_boundary": sum(value >= 30 for value in elapsed),
            "approached_30_second_boundary": sum(27 <= value < 30 for value in elapsed),
            "load_duration": _distribution(load_duration),
            "eval_duration": _distribution(eval_duration),
        },
        "generation": {
            "outputs_at_512": output_cap_hits,
        },
        "corrected_record_provenance": provenance,
        "http_requests": 0,
        "model_calls": 0,
    }


def evaluate_persisted_timeout_diagnostic_v2(
    *, root: Path = TIMEOUT_DIAGNOSTIC_V2_ROOT
) -> dict[str, object]:
    """Evaluate only v2 envelopes with versioned provenance."""

    cohort = select_frozen_stage_b_timeout_cohort()
    _assert_v2_cohort(cohort)
    return evaluate_persisted_timeout_diagnostic(
        root=root,
        expected_query_ids=cohort.query_ids,
        parser_version=TIMEOUT_DIAGNOSTIC_V2_PARSER_VERSION,
        response_extraction_version=TIMEOUT_DIAGNOSTIC_V2_EXTRACTION_VERSION,
    )


def _distribution(values: Sequence[float | int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "p50": None, "p95": None, "max": None}
    ordered = sorted(float(value) for value in values)

    def percentile(fraction: float) -> float:
        index = (len(ordered) - 1) * fraction
        lower = int(index)
        upper = min(lower + 1, len(ordered) - 1)
        weight = index - lower
        return ordered[lower] + (ordered[upper] - ordered[lower]) * weight

    return {
        "count": len(ordered),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": ordered[-1],
    }


def evaluate_stage_b_timeout_diagnostic(
    *,
    root: Path = TIMEOUT_DIAGNOSTIC_ROOT,
) -> dict[str, object]:
    """Evaluate diagnostic telemetry without replacing any Stage-B benchmark result."""

    records: list[TimeoutDiagnosticRecord] = []
    records_root = root / "records"
    for path in sorted(records_root.glob("*.json")) if records_root.is_dir() else ():
        try:
            records.append(
                TimeoutDiagnosticRecord.model_validate(
                    json.loads(path.read_text(encoding="utf-8"))
                )
            )
        except (OSError, json.JSONDecodeError, ValueError):
            continue
    failures = [item for item in records if item.status != "success"]
    successes = [item for item in records if item.status == "success"]
    timeout_counts: dict[str, int] = {}
    for item in failures:
        if item.failure_category is not None:
            timeout_counts[item.failure_category] = timeout_counts.get(item.failure_category, 0) + 1
    timeout_elapsed = [item.elapsed_seconds for item in failures]
    boundary = [
        item
        for item in failures
        if abs(item.elapsed_seconds - current_timeout_configuration().socket_timeout_seconds) <= 1.0
    ]
    prompt_distributions: dict[str, object] = {
        "original_timeout_cases": {"prompt_tokens": None, "evidence_tokens": None},
        "original_successful_stage_b_calls": {
            "prompt_tokens": None,
            "evidence_tokens": None,
        },
    }
    try:
        queries = {item.query_id: item for item in load_runtime_dev_queries()}
        _, tokenizer_fingerprint = load_generation_lock()
        tokenizer = LazyHuggingFaceTokenizer(
            identity=tokenizer_fingerprint.identity,
            revision=cast(str, tokenizer_fingerprint.revision),
        )
        original_timeout_ids = set(select_frozen_stage_b_timeout_cohort().query_ids)
        original_success_ids = {
            path.stem
            for path in STAGE_B_RESULTS_ROOT.glob("*.json")
            if json.loads(path.read_text(encoding="utf-8")).get("raw_output") is not None
        }
        distributions: dict[str, list[int]] = {
            "timeout_prompt": [],
            "timeout_evidence": [],
            "success_prompt": [],
            "success_evidence": [],
        }
        for query_id in original_timeout_ids | original_success_ids:
            context_path = STAGE_B_CONTEXT_CACHE_ROOT / f"{query_id}.json"
            if query_id not in queries or not context_path.is_file():
                continue
            context = json.loads(context_path.read_text(encoding="utf-8"))
            pack = ContextPack.model_validate(context["context_pack"])
            prompt = render_stage_b_generation_prompt(
                queries[query_id].query,
                pack,
                settings=STAGE_B_GENERATION_SETTINGS,
                jurisdiction_text="SA",
            )
            prompt_count = tokenizer.count(prompt.text)
            if query_id in original_timeout_ids:
                distributions["timeout_prompt"].append(prompt_count)
                distributions["timeout_evidence"].append(pack.token_count)
            if query_id in original_success_ids:
                distributions["success_prompt"].append(prompt_count)
                distributions["success_evidence"].append(pack.token_count)
        prompt_distributions = {
            "original_timeout_cases": {
                "prompt_tokens": _distribution(distributions["timeout_prompt"]),
                "evidence_tokens": _distribution(distributions["timeout_evidence"]),
            },
            "original_successful_stage_b_calls": {
                "prompt_tokens": _distribution(distributions["success_prompt"]),
                "evidence_tokens": _distribution(distributions["success_evidence"]),
            },
        }
    except (OSError, KeyError, ValueError):
        pass
    native_load = [
        item.native_metadata["load_duration"]
        for item in records
        if isinstance(item.native_metadata.get("load_duration"), (int, float))
    ]
    native_eval = [
        item.native_metadata["eval_duration"]
        for item in records
        if isinstance(item.native_metadata.get("eval_duration"), (int, float))
    ]
    output_tokens = [
        item.parsed_output_token_count
        for item in successes
        if item.parsed_output_token_count is not None
    ]
    if not records:
        diagnosis = "INSUFFICIENT_EVIDENCE"
    elif successes and failures:
        diagnosis = "INTERMITTENT_RUNTIME_FAILURE"
    elif boundary and len(boundary) == len(failures):
        diagnosis = "CLIENT_TIMEOUT_TOO_SHORT"
    elif timeout_counts.get("connection_error", 0) == len(failures):
        diagnosis = "NETWORK_OR_CONNECTION_FAILURE"
    else:
        diagnosis = "OLLAMA_RUNTIME_STALL"
    return {
        "status": "complete" if len(records) == TIMEOUT_DIAGNOSTIC_EXPECTED_COUNT else "incomplete",
        "expected": TIMEOUT_DIAGNOSTIC_EXPECTED_COUNT,
        "completed": len(records),
        "successes": len(successes),
        "repeated_failures": len(failures),
        "failure_by_subtype": dict(sorted(timeout_counts.items())),
        "elapsed_seconds": _distribution(timeout_elapsed),
        "failures_at_timeout_boundary": len(boundary),
        "timeout_boundary_tolerance_seconds": 1.0,
        "prompt_and_evidence_distributions": prompt_distributions,
        "successful_output_tokens": _distribution(output_tokens),
        "load_duration": _distribution(cast(list[float | int], native_load)),
        "eval_duration": _distribution(cast(list[float | int], native_eval)),
        "runtime_pattern": (
            "intermittent"
            if successes and failures
            else "deterministic_failure"
            if failures
            else "successful_replay"
        ),
        "diagnosis": diagnosis,
    }


def write_timeout_diagnostic_manifest(
    cohort: FrozenTimeoutCohort | None = None,
    *,
    path: Path = TIMEOUT_DIAGNOSTIC_MANIFEST,
) -> None:
    cohort = cohort or select_frozen_stage_b_timeout_cohort()
    candidate, tokenizer = load_generation_lock()
    lock = load_local_model_lock(LOCAL_OLLAMA_LOCK_PATH)
    write_text_free_artifact(
        path,
        {
            "schema_version": 1,
            "status": "prepared_not_executed",
            "cohort_count": len(cohort.query_ids),
            "cohort_ids_hash": cohort.query_ids_hash,
            "non_timeout_records": cohort.non_timeout_records,
            "frozen_fingerprint_mismatches": cohort.invalid_fingerprints,
            "timeout_configuration": current_timeout_configuration().model_dump(mode="json"),
            "model_tag": lock.model,
            "model_digest": lock.digest,
            "hf_revision": candidate.hf_revision,
            "tokenizer_id": tokenizer.identity,
            "tokenizer_revision": tokenizer.revision,
            "stage_b_template_hash": artifact_fingerprint(
                {"version": STAGE_B_PROMPT_TEMPLATE_VERSION}
            ),
            "stage_b_policy_hash": stage_b_generation_version_hash(STAGE_B_GENERATION_SETTINGS),
            "stage_b_results_hash": artifact_fingerprint(
                sorted(
                    path.name
                    for path in STAGE_B_RESULTS_ROOT.glob("*.json")
                )
            ),
        },
    )


def _assert_v2_cohort(cohort: FrozenTimeoutCohort) -> None:
    if len(cohort.query_ids) != TIMEOUT_DIAGNOSTIC_EXPECTED_COUNT:
        raise ValueError("v2 cohort hash/count mismatch")
    if cohort.query_ids_hash != TIMEOUT_DIAGNOSTIC_V2_COHORT_HASH:
        raise ValueError("v2 cohort hash/count mismatch")
    if cohort.invalid_fingerprints != 0 or cohort.raw_outputs_present != 0:
        raise ValueError("v2 cohort contains invalid or non-timeout records")


def write_timeout_diagnostic_v2_manifest(
    cohort: FrozenTimeoutCohort | None = None,
    *,
    path: Path = TIMEOUT_DIAGNOSTIC_V2_MANIFEST,
) -> None:
    """Write the text-free, not-yet-executed v2 diagnostic manifest."""

    frozen = cohort or select_frozen_stage_b_timeout_cohort()
    _assert_v2_cohort(frozen)
    candidate, tokenizer = load_generation_lock()
    lock = load_local_model_lock(LOCAL_OLLAMA_LOCK_PATH)
    write_text_free_artifact(
        path,
        {
            "schema_version": 2,
            "status": "prepared_not_executed",
            "namespace": TIMEOUT_DIAGNOSTIC_V2_ROOT.as_posix(),
            "cohort_count": len(frozen.query_ids),
            "cohort_ids_hash": frozen.query_ids_hash,
            "timeout_configuration": current_timeout_configuration().model_dump(mode="json"),
            "model_tag": lock.model,
            "model_digest": lock.digest,
            "hf_revision": candidate.hf_revision,
            "tokenizer_id": tokenizer.identity,
            "tokenizer_revision": tokenizer.revision,
            "response_extraction_version": TIMEOUT_DIAGNOSTIC_V2_EXTRACTION_VERSION,
            "parser_version": TIMEOUT_DIAGNOSTIC_V2_PARSER_VERSION,
            "stage_b_template_hash": artifact_fingerprint(
                {"version": STAGE_B_PROMPT_TEMPLATE_VERSION}
            ),
            "stage_b_schema_hash": artifact_fingerprint(generation_payload_schema()),
            "stage_b_policy_hash": stage_b_generation_version_hash(
                STAGE_B_GENERATION_SETTINGS
            ),
            "stage_b_results_hash": artifact_fingerprint(
                sorted(path.name for path in STAGE_B_RESULTS_ROOT.glob("*.json"))
            ),
        },
    )


def _response_from_value(value: object) -> TimeoutDiagnosticResponse:
    if isinstance(value, OllamaDiagnosticHTTPResponse):
        return TimeoutDiagnosticResponse(
            http_status=value.http_status,
            raw_text=extract_ollama_response_text(value.parsed_json),
            envelope_text=value.raw_text,
            native_metadata=value.native_metadata,
        )
    if isinstance(value, Mapping):
        mapping = cast(Mapping[str, object], value)
        raw = mapping.get("raw_text")
        if not isinstance(raw, str):
            raw = json.dumps(dict(mapping), ensure_ascii=False)
        try:
            extracted = extract_ollama_response_text(json.loads(raw))
        except (TypeError, ValueError):
            extracted = raw
        native = mapping.get("native_metadata")
        return TimeoutDiagnosticResponse(
            http_status=cast(int | None, mapping.get("http_status")),
            raw_text=extracted,
            envelope_text=raw,
            native_metadata=cast(dict[str, object], native) if isinstance(native, dict) else {},
        )
    raise ValueError("diagnostic transport returned an unsupported response")


def _verification_stage(
    response: TimeoutDiagnosticResponse,
    pack: ContextPack,
    resolver: CanonicalCorpusResolver,
) -> str:
    try:
        payload = parse_generation_payload(response.raw_text)
        if payload.decision is not GenerationDecision.ANSWER:
            return "generator_abstain"
        finalized = finalize_generation(
            pack,
            GenerationResult(
                decision=payload.decision,
                claims=tuple(
                    ModelOutputClaim(
                        mode=claim.mode,
                        text=getattr(claim, "text", None),
                        citations=tuple(
                            ModelOutputCitation(
                                evidence_id=citation.evidence_id,
                                quoted_text=citation.quoted_text,
                            )
                            for citation in claim.citations
                        ),
                    )
                    for claim in payload.claims
                ),
            ),
            resolver,
            jurisdiction_text="SA",
        )
    except (TypeError, ValueError):
        return "invalid_json_or_schema"
    if finalized.result.decision is GenerationDecision.ANSWER:
        return "verified_answer"
    if finalized.result.abstention_reason is not None:
        return finalized.result.abstention_reason.value
    return "post_generation_rejection"


def run_stage_b_timeout_diagnostic(
    *,
    resume: bool,
    root: Path = TIMEOUT_DIAGNOSTIC_ROOT,
) -> dict[str, object]:
    """Replay exactly the frozen timeout cohort; this function is never called by tests."""

    cohort = select_frozen_stage_b_timeout_cohort()
    queries = {item.query_id: item for item in load_runtime_dev_queries()}
    candidate, tokenizer_fingerprint = load_generation_lock()
    local_lock = load_local_model_lock(LOCAL_OLLAMA_LOCK_PATH)
    tokenizer = LazyHuggingFaceTokenizer(
        identity=tokenizer_fingerprint.identity,
        revision=cast(str, tokenizer_fingerprint.revision),
    )
    generator = OllamaGenerator(
        endpoint="http://localhost:11434/api/generate",
        model=local_lock.model,
        immutable_digest=local_lock.digest,
        transport=UrllibOllamaTransport(),
        local_lock_path=LOCAL_OLLAMA_LOCK_PATH,
        stage_b=True,
    )
    prompt_hash = artifact_fingerprint({"version": STAGE_B_PROMPT_TEMPLATE_VERSION})
    schema_hash = artifact_fingerprint(generation_payload_schema())
    policy_hash = stage_b_generation_version_hash(STAGE_B_GENERATION_SETTINGS)
    resolver = CanonicalCorpusResolver.from_json(
        CANONICAL_UNITS,
        CHUNKS,
        CORPUS_MANIFEST,
        document_paths=CANONICAL_DOCUMENTS,
    )
    for query_id in cohort.query_ids:
        record_path = _record_path(root, query_id)
        if resume and record_path.is_file():
            try:
                TimeoutDiagnosticRecord.model_validate(
                    json.loads(record_path.read_text(encoding="utf-8"))
                )
                continue
            except (OSError, json.JSONDecodeError, ValueError):
                pass
        context_payload = json.loads(
            (STAGE_B_CONTEXT_CACHE_ROOT / f"{query_id}.json").read_text(encoding="utf-8")
        )
        pack = ContextPack.model_validate(context_payload["context_pack"])
        frozen_path = STAGE_B_RESULTS_ROOT / f"{query_id}.json"
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        request_fingerprint = cast(str, frozen["fingerprint"])
        request = GenerationRequest(
            query=queries[query_id].query,
            context_pack=pack,
            settings=STAGE_B_GENERATION_SETTINGS,
            jurisdiction_text="SA",
        )
        prompt = generator.build_payload(request)["prompt"]
        prompt_tokens = tokenizer.count(cast(str, prompt))
        evidence_tokens = pack.token_count
        start = time.monotonic()
        try:
            generator.validate_local_lock()
            diagnostic_post = getattr(generator.transport, "post_json_diagnostic", None)
            if diagnostic_post is None:
                raise RuntimeError("diagnostic transport is unavailable")
            value = diagnostic_post(
                generator.endpoint,
                generator.build_payload(request),
            )
            elapsed = time.monotonic() - start
            response = _response_from_value(value)
            if response.envelope_text is None:
                raise ValueError("diagnostic response has no full Ollama envelope")
            _write_private_envelope(root, query_id, response.envelope_text)
            record = write_diagnostic_record(
                query_id=query_id,
                request_fingerprint=request_fingerprint,
                response=response,
                elapsed_seconds=elapsed,
                prompt_tokens=prompt_tokens,
                evidence_tokens=evidence_tokens,
                timeout_configuration=current_timeout_configuration(),
                monotonic_request_start=start,
                final_verification_stage=_verification_stage(response, pack, resolver),
                model_tag=local_lock.model,
                model_digest=local_lock.digest,
                hf_revision=cast(str, candidate.hf_revision),
                tokenizer_id=tokenizer_fingerprint.identity,
                tokenizer_revision=cast(str, tokenizer_fingerprint.revision),
                stage_b_prompt_hash=prompt_hash,
                stage_b_schema_hash=schema_hash,
                stage_b_policy_hash=policy_hash,
            )
        except Exception as error:
            elapsed = time.monotonic() - start
            response_received = (
                error.response_received
                if isinstance(error, OllamaDiagnosticTransportError)
                else False
            )
            record = write_diagnostic_record(
                query_id=query_id,
                request_fingerprint=request_fingerprint,
                error=error,
                response_received=response_received,
                elapsed_seconds=elapsed,
                prompt_tokens=prompt_tokens,
                evidence_tokens=evidence_tokens,
                timeout_configuration=current_timeout_configuration(),
                monotonic_request_start=start,
                model_tag=local_lock.model,
                model_digest=local_lock.digest,
                hf_revision=cast(str, candidate.hf_revision),
                tokenizer_id=tokenizer_fingerprint.identity,
                tokenizer_revision=cast(str, tokenizer_fingerprint.revision),
                stage_b_prompt_hash=prompt_hash,
                stage_b_schema_hash=schema_hash,
                stage_b_policy_hash=policy_hash,
            )
        _write_private_record(record_path, record)
    return timeout_diagnostic_status(root=root)


def run_stage_b_timeout_diagnostic_v2(
    *,
    resume: bool,
    root: Path = TIMEOUT_DIAGNOSTIC_V2_ROOT,
) -> dict[str, object]:
    """Replay the immutable timeout cohort into the isolated v2 namespace."""

    cohort = select_frozen_stage_b_timeout_cohort()
    _assert_v2_cohort(cohort)
    return run_stage_b_timeout_diagnostic(resume=resume, root=root)
