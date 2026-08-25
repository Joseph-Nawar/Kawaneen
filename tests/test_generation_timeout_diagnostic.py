from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest

from kawaneen.generation.contracts import parse_generation_payload
from kawaneen.generation.ollama import (
    OllamaDiagnosticHTTPResponse,
    UrllibOllamaTransport,
    extract_ollama_response_text,
)
from kawaneen.generation.timeout_diagnostic import (
    TIMEOUT_DIAGNOSTIC_EXPECTED_COUNT,
    TIMEOUT_DIAGNOSTIC_ROOT,
    TIMEOUT_DIAGNOSTIC_V2_ROOT,
    DiagnosticTimeoutError,
    TimeoutDiagnosticResponse,
    _response_from_value,
    classify_ollama_exception,
    current_timeout_configuration,
    evaluate_persisted_timeout_diagnostic,
    evaluate_stage_b_timeout_diagnostic,
    run_diagnostic_cases,
    run_stage_b_timeout_diagnostic_v2,
    select_frozen_stage_b_timeout_cohort,
    timeout_diagnostic_status,
    timeout_diagnostic_v2_status,
    write_diagnostic_record,
    write_timeout_diagnostic_v2_manifest,
)


def test_urllib_adapter_sends_frozen_socket_timeout(monkeypatch) -> None:
    calls: list[float | None] = []

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"response":"{}"}'

    def fake_urlopen(_request, *, timeout):
        calls.append(timeout)
        return Response()

    monkeypatch.setattr("kawaneen.generation.ollama.urlopen", fake_urlopen)
    from kawaneen.generation.ollama import UrllibOllamaTransport

    transport = UrllibOllamaTransport()
    transport.post_json("http://localhost:11434/api/generate", {})

    assert calls == [current_timeout_configuration().socket_timeout_seconds]


def test_diagnostic_transport_retains_complete_ollama_envelope(monkeypatch) -> None:
    envelope = json.dumps(
        {
            "response": '{"decision":"abstain","claims":[]}',
            "done": True,
            "done_reason": "stop",
            "eval_count": 4,
        }
    ).encode("utf-8")

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return envelope

    monkeypatch.setattr("kawaneen.generation.ollama.urlopen", lambda *_args, **_kwargs: Response())

    result = UrllibOllamaTransport().post_json_diagnostic("http://localhost:11434/api/generate", {})

    assert result.raw_text == envelope.decode("utf-8")
    assert result.parsed_json == json.loads(envelope)
    assert result.native_metadata == {
        "done": True,
        "done_reason": "stop",
        "eval_count": 4,
    }


def test_frozen_timeout_cohort_is_exactly_27_and_excludes_non_timeouts() -> None:
    cohort = select_frozen_stage_b_timeout_cohort()

    assert len(cohort.query_ids) == TIMEOUT_DIAGNOSTIC_EXPECTED_COUNT == 27
    assert cohort.query_ids_hash == (
        "c692ab9ca2790dcabfb5e082e52e5d324555f56bc7c47b1cc6a69d0c1878c554"
    )
    assert cohort.non_timeout_records == 133
    assert cohort.invalid_fingerprints == 0
    assert cohort.raw_outputs_present == 0


def test_current_stage_b_timeout_configuration_is_unchanged() -> None:
    configuration = current_timeout_configuration()

    assert configuration.http_client == "urllib.request"
    assert configuration.socket_timeout_seconds == 30.0
    assert configuration.connect_timeout_seconds is None
    assert configuration.read_timeout_seconds is None
    assert configuration.write_timeout_seconds is None
    assert configuration.pool_timeout_seconds is None
    assert configuration.overall_timeout_seconds is None
    assert configuration.retry_count == 0
    assert configuration.streaming is False


@pytest.mark.parametrize(
    ("error", "response_received", "expected"),
    (
        (DiagnosticTimeoutError("connect_timeout"), False, "connect_timeout"),
        (DiagnosticTimeoutError("read_timeout"), True, "read_timeout"),
        (DiagnosticTimeoutError("write_timeout"), False, "write_timeout"),
        (DiagnosticTimeoutError("pool_timeout"), False, "pool_timeout"),
        (TimeoutError("timed out"), False, "overall_timeout"),
        (ConnectionRefusedError("refused"), False, "connection_error"),
        (ValueError("bad JSON"), True, "malformed_response"),
    ),
)
def test_diagnostic_distinguishes_failure_categories(
    error: Exception, response_received: bool, expected: str
) -> None:
    assert classify_ollama_exception(error, response_received=response_received) == expected


def test_timeout_classifier_unwraps_nested_urlerror_and_preserves_http_errors() -> None:
    nested = URLError(URLError(TimeoutError("timed out")))
    http_error = HTTPError("http://localhost:11434/api/generate", 500, "server", {}, None)

    assert classify_ollama_exception(nested, response_received=False) == "overall_timeout"
    assert classify_ollama_exception(http_error, response_received=True) == "http_error"


def test_success_response_telemetry_captures_optional_ollama_metadata() -> None:
    response = TimeoutDiagnosticResponse(
        http_status=200,
        raw_text='{"decision":"abstain","claims":[]}',
        native_metadata={
            "done": True,
            "done_reason": "stop",
            "total_duration": 12,
            "eval_count": 3,
        },
    )

    record = write_diagnostic_record(
        query_id="query-private",
        request_fingerprint="a" * 64,
        response=response,
        elapsed_seconds=1.25,
        prompt_tokens=100,
        evidence_tokens=80,
        timeout_configuration=current_timeout_configuration(),
    )

    assert record.raw_output_present is True
    assert record.response_character_count == len(response.raw_text)
    assert record.http_status == 200
    assert record.native_metadata["done"] is True
    assert record.native_metadata["eval_count"] == 3
    assert record.generator_decision == "abstain"


def test_full_ollama_envelope_extracts_nested_response_and_retains_telemetry() -> None:
    inner = '{"decision":"abstain","claims":[]}'
    envelope = {
        "model": "qwen3:4b-instruct-2507-q4_K_M",
        "response": inner,
        "done": True,
        "done_reason": "length",
        "eval_count": 512,
    }
    envelope_text = json.dumps(envelope)

    extracted = extract_ollama_response_text(envelope)
    response = TimeoutDiagnosticResponse(
        raw_text=extracted,
        envelope_text=envelope_text,
        http_status=200,
        native_metadata=envelope,
    )
    record = write_diagnostic_record(
        query_id="query-private",
        request_fingerprint="e" * 64,
        response=response,
        elapsed_seconds=1.0,
        prompt_tokens=10,
        evidence_tokens=5,
        timeout_configuration=current_timeout_configuration(),
    )

    assert parse_generation_payload(response.raw_text).decision.value == "abstain"
    assert record.response_character_count == len(envelope_text)
    assert record.native_metadata["done_reason"] == "length"


def test_diagnostic_response_adapter_uses_shared_nested_extractor() -> None:
    inner = '{"decision":"abstain","claims":[]}'
    envelope_text = json.dumps({"response": inner, "done": True})
    normalized = _response_from_value(
        OllamaDiagnosticHTTPResponse(
            http_status=200,
            raw_text=envelope_text,
            parsed_json=json.loads(envelope_text),
            native_metadata={"done": True},
        )
    )

    assert normalized.raw_text == inner
    assert normalized.envelope_text == envelope_text


def test_malformed_inner_json_is_classified_without_network() -> None:
    response = TimeoutDiagnosticResponse(
        raw_text="{not-json",
        envelope_text=json.dumps({"response": "{not-json", "done": True}),
        http_status=200,
    )

    record = write_diagnostic_record(
        query_id="query-private",
        request_fingerprint="f" * 64,
        response=response,
        elapsed_seconds=1.0,
        prompt_tokens=10,
        evidence_tokens=5,
        timeout_configuration=current_timeout_configuration(),
    )

    assert record.failure_category == "malformed_response"
    assert record.parse_outcome == "invalid_pydantic_or_json"


def test_offline_evaluator_fails_closed_when_envelopes_are_absent(tmp_path: Path) -> None:
    result = evaluate_persisted_timeout_diagnostic(root=tmp_path)

    assert result["status"] == "insufficient_evidence"
    assert result["envelopes_available"] == 0
    assert result["missing_envelopes"] == 27
    assert result["http_requests"] == 0
    assert result["model_calls"] == 0


def test_offline_evaluator_rederives_old_flawed_record_without_http(tmp_path: Path) -> None:
    record = write_diagnostic_record(
        query_id="query-private",
        request_fingerprint="a" * 64,
        response=TimeoutDiagnosticResponse(
            raw_text="not-used-by-evaluator",
            http_status=200,
        ),
        elapsed_seconds=1.0,
        prompt_tokens=10,
        evidence_tokens=5,
        timeout_configuration=current_timeout_configuration(),
    )
    records = tmp_path / "records"
    envelopes = tmp_path / "envelopes"
    records.mkdir()
    envelopes.mkdir()
    (records / "query-private.json").write_text(
        json.dumps(record.model_dump(mode="json")), encoding="utf-8"
    )
    (envelopes / "query-private.json").write_text(
        json.dumps(
            {
                "response": '{"decision":"abstain","claims":[]}',
                "done": True,
                "done_reason": "stop",
            }
        ),
        encoding="utf-8",
    )

    result = evaluate_persisted_timeout_diagnostic(
        root=tmp_path,
        expected_query_ids=("query-private",),
    )

    assert result["status"] == "complete"
    assert result["valid_generated_responses"] == 1
    assert result["explicit_abstentions"] == 1
    assert result["done_reason"] == {"stop": {"explicit_abstention": 1}}
    assert result["http_requests"] == 0
    assert result["model_calls"] == 0
    assert (
        result["corrected_record_provenance"][0]["original_record_sha256"]
        == hashlib.sha256((records / "query-private.json").read_bytes()).hexdigest()
    )


def test_failure_telemetry_has_no_raw_source_or_answer_text() -> None:
    record = write_diagnostic_record(
        query_id="query-private",
        request_fingerprint="b" * 64,
        error=TimeoutError("timed out"),
        response_received=False,
        elapsed_seconds=30.0,
        prompt_tokens=100,
        evidence_tokens=80,
        timeout_configuration=current_timeout_configuration(),
    )

    dumped = record.model_dump(mode="json")
    assert record.failure_category == "overall_timeout"
    assert record.raw_output_present is False
    assert "source" not in json.dumps(dumped).lower()
    assert "answer" not in json.dumps(dumped).lower()


def test_status_reads_only_diagnostic_records(tmp_path: Path) -> None:
    status = timeout_diagnostic_status(root=tmp_path)

    assert status == {
        "expected": 27,
        "completed": 0,
        "success": 0,
        "timeout_by_subtype": {},
        "other_failure": 0,
        "corrupt": 0,
        "missing": 27,
    }


def test_deterministic_evaluator_reports_insufficient_evidence_without_replay(
    tmp_path: Path,
) -> None:
    result = evaluate_stage_b_timeout_diagnostic(root=tmp_path)

    assert result["status"] == "incomplete"
    assert result["completed"] == 0
    assert result["diagnosis"] == "INSUFFICIENT_EVIDENCE"


def test_corrupt_diagnostic_checkpoint_is_recomputed_and_resume_reuses_valid_record(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def runner(query_id: str):
        calls.append(query_id)
        return write_diagnostic_record(
            query_id=query_id,
            request_fingerprint="c" * 64,
            response=TimeoutDiagnosticResponse(
                http_status=200,
                raw_text='{"decision":"abstain","claims":[]}',
            ),
            elapsed_seconds=0.1,
            prompt_tokens=10,
            evidence_tokens=5,
            timeout_configuration=current_timeout_configuration(),
        )

    assert (
        run_diagnostic_cases(
            query_ids=("query-a", "query-b"), runner=runner, root=tmp_path, resume=True
        )
        == 2
    )
    assert calls == ["query-a", "query-b"]
    (tmp_path / "records" / "query-b.json").write_text("corrupt", encoding="utf-8")
    assert (
        run_diagnostic_cases(
            query_ids=("query-a", "query-b"), runner=runner, root=tmp_path, resume=True
        )
        == 2
    )
    assert calls == ["query-a", "query-b", "query-b"]


def test_v2_cohort_namespace_and_hash_are_frozen() -> None:
    cohort = select_frozen_stage_b_timeout_cohort()

    assert TIMEOUT_DIAGNOSTIC_V2_ROOT != TIMEOUT_DIAGNOSTIC_ROOT
    assert len(cohort.query_ids) == 27
    assert cohort.query_ids_hash == (
        "c692ab9ca2790dcabfb5e082e52e5d324555f56bc7c47b1cc6a69d0c1878c554"
    )


def test_v2_status_reads_only_v2_namespace(tmp_path: Path) -> None:
    status = timeout_diagnostic_v2_status(root=tmp_path)

    assert status["expected"] == 27
    assert status["completed"] == 0
    assert status["missing"] == 27


def test_v2_envelope_persistence_is_private_and_immutable(tmp_path: Path) -> None:
    from kawaneen.generation.timeout_diagnostic import _write_private_envelope

    _write_private_envelope(tmp_path, "query-a", '{"response":"{}"}')
    _write_private_envelope(tmp_path, "query-a", '{"response":"{}"}')

    with pytest.raises(ValueError, match="immutable diagnostic envelope"):
        _write_private_envelope(tmp_path, "query-a", '{"response":"changed"}')
    assert not list(tmp_path.glob("*.json"))
    assert (tmp_path / "envelopes" / "query-a.json").is_file()


def test_v2_manifest_is_text_free_and_does_not_touch_v1(tmp_path: Path) -> None:
    manifest = tmp_path / "v2-manifest.json"
    write_timeout_diagnostic_v2_manifest(path=manifest)

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["status"] == "prepared_not_executed"
    assert payload["cohort_count"] == 27
    assert "query_ids" not in json.dumps(payload)


def test_v2_status_and_evaluator_are_network_and_model_free(tmp_path: Path, monkeypatch) -> None:
    def fail(*_args, **_kwargs):
        raise AssertionError("network/model access is forbidden")

    monkeypatch.setattr("kawaneen.generation.ollama.urlopen", fail)
    monkeypatch.setattr("kawaneen.generation.ollama.OllamaGenerator.generate", fail)

    assert timeout_diagnostic_v2_status(root=tmp_path)["missing"] == 27
    result = evaluate_persisted_timeout_diagnostic(root=tmp_path)
    assert result["http_requests"] == 0
    assert result["model_calls"] == 0


def test_v2_runner_asserts_frozen_cohort_before_execution(monkeypatch, tmp_path: Path) -> None:
    from kawaneen.generation.timeout_diagnostic import FrozenTimeoutCohort

    wrong = FrozenTimeoutCohort(
        query_ids=("query-a",),
        query_ids_hash="0" * 64,
        non_timeout_records=0,
        invalid_fingerprints=0,
        raw_outputs_present=0,
    )
    monkeypatch.setattr(
        "kawaneen.generation.timeout_diagnostic.select_frozen_stage_b_timeout_cohort",
        lambda: wrong,
    )

    with pytest.raises(ValueError, match="cohort hash"):
        run_stage_b_timeout_diagnostic_v2(resume=True, root=tmp_path)


def test_diagnostic_records_are_separate_from_frozen_stage_b_results(tmp_path: Path) -> None:
    frozen = Path(
        "artifacts/private/phase10_generation/results/qwen-ollama-stage-b/"
        "query-2babbab84843348dc976b35b.json"
    )
    before = frozen.read_bytes()

    record = write_diagnostic_record(
        query_id="query-private",
        request_fingerprint="d" * 64,
        response=TimeoutDiagnosticResponse(
            http_status=200,
            raw_text='{"decision":"abstain","claims":[]}',
        ),
        elapsed_seconds=0.1,
        prompt_tokens=10,
        evidence_tokens=5,
        timeout_configuration=current_timeout_configuration(),
    )
    run_diagnostic_cases(
        query_ids=("query-private",),
        runner=lambda _query_id: record,
        root=tmp_path,
        resume=False,
    )

    assert frozen.read_bytes() == before
