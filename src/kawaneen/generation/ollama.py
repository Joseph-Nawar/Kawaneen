"""Lazy localhost Ollama adapter with immutable-digest gating."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field

from kawaneen.generation.abstention import invalid_generation_result
from kawaneen.generation.artifacts import write_text_free_artifact
from kawaneen.generation.contracts import (
    STAGE_B_GENERATION_SETTINGS,
    STAGE_C_GENERATION_SETTINGS,
    STAGE_D_GENERATION_SETTINGS,
    DirectClaim,
    GenerationPayload,
    GenerationRequest,
    GenerationResult,
    ModelOutput,
    ModelOutputCitation,
    ModelOutputClaim,
    generation_payload_schema,
    parse_generation_payload,
    parse_model_output,
    parse_stage_c_generation_payload,
    parse_stage_d_generation_payload,
)
from kawaneen.generation.prompt import (
    render_generation_prompt,
    render_stage_b_generation_prompt,
    render_stage_c_generation_prompt,
    render_stage_d_generation_prompt,
)
from kawaneen.generation.quote_registry import (
    QuoteRegistry,
    stage_c_result_from_payload,
    stage_d_result_from_payload,
)
from kawaneen.generation.stage_c import STAGE_C_TIMEOUT_SECONDS
from kawaneen.generation.stage_d import STAGE_D_TIMEOUT_SECONDS

_BARE_SHA256_DIGEST = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)
_CANONICAL_SHA256_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$", re.IGNORECASE)
LOCAL_OLLAMA_LOCK_PATH = Path("artifacts/private/phase10_generation/qwen-ollama-model-lock.json")


class OllamaTransport(Protocol):
    def get_json(self, endpoint: str) -> object: ...

    def post_json(self, endpoint: str, payload: dict[str, object]) -> object: ...


@dataclass(frozen=True, slots=True)
class OllamaDiagnosticHTTPResponse:
    """HTTP response details used only by the timeout diagnostic."""

    http_status: int | None
    raw_text: str
    parsed_json: object | None
    native_metadata: dict[str, object]


class OllamaDiagnosticTransportError(Exception):
    """Preserve transport-boundary details without changing normal generation."""

    def __init__(
        self,
        error: Exception,
        *,
        http_status: int | None = None,
        response_received: bool = False,
        raw_text: str | None = None,
    ) -> None:
        super().__init__(str(error))
        self.original_error = error
        self.http_status = http_status
        self.response_received = response_received
        self.raw_text = raw_text


class UrllibOllamaTransport:
    def __init__(self, *, timeout_seconds: float = 30.0) -> None:
        self.timeout_seconds = timeout_seconds

    def post_json(self, endpoint: str, payload: dict[str, object]) -> object:
        request = Request(
            endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.timeout_seconds) as response:
            value = json.loads(response.read().decode("utf-8"))
        return value

    def post_json_diagnostic(
        self, endpoint: str, payload: dict[str, object]
    ) -> OllamaDiagnosticHTTPResponse:
        """Make one non-streaming request while retaining diagnostic metadata."""

        request = Request(
            endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        response_status: int | None = None
        response_received = False
        raw_text: str | None = None
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                response_status = getattr(response, "status", None)
                response_received = True
                raw_text = response.read().decode("utf-8")
            if raw_text is None:
                raise ValueError("Ollama diagnostic response body is missing")
            parsed: object = json.loads(raw_text)
        except HTTPError as error:
            response_status = error.code
            response_received = True
            try:
                raw_text = error.read().decode("utf-8")
            except Exception:
                raw_text = None
            raise OllamaDiagnosticTransportError(
                error,
                http_status=response_status,
                response_received=response_received,
                raw_text=raw_text,
            ) from error
        except Exception as error:
            raise OllamaDiagnosticTransportError(
                error,
                http_status=response_status,
                response_received=response_received,
                raw_text=raw_text,
            ) from error
        native: dict[str, object] = {}
        if isinstance(parsed, Mapping):
            native = {
                key: parsed[key]
                for key in (
                    "done",
                    "done_reason",
                    "total_duration",
                    "load_duration",
                    "prompt_eval_count",
                    "prompt_eval_duration",
                    "eval_count",
                    "eval_duration",
                )
                if key in parsed
            }
        return OllamaDiagnosticHTTPResponse(
            http_status=response_status,
            raw_text=raw_text,
            parsed_json=cast(object, parsed),
            native_metadata=native,
        )

    def get_json(self, endpoint: str) -> object:
        request = Request(endpoint, headers={"Accept": "application/json"}, method="GET")
        with urlopen(request, timeout=self.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))


class OllamaModelIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str = Field(min_length=1)
    digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


def normalize_sha256_digest(value: object) -> str:
    """Normalize an Ollama API digest to the strict internal representation."""

    if not isinstance(value, str):
        raise ValueError("Ollama model digest must be a string")
    if _BARE_SHA256_DIGEST.fullmatch(value):
        return f"sha256:{value.lower()}"
    if _CANONICAL_SHA256_DIGEST.fullmatch(value):
        return f"sha256:{value.split(':', 1)[1].lower()}"
    raise ValueError("Ollama model digest must be a full SHA-256 digest")


def validate_ollama_endpoint(endpoint: str) -> tuple[str, str]:
    """Validate an HTTP Ollama endpoint and return its scheme and network location."""

    parsed = urlparse(endpoint)
    if parsed.scheme != "http" or not parsed.hostname:
        raise ValueError("Ollama endpoint must be an HTTP URL with a hostname")
    return parsed.scheme, parsed.netloc


def inspect_ollama_model(
    endpoint: str,
    expected_model: str,
    transport: OllamaTransport,
) -> OllamaModelIdentity:
    validate_ollama_endpoint(endpoint)
    response = transport.get_json(endpoint.rstrip("/") + "/api/tags")
    if not isinstance(response, Mapping):
        raise ValueError("Ollama tags response is not an object")
    response_mapping = cast(Mapping[str, object], response)
    models = response_mapping.get("models")
    if not isinstance(models, list):
        raise ValueError("Ollama tags response has no models")
    for raw_value in cast(list[object], models):
        if not isinstance(raw_value, Mapping):
            continue
        value = cast(Mapping[str, object], raw_value)
        name = value.get("name") or value.get("model")
        digest = value.get("digest")
        if name == expected_model:
            if not isinstance(digest, str):
                raise ValueError("Ollama model has no immutable digest")
            return OllamaModelIdentity(
                model=expected_model,
                digest=normalize_sha256_digest(digest),
            )
    raise ValueError(f"expected Ollama model tag is not installed: {expected_model}")


def write_local_model_lock(path: Path, identity: OllamaModelIdentity) -> None:
    write_text_free_artifact(path, identity.model_dump(mode="json"))


def load_local_model_lock(path: Path) -> OllamaModelIdentity:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("local Ollama model lock is unavailable") from error
    if not isinstance(payload, dict):
        raise ValueError("local Ollama model lock is not an object")
    return OllamaModelIdentity.model_validate(payload)


class OllamaGenerator:
    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        immutable_digest: str | None,
        transport: OllamaTransport | None = None,
        local_lock_path: Path | None = None,
        stage_b: bool = False,
        stage_c: bool = False,
        stage_d: bool = False,
    ) -> None:
        scheme, netloc = validate_ollama_endpoint(endpoint)
        if not model.strip():
            raise ValueError("Ollama model must not be blank")
        if (
            immutable_digest is not None
            and _CANONICAL_SHA256_DIGEST.fullmatch(immutable_digest) is None
        ):
            raise ValueError("Ollama digest must have the sha256:<64 hex> form")
        self.endpoint = endpoint
        self.model = model
        self.immutable_digest = immutable_digest
        self.transport = transport or UrllibOllamaTransport(
            timeout_seconds=(
                STAGE_C_TIMEOUT_SECONDS if stage_c else STAGE_D_TIMEOUT_SECONDS if stage_d else 30.0
            )
        )
        self.local_lock_path = local_lock_path
        self.stage_b = stage_b
        self.stage_c = stage_c
        self.stage_d = stage_d
        if sum((self.stage_b, self.stage_c, self.stage_d)) > 1:
            raise ValueError("Ollama generator cannot target multiple experiment stages")
        self.identity_endpoint = f"{scheme}://{netloc}"
        self.last_raw_response: str | None = None
        self.last_telemetry: dict[str, object] = {}

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.last_raw_response = None
        self.last_telemetry = {}
        started = time.monotonic()
        if self.immutable_digest is None:
            self.last_telemetry = {
                "elapsed_seconds": 0.0,
                "exception_class": "ValueError",
                "exception_subtype": "missing_immutable_digest",
                "failure_category": "configuration_error",
            }
            return invalid_generation_result("Ollama execution requires a locked immutable digest")
        try:
            self.validate_local_lock()
            payload = self.build_payload(request)
            response = self.transport.post_json(self.endpoint, payload)
            elapsed = time.monotonic() - started
            response_mapping = (
                cast(Mapping[str, object], response)
                if isinstance(response, Mapping)
                else cast(Mapping[str, object], {})
            )
            native: dict[str, object] = {
                key: response_mapping[key]
                for key in (
                    "done",
                    "done_reason",
                    "total_duration",
                    "load_duration",
                    "prompt_eval_count",
                    "prompt_eval_duration",
                    "eval_count",
                    "eval_duration",
                )
                if key in response_mapping
            }
            self.last_telemetry = {
                "elapsed_seconds": elapsed,
                "http_status": 200,
                "exception_class": None,
                "exception_subtype": None,
                "failure_category": None,
                **native,
            }
            raw = extract_ollama_response_text(cast(object, response))
            self.last_raw_response = raw
            if self.stage_b:
                return _result_from_payload(parse_generation_payload(raw))
            if self.stage_c:
                registry = request.quote_registry
                if not isinstance(registry, QuoteRegistry):
                    raise ValueError("Stage-C request is missing its QuoteRegistry")
                return stage_c_result_from_payload(parse_stage_c_generation_payload(raw), registry)
            if self.stage_d:
                registry = request.quote_registry
                if not isinstance(registry, QuoteRegistry):
                    raise ValueError("Stage-D request is missing its QuoteRegistry")
                return stage_d_result_from_payload(parse_stage_d_generation_payload(raw), registry)
            output = parse_model_output(raw)
            return _result_from_output(output)
        except Exception as error:  # fail closed; adapters do not retry
            self.last_telemetry = {
                **self.last_telemetry,
                "elapsed_seconds": time.monotonic() - started,
                "http_status": self.last_telemetry.get("http_status"),
                "exception_class": type(error).__name__,
                "exception_subtype": type(error).__module__ + "." + type(error).__name__,
                "failure_category": (
                    "timeout"
                    if isinstance(error, TimeoutError)
                    else "connection_error"
                    if isinstance(error, ConnectionError)
                    else "http_error"
                    if isinstance(error, HTTPError)
                    else "other"
                ),
            }
            return invalid_generation_result(str(error))

    def validate_local_lock(self) -> None:
        if self.local_lock_path is None:
            return
        lock = load_local_model_lock(self.local_lock_path)
        if lock.model != self.model or lock.digest != self.immutable_digest:
            raise ValueError("Ollama model lock does not match generator")
        installed = inspect_ollama_model(self.identity_endpoint, self.model, self.transport)
        if installed != lock:
            raise ValueError("installed Ollama model differs from lock")

    def build_payload(self, request: GenerationRequest) -> dict[str, object]:
        settings = (
            STAGE_B_GENERATION_SETTINGS
            if self.stage_b
            else STAGE_C_GENERATION_SETTINGS
            if self.stage_c
            else STAGE_D_GENERATION_SETTINGS
            if self.stage_d
            else request.settings
        )
        if self.stage_b:
            prompt = render_stage_b_generation_prompt(
                request.query,
                request.context_pack,
                settings=settings,
                jurisdiction_text=request.jurisdiction_text,
            )
            output_schema: object = generation_payload_schema()
        elif self.stage_c:
            registry = request.quote_registry
            if not isinstance(registry, QuoteRegistry):
                raise ValueError("Stage-C request is missing its QuoteRegistry")
            prompt = render_stage_c_generation_prompt(
                request.query,
                request.context_pack,
                registry=registry,
                settings=settings,
                jurisdiction_text=request.jurisdiction_text,
            )
            from kawaneen.generation.contracts import stage_c_generation_payload_schema

            output_schema = stage_c_generation_payload_schema()
        elif self.stage_d:
            registry = request.quote_registry
            if not isinstance(registry, QuoteRegistry):
                raise ValueError("Stage-D request is missing its QuoteRegistry")
            prompt = render_stage_d_generation_prompt(
                request.query,
                request.context_pack,
                registry=registry,
                settings=settings,
                jurisdiction_text=request.jurisdiction_text,
            )
            from kawaneen.generation.contracts import stage_d_generation_payload_schema

            output_schema = stage_d_generation_payload_schema()
        else:
            prompt = render_generation_prompt(
                request.query,
                request.context_pack,
                settings=settings,
                jurisdiction_text=request.jurisdiction_text,
            )
            output_schema = "json"
        return {
            "model": self.model,
            "prompt": prompt.text,
            "stream": False,
            "format": output_schema,
            "options": {
                "temperature": settings.temperature,
                "num_predict": settings.max_new_tokens,
            },
        }


def extract_ollama_response_text(response: object) -> str:
    """Extract the generated payload from an Ollama response envelope."""

    if isinstance(response, str):
        return response
    if isinstance(response, Mapping):
        mapping = cast(Mapping[str, object], response)
        value = mapping.get("response")
        if isinstance(value, str):
            return value
        return json.dumps(dict(mapping), ensure_ascii=False)
    raise ValueError("Ollama response is not JSON text or an object")


def _result_from_output(output: ModelOutput) -> GenerationResult:
    return GenerationResult(
        decision=output.decision,
        claims=output.claims,
    )


def _result_from_payload(output: GenerationPayload) -> GenerationResult:
    claims = tuple(
        ModelOutputClaim(
            mode=claim.mode,
            text=None if isinstance(claim, DirectClaim) else claim.text,
            citations=tuple(
                ModelOutputCitation(
                    evidence_id=citation.evidence_id,
                    quoted_text=citation.quoted_text,
                )
                for citation in claim.citations
            ),
        )
        for claim in output.claims
    )
    return GenerationResult(decision=output.decision, claims=claims)
