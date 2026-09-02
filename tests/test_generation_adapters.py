from __future__ import annotations

import json
from pathlib import Path

import pytest

from kawaneen.generation.contracts import (
    AbstentionReason,
    GenerationDecision,
    GenerationRequest,
    ModelCandidate,
)
from kawaneen.generation.ollama import (
    OllamaGenerator,
    inspect_ollama_model,
    load_local_model_lock,
    normalize_sha256_digest,
    write_local_model_lock,
)
from kawaneen.generation.registry import (
    default_model_registry,
    load_generation_lock,
    lock_hf_revision,
    lock_ollama_digest,
)
from kawaneen.generation.transformers import TransformersGenerator
from kawaneen.grounding.contracts import ContextPack


def request() -> GenerationRequest:
    return GenerationRequest(
        query="deadline",
        context_pack=ContextPack(
            query_id="q1",
            phase8_selection_sha256="a" * 64,
            canonical_corpus_hash="b" * 64,
            assembly_policy_version="phase9-v1",
            token_counter_identity="fake-v1",
            max_context_tokens=100,
            token_count=0,
            units=(),
            blocks=(),
            evidence=(),
            omissions=(),
        ),
    )


class FakeOllamaTransport:
    def __init__(self, response: object, identity_response: object | None = None) -> None:
        self.response = response
        self.identity_response = identity_response
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.get_calls: list[str] = []

    def post_json(self, endpoint: str, payload: dict[str, object]) -> object:
        self.calls.append((endpoint, payload))
        return self.response

    def get_json(self, endpoint: str) -> object:
        self.get_calls.append(endpoint)
        return self.identity_response


class FakeRuntime:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[tuple[str, object]] = []

    def generate(self, prompt: str, settings: object) -> str:
        self.calls.append((prompt, settings))
        return self.response


class FakeLoader:
    def __init__(self, runtime: FakeRuntime) -> None:
        self.runtime = runtime
        self.calls: list[ModelCandidate] = []

    def load(self, candidate: ModelCandidate) -> FakeRuntime:
        self.calls.append(candidate)
        return self.runtime


def model_json() -> str:
    return json.dumps(
        {
            "decision": "answer",
            "claims": [
                {
                    "text": "claim",
                    "citations": [{"evidence_id": "E001", "quoted_text": "quote"}],
                }
            ],
        }
    )


def test_ollama_constructor_is_side_effect_free_and_requires_immutable_digest() -> None:
    transport = FakeOllamaTransport({"response": model_json()})
    generator = OllamaGenerator(
        endpoint="http://localhost:11434/api/generate",
        model="qwen3:4b-instruct-2507-q4_K_M",
        immutable_digest="sha256:" + "a" * 64,
        transport=transport,
    )
    assert transport.calls == []

    result = generator.generate(request())

    assert result.decision is GenerationDecision.ANSWER
    assert len(transport.calls) == 1
    assert transport.calls[0][1]["stream"] is False
    assert generator.last_raw_response == model_json()

    unlocked = OllamaGenerator(
        endpoint="http://localhost:11434/api/generate",
        model="qwen3:4b-instruct-2507-q4_K_M",
        immutable_digest=None,
        transport=transport,
    )
    refused = unlocked.generate(request())
    assert refused.abstention_reason is AbstentionReason.INVALID_GENERATION
    assert len(transport.calls) == 1


def test_ollama_invalid_payload_fails_closed_without_retry() -> None:
    transport = FakeOllamaTransport({"response": "not-json"})
    generator = OllamaGenerator(
        endpoint="http://127.0.0.1:11434/api/generate",
        model="qwen3:4b-instruct-2507-q4_K_M",
        immutable_digest="sha256:" + "a" * 64,
        transport=transport,
    )

    result = generator.generate(request())

    assert result.abstention_reason is AbstentionReason.INVALID_GENERATION
    assert len(transport.calls) == 1


def test_ollama_rejects_non_immutable_digest_at_construction() -> None:
    with pytest.raises(ValueError):
        OllamaGenerator(
            endpoint="http://localhost:11434/api/generate",
            model="qwen3:4b-instruct-2507-q4_K_M",
            immutable_digest="latest",
        )


def test_ollama_accepts_compose_service_hostname() -> None:
    generator = OllamaGenerator(
        endpoint="http://ollama:11434/api/generate",
        model="qwen3:4b-instruct-2507-q4_K_M",
        immutable_digest="sha256:" + "a" * 64,
    )

    assert generator.identity_endpoint == "http://ollama:11434"


def test_transformers_loading_is_lazy_and_revision_locked() -> None:
    candidate = ModelCandidate(
        name="test",
        hf_identity="Qwen/Qwen3-4B-Instruct-2507",
        hf_revision="a" * 40,
        role="test",
    )
    runtime = FakeRuntime(model_json())
    loader = FakeLoader(runtime)
    generator = TransformersGenerator(candidate=candidate, loader=loader)
    assert loader.calls == []

    result = generator.generate(request())

    assert result.decision is GenerationDecision.ANSWER
    assert len(loader.calls) == 1
    assert len(runtime.calls) == 1


def test_registry_has_required_candidates_and_lock_helpers() -> None:
    candidates = default_model_registry()
    assert candidates[0].hf_identity == "Qwen/Qwen3-4B-Instruct-2507"
    assert candidates[0].ollama_model == "qwen3:4b-instruct-2507-q4_K_M"
    assert any(item.hf_identity == "QCRI/Fanar-1-9B-Instruct" for item in candidates)

    hf_locked = lock_hf_revision(candidates[0], "b" * 40)
    ollama_locked = lock_ollama_digest(hf_locked, "sha256:" + "c" * 64)
    assert hf_locked.hf_revision == "b" * 40
    assert ollama_locked.ollama_digest == "sha256:" + "c" * 64


def test_qwen_model_and_tokenizer_use_full_shared_immutable_revision() -> None:
    candidate, tokenizer = load_generation_lock()

    assert candidate.hf_identity == "Qwen/Qwen3-4B-Instruct-2507"
    assert candidate.hf_revision == "cdbee75f17c01a7cc42f958dc650907174af0554"
    assert tokenizer.identity == candidate.hf_identity
    assert tokenizer.revision == candidate.hf_revision


def test_ollama_identity_lock_requires_exact_tag_and_persists_digest(tmp_path: Path) -> None:
    digest = "d" * 64
    transport = FakeOllamaTransport(
        response={},
        identity_response={
            "models": [
                {"name": "qwen3:4b-instruct-2507-q4_K_M", "digest": digest},
            ]
        },
    )

    identity = inspect_ollama_model(
        "http://localhost:11434",
        "qwen3:4b-instruct-2507-q4_K_M",
        transport,
    )
    path = tmp_path / "ollama-lock.json"
    write_local_model_lock(path, identity)

    assert load_local_model_lock(path) == identity
    assert identity.digest == "sha256:" + digest
    assert transport.get_calls == ["http://localhost:11434/api/tags"]


@pytest.mark.parametrize(
    ("external", "expected"),
    (
        ("a" * 64, "sha256:" + "a" * 64),
        ("sha256:" + "b" * 64, "sha256:" + "b" * 64),
        ("SHA256:" + "C" * 64, "sha256:" + "c" * 64),
    ),
)
def test_normalize_sha256_digest_canonicalizes_api_values(external: str, expected: str) -> None:
    assert normalize_sha256_digest(external) == expected


@pytest.mark.parametrize("external", ("a" * 12, "sha256:" + "g" * 64, "sha256:" + "a" * 63))
def test_normalize_sha256_digest_rejects_short_or_malformed_values(external: str) -> None:
    with pytest.raises(ValueError):
        normalize_sha256_digest(external)


def test_ollama_identity_lock_rejects_tag_mismatch() -> None:
    transport = FakeOllamaTransport(
        response={},
        identity_response={
            "models": [
                {"name": "different-model", "digest": "sha256:" + "d" * 64},
            ]
        },
    )

    with pytest.raises(ValueError, match="expected Ollama model tag"):
        inspect_ollama_model(
            "http://localhost:11434",
            "qwen3:4b-instruct-2507-q4_K_M",
            transport,
        )


def test_ollama_generation_rejects_installed_digest_mismatch(tmp_path: Path) -> None:
    locked = "sha256:" + "a" * 64
    installed = "sha256:" + "b" * 64
    lock_path = tmp_path / "ollama-lock.json"
    write_local_model_lock(
        lock_path,
        inspect_ollama_model(
            "http://localhost:11434",
            "qwen3:4b-instruct-2507-q4_K_M",
            FakeOllamaTransport(
                response={},
                identity_response={
                    "models": [
                        {"name": "qwen3:4b-instruct-2507-q4_K_M", "digest": locked},
                    ]
                },
            ),
        ),
    )
    transport = FakeOllamaTransport(
        response={"response": model_json()},
        identity_response={
            "models": [
                {"name": "qwen3:4b-instruct-2507-q4_K_M", "digest": installed},
            ]
        },
    )
    generator = OllamaGenerator(
        endpoint="http://localhost:11434/api/generate",
        model="qwen3:4b-instruct-2507-q4_K_M",
        immutable_digest=locked,
        transport=transport,
        local_lock_path=lock_path,
    )

    result = generator.generate(request())

    assert result.abstention_reason is AbstentionReason.INVALID_GENERATION
    assert transport.calls == []
