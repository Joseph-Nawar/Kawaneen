"""Provider-side proposal boundary; no provider is loaded at import time."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, cast

from kawaneen.extraction.contracts import CandidateRegistry, SemanticProposal
from kawaneen.extraction.hybrid_prompt import (
    HYBRID_PROMPT_TEMPLATE_VERSION,
    HYBRID_QWEN_MODEL,
    HYBRID_QWEN_OLLAMA_DIGEST,
    HYBRID_RUNTIME_SETTINGS,
    render_hybrid_prompt,
)
from kawaneen.generation.ollama import (
    LOCAL_OLLAMA_LOCK_PATH,
    OllamaTransport,
    UrllibOllamaTransport,
    extract_ollama_response_text,
    inspect_ollama_model,
    load_local_model_lock,
)

QWEN_HF_ID = "Qwen/Qwen3-4B-Instruct-2507"
QWEN_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
QWEN_OLLAMA_TAG = "qwen3:4b-instruct-2507-q4_K_M"
QWEN_OLLAMA_DIGEST = HYBRID_QWEN_OLLAMA_DIGEST


class ExtractionProvider(Protocol):
    def propose(self, canonical_text: str, registry: CandidateRegistry) -> object: ...


def semantic_proposal_schema() -> dict[str, object]:
    return cast(dict[str, object], SemanticProposal.model_json_schema())


def parse_semantic_proposal(raw: object) -> SemanticProposal:
    value: object = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(value, Mapping):
        raise ValueError("provider response must be a JSON object")
    return SemanticProposal.model_validate(cast(Mapping[str, object], value))


class MockExtractionProvider:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls = 0

    def propose(self, canonical_text: str, registry: CandidateRegistry) -> object:
        del canonical_text, registry
        self.calls += 1
        return self.response


class OllamaExtractionProvider:
    """One-shot, locked localhost Ollama provider for the Phase 11B DEV run."""

    def __init__(
        self,
        *,
        endpoint: str = "http://127.0.0.1:11434/api/generate",
        model: str = HYBRID_QWEN_MODEL,
        immutable_digest: str = HYBRID_QWEN_OLLAMA_DIGEST,
        transport: OllamaTransport | None = None,
        local_lock_path: Path = LOCAL_OLLAMA_LOCK_PATH,
        prompt_template_version: str = HYBRID_PROMPT_TEMPLATE_VERSION,
    ) -> None:
        from urllib.parse import urlparse

        parsed = urlparse(endpoint)
        if parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Ollama endpoint must be localhost HTTP")
        self.endpoint = endpoint
        self.identity_endpoint = f"{parsed.scheme}://{parsed.netloc}"
        self.model = model
        self.immutable_digest = immutable_digest
        self.transport = transport or UrllibOllamaTransport(timeout_seconds=30.0)
        self.local_lock_path = local_lock_path
        self.prompt_template_version = prompt_template_version
        self.calls = 0
        self.last_raw_response: str | None = None

    def preflight(self) -> None:
        lock = load_local_model_lock(self.local_lock_path)
        if lock.model != self.model or lock.digest != self.immutable_digest:
            raise ValueError("local Ollama model lock does not match Phase 11B configuration")
        installed = inspect_ollama_model(self.identity_endpoint, self.model, self.transport)
        if installed != lock:
            raise ValueError("installed Ollama model differs from the local immutable lock")

    def propose(self, canonical_text: str, registry: CandidateRegistry) -> object:
        prompt = render_hybrid_prompt(
            canonical_text, registry, self.prompt_template_version
        )
        payload: dict[str, object] = {
            "model": self.model,
            "prompt": prompt.text,
            "stream": False,
            "format": semantic_proposal_schema(),
            "options": {
                "temperature": HYBRID_RUNTIME_SETTINGS["temperature"],
                "top_p": HYBRID_RUNTIME_SETTINGS["top_p"],
                "seed": HYBRID_RUNTIME_SETTINGS["seed"],
                "num_predict": HYBRID_RUNTIME_SETTINGS["num_predict"],
            },
        }
        self.calls += 1
        response = self.transport.post_json(self.endpoint, payload)
        self.last_raw_response = extract_ollama_response_text(response)
        return self.last_raw_response


class NoInferenceQwenProvider:
    """Configuration-only provider that makes accidental Phase 11A inference impossible."""

    def propose(self, canonical_text: str, registry: CandidateRegistry) -> object:
        del canonical_text, registry
        raise RuntimeError("Qwen/Ollama inference is intentionally disabled in Phase 11A")
