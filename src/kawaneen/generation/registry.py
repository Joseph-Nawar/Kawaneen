"""Explicit local model candidates and immutable lock helpers."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import cast

from kawaneen.generation.contracts import ModelCandidate, TokenizerFingerprint

_FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
_OLLAMA_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
GENERATION_LOCK_PATH = Path("data/manifests/generation/phase10_model_generator_lock.json")


def default_model_registry() -> tuple[ModelCandidate, ...]:
    return (
        ModelCandidate(
            name="qwen3-4b-instruct-2507",
            hf_identity="Qwen/Qwen3-4B-Instruct-2507",
            ollama_model="qwen3:4b-instruct-2507-q4_K_M",
            role="primary-local-candidate",
        ),
        ModelCandidate(
            name="fanar-1-9b-instruct",
            hf_identity="QCRI/Fanar-1-9B-Instruct",
            role="optional-arabic-challenger",
        ),
    )


def resolve_hf_revision(
    candidate: ModelCandidate,
    resolver: Callable[[str], str],
) -> ModelCandidate:
    return lock_hf_revision(candidate, resolver(candidate.hf_identity))


def resolve_hf_revision_from_hub(candidate: ModelCandidate) -> ModelCandidate:
    """Resolve only repository metadata; this does not download model weights."""

    from huggingface_hub import HfApi

    info = cast(object, HfApi().model_info(candidate.hf_identity))
    revision = getattr(info, "sha", None)
    if not isinstance(revision, str):
        raise ValueError(f"Hugging Face revision SHA unavailable for {candidate.hf_identity}")
    return lock_hf_revision(candidate, revision)


def lock_hf_revision(candidate: ModelCandidate, revision: str) -> ModelCandidate:
    if _FULL_SHA.fullmatch(revision) is None:
        raise ValueError("HF revision must be a full 40-character SHA")
    return candidate.model_copy(update={"hf_revision": revision})


def lock_ollama_digest(candidate: ModelCandidate, digest: str) -> ModelCandidate:
    if candidate.ollama_model is None:
        raise ValueError("candidate has no Ollama model name")
    if _OLLAMA_DIGEST.fullmatch(digest) is None:
        raise ValueError("Ollama digest must have the sha256:<64 hex> form")
    return candidate.model_copy(update={"ollama_digest": digest})


def load_generation_lock(
    path: Path = GENERATION_LOCK_PATH,
) -> tuple[ModelCandidate, TokenizerFingerprint]:
    """Load the immutable Qwen model and matching tokenizer metadata lock."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("generation lock is unavailable") from error
    if not isinstance(payload, dict):
        raise ValueError("generation lock is not an object")
    root = cast(dict[str, object], payload)
    qwen_value = root.get("qwen")
    tokenizer_value = root.get("tokenizer")
    if not isinstance(qwen_value, dict) or not isinstance(tokenizer_value, dict):
        raise ValueError("generation lock is missing Qwen/tokenizer records")
    qwen = cast(dict[str, object], qwen_value)
    tokenizer = cast(dict[str, object], tokenizer_value)
    hf_identity = qwen.get("hf_model_id")
    revision = qwen.get("hf_revision")
    model_tag = qwen.get("ollama_model_tag")
    if not all(isinstance(value, str) for value in (hf_identity, revision, model_tag)):
        raise ValueError("Qwen generation lock is incomplete")
    hf_identity_value = cast(str, hf_identity)
    revision_value = cast(str, revision)
    model_tag_value = cast(str, model_tag)
    candidate = lock_hf_revision(
        ModelCandidate(
            name=str(qwen.get("name", "qwen3-4b-instruct-2507")),
            hf_identity=hf_identity_value,
            ollama_model=model_tag_value,
            role=str(qwen.get("role", "primary-local-candidate")),
            ollama_digest=(
                str(qwen["ollama_digest"])
                if isinstance(qwen.get("ollama_digest"), str)
                else None
            ),
        ),
        revision_value,
    )
    tokenizer_identity = tokenizer.get("qwen_model_id")
    tokenizer_revision = tokenizer.get("immutable_revision")
    if not isinstance(tokenizer_identity, str) or not isinstance(tokenizer_revision, str):
        raise ValueError("Qwen tokenizer lock is incomplete")
    if tokenizer_identity != candidate.hf_identity or tokenizer_revision != candidate.hf_revision:
        raise ValueError("Qwen model and tokenizer revisions do not match")
    return candidate, TokenizerFingerprint(
        identity=tokenizer_identity,
        revision=lock_hf_revision(candidate, tokenizer_revision).hf_revision,
    )
