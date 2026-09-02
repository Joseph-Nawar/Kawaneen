"""Content-addressed identity for the frozen Kawaneen serving configuration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TypeVar, cast

from kawaneen.generation.contracts import STAGE_D_GENERATION_SETTINGS
from kawaneen.generation.prompt import (
    STAGE_D_PROMPT_TEMPLATE_VERSION,
    stage_d_generation_version_hash,
)

SCHEMA_VERSION = "phase16-serving-identity-v1"


@dataclass(frozen=True, slots=True)
class EmbeddingIdentity:
    model_id: str
    revision: str


@dataclass(frozen=True, slots=True)
class RetrievalIdentity:
    strategy: str
    sparse_top_k: int
    dense_top_k: int
    rrf_k: int
    sparse_weight: float
    dense_weight: float
    fused_candidate_count: int


@dataclass(frozen=True, slots=True)
class RerankerIdentity:
    model_id: str
    revision: str
    candidate_count: int
    serving_depth: int
    scoring_contract: str


@dataclass(frozen=True, slots=True)
class GeneratorIdentity:
    provider: str
    model: str
    revision: str
    hf_id: str
    hf_revision: str
    immutable_digest: str


@dataclass(frozen=True, slots=True)
class PromptIdentity:
    template_version: str
    version_hash: str


@dataclass(frozen=True, slots=True)
class AnswerabilityIdentity:
    version: str
    hash: str


@dataclass(frozen=True, slots=True)
class SourceConfigurationHashes:
    phase7_selection_sha256: str
    phase8_selection_sha256: str
    phase8_config_sha256: str
    phase8_model_lock_sha256: str


@dataclass(frozen=True, slots=True)
class ServingIdentity:
    schema_version: str
    corpus_version: str
    embedding: EmbeddingIdentity
    retrieval: RetrievalIdentity
    reranker: RerankerIdentity
    generator: GeneratorIdentity
    prompt: PromptIdentity
    answerability: AnswerabilityIdentity
    source_configuration_hashes: SourceConfigurationHashes
    _configuration_version: str = field(default="", repr=False, compare=False)

    @property
    def configuration_version(self) -> str:
        """Return the content address for the current identity fields."""

        return _content_hash(self._payload())

    @classmethod
    def build(cls, data_directory: Path = Path("data")) -> ServingIdentity:
        """Build identity from the public Phase 7/8/10 serving locks."""

        from kawaneen.api.composition import load_frozen_serving_configuration

        configuration = load_frozen_serving_configuration(data_directory)
        selected_path = (
            data_directory / "manifests" / "generation" / "phase10_selected_configuration.json"
        )
        selected = _load_object(selected_path)
        model = _object(selected.get("model"), "Phase 10 selected model")
        answerability = _object(selected.get("answerability"), "Phase 10 answerability")
        hf_id = _required_string(model.get("hf_id"), "Phase 10 model hf_id")
        hf_revision = _required_string(model.get("hf_revision"), "Phase 10 model hf_revision")
        ollama_digest = _required_string(model.get("ollama_digest"), "Phase 10 model ollama_digest")
        source_hashes = SourceConfigurationHashes(
            phase7_selection_sha256=_sha256(
                data_directory / "manifests" / "retrieval" / "phase7_dev_selection.json"
            ),
            phase8_selection_sha256=_sha256(
                data_directory / "manifests" / "retrieval" / "phase8_dev_selection.json"
            ),
            phase8_config_sha256=_sha256(
                data_directory.parent / "configs" / "retrieval" / "phase8_hybrid.toml"
            ),
            phase8_model_lock_sha256=_sha256(
                data_directory / "manifests" / "retrieval" / "phase8_model_lock.json"
            ),
        )
        identity_without_version = {
            "schema_version": SCHEMA_VERSION,
            "corpus_version": configuration.corpus_hash,
            "embedding": {
                "model_id": configuration.dense_model_id,
                "revision": configuration.dense_model_revision,
            },
            "retrieval": {
                "strategy": "hybrid_reranked",
                "sparse_top_k": configuration.fusion.sparse_top_k,
                "dense_top_k": configuration.fusion.dense_top_k,
                "rrf_k": configuration.fusion.rrf_k,
                "sparse_weight": configuration.fusion.sparse_weight,
                "dense_weight": configuration.fusion.dense_weight,
                "fused_candidate_count": configuration.fusion.candidate_k,
            },
            "reranker": {
                "model_id": configuration.reranker.model_id,
                "revision": configuration.reranker.model_revision,
                "candidate_count": configuration.reranker.candidate_count,
                "serving_depth": configuration.reranker.serving_depth,
                "scoring_contract": configuration.reranker.scoring_contract,
            },
            "generator": {
                "provider": "ollama",
                "model": _required_string(model.get("ollama_tag"), "Phase 10 model ollama_tag"),
                "revision": hf_revision,
                "hf_id": hf_id,
                "hf_revision": hf_revision,
                "immutable_digest": ollama_digest,
            },
            "prompt": {
                "template_version": STAGE_D_PROMPT_TEMPLATE_VERSION,
                "version_hash": stage_d_generation_version_hash(STAGE_D_GENERATION_SETTINGS),
            },
            "answerability": {
                "version": _required_string(answerability.get("version"), "answerability version"),
                "hash": _required_string(answerability.get("hash"), "answerability hash"),
            },
            "source_configuration_hashes": asdict(source_hashes),
        }
        configuration_version = _content_hash(identity_without_version)
        return cls.from_mapping(
            {**identity_without_version, "configuration_version": configuration_version}
        )

    @classmethod
    def from_mapping(cls, value: object) -> ServingIdentity:
        raw = _object(value, "serving identity")
        if raw.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
        configuration_version = _required_string(
            raw.get("configuration_version"), "configuration_version"
        )
        payload = dict(raw)
        payload.pop("configuration_version", None)
        expected = _content_hash(payload)
        if configuration_version != expected:
            raise ValueError("serving identity configuration_version does not match content")
        result = cls(
            schema_version=_required_string(raw.get("schema_version"), "schema_version"),
            corpus_version=_required_string(raw.get("corpus_version"), "corpus_version"),
            embedding=_nested(EmbeddingIdentity, raw.get("embedding"), "embedding"),
            retrieval=_nested(RetrievalIdentity, raw.get("retrieval"), "retrieval"),
            reranker=_nested(RerankerIdentity, raw.get("reranker"), "reranker"),
            generator=_nested(GeneratorIdentity, raw.get("generator"), "generator"),
            prompt=_nested(PromptIdentity, raw.get("prompt"), "prompt"),
            answerability=_nested(AnswerabilityIdentity, raw.get("answerability"), "answerability"),
            source_configuration_hashes=_nested(
                SourceConfigurationHashes,
                raw.get("source_configuration_hashes"),
                "source_configuration_hashes",
            ),
        )
        if result.configuration_version != configuration_version:
            raise ValueError("serving identity configuration_version does not match content")
        return result

    def to_dict(self) -> dict[str, object]:
        payload = self._payload()
        payload["configuration_version"] = self.configuration_version
        return payload

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    @property
    def source_paths(self) -> tuple[str, ...]:
        return (
            "data/manifests/retrieval/phase7_dev_selection.json",
            "data/manifests/retrieval/phase8_dev_selection.json",
            "configs/retrieval/phase8_hybrid.toml",
            "data/manifests/retrieval/phase8_model_lock.json",
        )

    def _payload(self) -> dict[str, object]:
        payload = cast(dict[str, object], asdict(self))
        payload.pop("_configuration_version", None)
        return payload


def load_serving_identity(path: Path) -> ServingIdentity:
    try:
        return ServingIdentity.from_mapping(json.loads(path.read_text(encoding="utf-8")))
    except FileNotFoundError as error:
        raise ValueError(f"serving identity is missing: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError("serving identity is not valid JSON") from error


def write_serving_identity(path: Path, identity: ServingIdentity) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(identity.canonical_bytes() + b"\n")


def verify_tracked_serving_identity(data_directory: Path, path: Path) -> ServingIdentity:
    expected = ServingIdentity.build(data_directory)
    actual = load_serving_identity(path)
    if actual != expected:
        raise ValueError("tracked serving identity differs from authoritative sources")
    return actual


T = TypeVar("T")


def _nested(type_: type[T], value: object, label: str) -> T:
    raw = _object(value, label)
    try:
        return type_(**raw)
    except TypeError as error:
        raise ValueError(f"{label} has invalid fields") from error


def _load_object(path: Path) -> dict[str, object]:
    try:
        return _object(json.loads(path.read_text(encoding="utf-8")), str(path))
    except FileNotFoundError as error:
        raise ValueError(f"authoritative identity source is missing: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"authoritative identity source is invalid JSON: {path}") from error


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return cast(dict[str, object], value)


def _required_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise ValueError(f"authoritative identity source is unavailable: {path}") from error


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _content_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


__all__ = [
    "AnswerabilityIdentity",
    "EmbeddingIdentity",
    "GeneratorIdentity",
    "PromptIdentity",
    "RerankerIdentity",
    "RetrievalIdentity",
    "ServingIdentity",
    "SourceConfigurationHashes",
    "load_serving_identity",
    "verify_tracked_serving_identity",
    "write_serving_identity",
]
