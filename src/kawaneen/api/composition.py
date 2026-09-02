"""Production serving composition over the frozen retrieval contracts.

This module deliberately reads only serving inputs and immutable model/configuration
locks.  It does not import an evaluation runner or load qrels.
"""

# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportArgumentType=false, reportPrivateUsage=false

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from kawaneen.api.runtime import ExpectedAssetUnavailable
from kawaneen.core.config import Settings
from kawaneen.observability.tracing import TraceObserver
from kawaneen.retrieval.hybrid.contracts import FusedCandidate, FusionConfig, RerankerConfig

if TYPE_CHECKING:
    from kawaneen.extraction.provider import ExtractionProvider
    from kawaneen.grounding.contracts import GeneratedDraft
    from kawaneen.retrieval.dense_models import DenseModelAdapter
    from kawaneen.retrieval.hybrid.reranker import BGERerankerAdapter
    from kawaneen.retrieval.serving import HybridServingRetriever


def _object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ExpectedAssetUnavailable("frozen retrieval configuration is unavailable") from error
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("frozen retrieval configuration is invalid") from error
    if not isinstance(value, dict):
        raise ValueError("frozen retrieval configuration must be an object")
    return cast(dict[str, object], value)


def _mapping(value: object, label: str) -> dict[str, object]:
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
        raise ExpectedAssetUnavailable("frozen retrieval configuration is unavailable") from error


@dataclass(frozen=True, slots=True)
class FrozenServingConfiguration:
    """The exact Phase 8 configuration permitted by the API boundary."""

    fusion: FusionConfig
    reranker: RerankerConfig
    phase8_selection_sha256: str
    phase8_config_sha256: str
    phase8_model_lock_sha256: str
    phase7_selection_sha256: str
    corpus_hash: str
    dense_model_id: str
    dense_model_revision: str


@dataclass(frozen=True, slots=True)
class ServingRetrievalBundle:
    retriever: HybridServingRetriever
    initialize: Callable[[], None]
    dense_model_id: str
    dense_revision: str
    reranker_model_id: str
    reranker_revision: str


@dataclass(frozen=True, slots=True)
class ServingGenerationBundle:
    generator: Callable[[str, object], GeneratedDraft | None]
    initialize: Callable[[], None]
    provider: str
    model: str
    revision: str


@dataclass(frozen=True, slots=True)
class ServingExtractionBundle:
    provider: ExtractionProvider
    initialize: Callable[[], None]
    provider_name: str
    model: str
    revision: str


def _local_ollama_identity(settings: Settings):
    from kawaneen.generation.ollama import load_local_model_lock

    path = (
        settings.artifacts_directory
        / "private"
        / "phase10_generation"
        / "qwen-ollama-model-lock.json"
    )
    if not path.is_file():
        raise ExpectedAssetUnavailable("locked local generation model is unavailable")
    return load_local_model_lock(path)


def build_stage_d_generation(settings: Settings) -> ServingGenerationBundle:
    """Build the locked Stage-D adapter without importing orchestration."""

    selected_path = (
        settings.data_directory / "manifests" / "generation" / "phase10_selected_configuration.json"
    )
    selected = _object(selected_path)
    if (
        selected.get("stage") != "stage_d"
        or selected.get("selection_status") != "selected_after_qualitative_review"
    ):
        raise ValueError("Phase 10 Stage-D selection is not active")
    model = _mapping(selected.get("model"), "Phase 10 Stage-D model")
    expected_model = _required_string(model.get("ollama_tag"), "Stage-D Ollama model")
    expected_digest = _required_string(model.get("ollama_digest"), "Stage-D Ollama digest")
    identity = _local_ollama_identity(settings)
    if identity.model != expected_model or identity.digest != expected_digest:
        raise ValueError("local Stage-D model does not match the frozen Phase 10 selection")
    from kawaneen.generation.ollama import OllamaDiagnosticTransportError, OllamaGenerator
    from kawaneen.generation.serving import StageDServingGenerator

    lock_path = (
        settings.artifacts_directory
        / "private"
        / "phase10_generation"
        / "qwen-ollama-model-lock.json"
    )
    provider = OllamaGenerator(
        endpoint=f"{settings.ollama_url.rstrip('/')}/api/generate",
        model=identity.model,
        immutable_digest=identity.digest,
        local_lock_path=lock_path,
        stage_d=True,
    )

    def initialize() -> None:
        try:
            provider.validate_local_lock()
        except (
            ConnectionError,
            FileNotFoundError,
            OllamaDiagnosticTransportError,
            OSError,
            TimeoutError,
        ) as error:
            raise ExpectedAssetUnavailable("locked local Stage-D model is unavailable") from error

    return ServingGenerationBundle(
        generator=StageDServingGenerator(provider),
        initialize=initialize,
        provider="ollama",
        model=identity.model,
        revision=_required_string(model.get("hf_revision"), "Stage-D model revision"),
    )


def build_hybrid_extraction(settings: Settings) -> ServingExtractionBundle:
    """Build the locked Phase-11 experimental provider when locally available."""

    config_path = (
        settings.data_directory
        / "manifests"
        / "extraction"
        / "phase11_hybrid_stage_b2_clean_config_v1.json"
    )
    config = _object(config_path)
    model = _required_string(config.get("model"), "Phase 11 hybrid model")
    digest = _required_string(config.get("ollama_digest"), "Phase 11 hybrid digest")
    hf_revision = _required_string(config.get("hf_revision"), "Phase 11 hybrid revision")
    identity = _local_ollama_identity(settings)
    if identity.model != model or identity.digest != digest:
        raise ValueError("local hybrid model does not match the frozen Phase 11 configuration")
    from kawaneen.extraction.provider import OllamaExtractionProvider
    from kawaneen.generation.ollama import OllamaDiagnosticTransportError

    lock_path = (
        settings.artifacts_directory
        / "private"
        / "phase10_generation"
        / "qwen-ollama-model-lock.json"
    )
    provider = OllamaExtractionProvider(
        endpoint=f"{settings.ollama_url.rstrip('/')}/api/generate",
        model=model,
        immutable_digest=digest,
        local_lock_path=lock_path,
    )

    def initialize() -> None:
        try:
            provider.preflight()
        except (
            ConnectionError,
            FileNotFoundError,
            OllamaDiagnosticTransportError,
            OSError,
            TimeoutError,
        ) as error:
            raise ExpectedAssetUnavailable(
                "locked hybrid extraction model is unavailable"
            ) from error

    return ServingExtractionBundle(
        provider=provider,
        initialize=initialize,
        provider_name="ollama",
        model=model,
        revision=hf_revision,
    )


def _single_vector_asset(root: Path) -> tuple[Path, Path]:
    vectors = sorted(root.glob("*/vectors.npy"))
    if not vectors:
        raise ExpectedAssetUnavailable("frozen dense retrieval vectors are unavailable")
    if len(vectors) != 1:
        raise ValueError("frozen dense retrieval vectors are ambiguous")
    vector_path = vectors[0]
    ids_path = vector_path.with_name("ids.json")
    if not ids_path.is_file():
        raise ExpectedAssetUnavailable("frozen dense retrieval IDs are unavailable")
    return vector_path, ids_path


def build_serving_retrieval(
    settings: Settings,
    configuration: FrozenServingConfiguration,
    *,
    dense_adapter: DenseModelAdapter | None = None,
    reranker_adapter: BGERerankerAdapter | None = None,
    observer: TraceObserver | None = None,
) -> ServingRetrievalBundle:
    """Build sparse+dense retrieval and a locked raw-logit reranker."""

    import numpy as np

    from kawaneen.retrieval.bm25 import BM25Index
    from kawaneen.retrieval.dense_models import BGEM3Adapter
    from kawaneen.retrieval.hybrid.reranker import BGERerankerAdapter
    from kawaneen.retrieval.qdrant_bootstrap import collection_name_for
    from kawaneen.retrieval.serving import HybridServingRetriever, load_serving_chunks
    from kawaneen.retrieval.vector_index import NumpyExactIndex

    private = settings.artifacts_directory / "private" / "phase7_retrieval"
    chunks = load_serving_chunks(private / "corpus" / "chunks.jsonl")
    vector_path, ids_path = _single_vector_asset(
        private / "embeddings" / "BAAI__bge-m3" / "arabic-raw-v1"
    )
    try:
        vectors = np.load(vector_path, allow_pickle=False)
        ids_value = json.loads(ids_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ExpectedAssetUnavailable("frozen dense retrieval vectors are unavailable") from error
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("frozen dense retrieval vectors are invalid") from error
    if not isinstance(ids_value, list) or any(not isinstance(item, str) for item in ids_value):
        raise ValueError("frozen dense retrieval IDs are invalid")
    dense_ids = tuple(cast(str, item) for item in ids_value)
    if len(dense_ids) != len(chunks) or set(dense_ids) != set(chunks):
        raise ValueError("frozen dense retrieval IDs do not match the serving corpus")
    if settings.dense_index_backend == "qdrant":
        from qdrant_client import QdrantClient

        from kawaneen.retrieval.qdrant_index import QdrantExactIndex

        dense_index = QdrantExactIndex.build(
            client=QdrantClient(url=settings.qdrant_url),
            collection_name=collection_name_for(configuration.corpus_hash),
            vectors=vectors,
            chunk_ids=dense_ids,
        )
    else:
        dense_index = NumpyExactIndex.build(vectors, dense_ids)
    dense = dense_adapter or BGEM3Adapter(revision=configuration.dense_model_revision)
    if dense.embedding_dimension and dense.embedding_dimension != vectors.shape[1]:
        raise ValueError("dense model dimension does not match the frozen index")

    policy_ids = {chunk.normalization_policy_id for chunk in chunks.values()}
    if len(policy_ids) != 1:
        raise ValueError("serving corpus has inconsistent sparse normalization policies")
    sparse_index = BM25Index.build(chunks.values(), next(iter(policy_ids)))
    reranker = reranker_adapter or BGERerankerAdapter(
        revision=configuration.reranker.model_revision,
        device=configuration.reranker.device,
        max_length=configuration.reranker.max_length,
    )

    def sparse_search(query: str, top_k: int):
        from kawaneen.retrieval.hybrid.contracts import SourceHit

        return tuple(
            SourceHit(item.chunk_id, item.score) for item in sparse_index.search(query, top_k=top_k)
        )

    def dense_search(query: str, top_k: int):
        from kawaneen.retrieval.hybrid.contracts import SourceHit

        vector = dense.encode_queries((query,), batch_size=1)[0]
        return tuple(
            SourceHit(item.chunk_id, item.score) for item in dense_index.search(vector, top_k=top_k)
        )

    def score(query: str, candidates: Sequence[FusedCandidate]) -> dict[str, float]:
        pairs = tuple((query, chunks[item.chunk_id].display_text) for item in candidates)
        values = reranker.score_pairs(pairs, batch_size=configuration.reranker.batch_size)
        return {item.chunk_id: float(value) for item, value in zip(candidates, values, strict=True)}

    retriever = HybridServingRetriever(
        chunks=chunks,
        sparse_search=sparse_search,
        dense_search=dense_search,
        reranker=score,
        fusion_config=configuration.fusion,
        observer=observer,
        reranker_model_id=configuration.reranker.model_id,
        reranker_model_revision=configuration.reranker.model_revision,
        reranker_scoring_contract=configuration.reranker.scoring_contract,
        reranker_serving_depth=configuration.reranker.serving_depth,
    )

    def initialize() -> None:
        try:
            dense.preload()
            reranker.preload()
            dense.encode_queries(("مرحبا",), batch_size=1)
            reranker.score_pairs((("مرحبا", "مرحبا"),), batch_size=1)
        except (FileNotFoundError, ImportError, OSError, RuntimeError) as error:
            raise ExpectedAssetUnavailable("retrieval model assets are unavailable") from error

    return ServingRetrievalBundle(
        retriever=retriever,
        initialize=initialize,
        dense_model_id=dense.model_id,
        dense_revision=dense.revision,
        reranker_model_id=configuration.reranker.model_id,
        reranker_revision=configuration.reranker.model_revision,
    )


def load_frozen_serving_configuration(data_directory: Path) -> FrozenServingConfiguration:
    """Load and cross-check the authoritative Phase 8 serving locks."""

    base = data_directory / "manifests" / "retrieval"
    final = _object(base / "phase8_final_manifest.json")
    selection_path = base / "phase8_dev_selection.json"
    selection = _object(selection_path)
    fusion_selection = _object(base / "phase8_dev_fusion_selection.json")
    model_lock_path = base / "phase8_model_lock.json"
    model_lock = _object(model_lock_path)

    if final.get("status") != "phase8_final_manifest_frozen":
        raise ValueError("Phase 8 final manifest is not frozen")
    if selection.get("status") != "phase8_dev_selection_frozen":
        raise ValueError("Phase 8 selection is not frozen")
    if fusion_selection.get("selected_fusion") != "rrf__s1_d025":
        raise ValueError("Phase 8 selected fusion is inconsistent")
    if fusion_selection.get("selected_dense_weight") != 0.25:
        raise ValueError("Phase 8 selected dense weight is inconsistent")
    if model_lock.get("status") != "immutable_model_revision_locked":
        raise ValueError("Phase 8 reranker lock is not immutable")

    final_hashes = _mapping(final.get("hashes"), "Phase 8 final hashes")
    selection_hash = _sha256(selection_path)
    model_lock_hash = _sha256(model_lock_path)
    if selection_hash != final_hashes.get("phase8_dev_selection_sha256"):
        raise ValueError("Phase 8 selection hash does not match the final manifest")
    if model_lock_hash != final_hashes.get("phase8_model_lock_sha256"):
        raise ValueError("Phase 8 model lock hash does not match the final manifest")

    final_configuration = _mapping(final.get("final_configuration"), "Phase 8 final configuration")
    selected_fusion = _mapping(final_configuration.get("fusion"), "Phase 8 final fusion")
    selected_reranker = _mapping(final_configuration.get("reranker"), "Phase 8 final reranker")
    selection_fusion = _mapping(selection.get("fusion"), "Phase 8 selected fusion")
    selection_reranker = _mapping(selection.get("reranker"), "Phase 8 selected reranker")
    if selected_fusion != selection_fusion or selected_reranker != selection_reranker:
        raise ValueError("Phase 8 final and selected configurations disagree")

    fusion = FusionConfig(
        sparse_weight=float(selected_fusion["sparse_weight"]),
        dense_weight=float(selected_fusion["dense_weight"]),
        rrf_k=int(selected_fusion["rrf_k"]),
        sparse_top_k=int(selected_fusion["sparse_top_k"]),
        dense_top_k=int(selected_fusion["dense_top_k"]),
        candidate_k=int(selected_fusion["candidate_k"]),
    )
    model_id = _required_string(selected_reranker.get("model_id"), "reranker model id")
    revision = _required_string(selected_reranker.get("revision"), "reranker revision")
    reranker = RerankerConfig(
        model_id=model_id,
        model_revision=revision,
        max_length=int(selected_reranker["max_length"]),
        candidate_count=int(selected_reranker["candidate_count"]),
        evaluation_depth=int(selected_reranker["evaluation_depth"]),
        serving_depth=int(selected_reranker["serving_depth"]),
        scoring_contract=_required_string(
            selected_reranker.get("scoring_contract"), "reranker scoring contract"
        ),
    )

    lock_model_id = _required_string(model_lock.get("model_id"), "reranker lock model id")
    lock_revision = _required_string(model_lock.get("revision"), "reranker lock revision")
    if (lock_model_id, lock_revision) != (reranker.model_id, reranker.model_revision):
        raise ValueError("Phase 8 reranker model lock disagrees with selected configuration")
    lock_contract = _mapping(model_lock.get("contract"), "Phase 8 reranker contract")
    for key, expected in {
        "candidate_count": reranker.candidate_count,
        "evaluation_depth": reranker.evaluation_depth,
        "max_length": reranker.max_length,
        "serving_depth": reranker.serving_depth,
        "scoring": "raw model logit",
    }.items():
        if lock_contract.get(key) != expected:
            raise ValueError(f"Phase 8 reranker lock disagrees for {key}")

    corpus_hash = _required_string(final_hashes.get("corpus_hash"), "Phase 8 corpus hash")
    phase7_selection_hash = _required_string(
        final_hashes.get("phase7_selection_sha256"), "Phase 7 selection hash"
    )
    phase8_config_hash = _required_string(
        final_hashes.get("phase8_config_sha256"), "Phase 8 config hash"
    )
    phase7_lock = _object(base / "phase7_model_lock.json")
    phase7_revisions = _mapping(phase7_lock.get("revisions"), "Phase 7 model revisions")
    dense_model_id = "BAAI/bge-m3"
    dense_model_revision = _required_string(
        phase7_revisions.get(dense_model_id), "Phase 7 BGE revision"
    )
    return FrozenServingConfiguration(
        fusion=fusion,
        reranker=reranker,
        phase8_selection_sha256=selection_hash,
        phase8_config_sha256=phase8_config_hash,
        phase8_model_lock_sha256=model_lock_hash,
        phase7_selection_sha256=phase7_selection_hash,
        corpus_hash=corpus_hash,
        dense_model_id=dense_model_id,
        dense_model_revision=dense_model_revision,
    )


__all__ = ["FrozenServingConfiguration", "load_frozen_serving_configuration"]
