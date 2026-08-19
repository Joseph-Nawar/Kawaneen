# pyright: basic, reportIndexIssue=false, reportArgumentType=false
"""Phase-8 DEV fusion orchestration and protected reranker entry points."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
import tomllib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from kawaneen.retrieval.bm25 import BM25Index
from kawaneen.retrieval.cache import load_cached_embeddings
from kawaneen.retrieval.corpus import load_phase7_release
from kawaneen.retrieval.dense_models import BGEM3Adapter
from kawaneen.retrieval.evaluation import evaluate_rankings
from kawaneen.retrieval.evidence import evidence_groups_to_chunks
from kawaneen.retrieval.hybrid.artifacts import write_json_atomic
from kawaneen.retrieval.hybrid.checkpoints import CheckpointStore, checkpoint_status
from kawaneen.retrieval.hybrid.contracts import (
    FusedCandidate,
    FusionConfig,
    RerankerConfig,
    SourceHit,
)
from kawaneen.retrieval.hybrid.evaluation import (
    candidate_complete_evidence_recall_at_k,
    candidate_recall_at_k,
    paired_comparison,
    provenance_fractions,
    rescue_damage_counts,
    select_reranker_pipeline,
)
from kawaneen.retrieval.hybrid.fusion import fuse_ranked_hits
from kawaneen.retrieval.hybrid.metadata import DocumentMetadata, metadata_coverage
from kawaneen.retrieval.hybrid.reranker import BGERerankerAdapter, rerank_candidates
from kawaneen.retrieval.models import RetrievalChunk
from kawaneen.retrieval.slices import QueryLengthBins, assign_slices
from kawaneen.retrieval.tokenization import represent
from kawaneen.retrieval.vector_index import NumpyExactIndex

PHASE8_CONFIG = Path("configs/retrieval/phase8_hybrid.toml")
PHASE7_SELECTION = Path("data/manifests/retrieval/phase7_dev_selection.json")
PHASE7_CORPUS = Path("data/manifests/retrieval/phase7_corpus_manifest.json")
PHASE7_MODEL_LOCK = Path("data/manifests/retrieval/phase7_model_lock.json")
PHASE7_BGE_CACHE = Path(
    "artifacts/private/phase7_retrieval/embeddings/BAAI__bge-m3/arabic-raw-v1/"
    "797830a20035acb251f33f9048725353c77fff417e8b58bab5f72252e6d7230b"
)
PHASE8_PRIVATE = Path("artifacts/private/phase8_retrieval")
PHASE8_METRICS = Path("data/evaluation/phase8_dev_fusion_metrics.json")
PHASE8_SELECTION = Path("data/manifests/retrieval/phase8_dev_fusion_selection.json")
PHASE8_MODEL_LOCK = Path("data/manifests/retrieval/phase8_model_lock.json")
PHASE8_METADATA = Path("data/manifests/retrieval/phase8_metadata_coverage.json")
PHASE8_RERANK_METRICS = Path("data/evaluation/phase8_dev_reranker_metrics.json")
PHASE8_DEV_SELECTION = Path("data/manifests/retrieval/phase8_dev_selection.json")
PHASE8_HOLDOUT_READINESS = Path("data/manifests/retrieval/phase8_holdout_readiness.json")
PHASE8_RERANKER_MODEL_ID = "BAAI/bge-reranker-v2-m3"
FULL_REVISION_RE = re.compile(r"[0-9a-f]{40}")
EXPECTED_PHASE7_SELECTION_SHA256 = (
    "bb9c58833f30ca9f4066bbb55339e2f74663fcaeae1a623892b6ce441b3a5fae"
)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_phase8_reranker_lock(path: Path = PHASE8_MODEL_LOCK) -> dict[str, Any]:
    """Load the Phase-8 reranker lock without loading model assets."""
    lock = _json(path)
    if lock.get("model_id") != PHASE8_RERANKER_MODEL_ID:
        raise ValueError("Phase-8 reranker model ID does not match the fixed contract")
    revision = lock.get("revision")
    if not isinstance(revision, str) or FULL_REVISION_RE.fullmatch(revision) is None:
        raise ValueError("Phase-8 reranker revision must be a full 40-character SHA")
    config = tomllib.loads(PHASE8_CONFIG.read_text(encoding="utf-8"))
    reranker_config = config.get("reranker", {})
    if reranker_config.get("model_id") != PHASE8_RERANKER_MODEL_ID:
        raise ValueError("Phase-8 config reranker model ID does not match the model lock")
    if reranker_config.get("model_revision") != revision:
        raise ValueError("Phase-8 config reranker revision does not match the model lock")
    return lock


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_phase7_inputs(
    *,
    selection_path: Path = PHASE7_SELECTION,
    corpus_path: Path = PHASE7_CORPUS,
    model_lock_path: Path = PHASE7_MODEL_LOCK,
    cache_path: Path = PHASE7_BGE_CACHE,
) -> dict[str, object]:
    if not selection_path.is_file() or _sha256(selection_path) != EXPECTED_PHASE7_SELECTION_SHA256:
        raise ValueError("Phase 7 selection SHA does not match the frozen selection SHA")
    for path in (corpus_path, model_lock_path):
        if not path.is_file():
            raise ValueError(f"required Phase 7 artifact is missing: {path}")
    selection = _json(selection_path)
    corpus = _json(corpus_path)
    lock = _json(model_lock_path)
    if selection.get("corpus_hash") != corpus.get("corpus_hash"):
        raise ValueError("Phase 7 corpus hash does not match frozen selection")
    if lock.get("revisions", {}).get("BAAI/bge-m3") != selection.get("model_revisions", {}).get(
        "BAAI/bge-m3"
    ):
        raise ValueError("Phase 7 BGE revision does not match frozen selection")
    if lock.get("contracts", {}).get("BAAI/bge-m3", {}).get("max_length") != 1536:
        raise ValueError("Phase 7 BGE max-length contract is not 1536")
    if not cache_path.is_dir():
        raise ValueError(f"required Phase 7 BGE cache is missing: {cache_path}")
    for filename in ("metadata.json", "ids.json", "vectors.npy"):
        if not (cache_path / filename).is_file():
            raise ValueError(f"required Phase 7 BGE cache file is missing: {cache_path / filename}")
    return {
        "selection": selection,
        "corpus": corpus,
        "model_lock": lock,
        "selection_sha256": _sha256(selection_path),
    }


def _configs() -> tuple[tuple[str, FusionConfig], ...]:
    return (
        ("bm25__phase7", FusionConfig(sparse_weight=1.0, dense_weight=0.0)),
        ("bge__phase7", FusionConfig(sparse_weight=0.0, dense_weight=1.0)),
        ("rrf__s1_d1", FusionConfig(sparse_weight=1.0, dense_weight=1.0)),
        ("rrf__s1_d025", FusionConfig(sparse_weight=1.0, dense_weight=0.25)),
        ("rrf__s1_d050", FusionConfig(sparse_weight=1.0, dense_weight=0.50)),
        ("rrf__s1_d075", FusionConfig(sparse_weight=1.0, dense_weight=0.75)),
        ("rrf__s1_d100", FusionConfig(sparse_weight=1.0, dense_weight=1.00)),
    )


def _metric_payload(result: Any) -> dict[str, object]:
    return {
        "metrics": dict(result.metrics),
        "sample_count": result.sample_count,
        "unanswerable_count": result.unanswerable_count,
        "slices": {
            dimension: {label: dict(values) for label, values in labels.items()}
            for dimension, labels in result.slices.items()
        },
    }


def _candidate_diagnostics(
    items: Sequence[Any],
    candidate_rows: Mapping[str, Sequence[Mapping[str, object]]],
    candidate_rankings: Mapping[str, Sequence[str]],
    bm25_rankings: Mapping[str, Sequence[str]],
    chunks: Sequence[RetrievalChunk],
) -> dict[str, object]:
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    unit_to_chunks: defaultdict[str, set[str]] = defaultdict(set)
    for chunk in chunks:
        for unit_id in chunk.source_unit_ids:
            unit_to_chunks[unit_id].add(chunk.chunk_id)
    qrels: dict[str, dict[str, int]] = {}
    recalls: list[float] = []
    cers: list[float] = []
    provenances: list[str] = []
    for item in items:
        if item.answerability.value != "answerable":
            continue
        qrels[item.query_id] = {qrel.chunk_id: int(qrel.grade) for qrel in item.chunk_qrels}
        ids = tuple(candidate_rankings.get(item.query_id, ()))
        recalls.append(candidate_recall_at_k(ids, qrels[item.query_id], 20))
        groups = evidence_groups_to_chunks(item, unit_to_chunks, chunks_by_id)
        cers.append(candidate_complete_evidence_recall_at_k(ids, tuple(groups.values()), 20))
        provenances.extend(
            str(row.get("provenance", "")) for row in candidate_rows.get(item.query_id, ())
        )
    return {
        "CandidateRecall@20": sum(recalls) / max(len(recalls), 1),
        "CandidateCompleteEvidenceRecall@20": sum(cers) / max(len(cers), 1),
        "provenance_fraction": provenance_fractions(provenances),
        "rescue_damage": rescue_damage_counts(bm25_rankings, candidate_rankings, qrels),
    }


def run_dev_fusion() -> dict[str, object]:
    """Run fixed DEV fusion experiments using only frozen Phase-7 inputs."""
    validated = validate_phase7_inputs()
    release = load_phase7_release()
    dev_items = release.split_items("dev")
    selection = validated["selection"]
    bins = QueryLengthBins.from_dict(selection["query_length_bins"])
    source_by_document = {chunk.document_id: chunk.source_id for chunk in release.chunks}
    bm25 = BM25Index.build(release.chunks, "arabic-light-v1", k1=1.2, b=0.75)
    vectors, ids = load_cached_embeddings(
        PHASE7_BGE_CACHE, fingerprint=str(selection["cache_fingerprints"]["bge"])
    )
    dense = NumpyExactIndex.build(vectors, ids)
    lock = validated["model_lock"]
    bge = BGEM3Adapter(
        revision=str(lock["revisions"]["BAAI/bge-m3"]), max_length=1536, device="cpu"
    )
    query_vectors = bge.encode_queries(
        tuple(represent(item.query_text, "arabic-raw-v1").search_text for item in dev_items),
        batch_size=1,
    )
    sparse_hits: dict[str, tuple[SourceHit, ...]] = {}
    dense_hits: dict[str, tuple[SourceHit, ...]] = {}
    for item, vector in zip(dev_items, query_vectors, strict=True):
        sparse_hits[item.query_id] = tuple(
            SourceHit(hit.chunk_id, hit.score) for hit in bm25.search(item.query_text, top_k=50)
        )
        dense_hits[item.query_id] = tuple(
            SourceHit(hit.chunk_id, hit.score) for hit in dense.search(vector, top_k=50)
        )
    bm25_rankings = {
        query_id: tuple(hit.chunk_id for hit in hits[:10]) for query_id, hits in sparse_hits.items()
    }
    bge_rankings = {
        query_id: tuple(hit.chunk_id for hit in hits[:10]) for query_id, hits in dense_hits.items()
    }
    methods: dict[str, object] = {}
    for name, config in _configs():
        candidate_rows: dict[str, list[dict[str, object]]] = {}
        rankings: dict[str, list[str]] = {}
        for item in dev_items:
            fused = fuse_ranked_hits(
                sparse=sparse_hits[item.query_id], dense=dense_hits[item.query_id], config=config
            )
            candidate_rows[item.query_id] = [
                {
                    "chunk_id": row.chunk_id,
                    "fused_rank": row.fused_rank,
                    "fused_score": row.fused_score,
                    "sparse_rank": row.sparse_rank,
                    "sparse_score": row.sparse_score,
                    "dense_rank": row.dense_rank,
                    "dense_score": row.dense_score,
                    "provenance": row.provenance,
                }
                for row in fused
            ]
            rankings[item.query_id] = [row.chunk_id for row in fused]
        effective = (
            bm25_rankings
            if name == "bm25__phase7"
            else bge_rankings
            if name == "bge__phase7"
            else rankings
        )
        diagnostic_rankings = effective
        result = evaluate_rankings(
            dev_items,
            effective,
            chunks=release.chunks,
            query_length_bins=bins,
            source_by_document=source_by_document,
        )
        if name == "bm25__phase7":
            candidate_rows = {
                query_id: [
                    {"chunk_id": hit.chunk_id, "provenance": "sparse-only"}
                    for hit in sparse_hits[query_id]
                ]
                for query_id, values in effective.items()
            }
            diagnostic_rankings = {
                query_id: tuple(hit.chunk_id for hit in sparse_hits[query_id])
                for query_id in effective
            }
        elif name == "bge__phase7":
            candidate_rows = {
                query_id: [
                    {"chunk_id": hit.chunk_id, "provenance": "dense-only"}
                    for hit in dense_hits[query_id]
                ]
                for query_id, values in effective.items()
            }
            diagnostic_rankings = {
                query_id: tuple(hit.chunk_id for hit in dense_hits[query_id])
                for query_id in effective
            }
        methods[name] = {
            **_metric_payload(result),
            "candidate_diagnostics": _candidate_diagnostics(
                dev_items, candidate_rows, diagnostic_rankings, bm25_rankings, release.chunks
            ),
        }
        write_json_atomic(
            PHASE8_PRIVATE / "dev" / "rankings" / f"{name}.json",
            {"rankings": rankings, "candidate_rows": candidate_rows, "per_query": result.per_query},
        )
    bm25_metrics = methods["bm25__phase7"]["metrics"]
    eligible: list[tuple[str, float, float]] = []
    for name, config in _configs()[2:]:
        metrics = methods[name]["metrics"]
        if (
            float(metrics["Recall@10"]) >= float(bm25_metrics["Recall@10"]) - 0.01
            and float(metrics["CompleteEvidenceRecall@10"])
            >= float(bm25_metrics["CompleteEvidenceRecall@10"]) - 0.01
        ):
            eligible.append((name, float(metrics["nDCG@10"]), config.dense_weight))
    eligible.sort(key=lambda row: (-row[1], row[2]))
    selected = eligible[0][0] if eligible else None
    selection_payload = {
        "schema_version": 1,
        "status": "provisional_dev_fusion_selection",
        "phase7_selection_sha256": validated["selection_sha256"],
        "primary_metric": "nDCG@10",
        "guardrails": {
            "Recall@10_max_regression": 0.01,
            "CompleteEvidenceRecall@10_max_regression": 0.01,
        },
        "selected_fusion": selected,
        "selected_dense_weight": next((row[2] for row in eligible if row[0] == selected), None),
        "reranker_evaluated": False,
        "final_phase8_selection_frozen": False,
    }
    write_json_atomic(PHASE8_SELECTION, selection_payload, text_free=True)
    write_json_atomic(
        PHASE8_METRICS,
        {
            "schema_version": 1,
            "status": "dev_fusion_complete",
            "phase7_selection_sha256": validated["selection_sha256"],
            "methods": methods,
            "selection_candidate": selection_payload,
            "bootstrap": {"replicates": 2000, "seed": 20260815, "confidence": 0.95},
        },
        text_free=True,
    )
    locked_revision: str | None = None
    if PHASE8_MODEL_LOCK.is_file():
        existing_lock = _json(PHASE8_MODEL_LOCK)
        existing_revision = existing_lock.get("revision")
        if existing_revision is not None:
            if (
                not isinstance(existing_revision, str)
                or FULL_REVISION_RE.fullmatch(existing_revision) is None
            ):
                raise ValueError("existing Phase-8 reranker revision is not a full SHA")
            locked_revision = existing_revision
    write_json_atomic(
        PHASE8_MODEL_LOCK,
        {
            "schema_version": 1,
            "status": (
                "immutable_model_revision_locked"
                if locked_revision
                else "reranker_revision_pending_before_real_execution"
            ),
            "model_id": PHASE8_RERANKER_MODEL_ID,
            "revision": locked_revision,
            "contract": {
                "query_contract": "original query text",
                "passage_contract": "exact chunk display_text",
                "max_length": 1024,
                "candidate_count": 20,
                "batch_size": 4,
                "device": "cpu",
                "scoring": "raw model logit",
                "evaluation_depth": 10,
                "serving_depth": 8,
            },
        },
        text_free=True,
    )
    document_ids = tuple(sorted({chunk.document_id for chunk in release.chunks}))
    write_json_atomic(
        PHASE8_METADATA,
        {
            **metadata_coverage(
                (DocumentMetadata(document_id=document_id) for document_id in document_ids),
                expected_document_ids=document_ids,
            ),
            "status": "structured_metadata_not_present_in_phase7_release",
        },
        text_free=True,
    )
    return {"status": "dev_fusion_complete", "methods": methods, "selection": selection_payload}


def phase8_status() -> dict[str, object]:
    selection_path = PHASE8_DEV_SELECTION if PHASE8_DEV_SELECTION.is_file() else PHASE8_SELECTION
    return {
        "selection": _json(selection_path) if selection_path.is_file() else {"status": "missing"},
        "reranker_checkpoints": checkpoint_status(PHASE8_PRIVATE / "rerank"),
        "real_reranker_loaded": False,
    }


def rerank_dev(*, resume: bool, device: str) -> dict[str, object]:
    """Manual real-reranker entry point; refuses until revision is locked."""
    lock = load_phase8_reranker_lock()
    revision = str(lock["revision"])
    selection = _json(PHASE8_SELECTION)
    selected_name = str(selection["selected_fusion"])
    private = _json(PHASE8_PRIVATE / "dev" / "rankings" / f"{selected_name}.json")
    release = load_phase7_release()
    adapter = BGERerankerAdapter(revision=str(revision), device=device, max_length=1024)
    config = RerankerConfig(model_id=str(lock["model_id"]), model_revision=revision, device=device)
    candidate_ids = tuple(
        sorted(row["chunk_id"] for rows in private["candidate_rows"].values() for row in rows)
    )
    fingerprint = adapter.fingerprint(
        _sha256(PHASE8_CONFIG),
        EXPECTED_PHASE7_SELECTION_SHA256,
        candidate_ids,
        config,
        corpus_hash=str(release.corpus_manifest["corpus_hash"]),
        chunk_policy_hash=str(release.corpus_manifest["chunk_policy_hashes"][0]),
    )
    store = CheckpointStore(PHASE8_PRIVATE / "rerank", fingerprint=fingerprint)
    chunks = {chunk.chunk_id: chunk for chunk in release.chunks}
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(adapter.model_id, revision=str(revision))
    completed = 0
    for item in release.split_items("dev"):
        rows = private["candidate_rows"].get(item.query_id, [])
        candidates = tuple(FusedCandidate(**row) for row in rows)
        ordered_ids = tuple(candidate.chunk_id for candidate in candidates)
        query_fingerprint = adapter.fingerprint(
            _sha256(PHASE8_CONFIG),
            EXPECTED_PHASE7_SELECTION_SHA256,
            ordered_ids,
            config,
            corpus_hash=str(release.corpus_manifest["corpus_hash"]),
            chunk_policy_hash=str(release.corpus_manifest["chunk_policy_hashes"][0]),
            query_id=item.query_id,
        )
        if resume and store.valid(item.query_id, ordered_ids, query_fingerprint=query_fingerprint):
            completed += 1
            continue

        def scorer(query: str, passage: str) -> float:
            return adapter.score_pairs(((query, passage),), batch_size=config.batch_size)[0]

        ranked, diagnostics = rerank_candidates(
            item.query_text,
            candidates,
            chunks,
            scorer=scorer,
            tokenizer=tokenizer,
            config=config,
        )
        store.write(
            item.query_id,
            {
                "candidate_chunk_ids": list(ordered_ids),
                "query_fingerprint": query_fingerprint,
                "ranked_chunk_ids": [row.chunk_id for row in ranked],
                "scores": [row.score for row in ranked],
                "diagnostics": asdict(diagnostics),
            },
        )
        completed += 1
    return {"status": "rerank_complete", "completed_queries": completed, "resume": resume}


def _ranking_payload(name: str) -> dict[str, Any]:
    return _json(PHASE8_PRIVATE / "dev" / "rankings" / f"{name}.json")


def _candidate_rankings(payload: Mapping[str, object]) -> dict[str, tuple[str, ...]]:
    raw = payload.get("candidate_rows")
    if not isinstance(raw, Mapping):
        raise ValueError("Phase-8 candidate artifact has no candidate rows")
    return {
        str(query_id): tuple(
            str(row["chunk_id"]) for row in cast(Sequence[Mapping[str, object]], rows)
        )
        for query_id, rows in raw.items()
    }


def _candidate_rows(payload: Mapping[str, object]) -> dict[str, tuple[Mapping[str, object], ...]]:
    raw = payload.get("candidate_rows")
    if not isinstance(raw, Mapping):
        raise ValueError("Phase-8 candidate artifact has no candidate rows")
    return {
        str(query_id): tuple(cast(Sequence[Mapping[str, object]], rows))
        for query_id, rows in raw.items()
    }


def _candidate_stage_metrics(
    items: Sequence[Any],
    rankings: Mapping[str, Sequence[str]],
    chunks: Sequence[RetrievalChunk],
) -> dict[str, float]:
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    unit_to_chunks: defaultdict[str, set[str]] = defaultdict(set)
    for chunk in chunks:
        for unit_id in chunk.source_unit_ids:
            unit_to_chunks[unit_id].add(chunk.chunk_id)
    recalls: list[float] = []
    cers: list[float] = []
    for item in items:
        if item.answerability.value != "answerable":
            continue
        qrels = {qrel.chunk_id: int(qrel.grade) for qrel in item.chunk_qrels}
        candidate_ids = tuple(rankings.get(item.query_id, ()))
        recalls.append(candidate_recall_at_k(candidate_ids, qrels, 20))
        groups = evidence_groups_to_chunks(item, unit_to_chunks, chunks_by_id)
        cers.append(
            candidate_complete_evidence_recall_at_k(candidate_ids, tuple(groups.values()), 20)
        )
    return {
        "CandidateRecall@20": sum(recalls) / max(len(recalls), 1),
        "CandidateCompleteEvidenceRecall@20": sum(cers) / max(len(cers), 1),
    }


def _percentile(values: Sequence[int], fraction: float) -> int:
    ordered = sorted(values)
    if not ordered:
        return 0
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _robustness_report(
    items: Sequence[Any],
    rankings: Mapping[str, Sequence[str]],
    *,
    chunks: Sequence[RetrievalChunk],
    bins: QueryLengthBins,
    source_by_document: Mapping[str, str],
) -> dict[str, object]:
    by_intent = {item.intent_id: item for item in items if item.variant_id is None}
    predicates = {
        "simple_arabic": lambda item: (
            item.language.value == "ar" and item.register.value == "simple"
        ),
        "egyptian_arabic": lambda item: (
            item.language.value == "ar" and item.register.value == "egyptian"
        ),
        "english": lambda item: item.language.value == "en",
        "arabic_english_code_switch": lambda item: item.language.value == "ar-en",
    }
    result: dict[str, object] = {}
    for label, predicate in predicates.items():
        variants = tuple(
            item
            for item in items
            if item.variant_id is not None
            and item.answerability.value == "answerable"
            and predicate(item)
            and item.base_intent_id in by_intent
            and by_intent[item.base_intent_id].answerability.value == "answerable"
        )
        parents = tuple(by_intent[item.base_intent_id] for item in variants)
        parent_result = evaluate_rankings(
            parents,
            rankings,
            chunks=chunks,
            query_length_bins=bins,
            source_by_document=source_by_document,
        )
        variant_result = evaluate_rankings(
            variants,
            rankings,
            chunks=chunks,
            query_length_bins=bins,
            source_by_document=source_by_document,
        )
        parent_metrics = dict(parent_result.metrics)
        variant_metrics = dict(variant_result.metrics)
        result[label] = {
            "sample_count": len(variants),
            "small_sample_warning": len(variants) < 10,
            "parent": parent_metrics,
            "variant": variant_metrics,
            "variant_minus_parent": {
                metric: variant_metrics[metric] - parent_metrics[metric]
                for metric in parent_metrics
            },
        }
    return result


def _provenance_relevant_top10(
    items: Sequence[Any],
    rankings: Mapping[str, Sequence[str]],
    rows: Mapping[str, Sequence[Mapping[str, object]]],
) -> dict[str, object]:
    counts = {label: 0 for label in ("sparse-only", "dense-only", "both")}
    query_counts = {label: 0 for label in counts}
    for item in items:
        if item.answerability.value != "answerable":
            continue
        relevant = {qrel.chunk_id for qrel in item.chunk_qrels if int(qrel.grade) > 0}
        provenance = {str(row["chunk_id"]): str(row["provenance"]) for row in rows[item.query_id]}
        seen: set[str] = set()
        for chunk_id in rankings[item.query_id][:10]:
            label = provenance.get(chunk_id)
            if chunk_id in relevant and label in counts:
                counts[label] += 1
                seen.add(label)
        for label in seen:
            query_counts[label] += 1
    return {"relevant_hit_count": counts, "query_count": query_counts}


def validate_phase8_dev_reranker_artifacts() -> dict[str, object]:
    """Validate persisted DEV checkpoints without loading a reranker model."""
    validated = validate_phase7_inputs()
    release = load_phase7_release()
    items = release.split_items("dev")
    selection = _json(PHASE8_SELECTION)
    if selection.get("selected_fusion") != "rrf__s1_d025":
        raise ValueError("Phase-8 DEV reranker candidate source is not the fixed weighted RRF")
    lock = load_phase8_reranker_lock()
    config = tomllib.loads(PHASE8_CONFIG.read_text(encoding="utf-8"))
    fusion_config = cast(Mapping[str, object], config["fusion"])
    if {
        "rrf_k": int(fusion_config["rrf_k"]),
        "sparse_top_k": int(fusion_config["sparse_top_k"]),
        "dense_top_k": int(fusion_config["dense_top_k"]),
        "candidate_k": int(fusion_config["candidate_k"]),
    } != {"rrf_k": 60, "sparse_top_k": 50, "dense_top_k": 50, "candidate_k": 20}:
        raise ValueError("Phase-8 selected RRF depths do not match the fixed contract")
    if list(fusion_config["sparse_weights"]) != [1.0] or list(fusion_config["dense_weights"]) != [
        0.25,
        0.5,
        0.75,
        1.0,
    ]:
        raise ValueError("Phase-8 RRF weight ladder changed")
    reranker_toml = cast(Mapping[str, object], config["reranker"])
    if {
        "model_id": reranker_toml["model_id"],
        "model_revision": reranker_toml["model_revision"],
        "max_length": int(reranker_toml["max_length"]),
        "candidate_count": int(reranker_toml["candidate_count"]),
        "scoring_contract": reranker_toml["scoring_contract"],
    } != {
        "model_id": "BAAI/bge-reranker-v2-m3",
        "model_revision": str(lock["revision"]),
        "max_length": 1024,
        "candidate_count": 20,
        "scoring_contract": "raw-logit-v1",
    }:
        raise ValueError("Phase-8 reranker contract changed from the locked execution")
    candidate_payload = _ranking_payload("rrf__s1_d025")
    candidate_rows = _candidate_rows(candidate_payload)
    revision = str(lock["revision"])
    reranker_config = RerankerConfig(
        model_id=str(lock["model_id"]), model_revision=revision, device="cpu"
    )
    adapter = BGERerankerAdapter(revision=revision, device="cpu", max_length=1024)
    candidate_ids = tuple(
        sorted(row["chunk_id"] for rows in candidate_rows.values() for row in rows)
    )
    global_fingerprint = adapter.fingerprint(
        _sha256(PHASE8_CONFIG),
        EXPECTED_PHASE7_SELECTION_SHA256,
        candidate_ids,
        reranker_config,
        corpus_hash=str(release.corpus_manifest["corpus_hash"]),
        chunk_policy_hash=str(release.corpus_manifest["chunk_policy_hashes"][0]),
    )
    checkpoint_root = PHASE8_PRIVATE / "rerank"
    manifest = _json(checkpoint_root / "manifest.json")
    manifest_queries = manifest.get("queries", {})
    if not isinstance(manifest_queries, Mapping):
        raise ValueError("Phase-8 reranker checkpoint manifest query table is invalid")
    expected_ids = {item.query_id for item in items}
    corrupt: set[str] = set()
    duplicate_candidates: set[str] = set()
    missing: set[str] = expected_ids - set(str(query_id) for query_id in manifest_queries)
    chunks_by_id = {chunk.chunk_id for chunk in release.chunks}
    for item in items:
        query_id = item.query_id
        rows = candidate_rows.get(query_id, ())
        candidate_chunk_ids = tuple(str(row["chunk_id"]) for row in rows)
        fused_ranks = tuple(int(row["fused_rank"]) for row in rows)
        if (
            len(candidate_chunk_ids) != 20
            or len(set(candidate_chunk_ids)) != 20
            or fused_ranks != tuple(range(1, 21))
        ):
            duplicate_candidates.add(query_id)
        entry = manifest_queries.get(query_id)
        if not isinstance(entry, Mapping) or entry.get("status") != "completed":
            missing.add(query_id)
            continue
        path = checkpoint_root / str(entry.get("path", ""))
        try:
            payload = _json(path)
            query_fingerprint = adapter.fingerprint(
                _sha256(PHASE8_CONFIG),
                EXPECTED_PHASE7_SELECTION_SHA256,
                candidate_chunk_ids,
                reranker_config,
                corpus_hash=str(release.corpus_manifest["corpus_hash"]),
                chunk_policy_hash=str(release.corpus_manifest["chunk_policy_hashes"][0]),
                query_id=query_id,
            )
            ranked_ids = tuple(str(value) for value in payload.get("ranked_chunk_ids", ()))
            scores = tuple(float(value) for value in payload.get("scores", ()))
            diagnostics = payload.get("diagnostics", {})
            valid = (
                payload.get("query_id") == query_id
                and payload.get("fingerprint") == global_fingerprint
                and payload.get("query_fingerprint") == query_fingerprint
                and tuple(payload.get("candidate_chunk_ids", ())) == candidate_chunk_ids
                and tuple(entry.get("candidate_chunk_ids", ())) == candidate_chunk_ids
                and len(ranked_ids) == 20
                and len(set(ranked_ids)) == 20
                and set(ranked_ids) == set(candidate_chunk_ids)
                and set(ranked_ids).issubset(chunks_by_id)
                and len(scores) == len(ranked_ids)
                and all(math.isfinite(score) for score in scores)
                and isinstance(diagnostics, Mapping)
                and len(diagnostics.get("pair_token_counts", ())) == 20
                and all(
                    isinstance(value, int) and value > 0
                    for value in diagnostics.get("pair_token_counts", ())
                )
                and int(diagnostics.get("truncated_count", -1)) >= 0
            )
        except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
            valid = False
        if not valid:
            corrupt.add(query_id)
    extra = set(str(query_id) for query_id in manifest_queries) - expected_ids
    manifest_fingerprint_match = manifest.get("fingerprint") == global_fingerprint
    issues = corrupt | missing | duplicate_candidates
    if not manifest_fingerprint_match:
        issues.update(expected_ids)
    return {
        "status": "validated" if not issues and not extra else "invalid",
        "expected_query_count": len(items),
        "completed_query_count": len(expected_ids - missing),
        "valid_query_count": len(expected_ids - issues),
        "invalid_query_count": len(issues),
        "missing_query_count": len(missing),
        "corrupt_query_count": len(corrupt),
        "duplicate_candidate_query_count": len(duplicate_candidates),
        "extra_manifest_query_count": len(extra),
        "manifest_fingerprint_match": manifest_fingerprint_match,
        "candidate_source": "rrf__s1_d025",
        "model_id": str(lock["model_id"]),
        "model_revision": revision,
        "max_length": reranker_config.max_length,
        "scoring_contract": reranker_config.scoring_contract,
        "phase7_selection_sha256": validated["selection_sha256"],
        "checkpoint_fingerprint": global_fingerprint,
        "private_artifact_audit": {
            "no_model_load": True,
            "no_score_recomputation": True,
            "valid_scores_finite": not corrupt,
        },
    }


def finalize_phase8_dev_selection() -> dict[str, object]:
    """Evaluate existing DEV reranker artifacts and freeze the final DEV choice."""
    validation = validate_phase8_dev_reranker_artifacts()
    if validation["status"] != "validated":
        raise ValueError("Phase-8 DEV reranker checkpoint validation failed")
    validated = validate_phase7_inputs()
    release = load_phase7_release()
    items = release.split_items("dev")
    bins = QueryLengthBins.from_dict(
        cast(Mapping[str, object], validated["selection"])["query_length_bins"]
    )
    source_by_document = {chunk.document_id: chunk.source_id for chunk in release.chunks}
    selected_payload = _ranking_payload("rrf__s1_d025")
    baseline_payloads = {
        "bm25": _ranking_payload("bm25__phase7"),
        "bge": _ranking_payload("bge__phase7"),
    }
    rankings = {
        "bm25": _candidate_rankings(baseline_payloads["bm25"]),
        "bge": _candidate_rankings(baseline_payloads["bge"]),
        "rrf": _candidate_rankings(selected_payload),
    }
    checkpoint_manifest = _json(PHASE8_PRIVATE / "rerank" / "manifest.json")
    reranked_rankings = {
        str(query_id): tuple(
            str(value)
            for value in _json(PHASE8_PRIVATE / "rerank" / str(entry["path"])).get(
                "ranked_chunk_ids", ()
            )
        )
        for query_id, entry in cast(
            Mapping[str, Mapping[str, object]], checkpoint_manifest["queries"]
        ).items()
    }
    rankings["rrf_reranked"] = reranked_rankings
    candidate_rows = _candidate_rows(selected_payload)
    method_results: dict[str, Any] = {}
    for name, method_rankings in rankings.items():
        evaluation = evaluate_rankings(
            items,
            method_rankings,
            chunks=release.chunks,
            query_length_bins=bins,
            source_by_document=source_by_document,
        )
        candidate_stage = _candidate_stage_metrics(
            items,
            {query_id: ids[:20] for query_id, ids in method_rankings.items()},
            release.chunks,
        )
        method_results[name] = {
            **_metric_payload(evaluation),
            **candidate_stage,
            "per_query": {query_id: dict(row) for query_id, row in evaluation.per_query.items()},
        }
    if (
        method_results["rrf"]["CandidateRecall@20"]
        != method_results["rrf_reranked"]["CandidateRecall@20"]
        or method_results["rrf"]["CandidateCompleteEvidenceRecall@20"]
        != method_results["rrf_reranked"]["CandidateCompleteEvidenceRecall@20"]
    ):
        raise ValueError("reranking changed the candidate-stage metrics")
    answerable_ids = sorted(method_results["rrf"]["per_query"])
    bootstrap_metrics = ("nDCG@10", "MRR@10", "Recall@10", "CompleteEvidenceRecall@10")
    comparisons: dict[str, object] = {}
    for label, left_name, right_name in (
        ("bm25_vs_rrf", "bm25", "rrf"),
        ("rrf_vs_rrf_reranked", "rrf", "rrf_reranked"),
        ("bm25_vs_rrf_reranked", "bm25", "rrf_reranked"),
    ):
        comparisons[label] = {
            metric: paired_comparison(
                [
                    method_results[left_name]["per_query"][query_id][metric]
                    for query_id in answerable_ids
                ],
                [
                    method_results[right_name]["per_query"][query_id][metric]
                    for query_id in answerable_ids
                ],
            )
            for metric in bootstrap_metrics
        }
    qrels = {
        item.query_id: {qrel.chunk_id: int(qrel.grade) for qrel in item.chunk_qrels}
        for item in items
        if item.answerability.value == "answerable"
    }
    rescue_damage = {
        "fusion_relative_to_bm25": rescue_damage_counts(rankings["bm25"], rankings["rrf"], qrels),
        "reranker_relative_to_rrf": rescue_damage_counts(
            rankings["rrf"], rankings["rrf_reranked"], qrels
        ),
    }
    provenance = {
        "candidate_fraction": provenance_fractions(
            [str(row["provenance"]) for rows in candidate_rows.values() for row in rows]
        ),
        "relevant_top10": {
            "rrf": _provenance_relevant_top10(items, rankings["rrf"], candidate_rows),
            "rrf_reranked": _provenance_relevant_top10(
                items, rankings["rrf_reranked"], candidate_rows
            ),
        },
    }
    decision = select_reranker_pipeline(
        method_results["rrf"]["metrics"], method_results["rrf_reranked"]["metrics"]
    )
    for method in method_results.values():
        method.pop("per_query", None)
    robustness = {
        name: _robustness_report(
            items,
            rankings[name],
            chunks=release.chunks,
            bins=bins,
            source_by_document=source_by_document,
        )
        for name in rankings
    }
    checkpoint_files = [
        _json(PHASE8_PRIVATE / "rerank" / str(entry["path"]))
        for entry in cast(
            Mapping[str, Mapping[str, object]], checkpoint_manifest["queries"]
        ).values()
    ]
    pair_lengths = [
        int(value)
        for payload in checkpoint_files
        for value in cast(
            Sequence[object],
            cast(Mapping[str, object], payload["diagnostics"])["pair_token_counts"],
        )
    ]
    truncated_count = sum(
        int(cast(Mapping[str, object], payload["diagnostics"])["truncated_count"])
        for payload in checkpoint_files
    )
    diagnostics = {
        "total_query_passage_pairs": len(pair_lengths),
        "token_lengths": {
            "p50": _percentile(pair_lengths, 0.50),
            "p90": _percentile(pair_lengths, 0.90),
            "p95": _percentile(pair_lengths, 0.95),
            "p99": _percentile(pair_lengths, 0.99),
            "max": max(pair_lengths, default=0),
        },
        "truncated_count": truncated_count,
        "truncated_fraction": truncated_count / max(len(pair_lengths), 1),
        "latency": {
            "recorded": False,
            "p50_ms": None,
            "p95_ms": None,
            "total_inference_ms": None,
            "download_time_excluded": True,
        },
        "checkpoint_reuse": {
            "completed_queries": validation["completed_query_count"],
            "valid_queries": validation["valid_query_count"],
            "recomputed_by_this_evaluation": 0,
        },
        "model_revision": validation["model_revision"],
        "model_artifact_cache_size_bytes": None,
    }
    metadata = _json(PHASE8_METADATA)
    qrel_hash = str(release.phase6_manifest["hashes"]["evidence_qrels"])
    metrics_payload = {
        "schema_version": 1,
        "status": "dev_reranker_evaluation_complete",
        "phase7_selection_sha256": validated["selection_sha256"],
        "corpus_hash": release.corpus_manifest["corpus_hash"],
        "release_hash": release.corpus_manifest["release_hash"],
        "qrel_hash": qrel_hash,
        "methods": method_results,
        "candidate_invariant": {
            "candidate_set_unchanged_after_reranking": True,
            "rrf": {
                "CandidateRecall@20": method_results["rrf"]["CandidateRecall@20"],
                "CandidateCompleteEvidenceRecall@20": method_results["rrf"][
                    "CandidateCompleteEvidenceRecall@20"
                ],
            },
            "rrf_reranked": {
                "CandidateRecall@20": method_results["rrf_reranked"]["CandidateRecall@20"],
                "CandidateCompleteEvidenceRecall@20": method_results["rrf_reranked"][
                    "CandidateCompleteEvidenceRecall@20"
                ],
            },
        },
        "selection_decision": decision,
        "bootstrap": {
            "replicates": 2000,
            "seed": 20260815,
            "confidence": 0.95,
            "comparisons": comparisons,
        },
        "rescue_damage": rescue_damage,
        "provenance_analysis": provenance,
        "slices": {name: method_results[name]["slices"] for name in method_results},
        "robustness_parent_to_variant": robustness,
        "reranker_diagnostics": diagnostics,
        "metadata": {
            "document_count": metadata["document_count"],
            "fields": metadata["fields"],
            "limitation": (
                "all six structured filter fields have 0% population; "
                "no metadata relevance experiment"
            ),
        },
    }
    write_json_atomic(PHASE8_RERANK_METRICS, metrics_payload, text_free=True)
    private_payload = {
        "schema_version": 1,
        "status": "dev_reranker_evaluation_private",
        "methods": {
            name: {
                "rankings": {query_id: list(ids) for query_id, ids in method_rankings.items()},
                "per_query": {
                    query_id: dict(row)
                    for query_id, row in evaluate_rankings(
                        items,
                        method_rankings,
                        chunks=release.chunks,
                        query_length_bins=bins,
                        source_by_document=source_by_document,
                    ).per_query.items()
                },
            }
            for name, method_rankings in rankings.items()
        },
    }
    write_json_atomic(PHASE8_PRIVATE / "dev" / "reranker_evaluation.json", private_payload)
    selection_manifest = {
        "schema_version": 1,
        "status": "phase8_dev_selection_frozen",
        "selected_pipeline": decision["selected_pipeline"],
        "frozen_inputs": {
            "bm25": {"normalization": "arabic-light-v1", "k1": 1.2, "b": 0.75},
            "bge": {
                "model_id": "BAAI/bge-m3",
                "normalization": "arabic-raw-v1",
                "max_length": 1536,
                "cache_reused": True,
            },
        },
        "fusion": {
            "sparse_weight": 1.0,
            "dense_weight": 0.25,
            "rrf_k": 60,
            "sparse_top_k": 50,
            "dense_top_k": 50,
            "candidate_k": 20,
        },
        "reranker": {
            "selected": decision["selected_pipeline"] == "rrf_reranked",
            "model_id": validation["model_id"],
            "revision": validation["model_revision"],
            "max_length": validation["max_length"],
            "candidate_count": 20,
            "scoring_contract": validation["scoring_contract"],
            "evaluation_depth": 10,
            "serving_depth": 8,
        },
        "dev_metric_evidence": metrics_payload["methods"],
        "selection_rule": decision,
        "metric_definitions": {
            "binary_relevance": "qrel grade > 0",
            "ndcg_gain": "2**rel - 1",
            "complete_evidence": "every required evidence group represented in top-k",
        },
        "bootstrap": {"replicates": 2000, "seed": 20260815, "confidence": 0.95},
        "hashes": {
            "corpus_hash": release.corpus_manifest["corpus_hash"],
            "release_hash": release.corpus_manifest["release_hash"],
            "qrel_hash": qrel_hash,
            "phase7_selection_sha256": validated["selection_sha256"],
            "phase8_config_sha256": _sha256(PHASE8_CONFIG),
            "phase8_model_lock_sha256": _sha256(PHASE8_MODEL_LOCK),
        },
        "metadata_limitation": metrics_payload["metadata"],
        "holdout_protocol": {
            "status": "not_executed",
            "allow_flag_required": True,
            "one_shot": True,
            "private_per_query_artifacts_required": True,
        },
    }
    if PHASE8_DEV_SELECTION.is_file():
        if _json(PHASE8_DEV_SELECTION) != selection_manifest:
            raise ValueError("Phase-8 DEV selection manifest is immutable")
    else:
        write_json_atomic(PHASE8_DEV_SELECTION, selection_manifest, text_free=True)
    readiness = {
        "schema_version": 1,
        "status": "phase8_holdout_ready_not_executed",
        "command": "uv run kawaneen retrieval phase8-holdout --allow-holdout --resume --device cpu",
        "selection_manifest": PHASE8_DEV_SELECTION.as_posix(),
        "private_root": (PHASE8_PRIVATE / "holdout").as_posix(),
        "per_query_fields": [
            "query_id",
            "candidate_chunk_ids",
            "candidate_scores",
            "fusion_provenance",
            "ranked_chunk_ids",
            "ranked_scores",
            "relevant_ranked_chunk_ids_at_10",
            "qrel_grades",
            "evidence_group_satisfaction",
            "slice_labels",
            "parent_intent_id",
            "latency_ms",
        ],
        "tracked_payload_is_text_free": True,
        "holdout_not_loaded": True,
    }
    if PHASE8_HOLDOUT_READINESS.is_file():
        if _json(PHASE8_HOLDOUT_READINESS) != readiness:
            raise ValueError("Phase-8 holdout readiness manifest is immutable")
    else:
        write_json_atomic(PHASE8_HOLDOUT_READINESS, readiness, text_free=True)
    return {
        "status": "phase8_dev_selection_frozen",
        "selection": selection_manifest,
        "selection_sha256": _sha256(PHASE8_DEV_SELECTION),
        "metrics": metrics_payload,
        "validation": validation,
    }


def phase8_holdout(*, allow_holdout: bool, resume: bool, device: str) -> dict[str, object]:
    """Execute the one-shot Phase-8 holdout path with private per-query capture."""
    if not allow_holdout:
        raise PermissionError("Phase-8 holdout evaluation requires --allow-holdout")
    if not PHASE8_DEV_SELECTION.is_file():
        raise ValueError("frozen Phase-8 DEV selection is required before holdout")
    selection = _json(PHASE8_DEV_SELECTION)
    if selection.get("status") != "phase8_dev_selection_frozen":
        raise ValueError("Phase-8 DEV selection is not frozen")
    holdout_root = PHASE8_PRIVATE / "holdout"
    holdout_root.mkdir(parents=True, exist_ok=True)
    manifest_path = holdout_root / "manifest.json"
    manifest = (
        _json(manifest_path)
        if manifest_path.is_file()
        else {
            "schema_version": 1,
            "status": "phase8_holdout_in_progress",
            "selection_sha256": _sha256(PHASE8_DEV_SELECTION),
            "queries": {},
        }
    )
    if manifest.get("selection_sha256") != _sha256(PHASE8_DEV_SELECTION):
        raise ValueError("Phase-8 holdout manifest selection hash mismatch")
    validated = validate_phase7_inputs()
    release = load_phase7_release(allow_holdout=True)
    items = release.split_items("holdout", allow_holdout=True)
    bins = QueryLengthBins.from_dict(
        cast(Mapping[str, object], validated["selection"])["query_length_bins"]
    )
    source_by_document = {chunk.document_id: chunk.source_id for chunk in release.chunks}
    unit_type_by_id = {
        unit_id: chunk.unit_type for chunk in release.chunks for unit_id in chunk.source_unit_ids
    }
    bm25 = BM25Index.build(release.chunks, "arabic-light-v1", k1=1.2, b=0.75)
    vectors, ids = load_cached_embeddings(
        PHASE7_BGE_CACHE,
        fingerprint=str(
            cast(Mapping[str, object], validated["selection"])["cache_fingerprints"]["bge"]
        ),
    )
    dense = NumpyExactIndex.build(vectors, ids)
    phase7_lock = cast(Mapping[str, object], validated["model_lock"])
    bge = BGEM3Adapter(
        revision=str(cast(Mapping[str, object], phase7_lock["revisions"])["BAAI/bge-m3"]),
        max_length=1536,
        device="cpu",
    )
    query_vectors = bge.encode_queries(
        tuple(represent(item.query_text, "arabic-raw-v1").search_text for item in items),
        batch_size=1,
    )
    selected_pipeline = str(selection["selected_pipeline"])
    reranker: BGERerankerAdapter | None = None
    tokenizer: Any | None = None
    if selected_pipeline == "rrf_reranked":
        lock = load_phase8_reranker_lock()
        reranker = BGERerankerAdapter(
            revision=str(lock["revision"]), device=device, max_length=1024
        )
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(reranker.model_id, revision=str(lock["revision"]))
        reranker._load()
    completed = 0
    reused = 0
    total_latency_ms = 0.0
    for index, item in enumerate(items):
        path = holdout_root / "rankings" / f"{item.query_id}.json"
        if resume and path.is_file():
            existing = _json(path)
            if (
                existing.get("query_id") == item.query_id
                and existing.get("selection_sha256") == manifest["selection_sha256"]
            ):
                completed += 1
                reused += 1
                total_latency_ms += float(existing.get("latency_ms", 0.0))
                continue
        started = time.perf_counter()
        sparse_hits = tuple(
            SourceHit(hit.chunk_id, hit.score) for hit in bm25.search(item.query_text, top_k=50)
        )
        dense_hits = tuple(
            SourceHit(hit.chunk_id, hit.score)
            for hit in dense.search(query_vectors[index], top_k=50)
        )
        fused = fuse_ranked_hits(
            sparse=sparse_hits,
            dense=dense_hits,
            config=FusionConfig(sparse_weight=1.0, dense_weight=0.25),
        )
        ranked_ids = tuple(candidate.chunk_id for candidate in fused)
        ranked_scores = tuple(candidate.fused_score for candidate in fused)
        reranker_scores: list[float] = []
        if reranker is not None and tokenizer is not None:
            chunks_by_id = {chunk.chunk_id: chunk for chunk in release.chunks}

            def scorer(query: str, passage: str) -> float:
                return reranker.score_pairs(((query, passage),), batch_size=4)[0]

            reranked, _diagnostics = rerank_candidates(
                item.query_text,
                fused,
                chunks_by_id,
                scorer=scorer,
                tokenizer=tokenizer,
                config=RerankerConfig(
                    model_revision=reranker.revision, device=device, max_length=1024
                ),
            )
            ranked_ids = tuple(row.chunk_id for row in reranked)
            reranker_scores = [row.score for row in reranked]
            ranked_scores = tuple(reranker_scores)
        latency_ms = (time.perf_counter() - started) * 1000.0
        qrels = {qrel.chunk_id: int(qrel.grade) for qrel in item.chunk_qrels}
        chunks_by_id = {chunk.chunk_id: chunk for chunk in release.chunks}
        unit_to_chunks: defaultdict[str, set[str]] = defaultdict(set)
        for chunk in release.chunks:
            for unit_id in chunk.source_unit_ids:
                unit_to_chunks[unit_id].add(chunk.chunk_id)
        groups = evidence_groups_to_chunks(item, unit_to_chunks, chunks_by_id)
        payload = {
            "query_id": item.query_id,
            "selection_sha256": manifest["selection_sha256"],
            "candidate_chunk_ids": list(
                fused_ids := tuple(candidate.chunk_id for candidate in fused)
            ),
            "candidate_scores": [candidate.fused_score for candidate in fused],
            "fusion_provenance": [candidate.provenance for candidate in fused],
            "ranked_chunk_ids": list(ranked_ids),
            "ranked_scores": list(ranked_scores),
            "reranker_scores": reranker_scores,
            "qrel_grades": [
                {"chunk_id": chunk_id, "grade": grade} for chunk_id, grade in qrels.items()
            ],
            "relevant_ranked_chunk_ids_at_10": [
                chunk_id for chunk_id in ranked_ids[:10] if qrels.get(chunk_id, 0) > 0
            ],
            "evidence_group_satisfaction": {
                "@5": bool(groups)
                and all(set(ranked_ids[:5]) & set(group) for group in groups.values()),
                "@10": bool(groups)
                and all(set(ranked_ids[:10]) & set(group) for group in groups.values()),
            },
            "slice_labels": assign_slices(item, bins, unit_type_by_id, source_by_document),
            "parent_intent_id": item.base_intent_id,
            "latency_ms": latency_ms,
            "candidate_fingerprint": hashlib.sha256(
                json.dumps(list(fused_ids), separators=(",", ":")).encode()
            ).hexdigest(),
        }
        write_json_atomic(path, payload)
        manifest["queries"][item.query_id] = {"path": path.name, "status": "completed"}
        write_json_atomic(manifest_path, manifest)
        completed += 1
        total_latency_ms += latency_ms
    manifest["status"] = "phase8_holdout_complete"
    write_json_atomic(manifest_path, manifest)
    return {
        "status": "phase8_holdout_complete",
        "completed_queries": completed,
        "reused_queries": reused,
        "total_query_latency_ms": total_latency_ms,
        "holdout_not_run_by_this_task": False,
    }
