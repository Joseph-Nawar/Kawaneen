"""Local, DEV-only runners for the Phase 15 reruns.

This module intentionally has no production imports or configuration hooks.  It
uses the frozen Phase 7 corpus, cached Phase 7 vectors, and the Phase 8 fusion
contracts while writing all new per-query material below the ignored Phase 15
private root.
"""

# pyright: basic

from __future__ import annotations

import hashlib
import json
import platform
import time
from collections.abc import Mapping, Sequence
from itertools import chain
from pathlib import Path
from typing import Any, cast

import numpy as np

from kawaneen.normalization import normalize_text
from kawaneen.normalization.policies import get_policy
from kawaneen.retrieval.cache import load_cached_embeddings
from kawaneen.retrieval.dense_models import BGEM3Adapter, DenseModelAdapter
from kawaneen.retrieval.hybrid.contracts import FusionConfig, SourceHit
from kawaneen.retrieval.hybrid.fusion import fuse_ranked_hits
from kawaneen.retrieval.hybrid.reranker import BGERerankerAdapter
from kawaneen.retrieval.models import RetrievalChunk
from kawaneen.retrieval.vector_index import NumpyExactIndex

from .evidence import write_json_atomic
from .inputs import Phase15InputRoots, load_dev_chunks, load_dev_query_records, load_dev_rankings
from .local_models import LocalInstructionModel, parse_json_object
from .runner import evaluate_dev_rankings
from .statistics import paired_bootstrap_delta

ARABIC_REVISION = "899f6e1b765915a72d5e4ace6bb2b221715550d8"
BGE_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
RERANKER_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
PHASE7_BGE_FINGERPRINT = "797830a20035acb251f33f9048725353c77fff417e8b58bab5f72252e6d7230b"
FALLBACK_MODEL = "abdelrahman-alkhodary/qwen2.5-1.5b-arabic-instruct"
FALLBACK_REVISION = "06d27020b3ac3d9058b7eebded9754c8e10fa6bd"
FALLBACK_OUTPUT_LIMIT = 512


def normalized_embedding_texts(texts: Sequence[str], normalization: str) -> tuple[str, ...]:
    """Apply one frozen Arabic normalization policy to every text in a side."""

    policy_id = (
        normalization if normalization.startswith("arabic-") else f"arabic-{normalization}-v1"
    )
    policy = get_policy(policy_id)
    return tuple(str(normalize_text(text, policy)) for text in texts)


def retrieval_chunks(rows: Sequence[Mapping[str, Any]]) -> tuple[RetrievalChunk, ...]:
    return tuple(
        RetrievalChunk(
            chunk_id=str(row["chunk_id"]),
            document_id=str(row.get("document_id", "")),
            source_id=str(row.get("source_id", "")),
            unit_type=str(row.get("unit_type", "")),
            display_text=str(row.get("display_text", "")),
            search_text=str(row.get("search_text", "")),
            source_unit_ids=tuple(
                str(item) for item in cast(Sequence[Any], row.get("source_unit_ids", ()))
            ),
            chunk_policy_hash=str(row.get("chunk_policy_hash", "")),
            normalization_policy_id=str(row.get("normalization_policy_id", "")),
            normalization_policy_hash=str(row.get("normalization_policy_hash", "")),
            token_count=int(row.get("token_count", 0)),
            source_spans=tuple(
                (int(span.get("start", 0)), int(span.get("end", 0)))
                for span in cast(Sequence[Mapping[str, Any]], row.get("source_spans", ()))
            ),
        )
        for row in rows
    )


def _metric_deltas(
    left: Mapping[str, Sequence[float]], right: Mapping[str, Sequence[float]]
) -> dict[str, object]:
    return {
        metric: paired_bootstrap_delta(values, right[metric]).__dict__
        for metric, values in left.items()
        if metric in right
    }


def _variant_metrics(
    records: Sequence[Mapping[str, Any]],
    rankings: Mapping[str, Sequence[str]],
    chunks: Sequence[Mapping[str, Any]],
) -> Mapping[str, tuple[float, ...]]:
    return evaluate_dev_rankings(records, rankings, chunks).metrics


def run_dialect_retrieval_matrix(
    roots: Phase15InputRoots,
    variants: Sequence[Mapping[str, Any]],
    *,
    base_intent_ids: Sequence[str],
) -> dict[str, object]:
    """Score all validated variants with BM25, BGE, frozen RRF, and reranking."""

    records = load_dev_query_records(roots)
    chunks_rows = load_dev_chunks(roots)
    chunks = retrieval_chunks(chunks_rows)
    by_intent = {str(record.get("intent_id")): record for record in records}
    base_records = [by_intent[intent_id] for intent_id in base_intent_ids]
    variant_models = [cast(dict[str, Any], item) for item in variants]
    variant_records_by_dialect: dict[str, list[dict[str, Any]]] = {
        dialect: [] for dialect in ("egyptian", "gulf_saudi", "levantine")
    }
    for item in variant_models:
        record = dict(by_intent[str(item["base_intent_id"])])
        record["query_id"] = str(item["variant_id"])
        record["query_text"] = str(item["text"])
        variant_records_by_dialect[str(item["dialect"])].append(record)

    from kawaneen.retrieval.bm25 import BM25Index

    bm25 = BM25Index.build(chunks, "arabic-light-v1", k1=1.2, b=0.75)
    bge_vectors, chunk_ids = load_cached_embeddings(
        roots.private_path(
            "phase7_retrieval/embeddings/BAAI__bge-m3/arabic-raw-v1/" + PHASE7_BGE_FINGERPRINT
        ),
        fingerprint=PHASE7_BGE_FINGERPRINT,
    )
    dense_index = NumpyExactIndex.build(bge_vectors, chunk_ids)
    bge = BGEM3Adapter(revision=BGE_REVISION, device="cpu")
    reranker = BGERerankerAdapter(revision=RERANKER_REVISION, device="cpu", max_length=1024)
    reranker.preload()
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}

    def systems_for(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, tuple[str, ...]]]:
        query_vectors = bge.encode_queries(
            tuple(str(row["query_text"]) for row in rows), batch_size=4
        )
        rankings: dict[str, dict[str, tuple[str, ...]]] = {
            name: {} for name in ("bm25", "bge-m3", "hybrid", "hybrid-reranker")
        }
        for row, query_vector in zip(rows, query_vectors, strict=True):
            query_id = str(row["query_id"])
            sparse = tuple(
                SourceHit(hit.chunk_id, hit.score)
                for hit in bm25.search(str(row["query_text"]), top_k=50)
            )
            dense = tuple(
                SourceHit(hit.chunk_id, hit.score)
                for hit in dense_index.search(query_vector, top_k=50)
            )
            fused = fuse_ranked_hits(
                sparse=sparse,
                dense=dense,
                config=FusionConfig(sparse_weight=1.0, dense_weight=0.25),
            )
            rankings["bm25"][query_id] = tuple(hit.chunk_id for hit in sparse[:10])
            rankings["bge-m3"][query_id] = tuple(hit.chunk_id for hit in dense[:10])
            rankings["hybrid"][query_id] = tuple(hit.chunk_id for hit in fused[:10])
            pairs = [
                (str(row["query_text"]), chunks_by_id[item.chunk_id].display_text) for item in fused
            ]
            scores = reranker.score_pairs(pairs, batch_size=4)
            reranked = sorted(
                zip(fused, scores, strict=True),
                key=lambda pair: (-pair[1], pair[0].fused_rank, pair[0].chunk_id),
            )
            rankings["hybrid-reranker"][query_id] = tuple(
                item[0].chunk_id for item in reranked[:10]
            )
        return rankings

    # Controls use already-frozen Phase 7/8 DEV rankings; only variant rows are newly scored.
    base_qids = tuple(str(record["query_id"]) for record in base_records)
    frozen_bm25 = load_dev_rankings(
        roots, "phase7_retrieval/dev/rankings/bm25__arabic-light-v1.json", base_qids
    )
    frozen_bge = load_dev_rankings(
        roots, "phase7_retrieval/dev/rankings/BAAI__bge-m3__arabic-raw-v1.json", base_qids
    )
    phase8_payload = json.loads(
        roots.private_path("phase8_retrieval/dev/reranker_evaluation.json").read_text(
            encoding="utf-8"
        )
    )
    phase8_methods = phase8_payload["methods"]
    frozen_hybrid = {qid: tuple(phase8_methods["rrf"]["rankings"][qid][:10]) for qid in base_qids}
    frozen_reranked = {
        qid: tuple(phase8_methods["rrf_reranked"]["rankings"][qid][:10]) for qid in base_qids
    }
    base_rankings = {
        "bm25": frozen_bm25,
        "bge-m3": frozen_bge,
        "hybrid": frozen_hybrid,
        "hybrid-reranker": frozen_reranked,
    }
    base_metrics = {
        name: _variant_metrics(base_records, ranking, chunks_rows)
        for name, ranking in base_rankings.items()
    }
    output: dict[str, object] = {
        "status": "RUN",
        "provenance": "PHASE15_DEV",
        "systems": {name: "RUN" for name in base_rankings},
        "model_revisions": {"bge-m3": BGE_REVISION, "reranker": RERANKER_REVISION},
        "device_runtime": {
            "bge-m3": "cpu/sentence-transformers",
            "reranker": "cpu/sentence-transformers",
        },
        "dialects": {},
    }
    dialect_output = cast(dict[str, object], output["dialects"])
    private_rankings: dict[str, dict[str, dict[str, tuple[str, ...]]]] = {}
    pooled: dict[str, dict[str, list[float]]] = {name: {} for name in base_rankings}
    for dialect, rows in variant_records_by_dialect.items():
        run_rankings = systems_for(rows)
        private_rankings[dialect] = run_rankings
        system_output: dict[str, object] = {}
        for system, ranking in run_rankings.items():
            metrics = _variant_metrics(rows, ranking, chunks_rows)
            baseline_metrics = base_metrics[system]
            selected_metrics = {
                metric: tuple(values)
                for metric, values in metrics.items()
                if metric in {"Recall@10", "MRR@10", "nDCG@10", "CompleteEvidenceRecall@10"}
            }
            baseline_selected = {
                metric: tuple(values)
                for metric, values in baseline_metrics.items()
                if metric in selected_metrics
            }
            system_output[system] = {
                "n": len(rows),
                "dialect_minus_msa": _metric_deltas(selected_metrics, baseline_selected),
            }
            for metric, values in selected_metrics.items():
                pooled[system].setdefault(metric, []).extend(values)
        dialect_output[dialect] = system_output
    pooled_output: dict[str, object] = {}
    for system, values in pooled.items():
        baseline = {
            metric: list(chain.from_iterable(base_metrics[system][metric] for _ in range(3)))
            for metric in values
        }
        pooled_output[system] = {
            "n": sum(len(rows) for rows in variant_records_by_dialect.values()),
            "dialect_minus_msa": _metric_deltas(values, baseline),
        }
    output["pooled"] = pooled_output
    write_json_atomic(
        roots.output_path("dialect/per_query_rankings.json"),
        {"provenance": "PHASE15_DEV", "rankings": private_rankings},
    )
    return output


def run_arabic_embedding(
    roots: Phase15InputRoots,
    *,
    normalizations: Sequence[str] = ("raw", "light", "aggressive"),
) -> dict[str, object]:
    """Run the exact Arabic model over the frozen DEV corpus and queries."""

    records = load_dev_query_records(roots)
    chunks = load_dev_chunks(roots)
    answerable = [
        record for record in records if str(record.get("answerability", "")).lower() == "answerable"
    ]
    query_ids = tuple(str(record["query_id"]) for record in answerable)
    adapter = DenseModelAdapter(
        model_id="omarelshehy/Arabic-Retrieval-v1.0",
        revision=ARABIC_REVISION,
        max_length=128,
        default_batch_size=1,
        embedding_dimension=768,
        device="cpu",
    )
    result: dict[str, object] = {
        "status": "RUN",
        "provenance": "PHASE15_DEV",
        "model_id": adapter.model_id,
        "revision": ARABIC_REVISION,
        "device_runtime": "cpu/sentence-transformers",
        "systems": {},
        "corpus_normalization_is_symmetric": True,
        "corpus_indexes": {},
    }
    systems = cast(dict[str, object], result["systems"])
    per_query: dict[str, Mapping[str, Sequence[float]]] = {}
    ranking_artifacts: dict[str, dict[str, tuple[str, ...]]] = {}
    for normalization in normalizations:
        name = f"arabic-retrieval-{normalization}"
        policy_id = (
            normalization if normalization.startswith("arabic-") else f"arabic-{normalization}-v1"
        )
        corpus_texts = normalized_embedding_texts(
            tuple(str(row.get("display_text", "")) for row in chunks), policy_id
        )
        vector_path = roots.output_path(f"embedding/arabic_corpus_vectors_{normalization}.npy")
        vector_path.parent.mkdir(parents=True, exist_ok=True)
        if vector_path.is_file():
            corpus_vectors = np.load(vector_path, allow_pickle=False)
        else:
            corpus_blocks = [
                adapter.encode_passages(corpus_texts[start : start + 4096], batch_size=256)
                for start in range(0, len(corpus_texts), 4096)
            ]
            corpus_vectors = np.vstack(corpus_blocks)
            np.save(vector_path, corpus_vectors, allow_pickle=False)
        index = NumpyExactIndex.build(corpus_vectors, tuple(str(row["chunk_id"]) for row in chunks))
        queries = normalized_embedding_texts(
            tuple(str(record["query_text"]) for record in answerable), policy_id
        )
        query_vectors = adapter.encode_queries(queries, batch_size=64)
        rankings = {
            qid: tuple(hit.chunk_id for hit in index.search(vector, top_k=10))
            for qid, vector in zip(query_ids, query_vectors, strict=True)
        }
        ranking_artifacts[name] = rankings
        evaluated = evaluate_dev_rankings(answerable, rankings, chunks)
        per_query[name] = evaluated.metrics
        systems[name] = {
            metric: {"mean": sum(values) / len(values), "n": len(values)}
            for metric, values in evaluated.metrics.items()
        }
        cast(dict[str, object], result["corpus_indexes"])[name] = {
            "path": vector_path.as_posix(),
            "normalization": policy_id,
            "row_count": len(corpus_texts),
        }

    frozen_rankings = {
        "E5": load_dev_rankings(
            roots,
            "phase7_retrieval/dev/rankings/intfloat__multilingual-e5-small__arabic-raw-v1.json",
            query_ids,
        ),
        "BGE-M3": load_dev_rankings(
            roots,
            "phase7_retrieval/dev/rankings/BAAI__bge-m3__arabic-raw-v1.json",
            query_ids,
        ),
    }
    frozen_metrics: dict[str, Mapping[str, Sequence[float]]] = {}
    for system, rankings in frozen_rankings.items():
        evaluated = evaluate_dev_rankings(answerable, rankings, chunks)
        frozen_metrics[system] = evaluated.metrics
        systems[f"{system}-raw"] = {
            metric: {"mean": sum(values) / len(values), "n": len(values)}
            for metric, values in evaluated.metrics.items()
        }
    result["paired_deltas"] = {}
    for name, metrics in per_query.items():
        if name == "arabic-retrieval-raw":
            continue
        result["paired_deltas"] = cast(dict[str, object], result["paired_deltas"])
        cast(dict[str, object], result["paired_deltas"])[name] = _metric_deltas(
            metrics, per_query["arabic-retrieval-raw"]
        )
    result["paired_model_comparisons"] = {
        f"arabic-retrieval-raw_vs_{system.lower().replace('-', '_')}_raw": _metric_deltas(
            per_query["arabic-retrieval-raw"], metrics
        )
        for system, metrics in frozen_metrics.items()
    }
    write_json_atomic(
        roots.output_path("embedding/per_query_rankings.json"),
        {"provenance": "PHASE15_DEV", "rankings": ranking_artifacts},
    )
    return result


def model_file_sha256(snapshot: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(snapshot.rglob("*")):
        if path.is_file() and not path.name.startswith("."):
            digest.update(path.relative_to(snapshot).as_posix().encode())
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def local_runtime_identity() -> dict[str, str]:
    return {
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }


def run_latency_experiment(roots: Phase15InputRoots) -> dict[str, object]:
    """Measure fixed batch-1 DEV operations end-to-end on one CPU runtime class."""

    records = load_dev_query_records(roots)
    chunks_rows = load_dev_chunks(roots)
    chunks = retrieval_chunks(chunks_rows)
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    selected = sorted(
        records,
        key=lambda record: hashlib.sha256(f"20260826:{record['query_id']}".encode()).hexdigest(),
    )[:20]
    texts = tuple(str(record["query_text"]) for record in selected)
    chunk_ids = tuple(chunk.chunk_id for chunk in chunks)
    from kawaneen.retrieval.bm25 import BM25Index
    from kawaneen.retrieval.keyword import KeywordIndex

    from .latency import measure_latency

    bm25 = BM25Index.build(chunks, "arabic-light-v1", k1=1.2, b=0.75)
    keyword = KeywordIndex.build(chunks, "arabic-light-v1")
    bge_vectors, _ = load_cached_embeddings(
        roots.private_path(
            "phase7_retrieval/embeddings/BAAI__bge-m3/arabic-raw-v1/" + PHASE7_BGE_FINGERPRINT
        ),
        fingerprint=PHASE7_BGE_FINGERPRINT,
    )
    e5_vectors, _ = load_cached_embeddings(
        roots.private_path(
            "phase7_retrieval/embeddings/intfloat__multilingual-e5-small/arabic-raw-v1/"
            "81e287f4ef1ba766059aa03886341136fdef926ec0124bceb39dc3d61a28830c"
        ),
        fingerprint="81e287f4ef1ba766059aa03886341136fdef926ec0124bceb39dc3d61a28830c",
    )
    arabic_vectors = np.load(
        roots.output_path("embedding/arabic_corpus_vectors_raw.npy"), allow_pickle=False
    )
    indexes = {
        "E5": NumpyExactIndex.build(e5_vectors, chunk_ids),
        "BGE-M3": NumpyExactIndex.build(bge_vectors, chunk_ids),
        "Arabic-Retrieval": NumpyExactIndex.build(arabic_vectors, chunk_ids),
    }
    adapters = {
        "E5": DenseModelAdapter(
            model_id="intfloat/multilingual-e5-small",
            revision="614241f622f53c4eeff9890bdc4f31cfecc418b3",
            embedding_dimension=384,
            default_batch_size=32,
            device="cpu",
        ),
        "BGE-M3": BGEM3Adapter(revision=BGE_REVISION, device="cpu"),
        "Arabic-Retrieval": DenseModelAdapter(
            model_id="omarelshehy/Arabic-Retrieval-v1.0",
            revision=ARABIC_REVISION,
            embedding_dimension=768,
            default_batch_size=1,
            device="cpu",
        ),
    }
    for adapter in adapters.values():
        adapter.preload()
    reranker = BGERerankerAdapter(revision=RERANKER_REVISION, device="cpu", max_length=1024)
    reranker.preload()

    def operation(name: str, index: int) -> object:
        text = texts[index]
        raw_text = normalized_embedding_texts((text,), "arabic-raw-v1")[0]
        light_text = normalized_embedding_texts((text,), "arabic-light-v1")[0]
        if name == "keyword":
            return keyword.search(light_text, top_k=10)
        if name == "BM25":
            return bm25.search(light_text, top_k=10)
        if name in indexes:
            query_vector = adapters[name].encode_queries((raw_text,), batch_size=1)[0]
            return indexes[name].search(query_vector, top_k=10)
        bge_query = adapters["BGE-M3"].encode_queries((raw_text,), batch_size=1)[0]
        dense = tuple(
            SourceHit(hit.chunk_id, hit.score)
            for hit in indexes["BGE-M3"].search(bge_query, top_k=50)
        )
        sparse = tuple(
            SourceHit(hit.chunk_id, hit.score) for hit in bm25.search(light_text, top_k=50)
        )
        fused = fuse_ranked_hits(
            sparse=sparse, dense=dense, config=FusionConfig(sparse_weight=1.0, dense_weight=0.25)
        )
        if name == "hybrid":
            return fused[:10]
        pairs = [(raw_text, chunks_by_id[item.chunk_id].display_text) for item in fused]
        scores = reranker.score_pairs(pairs, batch_size=4)
        return sorted(
            zip(fused, scores, strict=True), key=lambda pair: (-pair[1], pair[0].fused_rank)
        )[:10]

    summaries: dict[str, object] = {}
    for name in (
        "keyword",
        "BM25",
        "E5",
        "BGE-M3",
        "Arabic-Retrieval",
        "hybrid",
        "hybrid+reranker",
    ):
        timings = [
            measure_latency(
                lambda i=i, operation_name=name: operation(operation_name, i), samples=3, warmups=3
            )
            for i in range(len(selected))
        ]
        p50 = float(np.median([item.p50_ms for item in timings]))
        p95 = float(np.percentile([item.p95_ms for item in timings], 95))
        quality_rows = {
            str(record["query_id"]): tuple(
                hit.chunk_id for hit in cast(Sequence[Any], operation(name, index))
            )
            for index, record in enumerate(selected)
        }
        quality = evaluate_dev_rankings(selected, quality_rows, chunks).metrics
        summaries[name] = {
            "p50_ms": p50,
            "p95_ms": p95,
            "samples": len(selected),
            "batch_size": 1,
            "warmups": 3,
            "quality": {
                metric: {"mean": sum(values) / len(values), "n": len(values)}
                for metric, values in quality.items()
            },
            "timed_work": "normalization+encoding+retrieval"
            if name in indexes
            else (
                "normalization+BM25"
                if name in {"keyword", "BM25"}
                else "normalization+encoding+BM25+dense+RRF"
                if name == "hybrid"
                else "normalization+encoding+BM25+dense+RRF+reranker"
            ),
        }
    return {
        "status": "RUN",
        "provenance": "PHASE15_DEV",
        "protocol": {
            "fixed_subset_count": len(selected),
            "batch_size": 1,
            "warmups": 3,
            "samples_per_query": 3,
            "top_k": 10,
        },
        "runtime": local_runtime_identity()
        | {"device": "cpu", "runtime_class": "sentence-transformers/numpy"},
        "operations": summaries,
        "quality_subset": {
            "query_ids_sha256": hashlib.sha256(
                "\n".join(str(record["query_id"]) for record in selected).encode()
            ).hexdigest(),
            "metrics_use_same_fixed_subset": True,
        },
    }


def _snapshot_size(snapshot: Path) -> int:
    return sum(path.stat().st_size for path in snapshot.rglob("*") if path.is_file())


def _phase10_resolver(roots: Phase15InputRoots) -> Any:
    repo_root = roots.historical_private_root.parent.parent
    from kawaneen.grounding.dev import CANONICAL_DOCUMENTS, CORPUS_MANIFEST
    from kawaneen.grounding.provenance import CanonicalCorpusResolver

    return CanonicalCorpusResolver.from_json(
        repo_root
        / "artifacts/private/phase6_evaluation/ai-reviewed-v1/corpus/canonical_units.json",
        repo_root / "artifacts/private/phase7_retrieval/corpus/chunks.jsonl",
        repo_root / CORPUS_MANIFEST,
        document_paths=tuple(repo_root / path for path in CANONICAL_DOCUMENTS),
    )


def _fallback_metric(numerator: int, denominator: int) -> dict[str, object]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator if denominator else None,
    }


def run_fallback_generator(
    roots: Phase15InputRoots, subset: Mapping[str, Sequence[str]]
) -> dict[str, object]:
    """Run the fallback with the Phase-10 context and verification contracts."""

    records = {str(item["query_id"]): item for item in load_dev_query_records(roots)}
    context_root = roots.private_path("phase10_generation/context_packs/qwen-ollama-stage-c")
    output_root = roots.output_path("generator/fallback")
    output_root.mkdir(parents=True, exist_ok=True)
    model = LocalInstructionModel(FALLBACK_MODEL, FALLBACK_REVISION, dtype="bfloat16", device="mps")
    model.load()
    if model.snapshot is None:
        raise RuntimeError("fallback model did not expose its local immutable snapshot")
    all_ids = tuple(
        str(query_id)
        for group in (
            "answerable_gold_present_ids",
            "answerable_gold_absent_ids",
            "unanswerable_ids",
        )
        for query_id in subset[group]
    )
    if len(all_ids) != 80 or len(set(all_ids)) != 80:
        raise ValueError("matched generator subset must contain exactly 80 unique DEV IDs")

    from kawaneen.generation.contracts import (
        STAGE_C_GENERATION_SETTINGS,
        parse_stage_c_generation_payload,
    )
    from kawaneen.generation.postprocessing import finalize_generation
    from kawaneen.generation.prompt import render_stage_c_generation_prompt
    from kawaneen.generation.quote_registry import (
        build_quote_registry,
        stage_c_result_from_payload,
    )
    from kawaneen.grounding.contracts import ContextPack

    resolver = _phase10_resolver(roots)
    observations: list[dict[str, object]] = []
    for query_id in all_ids:
        record = records[query_id]
        context_path = context_root / f"{query_id}.json"
        payload = json.loads(context_path.read_text(encoding="utf-8"))
        pack = ContextPack.model_validate(payload["context_pack"])
        registry = build_quote_registry(pack)
        prompt = render_stage_c_generation_prompt(
            str(record["query_text"]),
            pack,
            registry=registry,
            settings=STAGE_C_GENERATION_SETTINGS,
            jurisdiction_text="SA",
        )
        started = time.perf_counter()
        raw = model.generate(prompt.text, max_new_tokens=FALLBACK_OUTPUT_LIMIT)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        parsed = parse_json_object(raw)
        finalization = None
        parsed_decision = "invalid"
        verification_reason = None
        if parsed is not None:
            try:
                parsed_payload = parse_stage_c_generation_payload(raw)
                parsed_decision = parsed_payload.decision.value
                generation_result = stage_c_result_from_payload(parsed_payload, registry)
                finalization = finalize_generation(pack, generation_result, resolver)
                verification_reason = (
                    finalization.result.detail or finalization.result.abstention_reason.value
                    if finalization.result.abstention_reason is not None
                    else None
                )
            except (TypeError, ValueError, KeyError) as error:
                verification_reason = str(error)
        final_result = finalization.result.model_dump(mode="json") if finalization else None
        verification = (
            finalization.verification.model_dump(mode="json")
            if finalization and finalization.verification is not None
            else None
        )
        result = {
            "query_id": query_id,
            "raw_output": raw,
            "parsed": parsed,
            "parsed_decision": parsed_decision,
            "final_result": final_result,
            "verification": verification,
            "verification_reason": verification_reason,
            "elapsed_ms": elapsed_ms,
        }
        write_json_atomic(output_root / f"{query_id}.json", result)
        observations.append(result)

    groups = {
        "answerable_gold_present": set(subset["answerable_gold_present_ids"]),
        "answerable_gold_absent": set(subset["answerable_gold_absent_ids"]),
        "explicit_unanswerable": set(subset["unanswerable_ids"]),
    }
    parsed_answers = {
        str(item["query_id"]) for item in observations if item["parsed_decision"] == "answer"
    }
    final_answers = {
        str(item["query_id"])
        for item in observations
        if isinstance(item["final_result"], dict)
        and item["final_result"].get("decision") == "answer"
    }
    final_abstains = {
        str(item["query_id"])
        for item in observations
        if isinstance(item["final_result"], dict)
        and item["final_result"].get("decision") == "abstain"
    }
    supported_answers: set[str] = set()
    complete_gold_use: set[str] = set()
    valid_citations = 0
    candidate_citations = 0
    gold_citations = 0
    verification_failures: dict[str, int] = {}
    evidence_by_query: dict[str, Any] = {}
    for item in observations:
        query_id = str(item["query_id"])
        if not isinstance(item["parsed"], dict):
            continue
        raw_claims = item["parsed"].get("claims", [])
        if isinstance(raw_claims, list):
            candidate_citations += sum(
                len(claim.get("quote_refs", ()))
                for claim in raw_claims
                if isinstance(claim, dict) and isinstance(claim.get("quote_refs"), list)
            )
        verification = item["verification"]
        if isinstance(verification, dict):
            valid = verification.get("valid_citations", [])
            invalid = verification.get("invalid_citations", [])
            if isinstance(valid, list):
                valid_citations += len(valid)
            if isinstance(invalid, list) and invalid:
                reason = str(invalid[0].get("reason", "other"))
                verification_failures[reason] = verification_failures.get(reason, 0) + 1
        context_payload = json.loads(
            (context_root / f"{query_id}.json").read_text(encoding="utf-8")
        )
        evidence_by_query[query_id] = context_payload["context_pack"].get("evidence", [])
        if query_id not in final_answers:
            continue
        qrels = {
            str(qrel["chunk_id"])
            for qrel in records[query_id].get("chunk_qrels", ())
            if "chunk_id" in qrel
        }
        evidence = {str(row["evidence_id"]): row for row in evidence_by_query[query_id]}
        cited_ids = {
            str(citation.get("evidence_id"))
            for claim in item["final_result"].get("claims", [])
            for citation in claim.get("citations", [])
        }
        cited_rows = [evidence[eid] for eid in cited_ids if eid in evidence]
        cited_chunks = {
            str(chunk_id)
            for row in cited_rows
            for chunk_id in row.get("contributing_chunk_ids", ())
        }
        if cited_chunks & qrels:
            supported_answers.add(query_id)
        gold_citations += sum(
            any(str(chunk_id) in qrels for chunk_id in row.get("contributing_chunk_ids", ()))
            for row in cited_rows
        )
        groups_by_query = records[query_id].get("evidence_groups", ())
        complete_gold_use.add(query_id) if groups_by_query and all(
            any(
                str(span.get("unit_id")) == str(row.get("unit_id"))
                for row in cited_rows
                for span in group.get("spans", ())
            )
            for group in groups_by_query
        ) else None

    answerable = groups["answerable_gold_present"] | groups["answerable_gold_absent"]
    invalid_count = sum(item["parsed_decision"] == "invalid" for item in observations)
    metrics = {
        "SupportedAnswerPrecision": _fallback_metric(len(supported_answers), len(final_answers)),
        "SupportedAnswerCoverage": _fallback_metric(len(supported_answers), len(answerable)),
        "FalseAnswerRate": _fallback_metric(
            len(final_answers & groups["explicit_unanswerable"]), 19
        ),
        "FalseAbstentionRate": _fallback_metric(
            len(final_abstains & groups["answerable_gold_present"]), 31
        ),
        "UnanswerableAbstentionRecall": _fallback_metric(
            len(final_abstains & groups["explicit_unanswerable"]), 19
        ),
        "CompleteGoldEvidenceUse": _fallback_metric(
            len(complete_gold_use & groups["answerable_gold_present"]), 31
        ),
        "ValidCitationRate": _fallback_metric(valid_citations, candidate_citations),
        "GoldCitationHitRate": _fallback_metric(gold_citations, valid_citations),
        "invalid_generation_rate": _fallback_metric(invalid_count, 80),
        "successful_verified_answer_count": len(final_answers),
    }
    elapsed = [float(cast(float, item["elapsed_ms"])) for item in observations]
    return {
        "status": "RUN",
        "provenance": "PHASE15_DEV",
        "model": {
            "model_id": FALLBACK_MODEL,
            "revision": FALLBACK_REVISION,
            "license": "apache-2.0",
            "runtime": "transformers",
            "device": "mps",
            "dtype": "bfloat16",
            "context_limit": 4096,
            "output_limit": FALLBACK_OUTPUT_LIMIT,
            "snapshot_sha256": model_file_sha256(model.snapshot),
            "disk_footprint_bytes": _snapshot_size(model.snapshot),
        },
        "matched_80": {key: len(value) for key, value in groups.items()},
        "same_ids_and_context_blocks": True,
        "context_contract": (
            "Phase10 qwen-ollama-stage-c ContextPack rendered by frozen Stage-C quote registry"
        ),
        "metrics": metrics,
        "outcome_counts": {
            "parsed_answer": len(parsed_answers),
            "parsed_abstain": sum(item["parsed_decision"] == "abstain" for item in observations),
            "invalid_generation": invalid_count,
            "verified_answer": len(final_answers),
            "verified_abstain": len(final_abstains),
        },
        "verification_failures": dict(sorted(verification_failures.items())),
        "latency": {
            "p50_ms": float(np.percentile(elapsed, 50)),
            "p95_ms": float(np.percentile(elapsed, 95)),
            "samples": len(elapsed),
        },
        "private_results": output_root.as_posix(),
        "alLaM": {"status": "BLOCKED_BEFORE_SCORING_NO_TRUSTWORTHY_4BIT_LOCAL_ARTIFACT"},
    }
