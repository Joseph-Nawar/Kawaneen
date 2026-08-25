from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from kawaneen.core.config import Settings


def _write_chunk(path: Path, chunk_id: str, document_id: str, *, append: bool = False) -> None:
    row = (
        json.dumps(
            {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "source_id": "fixture",
                "unit_type": "events",
                "display_text": f"نص {chunk_id}",
                "search_text": f"نص {chunk_id}",
                "source_unit_ids": [f"unit-{chunk_id}"],
                "source_spans": [{"unit_id": f"unit-{chunk_id}", "start": 0, "end": 7}],
                "chunk_policy_hash": "a" * 64,
                "normalization_policy_id": "arabic-light-v1",
                "normalization_policy_hash": "b" * 64,
                "token_count": 2,
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as handle:
        handle.write(row)


def test_synthetic_serving_retrieval_bundle_runs_locked_depths(tmp_path: Path) -> None:
    from kawaneen.api.composition import (
        build_serving_retrieval,
        load_frozen_serving_configuration,
    )
    from kawaneen.retrieval.dense_models import DenseModelAdapter

    corpus = tmp_path / "private" / "phase7_retrieval" / "corpus"
    corpus.mkdir(parents=True)
    chunks_path = corpus / "chunks.jsonl"
    _write_chunk(chunks_path, "c1", "d1")
    _write_chunk(chunks_path, "c2", "d2", append=True)

    vectors_root = (
        tmp_path
        / "private"
        / "phase7_retrieval"
        / "embeddings"
        / "BAAI__bge-m3"
        / "arabic-raw-v1"
        / "fixture"
    )
    vectors_root.mkdir(parents=True)
    np.save(vectors_root / "vectors.npy", np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    (vectors_root / "ids.json").write_text(json.dumps(["c1", "c2"]), encoding="utf-8")

    dense = DenseModelAdapter(
        model_id="fixture/dense",
        revision="fixture-revision",
        embedding_dimension=2,
        encoder=lambda texts, **kwargs: np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32),
    )

    class FakeReranker:
        def preload(self) -> None:
            return None

        def score_pairs(self, pairs: object, *, batch_size: int) -> tuple[float, ...]:
            return tuple(float(index) for index, _ in enumerate(pairs))

    configuration = load_frozen_serving_configuration(Path("data"))
    bundle = build_serving_retrieval(
        Settings(data_directory=Path("data"), artifacts_directory=tmp_path),
        configuration,
        dense_adapter=dense,
        reranker_adapter=FakeReranker(),  # type: ignore[arg-type]
    )

    bundle.initialize()
    result = bundle.retriever.search("نص", limit=2)

    assert result.summary.strategy == "hybrid_reranked"
    assert result.summary.hit_count == 2
    assert result.summary.returned_count == 2
    assert result.summary.score_type == "reranker_raw_logit"


def test_serving_chunk_loader_rejects_missing_and_malformed_inputs(tmp_path: Path) -> None:
    from kawaneen.retrieval.serving import load_serving_chunks

    with pytest.raises(FileNotFoundError):
        load_serving_chunks(tmp_path / "missing.jsonl")

    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_serving_chunks(empty)

    missing_id = tmp_path / "missing-id.jsonl"
    missing_id.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="chunk_id"):
        load_serving_chunks(missing_id)

    missing_units = tmp_path / "missing-units.jsonl"
    missing_units.write_text(json.dumps({"chunk_id": "c1"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source units"):
        load_serving_chunks(missing_units)

    bad_span = tmp_path / "bad-span.jsonl"
    bad_span.write_text(
        json.dumps({"chunk_id": "c1", "source_unit_ids": ["u1"], "source_spans": [{}]}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="span"):
        load_serving_chunks(bad_span)

    bad_spans = tmp_path / "bad-spans.jsonl"
    bad_spans.write_text(
        json.dumps({"chunk_id": "c1", "source_unit_ids": ["u1"], "source_spans": {}}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="spans"):
        load_serving_chunks(bad_spans)

    duplicate = tmp_path / "duplicate.jsonl"
    _write_chunk(duplicate, "c1", "d1")
    duplicate.write_text(duplicate.read_text(encoding="utf-8") * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        load_serving_chunks(duplicate)


def test_serving_retrieval_rejects_invalid_limit_and_incomplete_scores() -> None:
    from kawaneen.retrieval.hybrid.contracts import FusionConfig, SourceHit
    from kawaneen.retrieval.models import RetrievalChunk
    from kawaneen.retrieval.serving import HybridServingRetriever

    chunk = RetrievalChunk(
        chunk_id="c1",
        document_id="d1",
        source_id="fixture",
        unit_type="events",
        display_text="نص",
        search_text="نص",
        source_unit_ids=("u1",),
        chunk_policy_hash="a" * 64,
        normalization_policy_id="arabic-light-v1",
        normalization_policy_hash="b" * 64,
        token_count=1,
    )
    retriever = HybridServingRetriever(
        chunks={"c1": chunk},
        sparse_search=lambda query, top_k: (SourceHit("c1", 1.0),),
        dense_search=lambda query, top_k: (),
        reranker=lambda query, candidates: {},
        fusion_config=FusionConfig(sparse_weight=1.0, dense_weight=0.25),
    )

    with pytest.raises(ValueError, match="between 1 and 8"):
        retriever.search("q", limit=9)
    with pytest.raises(ValueError, match="every fused candidate"):
        retriever.search("q")
