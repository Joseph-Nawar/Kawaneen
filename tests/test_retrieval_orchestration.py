import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import kawaneen.retrieval.orchestrator as orchestrator
from kawaneen.evaluation.models import (
    Answerability,
    Difficulty,
    QueryCategory,
    QueryLanguage,
    QueryRegister,
    QueryType,
)
from kawaneen.normalization.policies import get_policy
from kawaneen.retrieval.cache import (
    CheckpointEncodingResult,
    embedding_cache_fingerprint,
    encode_corpus_checkpointed,
)
from kawaneen.retrieval.config import load_phase7_config
from kawaneen.retrieval.dense_models import DenseModelAdapter
from kawaneen.retrieval.latency import LatencySummary
from kawaneen.retrieval.models import RetrievalChunk, ScoredChunk
from kawaneen.retrieval.orchestrator import (
    _holdout_private_payload,
    choose_normalization_policy,
    freeze_selection_manifest,
    recover_holdout_artifacts,
    require_holdout_permission,
)
from kawaneen.retrieval.slices import QueryLengthBins


def test_normalization_selection_prefers_raw_when_difference_is_below_threshold() -> None:
    metrics = {
        "arabic-raw-v1": {"nDCG@10": 0.700, "Recall@10": 0.70},
        "arabic-light-v1": {"nDCG@10": 0.704, "Recall@10": 0.70},
    }
    assert choose_normalization_policy(metrics) == "arabic-raw-v1"


def test_freeze_selection_manifest_is_immutable(tmp_path: Path) -> None:
    path = tmp_path / "selection.json"
    payload = {
        "selected": {"keyword": "arabic-raw-v1"},
        "query_length_bins": {"short_max": 3, "medium_max": 5},
    }

    freeze_selection_manifest(path, payload)
    freeze_selection_manifest(path, payload)

    with pytest.raises(ValueError, match="immutable"):
        freeze_selection_manifest(path, {**payload, "selected": {"keyword": "arabic-light-v1"}})
    assert json.loads(path.read_text(encoding="utf-8")) == payload


def test_holdout_requires_explicit_permission() -> None:
    with pytest.raises(PermissionError, match="allow-holdout"):
        require_holdout_permission(False)
    require_holdout_permission(True)


def test_holdout_private_payload_preserves_rankings_metrics_and_sanitized_metadata() -> None:
    item = _mini_item("q-h")
    release = _mini_release((item,))
    ranked_hits = {"q-h": (ScoredChunk("chunk-0", 0.75),)}
    evaluation = orchestrator.evaluate_rankings(
        (item,),
        {"q-h": ("chunk-0",)},
        chunks=release.chunks,
        query_length_bins=QueryLengthBins(10, 20),
        source_by_document={"doc-0": "source-0"},
    )

    payload = _holdout_private_payload(
        items=(item,),
        chunks=release.chunks,
        ranked_hits=ranked_hits,
        evaluation=evaluation,
        retriever_id="bm25__arabic-light-v1",
        latency_ms={"q-h": 1.25},
    )

    row = payload["queries"][0]
    assert row["query_id"] == "q-h"
    assert row["parent_intent_id"] == "intent-0"
    assert row["ranked_chunk_ids"] == ["chunk-0"]
    assert row["ranked_scores"] == [0.75]
    assert row["qrels"] == [{"chunk_id": "chunk-0", "grade": 1}]
    assert row["latency_ms"] == 1.25
    assert row["evidence_group_satisfaction"] == {"@5": False, "@10": False}
    serialized = json.dumps(payload)
    assert "what is the rule" not in serialized
    assert "Arabic evidence" not in serialized
    assert "query_text" not in serialized
    assert "display_text" not in serialized


def test_holdout_recovery_compares_original_and_replay_without_tuning(
    monkeypatch, tmp_path: Path
) -> None:
    item = _mini_item("q-h")
    metrics = _metric_row(1, 0.5)
    methods = {
        "keyword": metrics,
        "bm25": metrics,
        "e5__arabic-raw-v1": metrics,
        "bge__arabic-raw-v1": metrics,
    }
    original_path = tmp_path / "holdout.json"
    replay_path = tmp_path / "replay.json"
    original_path.write_text(json.dumps({"methods": methods}), encoding="utf-8")
    config = replace(load_phase7_config(), private_root=tmp_path)
    payloads = {
        name: {"schema_version": 1, "split": "holdout", "retriever_id": name, "queries": []}
        for name in methods
    }
    monkeypatch.setattr(orchestrator, "load_phase7_config", lambda: config)
    monkeypatch.setattr(orchestrator, "HOLDOUT_METRICS_PATH", original_path)
    monkeypatch.setattr(orchestrator, "HOLDOUT_REPLAY_PATH", replay_path)
    monkeypatch.setattr(
        orchestrator,
        "_validate_holdout_recovery_gates",
        lambda: {"selection": {}, "model_lock": {}, "config": {}},
    )
    monkeypatch.setattr(
        orchestrator, "load_phase7_release", lambda **_kwargs: _mini_release((item,))
    )
    monkeypatch.setattr(orchestrator, "_write_final_manifest", lambda *_args: {})
    monkeypatch.setattr(orchestrator, "_comparison_payload", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(orchestrator, "robustness_parent_variant", lambda *_args: {})
    monkeypatch.setattr(
        orchestrator,
        "_run_holdout_evaluation",
        lambda **_kwargs: ({"methods": methods}, payloads),
    )

    result = recover_holdout_artifacts(allow_holdout=True)

    assert result["replay_reason"] == "artifact_recovery_after_instrumentation_defect"
    assert result["metric_comparison"]["all_match"] is True
    assert result["replay_not_used_for_tuning"] is True
    assert (
        json.loads(replay_path.read_text(encoding="utf-8"))["metric_comparison"]["all_match"]
        is True
    )


def test_refresh_persisted_robustness_adds_keyword_without_dropping_methods(
    monkeypatch, tmp_path: Path
) -> None:
    item = _mini_item("q-base")
    variant = _mini_item("q-variant", variant_id="v-1")
    release = _mini_release((item, variant))
    base_metrics = _metric_row(1, 0.5)["metrics"]
    variant_metrics = _metric_row(1, 0.25)["metrics"]
    dev_rows = {"q-base": base_metrics, "q-variant": variant_metrics}
    config = replace(load_phase7_config(), private_root=tmp_path)
    comparison_path = tmp_path / "comparison.json"
    replay_path = tmp_path / "replay.json"
    comparison_path.write_text(
        json.dumps({"robustness_parent_minus_variant": {"bm25": {"old": True}}}),
        encoding="utf-8",
    )
    private = tmp_path / "holdout-keyword.json"
    private.write_text(
        json.dumps(
            {
                "queries": [
                    {"query_id": query_id, "metrics": row}
                    for query_id, row in dev_rows.items()
                ]
            }
        ),
        encoding="utf-8",
    )
    replay_path.write_text(
        json.dumps(
            {
                "recovered_analysis": {
                    "private_artifacts": {"keyword": str(private)},
                    "robustness_parent_minus_variant": {"bm25": {}},
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(orchestrator, "load_phase7_config", lambda: config)
    monkeypatch.setattr(orchestrator, "load_phase7_release", lambda **_kwargs: release)
    monkeypatch.setattr(orchestrator, "COMPARISON_PATH", comparison_path)
    monkeypatch.setattr(orchestrator, "HOLDOUT_REPLAY_PATH", replay_path)
    monkeypatch.setattr(orchestrator, "_load_private_rows", lambda *_args: dev_rows)

    result = orchestrator.refresh_persisted_robustness_reports()

    assert set(result["dev"]) == {"keyword", "bm25", "e5", "bge"}
    assert result["dev"]["bm25"]
    assert result["holdout"]["keyword"] == result["dev"]["keyword"]


def test_cache_status_does_not_load_full_release(monkeypatch, tmp_path: Path) -> None:
    config = replace(load_phase7_config(), private_root=tmp_path)
    corpus_manifest_path = tmp_path / "corpus-manifest.json"
    corpus_manifest_path.write_text(
        json.dumps(
            {
                "chunk_count": 1,
                "corpus_hash": "corpus",
                "chunk_policy_hashes": ["chunk-policy"],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(orchestrator, "load_phase7_config", lambda: config)
    monkeypatch.setattr(orchestrator, "CORPUS_MANIFEST_PATH", corpus_manifest_path)
    monkeypatch.setattr(
        orchestrator,
        "load_phase7_release",
        lambda: pytest.fail("cache status must not load the Phase 6 release"),
    )
    fingerprint = embedding_cache_fingerprint(
        corpus_hash="corpus",
        policy_hash="chunk-policy",
        normalization_policy_hash=get_policy("arabic-raw-v1").policy_hash,
        model_id="BAAI/bge-m3",
        model_revision="5617a9f61b028005a4858fdac845db406aefb181",
        formatting_contract="bge-m3-dense-v1",
        max_length=1536,
        embedding_dimension=1024,
        normalize=True,
        dtype="float32",
    )
    cache_path = tmp_path / "embeddings" / "BAAI__bge-m3" / "arabic-raw-v1" / fingerprint
    encode_corpus_checkpointed(
        ("text",),
        ("chunk-0",),
        cache_path,
        fingerprint=fingerprint,
        encoder=lambda _texts, _batch_size: np.array([[1.0] + [0.0] * 1023], dtype=np.float32),
        embedding_dimension=1024,
        model_config={
            "model_id": "BAAI/bge-m3",
            "model_revision": "5617a9f61b028005a4858fdac845db406aefb181",
            "formatting_contract": "bge-m3-dense-v1",
            "max_length": 1536,
        },
    )

    status = orchestrator.cache_status(model="bge-m3")

    assert status["completed_blocks"] == 1
    assert status["completed_chunks"] == 1


def _metric_row(sample_count: int, value: float) -> dict[str, object]:
    metrics = {
        "Recall@1": value,
        "Recall@5": value,
        "Recall@10": value,
        "MRR@10": value,
        "nDCG@10": value,
        "Precision@5": value,
        "CompleteEvidenceRecall@5": value,
        "CompleteEvidenceRecall@10": value,
    }
    return {"sample_count": sample_count, "metrics": metrics}


def _report_method(sample_count: int, value: float) -> dict[str, object]:
    row = _metric_row(sample_count, value)
    return {
        **row,
        "unanswerable_score_distribution": {"count": 1, "max": value},
        "latency_ms": {"p50": value, "p95": value},
        "corpus_embedding_seconds": value,
        "index_build_seconds": value,
        "index_artifact_size_bytes": 1,
        "slices": {
            "base_vs_variant": {"base": row, "variant": row},
            "query_length": {"short": row},
        },
    }


def test_final_report_weights_dev_and_holdout_without_query_text(
    monkeypatch, tmp_path: Path
) -> None:
    methods = {
        "keyword__arabic-raw-v1": _report_method(2, 0.2),
        "bm25__arabic-light-v1": _report_method(2, 0.4),
        "e5__arabic-raw-v1": _report_method(2, 0.6),
        "bge__arabic-raw-v1": _report_method(2, 0.8),
    }
    holdout_methods = {
        "keyword": _report_method(1, 0.1),
        "bm25": _report_method(1, 0.3),
        "e5__arabic-raw-v1": _report_method(1, 0.5),
        "bge__arabic-raw-v1": _report_method(1, 0.7),
    }
    dev_path = tmp_path / "dev.json"
    holdout_path = tmp_path / "holdout.json"
    comparison_path = tmp_path / "comparison.json"
    selection_path = tmp_path / "selection.json"
    final_path = tmp_path / "final.json"
    dev_path.write_text(
        json.dumps(
            {
                "methods": methods,
                "corpus_hash": "corpus",
                "release_hash": "release",
                "model_lock_hash": "lock",
                "dense_diagnostics": {
                    "arabic-raw-v1": {"max_length": 512},
                    "bge": {"max_length": 1536},
                },
            }
        ),
        encoding="utf-8",
    )
    holdout_path.write_text(json.dumps({"methods": holdout_methods}), encoding="utf-8")
    comparison_path.write_text(
        json.dumps(
            {
                "comparisons": {"bm25_vs_e5": {}, "bm25_vs_bge": {}},
                "robustness_parent_minus_variant": {
                    "keyword": {},
                    "bm25": {},
                    "e5": {},
                    "bge": {},
                },
            }
        ),
        encoding="utf-8",
    )
    selection_path.write_text(
        json.dumps(
            {
                "selection": {
                    "keyword": "arabic-raw-v1",
                    "bm25": "arabic-light-v1",
                    "dense": "arabic-raw-v1",
                },
                "config_hash": "config",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(orchestrator, "DEV_METRICS_PATH", dev_path)
    monkeypatch.setattr(orchestrator, "HOLDOUT_METRICS_PATH", holdout_path)
    monkeypatch.setattr(orchestrator, "COMPARISON_PATH", comparison_path)
    monkeypatch.setattr(orchestrator, "HOLDOUT_REPLAY_PATH", tmp_path / "replay.json")
    monkeypatch.setattr(orchestrator, "SELECTION_PATH", selection_path)
    monkeypatch.setattr(orchestrator, "FINAL_MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(orchestrator, "FINAL_REPORT_PATH", final_path)

    report = orchestrator.build_final_report()

    assert report["primary_175_answerable_base_intents"]["bge"]["sample_count"] == 3
    assert report["primary_175_answerable_base_intents"]["bge"]["metrics"][
        "nDCG@10"
    ] == pytest.approx((2 * 0.8 + 1 * 0.7) / 3)
    assert report["holdout_robustness_parent_minus_variant"] is None
    assert "query_text" not in json.dumps(report)
    assert json.loads(final_path.read_text(encoding="utf-8"))["status"].startswith("phase7_")


def test_holdout_readiness_rejects_changed_selection(monkeypatch, tmp_path: Path) -> None:
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(
        json.dumps({"status": "dev_selection_frozen", "selection": {"dense": "wrong"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(orchestrator, "SELECTION_PATH", selection_path)
    monkeypatch.setattr(
        orchestrator, "_file_sha256", lambda _path: orchestrator.EXPECTED_SELECTION_SHA256
    )

    with pytest.raises(ValueError, match="does not match"):
        orchestrator.verify_holdout_readiness()


def test_holdout_readiness_rejects_existing_holdout(monkeypatch, tmp_path: Path) -> None:
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "status": "dev_selection_frozen",
                "selection": {
                    "keyword": "arabic-raw-v1",
                    "bm25": "arabic-light-v1",
                    "dense": "arabic-raw-v1",
                },
            }
        ),
        encoding="utf-8",
    )
    holdout_path = tmp_path / "holdout.json"
    holdout_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(orchestrator, "SELECTION_PATH", selection_path)
    monkeypatch.setattr(orchestrator, "HOLDOUT_METRICS_PATH", holdout_path)
    monkeypatch.setattr(orchestrator, "FINAL_MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(
        orchestrator, "_file_sha256", lambda _path: orchestrator.EXPECTED_SELECTION_SHA256
    )

    with pytest.raises(ValueError, match="one-shot"):
        orchestrator.verify_holdout_readiness()


def test_dense_sanity_audit_checks_contracts_and_writes_private_packet(
    monkeypatch, tmp_path: Path
) -> None:
    class DummyAdapter:
        def __init__(self, *, revision: str, max_length: int, device: str) -> None:
            self.revision = revision
            self.max_length = max_length
            self.device = device
            self.model_id = "dummy"
            self.formatting_contract = "dummy-v1"
            self.embedding_dimension = 4

        def format_query(self, text: str) -> str:
            return text

        def format_passage(self, text: str) -> str:
            return text

        def encode_queries(self, texts, *, batch_size: int) -> np.ndarray:
            del batch_size
            return np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (len(texts), 1))

    class PrefixedDummyAdapter(DummyAdapter):
        def format_query(self, text: str) -> str:
            return f"query: {text}"

        def format_passage(self, text: str) -> str:
            return f"passage: {text}"

    chunks = tuple(
        RetrievalChunk(
            chunk_id=f"chunk-{index}",
            document_id="doc",
            source_id="source",
            unit_type="paragraph",
            display_text=f"text-{index}",
            search_text=f"text-{index}",
            source_unit_ids=(f"unit-{index}",),
            chunk_policy_hash="policy",
            normalization_policy_id="arabic-raw-v1",
            normalization_policy_hash="normalization",
            token_count=1,
        )
        for index in range(4)
    )
    categories = (
        QueryCategory.EXACT_PROVISION,
        QueryCategory.DEFINITION,
        QueryCategory.DEADLINE,
        QueryCategory.CONDITIONS,
        QueryCategory.MULTI_EVIDENCE,
        QueryCategory.CASE_HOLDING,
        QueryCategory.AUTHORITY,
    )
    items = tuple(
        type(
            "Item",
            (),
            {
                "query_id": f"q-{index}",
                "query_text": f"query-{index}",
                "category": category,
                "answerability": Answerability.ANSWERABLE,
                "variant_id": "variant-6" if index == 6 else None,
                "chunk_qrels": (type("Qrel", (), {"chunk_id": "chunk-0", "grade": 1})(),),
            },
        )()
        for index, category in enumerate(categories)
    )
    release = type(
        "Release",
        (),
        {
            "chunks": chunks,
            "corpus_manifest": {"corpus_hash": "corpus", "release_hash": "release"},
            "split_items": lambda self, split: items if split == "dev" else (),
        },
    )()
    config = replace(load_phase7_config(), private_root=tmp_path, dense_device="cpu")
    lock = {"revisions": {"intfloat/multilingual-e5-small": "e5-rev", "BAAI/bge-m3": "bge-rev"}}
    vectors = np.eye(4, dtype=np.float32)
    monkeypatch.setattr(orchestrator, "load_phase7_config", lambda: config)
    monkeypatch.setattr(orchestrator, "load_phase7_release", lambda: release)
    monkeypatch.setattr(orchestrator, "_model_lock", lambda _config: lock)
    monkeypatch.setattr(orchestrator, "E5SmallAdapter", PrefixedDummyAdapter)
    monkeypatch.setattr(orchestrator, "BGEM3Adapter", DummyAdapter)
    monkeypatch.setattr(
        orchestrator,
        "_dense_cache_identity",
        lambda _release, _config, adapter, _policy, _revision: (
            tmp_path / adapter.model_id,
            f"{adapter.model_id}-fingerprint",
            tuple(chunk.search_text for chunk in chunks),
            tuple(chunk.chunk_id for chunk in chunks),
        ),
    )
    for adapter_name in ("dummy",):
        (tmp_path / adapter_name).mkdir()
        (tmp_path / adapter_name / "manifest.json").write_text(
            json.dumps(
                {
                    "model_config": {
                        "model_id": "dummy",
                        "model_revision": "bge-rev",
                        "max_length": config.bge_max_length,
                    }
                }
            ),
            encoding="utf-8",
        )
    monkeypatch.setattr(
        orchestrator,
        "load_cached_embeddings",
        lambda *_args, **_kwargs: (vectors, tuple(chunk.chunk_id for chunk in chunks)),
    )
    monkeypatch.setattr(
        orchestrator,
        "checkpoint_cache_status",
        lambda *_args, **_kwargs: {"completed_chunks": 4, "total_chunks": 4},
    )
    monkeypatch.setattr(orchestrator, "SANITY_MANIFEST_PATH", tmp_path / "audit.json")
    monkeypatch.setattr(orchestrator, "freeze_selection_manifest", lambda _path, payload: payload)
    monkeypatch.setattr(
        orchestrator, "_file_sha256", lambda _path: orchestrator.EXPECTED_SELECTION_SHA256
    )
    monkeypatch.setattr(orchestrator, "SELECTION_PATH", Path("real-selection.json"))
    monkeypatch.setattr(
        orchestrator,
        "json",
        json,
    )

    # The real frozen selection is read through a temporary replacement path.
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "selection": {
                    "keyword": "arabic-raw-v1",
                    "bm25": "arabic-light-v1",
                    "dense": "arabic-raw-v1",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(orchestrator, "SELECTION_PATH", selection_path)

    result = orchestrator.dense_sanity_audit()

    assert result["status"] == "dev_dense_sanity_passed"
    assert all(result["checks"][name] for name in ("formatting", "dimensions", "finite_normalized"))
    packet = json.loads((tmp_path / "dev" / "dense_sanity_packet.json").read_text(encoding="utf-8"))
    assert packet["sample_count"] == 7
    assert "query_text" in json.dumps(packet)


def _mini_chunk(chunk_id: str = "chunk-0") -> RetrievalChunk:
    return RetrievalChunk(
        chunk_id=chunk_id,
        document_id="doc-0",
        source_id="source-0",
        unit_type="paragraph",
        display_text="Arabic evidence",
        search_text="Arabic evidence",
        source_unit_ids=("unit-0",),
        source_spans=((0, 10),),
        chunk_policy_hash="policy",
        normalization_policy_id="arabic-raw-v1",
        normalization_policy_hash="normalization",
        token_count=2,
    )


def _mini_item(
    query_id: str,
    *,
    answerability: Answerability = Answerability.ANSWERABLE,
    variant_id: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        query_id=query_id,
        intent_id="intent-0",
        base_intent_id="intent-0",
        variant_id=variant_id,
        query_text="what is the rule",
        language=QueryLanguage.ARABIC,
        register=QueryRegister.FORMAL,
        category=QueryCategory.DEFINITION,
        query_type=QueryType.LEGAL_CONCEPT,
        jurisdiction="jurisdiction",
        answerability=answerability,
        difficulty=Difficulty.EASY,
        benchmark_source=None,
        source_document_ids=("doc-0",),
        evidence_groups=(),
        chunk_qrels=(SimpleNamespace(chunk_id="chunk-0", grade=1),),
    )


def _mini_release(items: tuple[SimpleNamespace, ...]) -> SimpleNamespace:
    chunks = (_mini_chunk(),)
    return SimpleNamespace(
        items=items,
        chunks=chunks,
        corpus_manifest={
            "corpus_hash": "corpus",
            "release_hash": "release",
            "chunk_policy_hashes": ["policy"],
        },
        split_items=lambda split, **_kwargs: items if split in {"dev", "holdout"} else (),
    )


def test_lexical_result_builder_covers_both_indexes_and_unanswerable_scores(
    monkeypatch, tmp_path: Path
) -> None:
    items = (_mini_item("q-0"), _mini_item("q-u", answerability=Answerability.UNANSWERABLE))
    release = _mini_release(items)
    config = replace(load_phase7_config(), private_root=tmp_path)

    class FakeIndex:
        def search(self, _query: str, *, top_k: int) -> tuple[ScoredChunk, ...]:
            assert top_k == 10
            return (ScoredChunk("chunk-0", 1.0),)

    class FakeFactory:
        @classmethod
        def build(cls, *_args, **_kwargs) -> FakeIndex:
            return FakeIndex()

    monkeypatch.setattr(orchestrator, "KeywordIndex", FakeFactory)
    monkeypatch.setattr(orchestrator, "BM25Index", FakeFactory)
    monkeypatch.setattr(
        orchestrator,
        "measure_latency",
        lambda *_args, **_kwargs: LatencySummary.from_samples(
            (1.0,), device="test", package_versions={}, threads=1
        ),
    )

    result = orchestrator._lexical_results(release, items, config)

    assert set(result) == {
        "keyword__arabic-raw-v1",
        "keyword__arabic-light-v1",
        "bm25__arabic-raw-v1",
        "bm25__arabic-light-v1",
    }
    assert result["bm25__arabic-light-v1"]["sample_count"] == 1
    assert result["bm25__arabic-light-v1"]["unanswerable_score_distribution"]["count"] == 1


def test_dense_result_uses_existing_cache_and_records_diagnostics(
    monkeypatch, tmp_path: Path
) -> None:
    items = (_mini_item("q-0"), _mini_item("q-u", answerability=Answerability.UNANSWERABLE))
    release = _mini_release(items)
    config = replace(load_phase7_config(), private_root=tmp_path)
    adapter = DenseModelAdapter(
        model_id="BAAI/bge-m3",
        revision="revision",
        max_length=1536,
        default_batch_size=2,
        formatting_contract="bge-m3-dense-v1",
        embedding_dimension=2,
        device="cpu",
        encoder=lambda texts, **_kwargs: np.tile(
            np.array([[1.0, 0.0]], dtype=np.float32), (len(texts), 1)
        ),
    )
    vectors = np.array([[1.0, 0.0]], dtype=np.float32)
    monkeypatch.setattr(
        orchestrator,
        "load_cached_embeddings",
        lambda *_args, **_kwargs: (vectors, ("chunk-0",)),
    )
    monkeypatch.setattr(orchestrator, "loaded_tokenizer", lambda _adapter: None)
    monkeypatch.setattr(
        orchestrator,
        "load_tokenizer",
        lambda _adapter: lambda texts, **_kwargs: {"input_ids": [[1, 2] for _ in texts]},
    )

    class NoFaiss:
        @classmethod
        def build(cls, *_args, **_kwargs):
            raise RuntimeError("no faiss")

    monkeypatch.setattr(orchestrator, "FaissExactIndex", NoFaiss)

    result, diagnostics = orchestrator._dense_result(
        release,
        items,
        config,
        adapter=adapter,
        policy_id="arabic-raw-v1",
        model_revision="revision",
        private_stage=None,
        query_length_bins=QueryLengthBins(10, 20),
        allow_corpus_encode=False,
    )

    assert result["sample_count"] == 1
    assert result["unanswerable_score_distribution"]["count"] == 1
    assert diagnostics["cache_status"] == "hit"
    assert diagnostics["backend"] == "numpy.exact-inner-product"
    assert diagnostics["corpus_token_diagnostics"]["fraction_above_model_maximum"] == 0.0


def test_model_lock_reuses_and_creates_immutable_contract(monkeypatch, tmp_path: Path) -> None:
    config = load_phase7_config()
    lock_path = tmp_path / "lock.json"
    monkeypatch.setattr(orchestrator, "MODEL_LOCK_PATH", lock_path)
    monkeypatch.setattr(
        orchestrator, "resolve_model_revision", lambda model_id: f"{model_id}-revision"
    )

    created = orchestrator._model_lock(config)
    reused = orchestrator._model_lock(config)

    assert created == reused
    assert created["contracts"]["BAAI/bge-m3"]["max_length"] == config.bge_max_length


def test_dense_cache_identity_is_deterministic_and_ordered(tmp_path: Path) -> None:
    config = replace(load_phase7_config(), private_root=tmp_path)
    release = _mini_release((_mini_item("q-0"),))
    adapter = DenseModelAdapter(
        model_id="BAAI/bge-m3",
        revision="revision",
        max_length=1536,
        formatting_contract="bge-m3-dense-v1",
        embedding_dimension=2,
    )

    first = orchestrator._dense_cache_identity(
        release, config, adapter, "arabic-raw-v1", "revision"
    )
    second = orchestrator._dense_cache_identity(
        release, config, adapter, "arabic-raw-v1", "revision"
    )

    assert first == second
    assert first[2] == ("Arabic evidence",)
    assert first[3] == ("chunk-0",)
    assert first[0].parts[-2] == "arabic-raw-v1"


def test_score_distribution_empty_is_text_free() -> None:
    assert orchestrator._score_distribution([]) == {"count": 0}


def test_cache_status_reports_missing_and_legacy_complete_cache(
    monkeypatch, tmp_path: Path
) -> None:
    config = replace(load_phase7_config(), private_root=tmp_path)
    manifest_path = tmp_path / "corpus.json"
    manifest_path.write_text(
        json.dumps({"chunk_count": 2, "corpus_hash": "corpus", "chunk_policy_hashes": ["policy"]}),
        encoding="utf-8",
    )
    lock = {"revisions": {"BAAI/bge-m3": "revision"}, "contracts": {}}
    monkeypatch.setattr(orchestrator, "load_phase7_config", lambda: config)
    monkeypatch.setattr(orchestrator, "_model_lock", lambda _config: lock)
    monkeypatch.setattr(orchestrator, "CORPUS_MANIFEST_PATH", manifest_path)

    missing = orchestrator.cache_status(model="bge-m3")
    assert missing["completed_blocks"] == 0
    assert missing["total_chunks"] == 2

    policy = get_policy("arabic-raw-v1")
    fingerprint = embedding_cache_fingerprint(
        corpus_hash="corpus",
        policy_hash="policy",
        normalization_policy_hash=policy.policy_hash,
        model_id="BAAI/bge-m3",
        model_revision="revision",
        formatting_contract="bge-m3-dense-v1",
        max_length=config.bge_max_length,
        embedding_dimension=1024,
        normalize=True,
        dtype="float32",
    )
    legacy = tmp_path / "embeddings" / "BAAI__bge-m3" / "arabic-raw-v1" / fingerprint
    from kawaneen.retrieval.cache import save_cached_embeddings

    save_cached_embeddings(
        legacy,
        np.tile(np.array([[1.0] + [0.0] * 1023], dtype=np.float32), (2, 1)),
        ("chunk-0", "chunk-1"),
        fingerprint=fingerprint,
    )
    complete = orchestrator.cache_status(model="bge-m3")
    assert complete["completed_blocks"] == 1
    assert complete["completed_chunks"] == 2


def test_comparison_payload_reports_bootstrap_and_complementarity() -> None:
    metric_names = (
        "Recall@1",
        "Recall@5",
        "Recall@10",
        "MRR@10",
        "nDCG@10",
        "Precision@5",
        "CompleteEvidenceRecall@5",
        "CompleteEvidenceRecall@10",
    )
    left = {
        "q1": {metric: 1.0 for metric in metric_names},
        "q2": {metric: 0.0 for metric in metric_names},
    }
    right = {
        "q1": {metric: 0.0 for metric in metric_names},
        "q2": {metric: 0.0 for metric in metric_names},
    }
    payload = orchestrator._comparison_payload(
        "bm25", "dense", left, right, seed=20260815, replicates=20
    )

    assert payload["sample_count"] == 2
    assert payload["metrics"]["Recall@10"]["wins"] == 1
    assert payload["complementarity_top10"]["lexical_succeeds_dense_fails"] == 1


def test_build_dev_comparison_writes_private_failure_packet(monkeypatch, tmp_path: Path) -> None:
    config = replace(load_phase7_config(), private_root=tmp_path)
    release = _mini_release((_mini_item("q-0"), _mini_item("q-u")))
    metric_names = (
        "Recall@1",
        "Recall@5",
        "Recall@10",
        "MRR@10",
        "nDCG@10",
        "Precision@5",
        "CompleteEvidenceRecall@5",
        "CompleteEvidenceRecall@10",
    )
    row = {metric: 0.0 for metric in metric_names}
    row["Recall@10"] = 1.0
    names = (
        "bm25__arabic-light-v1",
        "intfloat__multilingual-e5-small__arabic-raw-v1",
        "BAAI__bge-m3__arabic-raw-v1",
    )
    for name in names:
        path = tmp_path / "dev" / "rankings" / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "rankings": {"q-0": ["chunk-0"], "q-u": []},
                    "per_query": {"q-0": row, "q-u": row},
                }
            ),
            encoding="utf-8",
        )
    comparison_path = tmp_path / "comparison.json"
    monkeypatch.setattr(orchestrator, "load_phase7_release", lambda: release)
    monkeypatch.setattr(orchestrator, "COMPARISON_PATH", comparison_path)

    result = orchestrator.build_dev_comparison(
        config,
        {"selection": {"bm25": "arabic-light-v1", "dense": "arabic-raw-v1"}},
    )

    assert result["status"] == "dev_comparisons_complete"
    assert set(result["comparisons"]) == {"bm25_vs_e5", "bm25_vs_bge"}
    assert (tmp_path / "dev" / "failure_packet.json").is_file()


def test_evaluate_dev_reuses_mocked_cache_only_results(monkeypatch, tmp_path: Path) -> None:
    config = replace(load_phase7_config(), private_root=tmp_path)
    release = _mini_release((_mini_item("q-0"),))
    methods = {
        f"{method}__{policy}": _report_method(1, value)
        for method, value in (("keyword", 0.1), ("bm25", 0.2))
        for policy in config.normalization_policy_ids
    }
    dev_path = tmp_path / "dev.json"
    comparison_path = tmp_path / "comparison.json"
    monkeypatch.setattr(orchestrator, "load_phase7_config", lambda: config)
    monkeypatch.setattr(orchestrator, "load_phase7_release", lambda: release)
    monkeypatch.setattr(orchestrator, "DEV_METRICS_PATH", dev_path)
    monkeypatch.setattr(orchestrator, "COMPARISON_PATH", comparison_path)
    monkeypatch.setattr(
        orchestrator,
        "_model_lock",
        lambda _config: {"revisions": {model: "revision" for model in config.model_ids}},
    )
    monkeypatch.setattr(orchestrator, "_lexical_results", lambda *_args: methods)
    monkeypatch.setattr(
        orchestrator,
        "_dense_result",
        lambda *_args, **_kwargs: (_report_method(1, 0.3), {"cache_status": "hit"}),
    )
    monkeypatch.setattr(orchestrator, "build_dev_comparison", lambda *_args: {"status": "ok"})

    result = orchestrator.evaluate_dev()

    assert result["status"] == "dev_evaluation_complete"
    assert result["dense_selection_candidate"] == "arabic-raw-v1"
    assert result["methods"]["bge__arabic-raw-v1"]["sample_count"] == 1


def test_freeze_dev_selection_records_rule_and_evidence(monkeypatch, tmp_path: Path) -> None:
    config = replace(load_phase7_config(), tracked_manifest_root=tmp_path)
    methods = {
        f"{method}__{policy}": _report_method(1, value)
        for method, value in (("keyword", 0.1), ("bm25", 0.2), ("e5", 0.3))
        for policy in config.normalization_policy_ids
    }
    payload = {
        "status": "dev_evaluation_complete",
        "corpus_hash": "corpus",
        "release_hash": "release",
        "query_length_bins": {"short_max": 3, "medium_max": 8},
        "methods": methods,
        "dense_selection_candidate": "arabic-raw-v1",
        "dense_diagnostics": {
            "arabic-raw-v1": {"cache_fingerprint": "e5-raw"},
            "arabic-light-v1": {"cache_fingerprint": "e5-light"},
            "bge": {"cache_fingerprint": "bge-raw"},
        },
    }
    dev_path = tmp_path / "dev.json"
    selection_path = tmp_path / "selection.json"
    lock = {"revisions": {model: "revision" for model in config.model_ids}}
    dev_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(orchestrator, "DEV_METRICS_PATH", dev_path)
    monkeypatch.setattr(orchestrator, "SELECTION_PATH", selection_path)
    monkeypatch.setattr(orchestrator, "load_phase7_config", lambda: config)
    monkeypatch.setattr(orchestrator, "_model_lock", lambda _config: lock)

    result = orchestrator.freeze_dev_selection()

    assert result["status"] == "dev_selection_frozen"
    assert result["selection"] == {
        "keyword": "arabic-raw-v1",
        "bm25": "arabic-raw-v1",
        "dense": "arabic-raw-v1",
    }
    assert result["bootstrap_replicates"] == 2000
    assert result["normalization_selection_rule"]["primary"] == "dev nDCG@10"


def test_encode_corpus_reports_diagnostics_and_progress(monkeypatch, tmp_path: Path) -> None:
    config = replace(load_phase7_config(), private_root=tmp_path)
    release = _mini_release((_mini_item("q-0"),))
    adapter = DenseModelAdapter(
        model_id="BAAI/bge-m3",
        revision="revision",
        max_length=1536,
        embedding_dimension=2,
        encoder=lambda texts, **_kwargs: np.tile(
            np.array([[1.0, 0.0]], dtype=np.float32), (len(texts), 1)
        ),
    )
    monkeypatch.setattr(orchestrator, "load_phase7_config", lambda: config)
    monkeypatch.setattr(orchestrator, "load_phase7_release", lambda: release)
    monkeypatch.setattr(
        orchestrator,
        "_model_lock",
        lambda _config: {"revisions": {"BAAI/bge-m3": "revision"}},
    )
    monkeypatch.setattr(orchestrator, "BGEM3Adapter", lambda **_kwargs: adapter)
    monkeypatch.setattr(orchestrator, "load_tokenizer", lambda _adapter: None)
    monkeypatch.setattr(
        orchestrator,
        "_dense_cache_identity",
        lambda *_args: (
            tmp_path,
            "fingerprint",
            ("Arabic evidence",),
            ("chunk-0",),
        ),
    )

    def fake_encode(*_args, **kwargs):
        kwargs["progress_callback"]({"completed_blocks": 1, "total_blocks": 1})
        return CheckpointEncodingResult(np.array([[1.0, 0.0]], dtype=np.float32), 1, "complete")

    monkeypatch.setattr(orchestrator, "encode_checkpointed", fake_encode)
    result = orchestrator.encode_corpus(model="bge-m3", resume=True)

    assert result["status"] == "complete"
    assert result["cache_status"] == "complete"
    assert result["chunk_count"] == 1


def test_evaluate_holdout_assembles_mocked_cache_only_results(monkeypatch, tmp_path: Path) -> None:
    config = replace(load_phase7_config(), private_root=tmp_path)
    release = _mini_release((_mini_item("q-h"),))
    selection = {
        "selection": {
            "keyword": "arabic-raw-v1",
            "bm25": "arabic-light-v1",
            "dense": "arabic-raw-v1",
        },
        "query_length_bins": {"short_max": 10, "medium_max": 20},
    }
    selection_path = tmp_path / "selection.json"
    holdout_path = tmp_path / "holdout.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")

    class FakeIndex:
        def search(self, _query: str, *, top_k: int) -> tuple[ScoredChunk, ...]:
            assert top_k == 10
            return (ScoredChunk("chunk-0", 1.0),)

    class FakeFactory:
        @classmethod
        def build(cls, *_args, **_kwargs) -> FakeIndex:
            return FakeIndex()

    monkeypatch.setattr(orchestrator, "load_phase7_config", lambda: config)
    monkeypatch.setattr(orchestrator, "load_phase7_release", lambda **_kwargs: release)
    monkeypatch.setattr(orchestrator, "SELECTION_PATH", selection_path)
    monkeypatch.setattr(orchestrator, "HOLDOUT_METRICS_PATH", holdout_path)
    monkeypatch.setattr(orchestrator, "KeywordIndex", FakeFactory)
    monkeypatch.setattr(orchestrator, "BM25Index", FakeFactory)
    monkeypatch.setattr(orchestrator, "verify_holdout_readiness", lambda: {"status": "ok"})
    monkeypatch.setattr(
        orchestrator,
        "_model_lock",
        lambda _config: {
            "revisions": {
                "intfloat/multilingual-e5-small": "e5-revision",
                "BAAI/bge-m3": "bge-revision",
            }
        },
    )
    monkeypatch.setattr(
        orchestrator,
        "_dense_result",
        lambda *_args, **_kwargs: (_report_method(1, 0.4), {"cache_status": "hit"}),
    )
    monkeypatch.setattr(orchestrator, "_write_final_manifest", lambda *_args: {"status": "done"})

    result = orchestrator.evaluate_holdout(allow_holdout=True)

    assert result["status"] == "holdout_evaluation_complete"
    assert set(result["methods"]) == {"keyword", "bm25", "e5__arabic-raw-v1", "bge__arabic-raw-v1"}
    assert json.loads(holdout_path.read_text(encoding="utf-8"))["sample_count"] == 1


def test_retrieval_plan_and_report_reflect_available_tracked_artifacts(
    monkeypatch, tmp_path: Path
) -> None:
    config = replace(load_phase7_config(), tracked_manifest_root=tmp_path)
    assert orchestrator.retrieval_plan(config)["dense_max_lengths"]["BAAI/bge-m3"] == 1536

    paths = (
        "CORPUS_MANIFEST_PATH",
        "MODEL_LOCK_PATH",
        "SELECTION_PATH",
        "FINAL_MANIFEST_PATH",
        "DEV_METRICS_PATH",
        "HOLDOUT_METRICS_PATH",
        "HOLDOUT_REPLAY_PATH",
        "COMPARISON_PATH",
    )
    for index, name in enumerate(paths):
        path = tmp_path / f"artifact-{index}.json"
        path.write_text(json.dumps({"artifact": name}), encoding="utf-8")
        monkeypatch.setattr(orchestrator, name, path)
    report = orchestrator.retrieval_report()

    assert report["status"] == "available"
    assert len(report) == len(paths) + 1


def test_build_retrieval_corpus_serializes_chunk_provenance(monkeypatch, tmp_path: Path) -> None:
    release = _mini_release((_mini_item("q-0"),))
    chunks_path = tmp_path / "chunks.jsonl"
    manifest_path = tmp_path / "corpus.json"
    monkeypatch.setattr(orchestrator, "load_phase7_release", lambda: release)
    monkeypatch.setattr(orchestrator, "PHASE7_PRIVATE_CHUNKS", chunks_path)
    monkeypatch.setattr(orchestrator, "CORPUS_MANIFEST_PATH", manifest_path)

    result = orchestrator.build_retrieval_corpus()

    assert result["corpus_hash"] == "corpus"
    row = json.loads(chunks_path.read_text(encoding="utf-8"))
    assert row["source_spans"] == [{"end": 10, "start": 0, "unit_id": "unit-0"}]
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["release_hash"] == "release"


def test_retrieval_smoke_requires_twenty_items_and_reports_pipelines(
    monkeypatch, tmp_path: Path
) -> None:
    items = tuple(_mini_item(f"q-{index}") for index in range(20))
    release = _mini_release(items)
    config = replace(load_phase7_config(), private_root=tmp_path)
    monkeypatch.setattr(orchestrator, "load_phase7_config", lambda: config)
    monkeypatch.setattr(orchestrator, "load_phase7_release", lambda: release)
    monkeypatch.setattr(
        orchestrator,
        "_lexical_results",
        lambda *_args: {"keyword__arabic-raw-v1": {}, "bm25__arabic-raw-v1": {}},
    )

    result = orchestrator.retrieval_smoke()

    assert result["status"] == "passed"
    assert result["smoke_count"] == 20
    assert result["pipelines"] == ["bm25__arabic-raw-v1", "keyword__arabic-raw-v1"]


def test_real_model_smoke_uses_mocked_normalized_adapters(monkeypatch, tmp_path: Path) -> None:
    chunks = (_mini_chunk("chunk-0"), _mini_chunk("chunk-1"))
    items = (_mini_item("q-0"),)
    release = _mini_release(items)
    release.chunks = chunks
    config = replace(load_phase7_config(), private_root=tmp_path)

    def adapter_factory(*, revision: str, max_length: int, device: str) -> DenseModelAdapter:
        del max_length, device
        model_id = "intfloat/multilingual-e5-small" if revision == "e5" else "BAAI/bge-m3"
        return DenseModelAdapter(
            model_id=model_id,
            revision=revision,
            embedding_dimension=2,
            encoder=lambda texts, **_kwargs: np.tile(
                np.array([[1.0, 0.0]], dtype=np.float32), (len(texts), 1)
            ),
        )

    monkeypatch.setattr(orchestrator, "load_phase7_config", lambda: config)
    monkeypatch.setattr(orchestrator, "load_phase7_release", lambda: release)
    monkeypatch.setattr(
        orchestrator,
        "_model_lock",
        lambda _config: {
            "revisions": {
                "intfloat/multilingual-e5-small": "e5",
                "BAAI/bge-m3": "bge",
            }
        },
    )
    monkeypatch.setattr(orchestrator, "E5SmallAdapter", adapter_factory)
    monkeypatch.setattr(orchestrator, "BGEM3Adapter", adapter_factory)

    result = orchestrator.real_model_smoke()

    assert result["status"] == "passed"
    assert set(result["models"]) == {
        "intfloat/multilingual-e5-small",
        "BAAI/bge-m3",
    }


def test_dense_result_can_encode_a_missing_cache_when_explicitly_allowed(
    monkeypatch, tmp_path: Path
) -> None:
    release = _mini_release((_mini_item("q-0"),))
    config = replace(load_phase7_config(), private_root=tmp_path)
    adapter = DenseModelAdapter(
        model_id="intfloat/multilingual-e5-small",
        revision="revision",
        max_length=512,
        embedding_dimension=2,
        encoder=lambda texts, **_kwargs: np.tile(
            np.array([[1.0, 0.0]], dtype=np.float32), (len(texts), 1)
        ),
    )
    monkeypatch.setattr(
        orchestrator,
        "load_cached_embeddings",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(FileNotFoundError("missing")),
    )
    monkeypatch.setattr(
        orchestrator,
        "encode_checkpointed",
        lambda *_args, **_kwargs: CheckpointEncodingResult(
            np.array([[1.0, 0.0]], dtype=np.float32), 2, "complete"
        ),
    )
    monkeypatch.setattr(orchestrator, "loaded_tokenizer", lambda _adapter: None)

    result, diagnostics = orchestrator._dense_result(
        release,
        release.items,
        config,
        adapter=adapter,
        policy_id="arabic-raw-v1",
        model_revision="revision",
        private_stage="dev",
        query_length_bins=QueryLengthBins(10, 20),
        allow_corpus_encode=True,
    )

    assert result["sample_count"] == 1
    assert diagnostics["cache_status"] == "complete"
    assert (tmp_path / "dev" / "rankings").is_dir()


def test_holdout_readiness_success_validates_both_selected_caches(
    monkeypatch, tmp_path: Path
) -> None:
    config = replace(load_phase7_config(), private_root=tmp_path)
    selection_path = tmp_path / "selection.json"
    corpus_path = tmp_path / "corpus.json"
    selection_path.write_text(
        json.dumps(
            {
                "status": "dev_selection_frozen",
                "selection": {
                    "keyword": "arabic-raw-v1",
                    "bm25": "arabic-light-v1",
                    "dense": "arabic-raw-v1",
                },
            }
        ),
        encoding="utf-8",
    )
    corpus_path.write_text(
        json.dumps({"chunk_count": 1, "corpus_hash": "corpus", "chunk_policy_hashes": ["policy"]}),
        encoding="utf-8",
    )
    lock = {
        "revisions": {
            "intfloat/multilingual-e5-small": "e5-revision",
            "BAAI/bge-m3": "bge-revision",
        }
    }

    def adapter_factory(
        *, model_id: str, revision: str, max_length: int, **_kwargs
    ) -> DenseModelAdapter:
        return DenseModelAdapter(
            model_id=model_id,
            revision=revision,
            max_length=max_length,
            formatting_contract="e5-query-passage-v1"
            if model_id.startswith("intfloat")
            else "bge-m3-dense-v1",
            embedding_dimension=384 if model_id.startswith("intfloat") else 1024,
        )

    monkeypatch.setattr(orchestrator, "load_phase7_config", lambda: config)
    monkeypatch.setattr(orchestrator, "SELECTION_PATH", selection_path)
    monkeypatch.setattr(orchestrator, "CORPUS_MANIFEST_PATH", corpus_path)
    monkeypatch.setattr(orchestrator, "HOLDOUT_METRICS_PATH", tmp_path / "holdout.json")
    monkeypatch.setattr(orchestrator, "FINAL_MANIFEST_PATH", tmp_path / "final.json")
    monkeypatch.setattr(
        orchestrator, "_file_sha256", lambda _path: orchestrator.EXPECTED_SELECTION_SHA256
    )
    monkeypatch.setattr(orchestrator, "_model_lock", lambda _config: lock)
    monkeypatch.setattr(
        orchestrator,
        "E5SmallAdapter",
        lambda **kwargs: adapter_factory(model_id="intfloat/multilingual-e5-small", **kwargs),
    )
    monkeypatch.setattr(
        orchestrator,
        "BGEM3Adapter",
        lambda **kwargs: adapter_factory(model_id="BAAI/bge-m3", **kwargs),
    )
    monkeypatch.setattr(
        orchestrator,
        "load_cached_embeddings",
        lambda *_args, **_kwargs: (
            np.zeros((1, 384), dtype=np.float32),
            ("chunk-0",),
        ),
    )
    monkeypatch.setattr(
        orchestrator,
        "checkpoint_cache_status_from_manifest",
        lambda *_args, **_kwargs: {
            "completed_blocks": 1,
            "total_blocks": 1,
            "completed_chunks": 1,
            "total_chunks": 1,
        },
    )

    result = orchestrator.verify_holdout_readiness()

    assert result["status"] == "holdout_readiness_passed"
    assert result["cache_checks"]["e5"]["completed_chunks"] == 1
    assert result["cache_checks"]["bge"]["completed_chunks"] == 1
    assert result["private_per_query_artifacts"] == "forbidden"


def test_final_manifest_writer_records_artifact_hashes(monkeypatch, tmp_path: Path) -> None:
    config = replace(load_phase7_config(), tracked_manifest_root=tmp_path)
    model_lock_path = tmp_path / "model-lock.json"
    model_lock_path.write_text(json.dumps({"models": ["e5", "bge"]}), encoding="utf-8")
    dev_path = tmp_path / "dev.json"
    holdout_path = tmp_path / "holdout.json"
    comparison_path = tmp_path / "comparison.json"
    for path in (dev_path, holdout_path, comparison_path):
        path.write_text(json.dumps({"path": path.name}), encoding="utf-8")
    final_path = tmp_path / "final.json"
    release = _mini_release((_mini_item("q-0"),))
    selection = {"selection": {"keyword": "arabic-raw-v1"}}
    monkeypatch.setattr(orchestrator, "MODEL_LOCK_PATH", model_lock_path)
    monkeypatch.setattr(orchestrator, "DEV_METRICS_PATH", dev_path)
    monkeypatch.setattr(orchestrator, "HOLDOUT_METRICS_PATH", holdout_path)
    monkeypatch.setattr(orchestrator, "HOLDOUT_REPLAY_PATH", tmp_path / "replay.json")
    monkeypatch.setattr(orchestrator, "COMPARISON_PATH", comparison_path)
    monkeypatch.setattr(orchestrator, "FINAL_MANIFEST_PATH", final_path)

    result = orchestrator._write_final_manifest(config, release, selection)

    assert result["status"] == "phase7_experiment_complete"
    assert set(result["tracked_artifact_hashes"]) == {
        dev_path.as_posix(),
        holdout_path.as_posix(),
        comparison_path.as_posix(),
    }
    assert json.loads(final_path.read_text(encoding="utf-8"))["corpus_hash"] == "corpus"


def test_final_manifest_writer_allows_recovery_artifact_addition_without_selection_change(
    monkeypatch, tmp_path: Path
) -> None:
    config = replace(load_phase7_config(), tracked_manifest_root=tmp_path)
    model_lock_path = tmp_path / "model-lock.json"
    model_lock_path.write_text(json.dumps({"models": ["e5", "bge"]}), encoding="utf-8")
    dev_path = tmp_path / "dev.json"
    holdout_path = tmp_path / "holdout.json"
    comparison_path = tmp_path / "comparison.json"
    replay_path = tmp_path / "replay.json"
    for path in (dev_path, holdout_path, comparison_path):
        path.write_text(json.dumps({"path": path.name}), encoding="utf-8")
    final_path = tmp_path / "final.json"
    release = _mini_release((_mini_item("q-0"),))
    selection = {"selection": {"keyword": "arabic-raw-v1"}}
    monkeypatch.setattr(orchestrator, "MODEL_LOCK_PATH", model_lock_path)
    monkeypatch.setattr(orchestrator, "DEV_METRICS_PATH", dev_path)
    monkeypatch.setattr(orchestrator, "HOLDOUT_METRICS_PATH", holdout_path)
    monkeypatch.setattr(orchestrator, "COMPARISON_PATH", comparison_path)
    monkeypatch.setattr(orchestrator, "HOLDOUT_REPLAY_PATH", replay_path)
    monkeypatch.setattr(orchestrator, "FINAL_MANIFEST_PATH", final_path)

    orchestrator._write_final_manifest(config, release, selection)
    replay_path.write_text(json.dumps({"status": "recovered"}), encoding="utf-8")
    result = orchestrator._write_final_manifest(config, release, selection)

    assert replay_path.as_posix() in result["tracked_artifact_hashes"]
