import json
from pathlib import Path

import pytest

import kawaneen.retrieval.corpus as corpus_module
from kawaneen.chunking.policies import get_chunk_policy
from kawaneen.retrieval.corpus import (
    _build_complete_phase5_chunks,
    _load_chunks,
    _rekey_for_frozen_phase6_qrels,
    validate_qrel_chunks,
)
from kawaneen.retrieval.manifests import stable_hash
from kawaneen.retrieval.models import RetrievalChunk, RetrievalRelease


def _chunk(chunk_id: str) -> RetrievalChunk:
    return RetrievalChunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        source_id="alarb",
        unit_type="facts",
        display_text="النص الأصلي",
        search_text="النص الاصلي",
        source_unit_ids=("unit-1",),
        source_spans=((0, 2),),
        chunk_policy_hash="chunk-hash",
        normalization_policy_id="arabic-light-v1",
        normalization_policy_hash="norm-hash",
        token_count=2,
    )


def test_qrel_chunk_ids_must_exist_in_retrieval_corpus() -> None:
    with pytest.raises(ValueError, match="outside retrieval corpus"):
        validate_qrel_chunks({"query-1": ("missing",)}, (_chunk("present"),))


def test_duplicate_chunk_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate chunk"):
        validate_qrel_chunks({}, (_chunk("same"), _chunk("same")))


def test_tracked_corpus_manifest_does_not_contain_text(tmp_path: Path) -> None:
    from kawaneen.retrieval.manifests import build_corpus_manifest

    payload = build_corpus_manifest((_chunk("chunk-1"),), corpus_hash="corpus-hash")
    serialized = json.dumps(payload, ensure_ascii=False)
    assert "النص الأصلي" not in serialized
    assert "النص الاصلي" not in serialized
    (tmp_path / "manifest.json").write_text(serialized, encoding="utf-8")


def test_chunk_loader_uses_explicit_unit_metadata(tmp_path: Path) -> None:
    path = tmp_path / "chunks.jsonl"
    path.write_text(
        json.dumps(
            {
                "chunk_id": "chunk-1",
                "document_id": "doc-1",
                "source_id": "alarb",
                "unit_type": "FACTS",
                "display_text": "Original",
                "search_text": "original",
                "source_unit_ids": ["unit-1"],
                "source_spans": [{"start": 0, "end": 2}],
                "chunk_policy_hash": "chunk-hash",
                "normalization_policy_id": "arabic-light-v1",
                "normalization_policy_hash": "norm-hash",
                "token_count": 1,
                "provenance": {"source_id": "alarb", "source_field": "facts"},
            }
        ),
        encoding="utf-8",
    )
    chunks = _load_chunks(path, {"unit-1": {"document_id": "doc-1"}})

    assert chunks[0].unit_type == "facts"
    assert chunks[0].document_id == "doc-1"


def test_complete_phase5_builder_and_frozen_qrel_rekey() -> None:
    snapshot = {
        "units": [
            {
                "unit_id": "unit-1",
                "document_id": "doc-1",
                "unit_type": "facts",
                "text": "A short legal fact.",
                "provenance": {
                    "source_id": "alarb",
                    "source_version": "v1",
                    "source_path": "local",
                    "source_row": 1,
                    "source_field": "facts",
                },
            }
        ]
    }
    complete = _build_complete_phase5_chunks(snapshot)
    rekeyed = _rekey_for_frozen_phase6_qrels(complete)

    assert complete
    assert rekeyed[0].chunk_id != complete[0].chunk_id
    assert rekeyed[0].source_spans == complete[0].source_spans


def test_phase7_release_loader_reads_frozen_manifest_and_chunks(
    monkeypatch, tmp_path: Path
) -> None:
    phase6_root = tmp_path / "phase6"
    (phase6_root / "corpus").mkdir(parents=True)
    (phase6_root / "corpus" / "canonical_units.json").write_text(
        json.dumps({"units": []}), encoding="utf-8"
    )
    manifest_path = tmp_path / "phase6-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_version": "phase6-retrieval-eval-ai-reviewed-v1",
                "corpus_hash": "corpus-hash",
                "hashes": {"item_set": stable_hash([])},
            }
        ),
        encoding="utf-8",
    )
    chunks_path = tmp_path / "chunks.jsonl"
    raw = {
        "chunk_id": "chunk-1",
        "document_id": "doc-1",
        "source_id": "alarb",
        "unit_type": "facts",
        "display_text": "Original",
        "search_text": "original",
        "source_unit_ids": ["unit-1"],
        "source_spans": [{"start": 0, "end": 2}],
        "chunk_policy_hash": get_chunk_policy("legal-structure-v1").policy_hash,
        "normalization_policy_id": "arabic-light-v1",
        "normalization_policy_hash": "norm-hash",
        "token_count": 1,
        "provenance": {"source_id": "alarb", "source_field": "facts"},
    }
    chunks_path.write_text(json.dumps(raw) + "\n", encoding="utf-8")
    monkeypatch.setattr(corpus_module, "read_items_jsonl", lambda _path: ())

    release = corpus_module.load_phase7_release(
        phase6_root,
        phase6_manifest_path=manifest_path,
        chunks_path=chunks_path,
    )

    assert release.items == ()
    assert release.chunks[0].chunk_id == "chunk-1"
    assert release.corpus_manifest["corpus_hash"] == "corpus-hash"


def test_retrieval_release_guards_holdout_access() -> None:
    release = RetrievalRelease((), (), {}, {})
    assert release.split_items("dev") == ()
    with pytest.raises(PermissionError, match="allow_holdout"):
        release.split_items("holdout")
