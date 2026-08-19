from pathlib import Path

import pytest

from kawaneen.retrieval.manifests import build_corpus_manifest, hash_file, stable_hash
from kawaneen.retrieval.models import RetrievalChunk


def _chunk(chunk_id: str) -> RetrievalChunk:
    return RetrievalChunk(
        chunk_id=chunk_id,
        document_id="doc",
        source_id="source",
        unit_type="paragraph",
        display_text="text",
        search_text="text",
        source_unit_ids=(chunk_id,),
        source_spans=(),
        chunk_policy_hash="policy",
        normalization_policy_id="arabic-raw-v1",
        normalization_policy_hash="normalization",
        token_count=1,
    )


def test_manifest_and_file_hashes_are_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "payload"
    path.write_text("payload", encoding="utf-8")
    assert hash_file(path) == hash_file(str(path))
    assert stable_hash({"b": 2, "a": 1}) == stable_hash({"a": 1, "b": 2})
    manifest = build_corpus_manifest((_chunk("b"), _chunk("a")), corpus_hash="corpus")
    assert manifest["chunk_count"] == 2
    assert manifest["chunk_ids_hash"] == stable_hash(["a", "b"])


def test_manifest_rejects_duplicate_chunk_ids() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        build_corpus_manifest((_chunk("same"), _chunk("same")), corpus_hash="corpus")
