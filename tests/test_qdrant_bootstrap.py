from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def _write_seed(root: Path) -> None:
    corpus = root / "corpus"
    embeddings = root / "embeddings" / "BAAI__bge-m3" / "arabic-raw-v1" / "fixture"
    corpus.mkdir(parents=True)
    embeddings.mkdir(parents=True)
    rows = []
    for index in range(2):
        rows.append(
            json.dumps(
                {
                    "chunk_id": f"c{index}",
                    "document_id": f"d{index}",
                    "source_id": "fixture",
                    "unit_type": "events",
                    "display_text": f"نص {index}",
                    "search_text": f"نص {index}",
                    "source_unit_ids": [f"u{index}"],
                    "source_spans": [{"unit_id": f"u{index}", "start": 0, "end": 5}],
                    "chunk_policy_hash": "a" * 64,
                    "normalization_policy_id": "arabic-light-v1",
                    "normalization_policy_hash": "b" * 64,
                    "token_count": 2,
                },
                ensure_ascii=False,
            )
        )
    (corpus / "chunks.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (embeddings / "ids.json").write_text(json.dumps(["c0", "c1"]), encoding="utf-8")
    np.save(embeddings / "vectors.npy", np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    (root / "manifest.json").write_text(json.dumps({"corpus_hash": "a" * 64}), encoding="utf-8")


def test_collection_name_is_stable_and_owned() -> None:
    from kawaneen.retrieval.qdrant_bootstrap import collection_name_for

    assert collection_name_for("a" * 64) == "kawaneen_aaaaaaaaaaaa_bge_m3"
    with pytest.raises(ValueError, match="SHA-256"):
        collection_name_for("A" * 64)


def test_seed_loader_validates_frozen_identity_and_vectors(tmp_path: Path) -> None:
    from kawaneen.retrieval.qdrant_bootstrap import load_qdrant_seed

    root = tmp_path / "phase7_retrieval"
    _write_seed(root)
    seed = load_qdrant_seed(
        root,
        expected_corpus_hash="a" * 64,
        expected_model_revision="b" * 40,
        expected_dimension=2,
    )
    assert seed.collection_name == "kawaneen_aaaaaaaaaaaa_bge_m3"
    assert seed.chunk_ids == ("c0", "c1")
    assert seed.vectors.dtype == np.float32

    with pytest.raises(ValueError, match="corpus"):
        load_qdrant_seed(
            root,
            expected_corpus_hash="c" * 64,
            expected_model_revision="b" * 40,
            expected_dimension=2,
        )

    with pytest.raises(ValueError, match="model revision"):
        load_qdrant_seed(
            root,
            expected_corpus_hash="a" * 64,
            expected_model_revision="",
            expected_dimension=2,
        )


def test_seed_qdrant_collection_creates_and_reuses_owned_collection(tmp_path: Path) -> None:
    from kawaneen.retrieval.qdrant_bootstrap import load_qdrant_seed, seed_qdrant_collection

    root = tmp_path / "phase7_retrieval"
    _write_seed(root)
    seed = load_qdrant_seed(
        root,
        expected_corpus_hash="a" * 64,
        expected_model_revision="b" * 40,
        expected_dimension=2,
    )

    class Client:
        def __init__(self) -> None:
            self.info = None
            self.points: list[object] = []

        def get_collection(self, collection_name: str) -> object:
            del collection_name
            if self.info is None:
                raise RuntimeError("missing collection")
            return self.info

        def create_collection(self, *, collection_name: str, vectors_config: object) -> None:
            self.info = type(
                "Info",
                (),
                {
                    "config": type(
                        "Config", (), {"params": type("Params", (), {"vectors": vectors_config})()}
                    )()
                },
            )()
            self.collection_name = collection_name

        def upsert(self, *, collection_name: str, points: list[object], wait: bool) -> None:
            assert collection_name == seed.collection_name
            assert wait is True
            self.points = points

        def delete_collection(self, collection_name: str) -> None:
            assert collection_name == seed.collection_name
            self.info = None
            self.points = []

        def count(self, *, collection_name: str, exact: bool) -> object:
            assert collection_name == seed.collection_name
            assert exact is True
            return type("Count", (), {"count": len(self.points)})()

        def scroll(
            self, *, collection_name: str, limit: int, with_payload: bool
        ) -> tuple[list[object], None]:
            assert collection_name == seed.collection_name
            assert limit == 1
            assert with_payload is True
            return self.points[:1], None

    client = Client()
    assert seed_qdrant_collection(client, seed) == seed.collection_name
    assert len(client.points) == 2
    assert seed_qdrant_collection(client, seed) == seed.collection_name
    client.points = []
    assert seed_qdrant_collection(client, seed) == seed.collection_name
