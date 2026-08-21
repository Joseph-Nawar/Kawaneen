from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from kawaneen.grounding.inputs import load_frozen_phase8_dev_rankings
from kawaneen.grounding.provenance import CanonicalCorpusResolver


def write_corpus(tmp_path: Path) -> tuple[Path, Path]:
    canonical = tmp_path / "canonical_units.json"
    canonical.write_text(
        json.dumps(
            {
                "summary": {"corpus_hash": "b" * 64},
                "units": [
                    {
                        "unit_id": "u-1",
                        "document_id": "doc-1",
                        "ordinal": 2,
                        "text": "النص الأصلي ِ",
                        "unit_type": "events",
                        "provenance": {
                            "source_id": "alarb",
                            "source_version": "v1",
                            "source_path": "private",
                            "source_row": 1,
                            "source_field": "events",
                            "split": "",
                        },
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text(
        json.dumps(
            {
                "chunk_id": "chunk-1",
                "document_id": "doc-1",
                "display_text": "FORGED RETRIEVAL TEXT",
                "search_text": "FORGED NORMALIZED TEXT",
                "source_unit_ids": ["u-1"],
                "source_spans": [{"unit_id": "u-1", "start": 0, "end": 12}],
                "provenance": {
                    "source_id": "forged-source",
                    "document_title": "invented title",
                    "page": "99",
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return canonical, chunks


def test_resolver_uses_canonical_text_and_metadata_not_retrieval_metadata(
    tmp_path: Path,
) -> None:
    canonical, chunks = write_corpus(tmp_path)
    resolved = CanonicalCorpusResolver.from_json(canonical, chunks).resolve_chunk("chunk-1")
    assert resolved.units[0].display_text == "النص الأصلي ِ"
    assert resolved.units[0].source.source_id == "alarb"
    assert resolved.units[0].source.document_title is None
    assert resolved.units[0].source.page is None


def test_resolver_wires_authoritative_document_metadata(tmp_path: Path) -> None:
    canonical, chunks = write_corpus(tmp_path)
    documents = tmp_path / "documents.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "document_id": "doc-1",
                    "title": "Canonical title",
                    "source_id": "alarb",
                    "raw_article_label": "المادة 1",
                    "source_metadata_json": '{"details_url":"https://example.test/doc-1"}',
                }
            ],
            schema=pa.schema(
                [
                    ("document_id", pa.string()),
                    ("title", pa.string()),
                    ("source_id", pa.string()),
                    ("raw_article_label", pa.string()),
                    ("source_metadata_json", pa.string()),
                ]
            ),
        ),
        documents,
    )
    resolved = CanonicalCorpusResolver.from_json(
        canonical,
        chunks,
        document_paths=(documents,),
    ).resolve_chunk("chunk-1")
    assert resolved.units[0].source.document_title == "Canonical title"
    assert resolved.units[0].source.article == "المادة 1"
    assert resolved.units[0].source.source_url == "https://example.test/doc-1"


def test_unknown_chunk_and_unknown_unit_fail_closed(tmp_path: Path) -> None:
    canonical, chunks = write_corpus(tmp_path)
    resolver = CanonicalCorpusResolver.from_json(canonical, chunks)
    with pytest.raises(ValueError, match="unknown chunk"):
        resolver.resolve_chunk("missing")

    chunks.write_text(
        json.dumps(
            {
                "chunk_id": "bad",
                "document_id": "doc-1",
                "source_unit_ids": ["missing-unit"],
                "source_spans": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown canonical unit"):
        CanonicalCorpusResolver.from_json(canonical, chunks).resolve_chunk("bad")


def test_frozen_ranking_reader_reads_only_persisted_top_eight(tmp_path: Path) -> None:
    selection = tmp_path / "selection.json"
    selection.write_text(
        '{"status":"phase8_dev_selection_frozen",'
        '"selected_pipeline":"rrf_reranked","reranker":{"serving_depth":8}}\n',
        encoding="utf-8",
    )
    selection_sha = hashlib.sha256(selection.read_bytes()).hexdigest()
    rerank_root = tmp_path / "rerank"
    rerank_root.mkdir()
    (rerank_root / "manifest.json").write_text(
        json.dumps(
            {
                "queries": {
                    "q-1": {"status": "completed", "path": "q-1.json"},
                    "q-2": {"status": "completed", "path": "q-2.json"},
                }
            }
        ),
        encoding="utf-8",
    )
    for query_id in ("q-1", "q-2"):
        (rerank_root / f"{query_id}.json").write_text(
            json.dumps(
                {
                    "query_id": query_id,
                    "ranked_chunk_ids": [f"{query_id}-c{i}" for i in range(1, 11)],
                }
            ),
            encoding="utf-8",
        )

    result = load_frozen_phase8_dev_rankings(
        selection_path=selection,
        rerank_root=rerank_root,
        expected_selection_sha256=selection_sha,
    )
    assert [(item.query_id, item.rank, item.chunk_id) for item in result] == [
        ("q-1", i, f"q-1-c{i}") for i in range(1, 9)
    ] + [("q-2", i, f"q-2-c{i}") for i in range(1, 9)]
