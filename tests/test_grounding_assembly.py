from __future__ import annotations

import json
from pathlib import Path

from kawaneen.grounding.assembly import ContextAssembler
from kawaneen.grounding.contracts import RetrievalInput
from kawaneen.grounding.provenance import CanonicalCorpusResolver
from kawaneen.grounding.rendering import render_context


class FakeCounter:
    identity = "fake-codepoint-v1"

    def count(self, text: str) -> int:
        return len(text)


def resolver(tmp_path: Path) -> CanonicalCorpusResolver:
    units = []
    for document_id, rows in {
        "doc-a": [
            ("a1", 1, "A1", ("Intro",)),
            ("a2", 2, "A2", ("Intro",)),
            ("a3", 3, "A3", ("Rules",)),
        ],
        "doc-b": [("b1", 1, "A1", ())],
    }.items():
        for unit_id, ordinal, text, _heading in rows:
            units.append(
                {
                    "unit_id": unit_id,
                    "document_id": document_id,
                    "ordinal": ordinal,
                    "text": text,
                    "unit_type": "events",
                    "heading_path": list(_heading),
                    "provenance": {
                        "source_id": "source-a",
                        "source_version": "v1",
                        "source_path": "private",
                        "source_row": ordinal,
                        "source_field": "events",
                        "split": "",
                    },
                }
            )
    canonical = tmp_path / "canonical.json"
    canonical.write_text(
        json.dumps({"summary": {"corpus_hash": "b" * 64}, "units": units}),
        encoding="utf-8",
    )
    chunks = tmp_path / "chunks.jsonl"
    rows = [
        ("c-a1", "doc-a", ["a1"]),
        ("c-a2", "doc-a", ["a2"]),
        ("c-a3", "doc-a", ["a3"]),
        ("c-a23", "doc-a", ["a2", "a3"]),
        ("c-b1", "doc-b", ["b1"]),
    ]
    chunks.write_text(
        "".join(
            json.dumps(
                {
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "display_text": "forged",
                    "source_unit_ids": unit_ids,
                    "source_spans": [],
                }
            )
            + "\n"
            for chunk_id, document_id, unit_ids in rows
        ),
        encoding="utf-8",
    )
    return CanonicalCorpusResolver.from_json(canonical, chunks)


def inputs(*chunk_ids: str) -> tuple[RetrievalInput, ...]:
    return tuple(
        RetrievalInput(query_id="q1", rank=rank, chunk_id=chunk_id)
        for rank, chunk_id in enumerate(chunk_ids, start=1)
    )


def assemble(
    tmp_path: Path,
    ranked: tuple[RetrievalInput, ...],
    max_tokens: int = 10_000,
):
    return ContextAssembler(
        resolver(tmp_path), FakeCounter(), max_context_tokens=max_tokens
    ).assemble(
        query_id="q1",
        ranked_inputs=ranked,
        phase8_selection_sha256="a" * 64,
        canonical_corpus_hash="b" * 64,
    )


def test_overlapping_and_exact_duplicate_chunks_keep_each_canonical_unit_once(
    tmp_path: Path,
) -> None:
    pack = assemble(tmp_path, inputs("c-a23", "c-a2", "c-a1"))
    assert [unit.unit_id for unit in pack.units] == ["a1", "a2", "a3"]
    assert len(pack.blocks) == 2
    assert pack.evidence[1].contributing_ranks == (1, 2)
    assert render_context(pack).count("A2") == 1


def test_document_groups_use_best_rank_but_units_use_canonical_order(tmp_path: Path) -> None:
    pack = assemble(tmp_path, inputs("c-a3", "c-b1", "c-a1"))
    assert [block.document_id for block in pack.blocks] == ["doc-a", "doc-a", "doc-b"]
    assert [unit.unit_id for unit in pack.units] == ["a1", "a3", "b1"]


def test_identical_text_in_different_documents_remains_two_units(tmp_path: Path) -> None:
    pack = assemble(tmp_path, inputs("c-a1", "c-b1"))
    assert [unit.unit_id for unit in pack.units] == ["a1", "b1"]
    assert [unit.document_id for unit in pack.units] == ["doc-a", "doc-b"]


def test_equal_retrieval_ranks_use_deterministic_chunk_and_document_ties(tmp_path: Path) -> None:
    pack = ContextAssembler(resolver(tmp_path), FakeCounter(), max_context_tokens=1000).assemble(
        query_id="q1",
        ranked_inputs=(
            RetrievalInput(query_id="q1", rank=1, chunk_id="c-b1"),
            RetrievalInput(query_id="q1", rank=1, chunk_id="c-a1"),
        ),
        phase8_selection_sha256="a" * 64,
        canonical_corpus_hash="b" * 64,
    )
    assert [block.document_id for block in pack.blocks] == ["doc-a", "doc-b"]


def test_heading_changes_and_non_adjacent_units_make_separate_blocks(tmp_path: Path) -> None:
    pack = assemble(tmp_path, inputs("c-a1", "c-a3"))
    assert [block.heading_path for block in pack.blocks] == [("Intro",), ("Rules",)]
    assert [unit.unit_id for unit in pack.units] == ["a1", "a3"]


def test_exact_token_budget_is_allowed_and_over_budget_omits_whole_unit(tmp_path: Path) -> None:
    full = assemble(tmp_path, inputs("c-a1"), max_tokens=10_000)
    exact = assemble(tmp_path, inputs("c-a1"), max_tokens=full.token_count)
    assert exact.token_count == full.token_count
    assert [unit.unit_id for unit in exact.units] == ["a1"]

    over = assemble(tmp_path, inputs("c-a1", "c-a2"), max_tokens=full.token_count)
    assert [unit.unit_id for unit in over.units] == ["a1"]
    assert over.omissions[0].unit_id == "a2"
    assert "A2" not in render_context(over)


def test_empty_retrieval_is_a_valid_empty_pack(tmp_path: Path) -> None:
    pack = assemble(tmp_path, ())
    assert pack.units == ()
    assert pack.blocks == ()
    assert pack.evidence == ()
    assert pack.token_count == 0


def test_long_unit_is_skipped_without_partial_text(tmp_path: Path) -> None:
    pack = assemble(tmp_path, inputs("c-a1"), max_tokens=1)
    assert pack.units == ()
    assert pack.omissions[0].unit_id == "a1"
    assert "A1" not in render_context(pack)
