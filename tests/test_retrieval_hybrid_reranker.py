import json
import re
import tomllib
from pathlib import Path

import pytest

from kawaneen.cli import build_parser
from kawaneen.retrieval.hybrid.checkpoints import CheckpointStore, checkpoint_status
from kawaneen.retrieval.hybrid.contracts import FusedCandidate, RerankerConfig
from kawaneen.retrieval.hybrid.orchestration import (
    PHASE8_CONFIG,
    PHASE8_MODEL_LOCK,
    load_phase8_reranker_lock,
)
from kawaneen.retrieval.hybrid.reranker import (
    BGERerankerAdapter,
    rerank_candidates,
)
from kawaneen.retrieval.models import RetrievalChunk


def _chunk(chunk_id: str, text: str) -> RetrievalChunk:
    return RetrievalChunk(
        chunk_id=chunk_id,
        document_id="doc",
        source_id="source",
        unit_type="facts",
        display_text=text,
        search_text="normalized and different",
        source_unit_ids=("unit",),
        chunk_policy_hash="chunk",
        normalization_policy_id="raw",
        normalization_policy_hash="norm",
        token_count=len(text.split()),
    )


def _candidate(chunk_id: str, rank: int) -> FusedCandidate:
    return FusedCandidate(
        chunk_id=chunk_id,
        fused_rank=rank,
        fused_score=1 / rank,
        sparse_rank=rank,
        sparse_score=1.0,
        dense_rank=None,
        dense_score=None,
        provenance="sparse-only",
    )


class FakeTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        return list(range(len(text.split()) + (2 if add_special_tokens else 0)))

    def decode(self, ids: list[int]) -> str:
        return " ".join(f"p{value}" for value in ids)


def test_reranker_uses_original_query_and_exact_display_text() -> None:
    calls: list[tuple[str, str]] = []
    chunks = {"a": _chunk("a", "display text")}
    reranked, diagnostics = rerank_candidates(
        "original query",
        (_candidate("a", 1),),
        chunks,
        scorer=lambda query, passage: calls.append((query, passage)) or 0.5,
        tokenizer=FakeTokenizer(),
        config=RerankerConfig(max_length=20),
    )

    assert calls == [("original query", "display text")]
    assert reranked[0].score == 0.5
    assert diagnostics.truncated_count == 0


def test_reranker_truncates_only_passage_and_tie_breaks_by_fused_rank_then_id() -> None:
    chunks = {"a": _chunk("a", "one two three four five"), "b": _chunk("b", "one")}
    result, diagnostics = rerank_candidates(
        "query words",
        (_candidate("b", 2), _candidate("a", 1)),
        chunks,
        scorer=lambda _query, _passage: 1.0,
        tokenizer=FakeTokenizer(),
        config=RerankerConfig(max_length=5),
    )

    assert [item.chunk_id for item in result] == ["a", "b"]
    assert diagnostics.truncated_count == 1
    assert diagnostics.max_pair_tokens <= 5


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_reranker_rejects_non_finite_scores(score: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        rerank_candidates(
            "q",
            (_candidate("a", 1),),
            {"a": _chunk("a", "text")},
            scorer=lambda _query, _passage: score,
            tokenizer=FakeTokenizer(),
            config=RerankerConfig(),
        )


def test_model_and_candidate_fingerprint_changes_when_contract_changes() -> None:
    adapter = BGERerankerAdapter(revision="revision-sha")
    first = adapter.fingerprint(
        "config", "selection", ("a", "b"), RerankerConfig(model_revision="revision-sha")
    )
    second = adapter.fingerprint(
        "changed", "selection", ("a", "b"), RerankerConfig(model_revision="revision-sha")
    )
    assert first != second


def test_phase8_reranker_lock_requires_full_revision_and_config_alignment() -> None:
    lock = load_phase8_reranker_lock()
    revision = lock["revision"]
    assert lock["model_id"] == "BAAI/bge-reranker-v2-m3"
    assert re.fullmatch(r"[0-9a-f]{40}", revision)

    config = tomllib.loads(PHASE8_CONFIG.read_text(encoding="utf-8"))
    assert config["reranker"]["model_id"] == lock["model_id"]
    assert config["reranker"]["model_revision"] == revision

    adapter = BGERerankerAdapter(revision=revision)
    fingerprint = adapter.fingerprint(
        "config", "selection", ("a",), RerankerConfig(model_revision=revision)
    )
    changed_revision = "0" * 40
    changed = BGERerankerAdapter(revision=changed_revision).fingerprint(
        "config", "selection", ("a",), RerankerConfig(model_revision=changed_revision)
    )
    assert fingerprint != changed


def test_phase8_rerank_lock_gate_accepts_locked_revision_without_loading_model() -> None:
    lock = load_phase8_reranker_lock(PHASE8_MODEL_LOCK)
    assert lock["revision"] == "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
    args = build_parser().parse_args(
        ["retrieval", "phase8-rerank-dev", "--resume", "--device", "cpu"]
    )
    assert args.retrieval_command == "phase8-rerank-dev"
    assert args.resume is True
    assert args.device == "cpu"


def test_phase8_reranker_lock_rejects_short_revision(tmp_path: Path) -> None:
    path = tmp_path / "phase8_model_lock.json"
    path.write_text(
        json.dumps({"model_id": "BAAI/bge-reranker-v2-m3", "revision": "953dc6f"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="full 40-character SHA"):
        load_phase8_reranker_lock(path)


def test_checkpoint_store_resumes_valid_queries_and_recomputes_corrupt(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path, fingerprint="fp")
    store.write("q1", {"ranked_chunk_ids": ["a"], "scores": [1.0]})
    store.write("q2", {"ranked_chunk_ids": ["b"], "scores": [0.5]})
    (tmp_path / "q2.json").write_text("{corrupt", encoding="utf-8")

    assert store.valid("q1", ("a",))
    assert not store.valid("q2", ("b",))
    # Manifest-only status reports completed entries; validity of each file is
    # checked only when the executor considers that query for resume.
    assert checkpoint_status(tmp_path)["valid_count"] == 2


def test_status_reads_only_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = {"schema_version": 1, "fingerprint": "fp", "queries": {"q1": "q1.json"}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    original_read_text = Path.read_text

    def manifest_only(path: Path, *args: object, **kwargs: object) -> str:
        if path.name != "manifest.json":
            pytest.fail("query loaded")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", manifest_only)
    assert checkpoint_status(tmp_path)["total_count"] == 1
