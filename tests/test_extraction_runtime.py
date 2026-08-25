from pathlib import Path

from kawaneen.extraction.checkpoints import (
    ExtractionCheckpoint,
    ExtractionCheckpointStore,
    extraction_fingerprint,
)
from kawaneen.extraction.tokenizer import pinned_local_tokenizer


def test_checkpoint_completed_lifecycle_is_resumable(tmp_path: Path) -> None:
    fingerprint = extraction_fingerprint(
        source_unit_hash="a" * 64,
        extractor_configuration="deterministic-v1",
        candidate_version="phase11-candidates-v1",
        prompt_hash="b" * 64,
        schema_hash="c" * 64,
        qwen_model="none",
        qwen_digest="none",
        tokenizer_revision="none",
        semantic_validation_policy="phase11-span-v1",
    )
    result_path = tmp_path / "result.json"
    result_path.write_text(
        '{"artifact_type":"phase11_extraction_result","lifecycle_state":"complete"}',
        encoding="utf-8",
    )
    checkpoint = ExtractionCheckpoint(
        checkpoint_id="u1",
        record_id="u1",
        result_path=result_path.as_posix(),
        fingerprint=fingerprint,
        source_unit_hash="a" * 64,
        extractor_configuration="deterministic-v1",
        candidate_version="phase11-candidates-v1",
        prompt_hash="b" * 64,
        schema_hash="c" * 64,
        qwen_model="none",
        qwen_digest="none",
        tokenizer_revision="none",
        semantic_validation_policy="phase11-span-v1",
        lifecycle_state="complete",
        context_prepared=True,
        extraction_attempted=True,
        extraction_completed=True,
        final_validation_completed=True,
    )
    store = ExtractionCheckpointStore(tmp_path / "checkpoints")
    store.write(checkpoint)
    assert store.valid("u1", fingerprint)
    assert store.status() == {"completed": 1, "incomplete": 0, "failed": 0, "corrupt": 0}


def test_incomplete_or_fingerprint_mismatch_is_not_reusable(tmp_path: Path) -> None:
    fingerprint = "d" * 64
    checkpoint = ExtractionCheckpoint(
        checkpoint_id="u2",
        record_id="u2",
        result_path=(tmp_path / "missing.json").as_posix(),
        fingerprint=fingerprint,
        source_unit_hash="a" * 64,
        extractor_configuration="hybrid-qwen-v1",
        candidate_version="phase11-candidates-v1",
        prompt_hash="b" * 64,
        schema_hash="c" * 64,
        qwen_model="qwen3:4b-instruct-2507-q4_K_M",
        qwen_digest=None,
        tokenizer_revision="cdbee75f17c01a7cc42f958dc650907174af0554",
        semantic_validation_policy="phase11-span-v1",
        lifecycle_state="incomplete",
    )
    store = ExtractionCheckpointStore(tmp_path / "checkpoints")
    store.write(checkpoint)
    assert not store.valid("u2", fingerprint)
    assert not store.valid("u2", "e" * 64)


def test_pinned_tokenizer_is_local_only_and_lazy() -> None:
    tokenizer = pinned_local_tokenizer()
    assert tokenizer.identity == "Qwen/Qwen3-4B-Instruct-2507"
    assert tokenizer.revision == "cdbee75f17c01a7cc42f958dc650907174af0554"
