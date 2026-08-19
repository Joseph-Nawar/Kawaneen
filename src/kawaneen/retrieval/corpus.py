# pyright: basic
"""Read-only Phase 5/6 release and retrieval corpus construction."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from kawaneen.chunking.models import SourceSpan, deterministic_chunk_id
from kawaneen.chunking.policies import get_chunk_policy
from kawaneen.chunking.structure import split_exact_spans
from kawaneen.corpus.models import CanonicalUnit
from kawaneen.evaluation.serialization import read_items_jsonl
from kawaneen.normalization.policies import get_policy, normalize_text
from kawaneen.retrieval.manifests import build_corpus_manifest, stable_hash
from kawaneen.retrieval.models import RetrievalChunk, RetrievalRelease

PHASE6_MANIFEST = Path("data/manifests/evaluation/phase6_ai_reviewed_v1_manifest.json")
PHASE6_ROOT = Path("artifacts/private/phase6_evaluation/ai-reviewed-v1")
PHASE5_CHUNKS = Path("artifacts/private/phase5_chunking/chunks/legal-structure-v1/chunks.jsonl")
PHASE7_PRIVATE_CHUNKS = Path("artifacts/private/phase7_retrieval/corpus/chunks.jsonl")


def validate_qrel_chunks(
    qrels: Mapping[str, tuple[str, ...]], chunks: tuple[RetrievalChunk, ...]
) -> None:
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("duplicate chunk IDs in retrieval corpus")
    known = set(chunk_ids)
    missing = sorted(
        {chunk_id for ids in qrels.values() for chunk_id in ids if chunk_id not in known}
    )
    if missing:
        raise ValueError(f"qrel chunk IDs outside retrieval corpus: {missing[:3]}")


def _load_chunks(
    path: Path, unit_metadata: Mapping[str, Mapping[str, object]]
) -> tuple[RetrievalChunk, ...]:
    chunks: list[RetrievalChunk] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        unit_ids = tuple(str(value) for value in raw["source_unit_ids"])
        metadata = [unit_metadata[unit_id] for unit_id in unit_ids] if unit_metadata else []
        document_ids = (
            {str(item["document_id"]) for item in metadata}
            if metadata
            else {str(raw["document_id"])}
        )
        if len(document_ids) != 1:
            raise ValueError(f"chunk spans multiple documents: {raw['chunk_id']}")
        chunks.append(
            RetrievalChunk(
                chunk_id=str(raw["chunk_id"]),
                document_id=next(iter(document_ids)),
                source_id=str(raw.get("source_id", raw["provenance"]["source_id"])),
                unit_type=str(raw.get("unit_type", raw["provenance"]["source_field"])).lower(),
                display_text=str(raw["display_text"]),
                search_text=str(raw["search_text"]),
                source_unit_ids=unit_ids,
                chunk_policy_hash=str(raw["chunk_policy_hash"]),
                normalization_policy_id=str(raw["normalization_policy_id"]),
                normalization_policy_hash=str(raw["normalization_policy_hash"]),
                token_count=int(raw["token_count"]),
                source_spans=tuple(
                    (int(span["start"]), int(span["end"])) for span in raw["source_spans"]
                ),
            )
        )
    return tuple(chunks)


def _build_complete_phase5_chunks(
    snapshot: Mapping[str, object],
) -> tuple[RetrievalChunk, ...]:
    raw_units = snapshot["units"]
    if not isinstance(raw_units, list):
        raise ValueError("Phase 6 corpus snapshot has no unit list")
    units = tuple(CanonicalUnit.model_validate(item) for item in raw_units)
    legal_policy = get_chunk_policy("legal-structure-v1")
    normalization_policy = get_policy("arabic-light-v1")
    chunks: list[RetrievalChunk] = []
    for unit in units:
        for span in split_exact_spans(unit.text, unit.unit_id):
            display = unit.text[span.start : span.end]
            normalized = normalization_policy
            search_text = unit.text[span.start : span.end]
            normalized_text = normalize_text(search_text, normalized)
            if not isinstance(normalized_text, str):
                raise TypeError("chunk normalization unexpectedly returned an audit result")
            chunks.append(
                RetrievalChunk(
                    chunk_id=deterministic_chunk_id(
                        "legal-structure-v1", unit.unit_id, (span,), legal_policy.policy_hash
                    ),
                    document_id=unit.document_id,
                    source_id=unit.provenance.source_id,
                    unit_type=unit.unit_type.value,
                    display_text=display,
                    search_text=normalized_text,
                    source_unit_ids=(unit.unit_id,),
                    chunk_policy_hash=legal_policy.policy_hash,
                    normalization_policy_id=normalized.policy_id,
                    normalization_policy_hash=normalized.policy_hash,
                    token_count=len(display.split()),
                    source_spans=((span.start, span.end),),
                )
            )
    return tuple(chunks)


def _rekey_for_frozen_phase6_qrels(
    chunks: tuple[RetrievalChunk, ...],
) -> tuple[RetrievalChunk, ...]:
    """Preserve Phase-6 qrel IDs when that release predates the policy-hash ID component."""

    return tuple(
        RetrievalChunk(
            chunk_id=deterministic_chunk_id(
                "legal-structure-v1",
                chunk.source_unit_ids[0],
                tuple(
                    SourceSpan(unit_id, start, end)
                    for unit_id, (start, end) in zip(
                        chunk.source_unit_ids, chunk.source_spans, strict=True
                    )
                ),
                "",
            ),
            document_id=chunk.document_id,
            source_id=chunk.source_id,
            unit_type=chunk.unit_type,
            display_text=chunk.display_text,
            search_text=chunk.search_text,
            source_unit_ids=chunk.source_unit_ids,
            chunk_policy_hash=chunk.chunk_policy_hash,
            normalization_policy_id=chunk.normalization_policy_id,
            normalization_policy_hash=chunk.normalization_policy_hash,
            token_count=chunk.token_count,
            source_spans=chunk.source_spans,
        )
        for chunk in chunks
    )


def load_phase7_release(
    phase6_root: Path = PHASE6_ROOT,
    *,
    phase6_manifest_path: Path = PHASE6_MANIFEST,
    chunks_path: Path = PHASE5_CHUNKS,
    allow_holdout: bool = False,
) -> RetrievalRelease:
    manifest = json.loads(phase6_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_version") != "phase6-retrieval-eval-ai-reviewed-v1":
        raise ValueError("unexpected Phase 6 release version")
    items = read_items_jsonl(phase6_root / "draft" / "selected_and_variants.jsonl")
    item_hash = stable_hash([item.model_dump(mode="json") for item in items])
    if item_hash != manifest["hashes"]["item_set"]:
        raise ValueError("Phase 6 item-set hash mismatch")
    if not allow_holdout:
        # Access is intentionally checked even though the release loader retains both splits.
        _ = tuple(item for item in items if item.split.value == "dev")
    snapshot = json.loads(
        (phase6_root / "corpus" / "canonical_units.json").read_text(encoding="utf-8")
    )
    unit_metadata = {str(item["unit_id"]): item for item in snapshot["units"]}
    effective_chunks_path = (
        PHASE7_PRIVATE_CHUNKS
        if chunks_path == PHASE5_CHUNKS and PHASE7_PRIVATE_CHUNKS.is_file()
        else chunks_path
    )
    chunks = _load_chunks(
        effective_chunks_path, unit_metadata if effective_chunks_path == chunks_path else {}
    )
    chunk_policy = get_chunk_policy("legal-structure-v1")
    if any(chunk.chunk_policy_hash != chunk_policy.policy_hash for chunk in chunks):
        raise ValueError("retrieval corpus contains a different chunk policy")
    qrels = {
        item.query_id: tuple(qrel.chunk_id for qrel in item.chunk_qrels)
        for item in items
        if item.answerability.value == "answerable"
    }
    chunk_id_contract = "phase5-policy-hash"
    try:
        validate_qrel_chunks(qrels, chunks)
    except ValueError as exc:
        if "outside retrieval corpus" not in str(exc):
            raise
        complete_chunks = _build_complete_phase5_chunks(snapshot)
        try:
            validate_qrel_chunks(qrels, complete_chunks)
            chunks = complete_chunks
        except ValueError:
            compatibility_chunks = _rekey_for_frozen_phase6_qrels(complete_chunks)
            validate_qrel_chunks(qrels, compatibility_chunks)
            chunks = compatibility_chunks
            chunk_id_contract = "phase6-qrel-empty-policy-hash-compatibility"
    corpus_manifest = build_corpus_manifest(
        chunks,
        corpus_hash=str(manifest["corpus_hash"]),
        release_hash=str(manifest["hashes"]["item_set"]),
    )
    corpus_manifest["chunk_id_contract"] = chunk_id_contract
    return RetrievalRelease(
        items=items,
        chunks=chunks,
        phase6_manifest=manifest,
        corpus_manifest=corpus_manifest,
    )
