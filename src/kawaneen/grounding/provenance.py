"""Canonical corpus resolution for Phase-9 grounding."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

from kawaneen.corpus.models import CanonicalUnit
from kawaneen.grounding.contracts import (
    CanonicalEvidenceUnit,
    CanonicalSourceSpan,
    ResolvedChunk,
    SourceRecord,
)


@dataclass(frozen=True, slots=True)
class _ChunkRecord:
    source_unit_ids: tuple[str, ...]
    source_spans: tuple[CanonicalSourceSpan, ...]


def _json_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return cast(dict[str, object], value)


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not an object")
    return cast(dict[str, object], value)


@dataclass(frozen=True, slots=True)
class CanonicalCorpusResolver:
    """Resolve chunks to exact canonical unit text and canonical metadata."""

    units_by_id: Mapping[str, CanonicalEvidenceUnit]
    chunks_by_id: Mapping[str, _ChunkRecord]
    document_sources_by_id: Mapping[str, SourceRecord]
    corpus_hash: str | None = None
    chunk_policy_hash: str | None = None

    @classmethod
    def from_json(
        cls,
        canonical_units_path: Path,
        chunks_path: Path,
        corpus_manifest_path: Path | None = None,
        document_paths: tuple[Path, ...] = (),
    ) -> CanonicalCorpusResolver:
        snapshot = _json_object(canonical_units_path)
        raw_units_value = snapshot.get("units")
        if not isinstance(raw_units_value, list):
            raise ValueError("canonical corpus has no units")
        raw_units = cast(list[object], raw_units_value)
        document_sources = _load_document_sources(document_paths)
        units: dict[str, CanonicalEvidenceUnit] = {}
        for raw_value in raw_units:
            raw = _mapping(raw_value, "canonical unit")
            canonical_payload = dict(raw)
            canonical_payload.pop("heading_path", None)
            unit = CanonicalUnit.model_validate(canonical_payload)
            if unit.unit_id in units:
                raise ValueError(f"duplicate canonical unit: {unit.unit_id}")
            units[unit.unit_id] = CanonicalEvidenceUnit(
                unit_id=unit.unit_id,
                document_id=unit.document_id,
                ordinal=unit.ordinal,
                display_text=unit.text,
                heading_path=tuple(
                    str(value)
                    for value in cast(list[object], raw.get("heading_path", []))
                ),
                source=document_sources.get(
                    unit.document_id,
                    SourceRecord(
                        document_id=unit.document_id,
                        source_id=unit.provenance.source_id,
                    ),
                ),
            )

        chunks: dict[str, _ChunkRecord] = {}
        chunk_policy_hashes: set[str] = set()
        for line in chunks_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            raw = _json_line(line)
            chunk_id = str(raw.get("chunk_id", ""))
            if not chunk_id:
                raise ValueError("retrieval chunk has no chunk_id")
            if chunk_id in chunks:
                raise ValueError(f"duplicate retrieval chunk: {chunk_id}")
            raw_unit_ids_value = raw.get("source_unit_ids", [])
            if not isinstance(raw_unit_ids_value, list):
                raise ValueError(f"retrieval chunk source units are invalid: {chunk_id}")
            raw_unit_ids = cast(list[object], raw_unit_ids_value)
            source_unit_ids = tuple(str(value) for value in raw_unit_ids)
            if not source_unit_ids:
                raise ValueError(f"retrieval chunk has no source units: {chunk_id}")
            raw_spans_value = raw.get("source_spans", [])
            if not isinstance(raw_spans_value, list):
                raise ValueError(f"retrieval chunk source spans are invalid: {chunk_id}")
            raw_spans = cast(list[object], raw_spans_value)
            chunks[chunk_id] = _ChunkRecord(
                source_unit_ids=source_unit_ids,
                source_spans=tuple(CanonicalSourceSpan.model_validate(span) for span in raw_spans),
            )
            policy_hash = raw.get("chunk_policy_hash")
            if isinstance(policy_hash, str) and policy_hash:
                chunk_policy_hashes.add(policy_hash)

        corpus_hash = None
        if corpus_manifest_path is not None:
            manifest = _json_object(corpus_manifest_path)
            value = manifest.get("corpus_hash")
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError("canonical corpus manifest has no valid corpus hash")
            corpus_hash = value
        else:
            summary_value = snapshot.get("summary", {})
            if isinstance(summary_value, dict):
                summary = cast(dict[str, object], summary_value)
                if isinstance(summary.get("corpus_hash"), str):
                    corpus_hash = str(summary["corpus_hash"])
        if len(chunk_policy_hashes) > 1:
            raise ValueError("canonical chunk inputs contain multiple chunk-policy hashes")
        return cls(
            MappingProxyType(units),
            MappingProxyType(chunks),
            MappingProxyType(document_sources),
            corpus_hash,
            next(iter(chunk_policy_hashes), None),
        )

    def resolve_chunk(self, chunk_id: str) -> ResolvedChunk:
        raw = self.chunks_by_id.get(chunk_id)
        if raw is None:
            raise ValueError(f"unknown chunk: {chunk_id}")
        unit_ids = raw.source_unit_ids
        missing = [unit_id for unit_id in unit_ids if unit_id not in self.units_by_id]
        if missing:
            raise ValueError(f"unknown canonical unit for chunk {chunk_id}: {missing[0]}")
        units = tuple(self.units_by_id[unit_id] for unit_id in unit_ids)
        document_ids = {unit.document_id for unit in units}
        if len(document_ids) != 1:
            raise ValueError(f"chunk spans multiple canonical documents: {chunk_id}")
        ordered_units = tuple(
            sorted(units, key=lambda unit: (unit.ordinal or 0, unit.unit_id))
        )
        return ResolvedChunk(
            chunk_id=chunk_id,
            document_id=next(iter(document_ids)),
            source_unit_ids=unit_ids,
            source_spans=raw.source_spans,
            units=ordered_units,
        )


def _json_line(line: str) -> dict[str, object]:
    value = json.loads(line)
    if not isinstance(value, dict):
        raise ValueError("retrieval chunk row must be an object")
    return cast(dict[str, object], value)


def _load_document_sources(paths: tuple[Path, ...]) -> dict[str, SourceRecord]:
    """Load only authoritative document fields from canonical Parquet records."""

    if not paths:
        return {}
    try:
        import pyarrow.parquet as parquet_module
    except ImportError as error:  # pragma: no cover - project dependency is required
        raise ValueError("canonical document metadata requires pyarrow") from error

    parquet: Any = cast(Any, parquet_module)
    sources: dict[str, SourceRecord] = {}
    for path in paths:
        table: Any = parquet.read_table(path)
        for raw_value in cast(list[dict[str, Any]], table.to_pylist()):
            document_id = _optional_text(raw_value.get("document_id"))
            if document_id is None:
                raise ValueError(f"canonical document record has no document_id: {path}")
            if document_id in sources:
                raise ValueError(f"duplicate canonical document: {document_id}")
            source_metadata = _source_metadata(raw_value.get("source_metadata_json"))
            sources[document_id] = SourceRecord(
                document_id=document_id,
                source_id=_optional_text(raw_value.get("source_id")),
                document_title=_optional_text(raw_value.get("title")),
                article=_optional_text(raw_value.get("raw_article_label")),
                source_url=_optional_text(source_metadata.get("details_url")),
            )
    return sources


def _source_metadata(value: object) -> dict[str, object]:
    if not isinstance(value, str) or not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return (
        cast(dict[str, object], parsed)
        if isinstance(parsed, dict)
        else {}
    )


def _optional_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None
