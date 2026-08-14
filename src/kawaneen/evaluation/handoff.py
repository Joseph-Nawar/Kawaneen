"""Private, lossless canonical source-text handoff artifacts."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from kawaneen.corpus.models import CanonicalUnit
from kawaneen.evaluation.corpus import EvaluationCorpus
from kawaneen.evaluation.models import DatasetItem

_SHARD_LIMIT = 23 * 1024 * 1024


def _row(unit: CanonicalUnit) -> dict[str, object]:
    return {
        "source_id": unit.provenance.source_id,
        "source_version": unit.provenance.source_version,
        "document_id": unit.document_id,
        "unit_id": unit.unit_id,
        "unit_type": unit.unit_type.value,
        "display_text": unit.text,
    }


def _write_shards(rows: tuple[dict[str, object], ...], root: Path) -> list[dict[str, object]]:
    root.mkdir(parents=True, exist_ok=True)
    shards: list[dict[str, object]] = []
    current: list[str] = []
    current_size = 0
    shard_index = 1
    for row in rows:
        line = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        encoded_size = len(line.encode("utf-8"))
        if current and current_size + encoded_size > _SHARD_LIMIT:
            name = f"canonical_review_{shard_index:04d}.jsonl"
            path = root / name
            path.write_text("".join(current), encoding="utf-8")
            shards.append(
                {
                    "name": name,
                    "row_count": len(current),
                    "size_bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
            shard_index += 1
            current = []
            current_size = 0
        current.append(line)
        current_size += encoded_size
    if current:
        name = f"canonical_review_{shard_index:04d}.jsonl"
        path = root / name
        path.write_text("".join(current), encoding="utf-8")
        shards.append(
            {
                "name": name,
                "row_count": len(current),
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return shards


def write_handoff_artifacts(
    corpus: EvaluationCorpus, items: tuple[DatasetItem, ...], root: Path
) -> dict[str, object]:
    all_rows = tuple(
        _row(unit)
        for unit in sorted(
            corpus.units,
            key=lambda unit: (unit.provenance.source_id, unit.document_id, unit.unit_id),
        )
    )
    shard_root = root / "canonical_review_shards"
    shards = _write_shards(all_rows, shard_root)
    referenced_docs = {document_id for item in items for document_id in item.source_document_ids}
    context_rows = tuple(row for row in all_rows if row["document_id"] in referenced_docs)
    context_path = root / "phase6_review_source_context.jsonl"
    context_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in context_rows),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "corpus_hash": corpus.corpus_hash,
        "included_fields": [
            "source_id",
            "source_version",
            "document_id",
            "unit_id",
            "unit_type",
            "display_text",
        ],
        "row_count": len(all_rows),
        "source_counts": dict(sorted(Counter(str(row["source_id"]) for row in all_rows).items())),
        "shards": shards,
        "context": {
            "path": context_path.as_posix(),
            "row_count": len(context_rows),
            "size_bytes": context_path.stat().st_size,
            "sha256": hashlib.sha256(context_path.read_bytes()).hexdigest(),
        },
    }
    manifest_path = root / "canonical_review_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return {"manifest": manifest_path.as_posix(), "shards": shards, "context": manifest["context"]}
