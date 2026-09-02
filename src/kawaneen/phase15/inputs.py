"""Read-only loaders for historical DEV artifacts used by Phase 15."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

HOLDOUT_MARKER = "holdout"
DEV_QUERY_RELATIVE_PATH = Path("phase6_evaluation/ai-reviewed-v1/draft/selected_and_variants.jsonl")
DEV_CHUNK_RELATIVE_PATH = Path("phase7_retrieval/corpus/chunks.jsonl")


def _assert_safe_relative(relative: Path) -> None:
    if relative.is_absolute() or HOLDOUT_MARKER in {part.lower() for part in relative.parts}:
        raise ValueError(
            "HOLDOUT paths are forbidden; Phase 15 input loaders accept DEV-only paths"
        )


@dataclass(frozen=True, slots=True)
class Phase15InputRoots:
    """Separate historical read root from the Phase 15 private output root."""

    historical_private_root: Path
    output_root: Path

    def private_path(self, relative: str | Path) -> Path:
        path = Path(relative)
        _assert_safe_relative(path)
        return self.historical_private_root / path

    def output_path(self, relative: str | Path) -> Path:
        path = Path(relative)
        _assert_safe_relative(path)
        return self.output_root / "artifacts/private/phase15_evaluation" / path


def _read_json(path: Path) -> dict[str, Any]:
    value: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return cast(dict[str, Any], value)


def load_dev_query_records(roots: Phase15InputRoots) -> tuple[dict[str, Any], ...]:
    """Load only records explicitly marked ``split=dev``."""

    path = roots.private_path(DEV_QUERY_RELATIVE_PATH)
    if not path.is_file():
        raise FileNotFoundError(f"missing required DEV query/qrel artifact: {path}")
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parsed: Any = json.loads(line)
        if not isinstance(parsed, dict):
            raise ValueError(f"invalid query record in {path}")
        record = cast(dict[str, Any], parsed)
        if str(record.get("split", "")).lower() == "dev":
            records.append(record)
    if len({str(record.get("query_id")) for record in records}) != len(records):
        raise ValueError("DEV query records must have unique query IDs")
    if not records:
        raise ValueError("required DEV query/qrel artifact contains no DEV records")
    return tuple(records)


def load_dev_chunks(roots: Phase15InputRoots) -> tuple[dict[str, Any], ...]:
    path = roots.private_path(DEV_CHUNK_RELATIVE_PATH)
    if not path.is_file():
        raise FileNotFoundError(f"missing required DEV chunk artifact: {path}")
    chunks: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parsed: Any = json.loads(line)
        if not isinstance(parsed, dict):
            raise ValueError(f"invalid chunk record in {path}")
        chunk = cast(dict[str, Any], parsed)
        chunks.append(chunk)
    if len({str(chunk.get("chunk_id")) for chunk in chunks}) != len(chunks):
        raise ValueError("DEV chunks must have unique chunk IDs")
    return tuple(chunks)


def load_dev_rankings(
    roots: Phase15InputRoots,
    relative: str | Path,
    expected_query_ids: tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    path = roots.private_path(relative)
    if not path.is_file():
        raise FileNotFoundError(f"missing required DEV ranking artifact: {path}")
    payload = _read_json(path)
    raw_rankings_value: Any = payload.get("rankings")
    if not isinstance(raw_rankings_value, dict):
        raise ValueError(f"ranking artifact has no rankings object: {path}")
    raw_rankings = cast(dict[str, Any], raw_rankings_value)
    expected = set(expected_query_ids)
    result: dict[str, tuple[str, ...]] = {}
    for query_id in expected_query_ids:
        raw = raw_rankings.get(query_id)
        if raw is None:
            raise ValueError(f"ranking artifact is missing DEV query {query_id}: {path}")
        if not isinstance(raw, list):
            raise ValueError(f"ranking for {query_id} is not a list: {path}")
        raw_items = cast(list[Any], raw)
        result[query_id] = tuple(str(item) for item in raw_items)
    unexpected = set(str(key) for key in raw_rankings) - expected
    if unexpected and any(key.lower().startswith("holdout") for key in unexpected):
        raise ValueError(f"ranking artifact contains HOLDOUT query IDs: {path}")
    return result
