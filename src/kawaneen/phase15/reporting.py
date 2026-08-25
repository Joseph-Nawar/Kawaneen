"""Aggregate-only Phase 15 artifact serialization."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .evidence import write_json_atomic

SENSITIVE_KEYS = {
    "query_text",
    "source_text",
    "evidence_text",
    "snippet",
    "raw_output",
    "response_text",
    "dialect_text",
    "context_text",
    "prompt_text",
}


def assert_text_free(value: Any, *, key: str | None = None) -> None:
    if key in SENSITIVE_KEYS:
        raise ValueError(f"tracked Phase 15 artifacts cannot contain private text field: {key}")
    if isinstance(value, Mapping):
        for child_key, child_value in value.items():
            assert_text_free(child_value, key=str(child_key))
    elif isinstance(value, (list, tuple)):
        for child in value:
            assert_text_free(child)


def write_aggregate_artifact(root: Path, filename: str, payload: Mapping[str, Any]) -> Path:
    if "/" in filename or "\\" in filename or not filename.endswith(".json"):
        raise ValueError("aggregate artifacts must be one JSON file name")
    assert_text_free(payload)
    destination = root / "data/evaluation" / filename
    write_json_atomic(destination, payload)
    return destination


def metric_status_artifact(
    *, status: str, provenance: str = "PHASE15_DEV", reason: str | None = None
) -> dict[str, object]:
    payload: dict[str, object] = {"status": status, "provenance": provenance}
    if reason is not None:
        payload["reason"] = reason
    return payload
