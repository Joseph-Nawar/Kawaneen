"""Deterministic JSON and flattened CSV downloads for extraction results."""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Sequence
from typing import TypeAlias

from kawaneen.api.contracts import ExtractionResponse

ExtractionItem: TypeAlias = tuple[str, ExtractionResponse]


def extraction_json(items: Sequence[ExtractionItem]) -> bytes:
    payload = [
        {"segment_id": segment_id, "response": response.model_dump(mode="json")}
        for segment_id, response in items
    ]
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")


def extraction_csv(items: Sequence[ExtractionItem]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=("segment_id", "field", "value"))
    writer.writeheader()
    for segment_id, response in items:
        result = response.result
        for field in ("obligations", "prohibitions", "permissions", "deadlines", "exceptions", "regulated_entities"):
            value = getattr(result, field)
            writer.writerow({"segment_id": segment_id, "field": field, "value": json.dumps(value, ensure_ascii=False, default=str)})
    return output.getvalue().encode("utf-8")
