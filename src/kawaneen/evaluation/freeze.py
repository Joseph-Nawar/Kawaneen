"""Immutable private v1 materialization after all review gates pass."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from kawaneen.evaluation.models import DatasetItem
from kawaneen.evaluation.serialization import write_items_jsonl


class FrozenMutationError(ValueError):
    pass


def _items_hash(items: tuple[DatasetItem, ...]) -> str:
    payload = [item.model_dump(mode="json") for item in items]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def freeze_items(
    items: tuple[DatasetItem, ...], *, private_root: Path, corpus_hash: str
) -> dict[str, object]:
    version = "phase6-retrieval-eval-v1"
    root = private_root / "frozen" / version
    manifest_path = root / "manifest.json"
    item_hash = _items_hash(items)
    if manifest_path.is_file():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior.get("item_set_hash") != item_hash or prior.get("corpus_hash") != corpus_hash:
            raise FrozenMutationError("immutable v1 manifest would be mutated")
        return prior
    root.mkdir(parents=True, exist_ok=True)
    write_items_jsonl(root / "items.jsonl", items)
    dev_ids = sorted(item.query_id for item in items if item.split.value == "dev")
    holdout_ids = sorted(item.query_id for item in items if item.split.value == "holdout")
    qrel_payload = [
        {
            "query_id": item.query_id,
            "qrels": [qrel.model_dump(mode="json") for qrel in item.chunk_qrels],
        }
        for item in items
    ]
    (root / "qrels.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in qrel_payload),
        encoding="utf-8",
    )
    (root / "dev_ids.json").write_text(json.dumps(dev_ids, sort_keys=True), encoding="utf-8")
    (root / "holdout_ids.json").write_text(
        json.dumps(holdout_ids, sort_keys=True), encoding="utf-8"
    )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "dataset_version": version,
        "corpus_hash": corpus_hash,
        "item_set_hash": item_hash,
        "dev_ids_hash": hashlib.sha256(",".join(dev_ids).encode()).hexdigest(),
        "holdout_ids_hash": hashlib.sha256(",".join(holdout_ids).encode()).hexdigest(),
        "qrels_hash": hashlib.sha256(
            "".join(json.dumps(record, sort_keys=True) for record in qrel_payload).encode()
        ).hexdigest(),
        "review_state_hash": hashlib.sha256(
            "".join(item.review.model_dump_json() for item in items).encode()
        ).hexdigest(),
        "policy_versions": {
            "content_policy": "phase5-source-content-policy-v1",
            "chunk_policy": "legal-structure-v1",
        },
        "item_count": len(items),
        "status": "frozen",
    }
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
