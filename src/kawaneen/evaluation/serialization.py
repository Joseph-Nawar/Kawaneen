"""Private JSONL serialization for evaluation records and source snapshots."""

from __future__ import annotations

import json
from pathlib import Path

from kawaneen.evaluation.corpus import EvaluationCorpus, corpus_summary
from kawaneen.evaluation.models import DatasetItem


def write_items_jsonl(path: Path, items: tuple[DatasetItem, ...]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(item.model_dump_json() + "\n" for item in items), encoding="utf-8")
    return path


def read_items_jsonl(path: Path) -> tuple[DatasetItem, ...]:
    return tuple(
        DatasetItem.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def write_private_corpus_snapshot(corpus: EvaluationCorpus, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": corpus_summary(corpus),
        "units": [unit.model_dump(mode="json") for unit in corpus.units],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return path
