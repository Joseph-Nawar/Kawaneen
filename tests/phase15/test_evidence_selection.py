from __future__ import annotations

import json
from pathlib import Path

import pytest

from kawaneen.phase15.contracts import (
    GeneratorSubsetManifest,
)
from kawaneen.phase15.evidence import (
    build_evidence_registry,
    verify_evidence_registry,
    write_json_atomic,
)
from kawaneen.phase15.selection import (
    ReviewCandidate,
    build_dialect_manifest,
    select_dialect_base_intents,
    select_generator_subset,
    select_review_cases,
)


def test_evidence_registry_hashes_tracked_historical_files_only(tmp_path: Path) -> None:
    for phase in (*range(3, 12), 14):
        path = tmp_path / "data" / "evaluation" / f"phase{phase}_frozen.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"phase": phase}), encoding="utf-8")
    private = tmp_path / "artifacts" / "private" / "phase15_evaluation" / "raw.json"
    private.parent.mkdir(parents=True, exist_ok=True)
    private.write_text("private", encoding="utf-8")

    registry = build_evidence_registry(tmp_path)
    assert {entry.phase for entry in registry.entries} == {str(p) for p in (*range(3, 12), 14)}
    assert all("private" not in entry.path for entry in registry.entries)
    write_json_atomic(tmp_path / "registry.json", registry.model_dump(mode="json"))
    assert verify_evidence_registry(tmp_path, tmp_path / "registry.json") is True

    (tmp_path / "data" / "evaluation" / "phase3_frozen.json").write_text(
        "changed", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_evidence_registry(tmp_path, tmp_path / "registry.json")


def test_deterministic_dev_selection_has_no_holdout() -> None:
    records = [
        {"id": f"q{i}", "split": "dev", "answerable": True, "gold_present_in_top8": i < 31}
        for i in range(61)
    ] + [
        {"id": f"u{i}", "split": "dev", "answerable": False, "gold_present_in_top8": False}
        for i in range(19)
    ]
    records.append({"id": "holdout", "split": "holdout", "answerable": True})

    subset = select_generator_subset(records, seed=20260826)
    assert isinstance(subset, GeneratorSubsetManifest)
    assert len(set(subset.answerable_gold_present_ids)) == 31
    assert "holdout" not in subset.model_dump_json()

    dialect = select_dialect_base_intents(records, seed=20260826)
    assert len(dialect) == 20
    assert all(item != "holdout" for item in dialect)


def test_review_selection_is_exact_and_rejects_holdout() -> None:
    candidates = [
        ReviewCandidate(
            case_id=f"case-{i}",
            language="ar" if i % 2 else "en",
            pipeline_stage="retrieval",
            legal_category="civil",
            answerability="answerable",
            severity="high" if i % 3 == 0 else "medium",
        )
        for i in range(130)
    ]
    packet = select_review_cases(candidates, seed=20260826)
    assert len(packet) == 120
    assert len({case.case_id for case in packet}) == 120

    with pytest.raises(ValueError, match="HOLDOUT"):
        select_review_cases([candidates[0].model_copy(update={"holdout": True})], seed=1)
    with pytest.raises(ValueError, match="exactly 120"):
        select_review_cases(candidates, count=1)
    with pytest.raises(ValueError, match="at least 120"):
        select_review_cases(candidates[:1])


def test_dialect_manifest_and_selection_requirements() -> None:
    with pytest.raises(ValueError, match="at least 20"):
        select_dialect_base_intents([{"id": "one", "split": "dev"}])
    with pytest.raises(ValueError, match="Egyptian"):
        build_dialect_manifest([], {"egyptian": [], "gulf": [], "levantine": []})
    base_ids = [f"base-{i}" for i in range(20)]
    manifest = build_dialect_manifest(
        base_ids,
        {
            "egyptian": [f"eg-{i}" for i in range(20)],
            "gulf_saudi": [f"gulf-{i}" for i in range(20)],
            "levantine": [f"lev-{i}" for i in range(20)],
        },
        {"eg-0": "hash"},
    )
    assert manifest.dialect_counts == {"egyptian": 20, "gulf_saudi": 20, "levantine": 20}
