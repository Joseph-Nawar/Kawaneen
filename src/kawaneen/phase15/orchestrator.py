"""Phase 15 command orchestration with explicit DEV-only gates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .contracts import PHASE15_BASE_SHA
from .evidence import (
    build_evidence_registry,
    build_experiment_plan,
    verify_evidence_registry,
    write_json_atomic,
)

TRACKED_MANIFEST_ROOT = Path("data/manifests/evaluation")
TRACKED_EVALUATION_ROOT = Path("data/evaluation")
PRIVATE_ROOT = Path("artifacts/private/phase15_evaluation")
FORBIDDEN_FINAL_ARTIFACTS = (
    TRACKED_EVALUATION_ROOT / "phase15_error_analysis.json",
    TRACKED_EVALUATION_ROOT / "phase15_research_questions.json",
    Path("docs/reports/phase-15-evaluation-and-experiment-report.md"),
)


def _path(root: Path, relative: Path) -> Path:
    return root / relative


def _write_frozen(root: Path, relative: Path, payload: dict[str, Any]) -> None:
    destination = _path(root, relative)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != encoded:
            raise ValueError(f"frozen Phase 15 artifact already exists and differs: {relative}")
        return
    write_json_atomic(destination, payload)


def phase15_plan(root: Path = Path(".")) -> dict[str, Any]:
    plan = build_experiment_plan()
    payload = plan.model_dump(mode="json")
    _write_frozen(root, TRACKED_MANIFEST_ROOT / "phase15_experiment_plan.json", payload)
    return payload


def phase15_freeze(root: Path = Path(".")) -> dict[str, Any]:
    """Freeze plan and historical evidence before any new DEV scoring."""

    plan = phase15_plan(root)
    registry_path = _path(root, TRACKED_MANIFEST_ROOT / "phase15_evidence_registry.json")
    if registry_path.exists():
        verify_evidence_registry(root, registry_path)
        registry_payload = json.loads(registry_path.read_text(encoding="utf-8"))
    else:
        registry = build_evidence_registry(root)
        registry_payload = registry.model_dump(mode="json")
        _write_frozen(root, TRACKED_MANIFEST_ROOT / "phase15_evidence_registry.json", registry_payload)
    return {
        "status": "frozen",
        "base_sha": plan["base_sha"],
        "seed": plan["seed"],
        "bootstrap_replicates": plan["bootstrap_replicates"],
        "registry_entries": len(registry_payload["entries"]),
        "protected_final_artifacts_absent": all(not _path(root, item).exists() for item in FORBIDDEN_FINAL_ARTIFACTS),
        "private_root": _path(root, PRIVATE_ROOT).as_posix(),
        "holdout_runs_permitted": False,
    }


def phase15_synthesize(root: Path = Path(".")) -> dict[str, Any]:
    """Verify frozen historical inputs; never create final report artifacts."""

    registry_path = _path(root, TRACKED_MANIFEST_ROOT / "phase15_evidence_registry.json")
    verify_evidence_registry(root, registry_path)
    return {
        "status": "historical evidence verified",
        "provenance": "HISTORICAL_FROZEN",
        "registry": registry_path.as_posix(),
        "final_report_created": False,
    }


def assert_no_protected_artifacts(root: Path = Path(".")) -> None:
    for relative in FORBIDDEN_FINAL_ARTIFACTS:
        if _path(root, relative).exists():
            raise ValueError(f"human-gated final artifact exists too early: {relative}")
