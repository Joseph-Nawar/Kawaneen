"""Phase 15 command orchestration with explicit DEV-only gates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .contracts import ALLAM_MODEL, ModelLock, ReviewCase
from .embedding import create_arabic_model_lock
from .evidence import (
    build_evidence_registry,
    build_experiment_plan,
    verify_evidence_registry,
    write_json_atomic,
)
from .reporting import metric_status_artifact, write_aggregate_artifact
from .review import ReviewStore, default_review_paths, prepare_review_packet

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
    model_lock = phase15_model_lock(root)
    registry_path = _path(root, TRACKED_MANIFEST_ROOT / "phase15_evidence_registry.json")
    if registry_path.exists():
        verify_evidence_registry(root, registry_path)
        registry_payload = json.loads(registry_path.read_text(encoding="utf-8"))
    else:
        registry = build_evidence_registry(root)
        registry_payload = registry.model_dump(mode="json")
        _write_frozen(
            root, TRACKED_MANIFEST_ROOT / "phase15_evidence_registry.json", registry_payload
        )
    return {
        "status": "frozen",
        "base_sha": plan["base_sha"],
        "seed": plan["seed"],
        "bootstrap_replicates": plan["bootstrap_replicates"],
        "registry_entries": len(registry_payload["entries"]),
        "model_lock": model_lock["path"],
        "protected_final_artifacts_absent": all(
            not _path(root, item).exists() for item in FORBIDDEN_FINAL_ARTIFACTS
        ),
        "private_root": _path(root, PRIVATE_ROOT).as_posix(),
        "holdout_runs_permitted": False,
    }


def phase15_model_lock(root: Path = Path(".")) -> dict[str, Any]:
    """Write immutable public revisions and the ALLaM preflight stop state."""

    arabic = create_arabic_model_lock("899f6e1b765915a72d5e4ace6bb2b221715550d8")
    allam_revision = "a28dd1e67420cde72d3629c8633a974cf7d9c366"
    fallback = ModelLock(
        model_id="hammh0a/Hala-1.2B",
        revision="10e586c0899f9b97c5764e6520ccd7c199ae0e60",
        dtype="bf16",
        batch_size=1,
        runtime="transformers",
        device="cpu-or-mps",
    )
    payload: dict[str, Any] = {
        "schema_version": "phase15-model-lock-v1",
        "provenance": "PHASE15_DEV",
        "arabic_embedding": arabic.model_dump(mode="json"),
        "allam": {
            "model_id": ALLAM_MODEL,
            "revision": allam_revision,
            "status": "BLOCKED_BEFORE_SCORING_NO_TRUSTWORTHY_4BIT_LOCAL_ARTIFACT",
            "full_precision_forbidden": True,
            "quantization_bits": 4,
            "artifact_sha256": None,
            "runtime": None,
            "device": "M5-16GB",
            "context_limit": 4096,
            "output_limit": 256,
            "disk_footprint_bytes": None,
            "bounded_preflight": "not_run",
        },
        "fallback_preregistered_before_results": fallback.model_dump(mode="json"),
        "no_model_shopping_after_dev_results": True,
    }
    relative = TRACKED_MANIFEST_ROOT / "phase15_model_lock.json"
    _write_frozen(root, relative, payload)
    return {"path": _path(root, relative).as_posix(), "allam_status": payload["allam"]["status"]}


def phase15_review_prepare(root: Path = Path(".")) -> dict[str, Any]:
    packet_path, progress_path, manifest_path = default_review_paths(root)
    if packet_path.exists():
        return {
            "status": "already prepared",
            "manifest": manifest_path.as_posix(),
            "packet": packet_path.as_posix(),
        }
    candidate_path = root / PRIVATE_ROOT / "review_candidates.json"
    if not candidate_path.is_file():
        raise RuntimeError(
            "cannot prepare review packet: private DEV review_candidates.json is missing; "
            "no cases are fabricated"
        )
    payload = json.loads(candidate_path.read_text(encoding="utf-8"))
    cases = tuple(ReviewCase.model_validate(item) for item in payload.get("cases", ()))
    manifest = prepare_review_packet(cases, packet_path, manifest_path)
    return {
        "status": "prepared",
        "manifest": manifest_path.as_posix(),
        "packet": packet_path.as_posix(),
        "progress": progress_path.as_posix(),
        "case_count": manifest["case_count"],
    }


def phase15_review_status(root: Path = Path(".")) -> dict[str, Any]:
    packet_path, progress_path, manifest_path = default_review_paths(root)
    if not packet_path.is_file():
        return {
            "packet_present": False,
            "reviewed": 0,
            "total": 120,
            "progress": "0 / 120",
            "packet_path": packet_path.as_posix(),
            "progress_path": progress_path.as_posix(),
            "manifest_path": manifest_path.as_posix(),
        }
    return ReviewStore(packet_path, progress_path).status()


def phase15_finalize(root: Path = Path(".")) -> dict[str, Any]:
    packet_path, progress_path, _manifest_path = default_review_paths(root)
    if not packet_path.is_file():
        raise RuntimeError("phase15 finalize requires the private 120-case review packet")
    store = ReviewStore(packet_path, progress_path)
    store.require_finalize_ready()
    raise RuntimeError("phase15 final report is intentionally disabled at the Phase 15 human gate")


def phase15_unavailable_experiment(root: Path, experiment: str) -> dict[str, Any]:
    expected = root / PRIVATE_ROOT / "inputs" / f"{experiment}.json"
    raise RuntimeError(
        f"{experiment} is DEV-only and requires a prepared private input at {expected}; "
        "no protected HOLDOUT access or synthetic result is permitted"
    )


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


def write_phase15_status_artifacts(root: Path = Path(".")) -> tuple[str, ...]:
    """Record honest aggregate gates when private experiment inputs are unavailable."""

    reason = (
        "No new Phase 15 DEV scoring was executed before the mandatory human-review stop; "
        "this status is not a metric result."
    )
    filenames = (
        "phase15_dialect_manifest.json",
        "phase15_generator_subset_manifest.json",
        "phase15_embedding_metrics.json",
        "phase15_dialect_metrics.json",
        "phase15_reranking_metrics.json",
        "phase15_generator_metrics.json",
        "phase15_citation_counterfactual.json",
        "phase15_abstention_sensitivity.json",
        "phase15_latency_metrics.json",
    )
    payload = metric_status_artifact(status="NOT_RUN", reason=reason)
    manifest_filenames = {"phase15_dialect_manifest.json", "phase15_generator_subset_manifest.json"}
    paths: list[str] = []
    for filename in filenames:
        if filename in manifest_filenames:
            relative = TRACKED_MANIFEST_ROOT / filename
            _write_frozen(root, relative, payload)
            paths.append(_path(root, relative).as_posix())
        else:
            paths.append(write_aggregate_artifact(root, filename, payload).as_posix())
    return tuple(paths)


def assert_no_protected_artifacts(root: Path = Path(".")) -> None:
    for relative in FORBIDDEN_FINAL_ARTIFACTS:
        if _path(root, relative).exists():
            raise ValueError(f"human-gated final artifact exists too early: {relative}")
