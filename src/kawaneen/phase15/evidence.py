"""Hash-anchored historical evidence and atomic aggregate artifact writes."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any

from .contracts import (
    PHASE15_BASE_SHA,
    ArtifactHash,
    EvidenceRegistry,
    ExperimentPlan,
)

RESEARCH_QUESTIONS = (
    "Does structure-aware chunking outperform fixed chunks for Arabic legal retrieval?",
    "How much does light Arabic normalization affect lexical and dense retrieval?",
    "Does hybrid retrieval outperform dense-only retrieval across Arabic and English queries?",
    "How much does reranking improve hard legal queries?",
    "How robust is retrieval to dialectal paraphrasing?",
    "Can citation verification meaningfully reduce unsupported answers?",
    "How accurately can a zero-cost local model answer from Arabic legal evidence?",
)

HARD_PROHIBITIONS = (
    "No protected Phase 3, Phase 8, or Phase 11 HOLDOUT reruns or access for tuning.",
    "Historical HOLDOUT metrics may be cited only from already frozen tracked artifacts.",
    "Do not mutate frozen Phase 3-14 result artifacts.",
    "Do not change production retrieval, chunking, generator, API, UI, or Stage-D policy.",
    "No ALLaM or threshold production promotion.",
    "No model shopping after DEV outcomes are visible.",
    "Phase 6 AI-reviewed data is not human gold or expert gold.",
    "No external upload of private query or source text.",
    "Tracked new outputs must be aggregate and text-free.",
    "Negative and inconclusive findings remain in the report.",
)


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON with replace semantics so interrupted progress cannot corrupt it."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_experiment_plan(base_sha: str = PHASE15_BASE_SHA) -> ExperimentPlan:
    return ExperimentPlan(
        base_sha=base_sha,
        research_questions=RESEARCH_QUESTIONS,
        hard_prohibitions=HARD_PROHIBITIONS,
    )


def _historical_paths(root: Path) -> tuple[Path, ...]:
    paths: set[Path] = set()
    for phase in (*range(3, 12), 14):
        patterns = (f"phase{phase}_*.json", f"phase{phase}-*.json")
        for pattern in patterns:
            paths.update((root / "data" / "evaluation").glob(pattern))
            paths.update((root / "data" / "manifests").rglob(pattern))
    return tuple(
        sorted(
            path
            for path in paths
            if path.is_file()
            and "artifacts/private" not in path.as_posix()
            and "private" not in path.relative_to(root).parts
        )
    )


def build_evidence_registry(root: Path, *, base_sha: str = PHASE15_BASE_SHA) -> EvidenceRegistry:
    """Hash frozen tracked artifacts without reading private query/source text."""

    entries: list[ArtifactHash] = []
    seen_phases: set[str] = set()
    for path in _historical_paths(root):
        relative = path.relative_to(root).as_posix()
        phase = next(
            (
                part.removeprefix("phase").split("_")[0].split("-")[0]
                for part in path.name.split(".")[:1]
            ),
            "unknown",
        )
        if not phase.isdigit():
            continue
        seen_phases.add(phase)
        entries.append(
            ArtifactHash(
                phase=phase,
                path=relative,
                sha256=sha256_file(path),
            )
        )
    required = {str(item) for item in (*range(3, 12), 14)}
    missing = required - seen_phases
    if missing:
        raise ValueError(f"missing tracked frozen evidence for phases: {sorted(missing)}")
    return EvidenceRegistry(base_sha=base_sha, entries=tuple(entries))


def verify_evidence_registry(root: Path, registry_path: Path) -> bool:
    """Verify every frozen registry hash against the current tracked file."""

    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    registry = EvidenceRegistry.model_validate(payload)
    if not registry.registry_read_only:
        raise ValueError("evidence registry must remain read-only")
    for entry in registry.entries:
        path = root / entry.path
        if not path.is_file():
            raise ValueError(f"registered evidence is missing: {entry.path}")
        if "artifacts/private" in entry.path or "private" in Path(entry.path).parts:
            raise ValueError(f"private evidence path is registered: {entry.path}")
        actual = sha256_file(path)
        if actual != entry.sha256:
            raise ValueError(
                f"hash mismatch for {entry.path}: expected {entry.sha256}, got {actual}"
            )
    return True
