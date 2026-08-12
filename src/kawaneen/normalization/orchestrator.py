"""Phase 4 normalization experiment orchestration and conservative selection."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pyarrow as _pyarrow

from kawaneen.corpus.models import CanonicalUnit
from kawaneen.corpus.serialization import write_json, write_parquet
from kawaneen.normalization.challenge import PRIVATE_ROOT, PrivateChallenge, build_private_challenge
from kawaneen.normalization.corpus import (
    ELIGIBLE_SOURCES,
    freeze_candidate_policy,
    load_candidate_units,
    select_representative_subset,
)
from kawaneen.normalization.diagnostics import CorpusDiagnostics, diagnose_policy
from kawaneen.normalization.models import NormalizationPolicy
from kawaneen.normalization.policies import (
    all_policies,
    get_policy,
    normalize_text,
)
from kawaneen.normalization.records import NormalizedRecord, validate_record_contract
from kawaneen.normalization.retrieval import AblationReport, run_ablation
from kawaneen.normalization.safety import validate_identifier_safety

pa: Any = _pyarrow

CANONICAL_ROOT = Path("data/interim/canonical")
MANIFEST_ROOT = Path("data/manifests/normalization")
METRICS_PATH = Path("data/evaluation/phase4_normalization_metrics.json")
MAX_DISTINCT_FORM_COLLISION_RATE = 0.10
MAX_UNIT_COLLISION_RATE = 0.20
MEANINGFUL_GAIN = 0.02
CONTROL_REGRESSION_TOLERANCE = 0.02


@dataclass(frozen=True, slots=True)
class SelectionEvidence:
    policy_metrics: dict[str, dict[str, float]]
    slice_metrics: dict[str, dict[str, dict[str, float]]]
    paired_confidence_intervals: dict[str, dict[str, float | int]]
    policy_gate_status: dict[str, dict[str, object]]


@dataclass(frozen=True, slots=True)
class SelectionDecision:
    selected_policy_id: str
    rationale: str
    eligible_policies: tuple[str, ...]
    rejected_policies: dict[str, tuple[str, ...]]

    def to_sanitized_dict(self) -> dict[str, object]:
        return {
            "selected_policy_id": self.selected_policy_id,
            "rationale": self.rationale,
            "eligible_policies": list(self.eligible_policies),
            "rejected_policies": {
                policy: list(reasons) for policy, reasons in self.rejected_policies.items()
            },
        }


def normalization_plan() -> dict[str, object]:
    return {
        "schema_version": 1,
        "phase": "phase-4-arabic-normalization",
        "private_root": PRIVATE_ROOT.as_posix(),
        "policies": [
            {
                "policy_id": policy.policy_id,
                "version": policy.version,
                "policy_hash": policy.policy_hash,
                "transforms": list(policy.transforms),
                "config": dict(policy.config),
            }
            for policy in all_policies()
        ],
        "scope_exclusions": [
            "chunking",
            "embeddings",
            "production retrieval",
            "Phase 6 human evaluation",
            "NFKC",
            "stemming",
            "lemmatization",
            "transliteration",
            "legal abbreviation expansion",
        ],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hashes() -> dict[str, str]:
    inventory = json.loads(
        Path("data/manifests/canonical/inventory.json").read_text(encoding="utf-8")
    )
    return {
        str(file["path"]): _sha256(Path(str(file["path"])))
        for source in inventory["sources"]
        for file in source["files"]
    }


def _normalized_schema() -> Any:
    return pa.schema(
        [
            ("unit_id", pa.string()),
            ("document_id", pa.string()),
            ("unit_type", pa.string()),
            ("display_text", pa.string()),
            ("search_text", pa.string()),
            ("policy_id", pa.string()),
            ("policy_hash", pa.string()),
            ("source_text_sha256", pa.string()),
            ("search_text_sha256", pa.string()),
            ("source_id", pa.string()),
            ("source_version", pa.string()),
            ("source_path", pa.string()),
            ("source_row", pa.int64()),
            ("source_field", pa.string()),
            ("split", pa.string()),
            ("ordinal", pa.int64()),
            ("transform_counts_json", pa.string()),
        ]
    )


def _materialize_policy(
    units: Sequence[CanonicalUnit], policy: NormalizationPolicy
) -> tuple[list[NormalizedRecord], dict[str, int]]:
    records: list[NormalizedRecord] = []
    safety_failures = 0
    for unit in units:
        record = NormalizedRecord.from_canonical(unit, policy)
        validate_record_contract(record, unit.text)
        safety_failures += not validate_identifier_safety(unit.text, record.search_text).safe
        records.append(record)
    return records, {"identifier_safety_failures": safety_failures}


def _write_private_view(records: Sequence[NormalizedRecord], path: Path) -> dict[str, object]:
    rows = [
        {
            **record.model_dump(exclude={"provenance", "transform_counts"}),
            "unit_type": record.unit_type.value,
            "source_id": record.provenance.source_id,
            "source_version": record.provenance.source_version,
            "source_path": record.provenance.source_path,
            "source_row": record.provenance.source_row,
            "source_field": record.provenance.source_field,
            "split": record.provenance.split,
            "transform_counts_json": json.dumps(
                record.transform_counts, ensure_ascii=False, sort_keys=True
            ),
        }
        for record in records
    ]
    return write_parquet(rows, path, _normalized_schema())


def _gate_status(
    units: Sequence[CanonicalUnit],
    policy: NormalizationPolicy,
    diagnostics: CorpusDiagnostics,
    safety_failures: int,
) -> dict[str, object]:
    idempotency_failures = 0
    determinism_failures = 0
    for unit in units:
        first = normalize_text(unit.text, policy)
        second = normalize_text(unit.text, policy)
        if first != second:
            determinism_failures += 1
        if not isinstance(first, str) or normalize_text(first, policy) != first:
            idempotency_failures += 1
    failures: list[str] = []
    if safety_failures:
        failures.append("identifier_safety")
    if determinism_failures:
        failures.append("determinism")
    if idempotency_failures:
        failures.append("idempotency")
    if diagnostics.distinct_form_collision_rate > MAX_DISTINCT_FORM_COLLISION_RATE:
        failures.append("distinct_form_collision_rate")
    if diagnostics.unit_collision_rate > MAX_UNIT_COLLISION_RATE:
        failures.append("unit_collision_rate")
    return {
        "eligible": not failures,
        "failures": failures,
        "preservation_checked": len(units),
        "determinism_failures": determinism_failures,
        "idempotency_failures": idempotency_failures,
        "identifier_safety_failures": safety_failures,
        "distinct_form_collision_rate": diagnostics.distinct_form_collision_rate,
        "unit_collision_rate": diagnostics.unit_collision_rate,
    }


def _interval_improvement(
    evidence: SelectionEvidence, less_destructive: str, more_destructive: str, metric: str
) -> tuple[float, float, float]:
    key = f"{less_destructive}__vs__{more_destructive}__{metric}"
    interval = evidence.paired_confidence_intervals.get(key)
    if interval is not None:
        estimate = -float(interval["estimate"])
        lower = -float(interval["upper"])
        upper = -float(interval["lower"])
        return estimate, lower, upper
    estimate = (
        evidence.policy_metrics[more_destructive][metric]
        - evidence.policy_metrics[less_destructive][metric]
    )
    return estimate, estimate, estimate


def _meaningful_improvement(
    evidence: SelectionEvidence, less_destructive: str, more_destructive: str
) -> bool:
    estimate, lower, _upper = _interval_improvement(
        evidence, less_destructive, more_destructive, "mrr_at_10"
    )
    recall_estimate, recall_lower, _ = _interval_improvement(
        evidence, less_destructive, more_destructive, "recall_at_10"
    )
    ndcg_estimate, ndcg_lower, _ = _interval_improvement(
        evidence, less_destructive, more_destructive, "ndcg_at_10"
    )
    return (
        (estimate >= MEANINGFUL_GAIN and lower > 0)
        or (recall_estimate >= MEANINGFUL_GAIN and recall_lower > 0)
        or (ndcg_estimate >= MEANINGFUL_GAIN and ndcg_lower > 0)
    )


def _no_slice_regression(
    evidence: SelectionEvidence, less_destructive: str, more_destructive: str
) -> bool:
    for slice_name in ("unchanged_control", "collision_risk"):
        left = evidence.slice_metrics.get(less_destructive, {}).get(slice_name, {})
        right = evidence.slice_metrics.get(more_destructive, {}).get(slice_name, {})
        if (
            left
            and right
            and right.get("mrr_at_10", 0.0) - left.get("mrr_at_10", 0.0)
            < -CONTROL_REGRESSION_TOLERANCE
        ):
            return False
    return True


def select_policy(evidence: SelectionEvidence) -> SelectionDecision:
    order = ("arabic-raw-v1", "arabic-light-v1", "arabic-aggressive-v1")
    rejected: dict[str, tuple[str, ...]] = {}
    for policy, status in evidence.policy_gate_status.items():
        if not bool(status.get("eligible", False)):
            failures = status.get("failures", [])
            if isinstance(failures, list):
                rejected[policy] = tuple(str(reason) for reason in cast(list[object], failures))
            else:
                rejected[policy] = ("invalid_gate_status",)
    eligible = tuple(
        policy for policy in order if policy not in rejected and policy in evidence.policy_metrics
    )
    if not eligible:
        raise ValueError("no normalization policy passed hard gates")
    selected = "arabic-raw-v1" if "arabic-raw-v1" in eligible else eligible[0]
    if (
        "arabic-light-v1" in eligible
        and selected == "arabic-raw-v1"
        and _meaningful_improvement(evidence, "arabic-raw-v1", "arabic-light-v1")
        and _no_slice_regression(evidence, "arabic-raw-v1", "arabic-light-v1")
    ):
        selected = "arabic-light-v1"
    if (
        "arabic-aggressive-v1" in eligible
        and selected == "arabic-light-v1"
        and _meaningful_improvement(evidence, "arabic-light-v1", "arabic-aggressive-v1")
        and _no_slice_regression(evidence, "arabic-light-v1", "arabic-aggressive-v1")
    ):
        selected = "arabic-aggressive-v1"
    rationale = (
        "Selected the least destructive policy passing all hard gates; more destructive policies "
        "were promoted only for meaningful paired retrieval improvement without "
        "control/collision regression."
    )
    return SelectionDecision(
        selected_policy_id=selected,
        rationale=rationale,
        eligible_policies=eligible,
        rejected_policies=rejected,
    )


def _challenge_summary(challenge: PrivateChallenge) -> dict[str, object]:
    counts = Counter(item.phenomenon for item in challenge.items)
    return {
        "construction_version": challenge.construction_version,
        "seed": challenge.seed,
        "query_count": len(challenge.items),
        "phenomenon_counts": dict(sorted(counts.items())),
        "multi_relevant_qrel_count": sum(
            len(relevant) > 1 for relevant in challenge.qrels.values()
        ),
        "private_artifact_root": PRIVATE_ROOT.as_posix(),
    }


def _write_private_results(report: AblationReport) -> None:
    result_root = PRIVATE_ROOT / "retrieval_results"
    result_root.mkdir(parents=True, exist_ok=True)
    for policy_id, results in report.private_results.items():
        write_json(result_root / f"{policy_id}.json", results)


def run_phase4_experiment() -> dict[str, object]:
    """Run the local private Phase 4 experiment and write sanitized aggregate evidence."""

    before_hashes = _canonical_hashes()
    full_units = load_candidate_units(CANONICAL_ROOT, ELIGIBLE_SOURCES)
    units = select_representative_subset(full_units)
    candidate_policy = freeze_candidate_policy(units)
    challenge = build_private_challenge(units, output_root=PRIVATE_ROOT / "challenge")
    diagnostics: dict[str, CorpusDiagnostics] = {}
    gates: dict[str, dict[str, object]] = {}
    for policy in all_policies():
        current_diagnostics = diagnose_policy(units, policy)
        records, safety = _materialize_policy(units, policy)
        diagnostics[policy.policy_id] = current_diagnostics
        gates[policy.policy_id] = _gate_status(
            units, policy, current_diagnostics, safety["identifier_safety_failures"]
        )
        _write_private_view(records, PRIVATE_ROOT / "views" / policy.policy_id / "units.parquet")
        del records
    evidence_report = run_ablation(units, challenge, all_policies())
    evidence = SelectionEvidence(
        policy_metrics=evidence_report.policy_metrics,
        slice_metrics=evidence_report.slice_metrics,
        paired_confidence_intervals=evidence_report.paired_confidence_intervals,
        policy_gate_status=gates,
    )
    decision = select_policy(evidence)
    selected_records, _selected_safety = _materialize_policy(
        units, get_policy(decision.selected_policy_id)
    )
    _write_private_view(selected_records, PRIVATE_ROOT / "selected" / "units.parquet")
    _write_private_results(evidence_report)
    after_hashes = _canonical_hashes()
    if before_hashes != after_hashes:
        raise RuntimeError("canonical file hashes changed during Phase 4 experiment")

    manifest = {
        "schema_version": 1,
        "status": "phase4_experiment_complete",
        "candidate_policy": candidate_policy.to_sanitized_dict(),
        "full_eligible_candidate_count": len(full_units),
        "scope_note": (
            "deterministic balanced subset selected before results because full local corpus "
            "was impractical"
        ),
        "challenge": _challenge_summary(challenge),
        "policies": {
            policy_id: {
                "policy_hash": policy.policy_hash,
                "diagnostics": diagnostics[policy_id].to_sanitized_dict(),
                "gate_status": gates[policy_id],
            }
            for policy_id, policy in ((item.policy_id, item) for item in all_policies())
        },
        "retrieval_scope": {
            "candidate_count": evidence_report.candidate_count,
            "tokenizer": "unicode-word-or-single-punctuation-v1",
            "bm25": {"k1": evidence_report.k1, "b": evidence_report.b},
            "seed": evidence_report.seed,
        },
        "selection": decision.to_sanitized_dict(),
        "canonical_hashes": before_hashes,
        "revalidation_required": "Phase 7 human evaluation set",
    }
    metrics = {
        "schema_version": 1,
        "status": "phase4_experiment_complete",
        "candidate_count": evidence_report.candidate_count,
        "challenge_query_count": len(evidence_report.challenge_query_ids),
        "policy_metrics": evidence_report.policy_metrics,
        "slice_metrics": evidence_report.slice_metrics,
        "pairwise_wins_ties_losses": evidence_report.pairwise_wins_ties_losses,
        "paired_confidence_intervals": evidence_report.paired_confidence_intervals,
        "selected_policy_id": decision.selected_policy_id,
    }
    write_json(
        MANIFEST_ROOT / "policies.json",
        {"schema_version": 1, "policies": normalization_plan()["policies"]},
    )
    write_json(MANIFEST_ROOT / "phase4_manifest.json", manifest)
    write_json(METRICS_PATH, metrics)
    return {"manifest": manifest, "metrics": metrics}
