from __future__ import annotations

import json
import subprocess
import tomllib
from pathlib import Path

from kawaneen.normalization.policies import policy_configurations

ROOT = Path(__file__).parents[1]


def test_versioned_policy_toml_mirrors_code_configuration() -> None:
    config = tomllib.loads(
        (ROOT / "configs/normalization/policies.toml").read_text(encoding="utf-8")
    )
    file_policies = {item["policy_id"]: item for item in config["policies"]}
    code_policies = {item["policy_id"]: item for item in policy_configurations()}
    assert config["schema_version"] == 1
    assert set(file_policies) == set(code_policies)
    for policy_id, code_policy in code_policies.items():
        file_policy = file_policies[policy_id]
        assert file_policy["version"] == code_policy["version"]
        assert tuple(file_policy["transforms"]) == code_policy["transforms"]


def test_sanitized_phase4_outputs_have_no_text_fields() -> None:
    manifest = json.loads(
        (ROOT / "data/manifests/normalization/phase4_manifest.json").read_text(encoding="utf-8")
    )
    metrics = json.loads(
        (ROOT / "data/evaluation/phase4_normalization_metrics.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "phase4_experiment_complete"
    assert metrics["status"] == "phase4_experiment_complete"
    serialized = json.dumps((manifest, metrics), ensure_ascii=False)
    for forbidden in ("display_text", "search_text", "query_text", "source_text", "qrels"):
        assert forbidden not in serialized
    assert manifest["challenge"]["private_artifact_root"] == (
        "artifacts/private/phase4_normalization"
    )


def test_sensitivity_outputs_are_sanitized_and_keep_primary_frozen() -> None:
    manifest = json.loads(
        (ROOT / "data/manifests/normalization/phase4_sensitivity_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    metrics = json.loads(
        (ROOT / "data/evaluation/phase4_sensitivity_metrics.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "phase4_sensitivity_validation_complete"
    assert manifest["primary_challenge_version"] == "phase4-primary-challenge-v1"
    assert manifest["probe"]["query_count"] == 60
    assert metrics["decision"]["primary_selected_policy_id"] == "arabic-raw-v1"
    assert metrics["decision"]["validation_selected_policy_id"] == "arabic-light-v1"
    serialized = json.dumps((manifest, metrics), ensure_ascii=False)
    for forbidden in ("query_text", "source_display_text", "display_text", "search_text"):
        assert forbidden not in serialized
    assert metrics["tokenizer_hidden_normalization"] is False


def test_private_phase4_artifacts_are_ignored_and_untracked() -> None:
    private_paths = (
        "artifacts/private/phase4_normalization",
        "artifacts/private/phase4_normalization/sensitivity_validation/probe/challenge_items.jsonl",
        "artifacts/private/phase4_normalization/sensitivity_validation/probe/qrels.json",
        "artifacts/private/phase4_normalization/sensitivity_validation/primary_sensitivity_audit.json",
    )
    for private_path in private_paths:
        ignored = subprocess.run(["git", "check-ignore", "-q", private_path], cwd=ROOT, check=False)
        assert ignored.returncode == 0
    tracked = subprocess.run(
        ["git", "ls-files", "artifacts/private/phase4_normalization"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert tracked.stdout == ""


def test_phase4_docs_state_private_scope_and_phase7_revalidation() -> None:
    phase = (ROOT / "docs/phases/phase-04-arabic-normalization.md").read_text(encoding="utf-8")
    report = (ROOT / "docs/reports/phase-04-arabic-normalization-report.md").read_text(
        encoding="utf-8"
    )
    combined = phase + report
    assert "artifacts/private/phase4_normalization/" in combined
    assert "Phase 7" in combined
    assert "arabic-raw-v1" in combined
    assert "arabic-light-v1" in combined
    assert "arabic-aggressive-v1" in combined
    assert "NFKC" in combined
    sensitivity_report = (
        ROOT / "docs/reports/phase-04-sensitivity-validation-report.md"
    ).read_text(encoding="utf-8")
    assert "phase4-primary-challenge-v1" in sensitivity_report
    assert "arabic-light-v1" in sensitivity_report
    assert "60" in sensitivity_report
