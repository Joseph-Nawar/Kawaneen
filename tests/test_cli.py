from __future__ import annotations

import json
import subprocess
import sys


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "kawaneen", *args],
        check=False,
        capture_output=True,
        text=True,
    )


def test_version_command() -> None:
    result = run_cli("--version")
    assert result.returncode == 0
    assert result.stdout.startswith("kawaneen ")


def test_doctor_command_reports_foundation_status() -> None:
    result = run_cli("doctor")
    assert result.returncode == 0
    assert "Kawaneen foundation: ready" in result.stdout


def test_data_plan_is_offline_and_contains_permitted_sources() -> None:
    result = run_cli("data", "plan")
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert {row["source_id"] for row in payload} == {"alarb", "arabiccr", "saudi-moj-derived"}


def test_data_status_is_offline() -> None:
    result = run_cli("data", "status")
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert {row["source_id"] for row in payload} == {"alarb", "arabiccr", "saudi-moj-derived"}
    assert all(isinstance(row["raw_present"], bool) for row in payload)


def test_denied_source_acquisition_fails_closed() -> None:
    result = run_cli("data", "acquire", "alcd", "--purpose", "evaluation")
    assert result.returncode == 1
    assert "denied" in result.stderr


def test_data_manifest_validate_command() -> None:
    result = run_cli("data", "manifest", "validate")
    assert result.returncode == 0
    assert "manifests valid" in result.stdout


def test_data_rebuild_requires_explicit_auto_flag() -> None:
    result = run_cli("data", "rebuild")
    assert result.returncode == 1
    assert "requires --auto" in result.stderr


def test_data_verify_all_reports_unacquired_sources_without_bypass() -> None:
    result = run_cli("data", "verify")
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    moj = next(item for item in payload if item["source_id"] == "saudi-moj-derived")
    assert moj["row_counts"]["data/train-00000-of-00001.parquet"] == 3185


def test_statutory_audit_reports_sanitized_quality_counts() -> None:
    result = run_cli("data", "audit-statutory", "saudi-moj-derived")
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["total_rows"] == 3185
    assert payload["unique_law_names"] == 71
    assert "text" not in payload
