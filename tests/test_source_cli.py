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


def test_sources_validate_command() -> None:
    result = run_cli("sources", "validate")
    assert result.returncode == 0
    assert "Source registry valid" in result.stdout


def test_sources_summary_command() -> None:
    result = run_cli("sources", "summary")
    assert result.returncode == 0
    assert "Sources:" in result.stdout
    assert "Decisions:" in result.stdout


def test_sources_summary_json_command() -> None:
    result = run_cli("sources", "summary", "--format", "json")
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["source_count"] >= 12
    assert "decisions" in payload
