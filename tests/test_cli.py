from __future__ import annotations

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
