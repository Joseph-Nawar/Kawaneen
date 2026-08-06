from __future__ import annotations

import subprocess
import sys

import kawaneen


def test_package_exposes_installed_version() -> None:
    assert kawaneen.__version__


def test_import_has_no_filesystem_side_effect(tmp_path) -> None:
    script = "import kawaneen"
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout == ""
    assert result.stderr == ""
    assert tuple(tmp_path.iterdir()) == ()
