from __future__ import annotations

import subprocess
from pathlib import Path


def test_tracked_tests_have_no_machine_specific_paths() -> None:
    files = subprocess.check_output(["git", "ls-files", "tests"], text=True).splitlines()
    offenders = []
    machine_path_fragments = ("/" + "Users/", "/" + "Volumes/")
    for filename in files:
        text = Path(filename).read_text(encoding="utf-8")
        if any(fragment in text for fragment in machine_path_fragments):
            offenders.append(filename)
    assert offenders == []
