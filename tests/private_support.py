from __future__ import annotations

import os
from pathlib import Path

import pytest


def private_repo_path(*parts: str) -> Path:
    root = Path(os.environ.get("KAWANEEN_PRIVATE_TEST_ROOT", "artifacts/private"))
    path = root.joinpath(*parts)
    if not path.exists():
        pytest.skip(f"private artifact is unavailable: {path}")
    return path


def external_review_path(filename: str) -> Path:
    root = os.environ.get("KAWANEEN_PRIVATE_EXTERNAL_ROOT")
    if not root:
        pytest.skip("set KAWANEEN_PRIVATE_EXTERNAL_ROOT for external review artifacts")
    path = Path(root) / filename
    if not path.is_file():
        pytest.skip(f"external review artifact is unavailable: {path}")
    return path
