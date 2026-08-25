from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[2]
CASES_PATH = ROOT / "data" / "regression" / "phase14_cases.json"
LOCK_PATH = ROOT / "data" / "manifests" / "testing" / "phase14_regression_lock.json"


def load_cases() -> list[dict[str, Any]]:
    value = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    return value["cases"]


def load_lock() -> dict[str, Any]:
    return json.loads(LOCK_PATH.read_text(encoding="utf-8"))
