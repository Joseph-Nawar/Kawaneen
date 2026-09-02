from __future__ import annotations

import runpy
from pathlib import Path


def test_review_app_script_imports_without_package_context() -> None:
    review_app = Path(__file__).parents[2] / "src/kawaneen/phase15/review_app.py"

    runpy.run_path(str(review_app), run_name="phase15_review_app_smoke")
