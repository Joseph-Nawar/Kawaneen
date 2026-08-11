import json
from pathlib import Path

import pytest

from kawaneen.parsing.qualification import (
    HoldoutAlreadyEvaluatedError,
    create_frozen_split,
    evaluate_holdout_once,
    select_development_configuration,
)


def test_split_is_deterministic_and_stratified(tmp_path: Path) -> None:
    selection = {
        "selection": [
            {"id": "moj-1", "category": "born_digital_arabic"},
            {"id": "moj-2", "category": "born_digital_arabic"},
            {"id": "sama-1", "category": "image_only_scan"},
            {"id": "sama-2", "category": "image_only_scan"},
        ]
    }
    output = tmp_path / "split.json"
    first = create_frozen_split(selection, output, development_fraction=0.5)
    second = create_frozen_split(selection, output, development_fraction=0.5)
    assert first == second
    assert set(first["development"]) | set(first["holdout"]) == {
        "moj-1",
        "moj-2",
        "sama-1",
        "sama-2",
    }
    assert set(first["development"]) & set(first["holdout"]) == set()


def test_existing_split_cannot_change_after_creation(tmp_path: Path) -> None:
    output = tmp_path / "split.json"
    output.write_text(
        json.dumps({"development": ["fixed"], "holdout": ["other"]}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="frozen"):
        create_frozen_split({"selection": []}, output)


def test_development_selection_never_reads_holdout_results() -> None:
    selected = select_development_configuration(
        "sama",
        candidates=("150dpi_full_page", "200dpi_layout"),
        development_results={
            "150dpi_full_page": {"cer": 0.2},
            "200dpi_layout": {"cer": 0.1},
        },
    )
    assert selected == "200dpi_layout"


def test_holdout_evaluation_is_one_shot(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.json"
    evaluate_holdout_once("route", "config", ("p1",), ledger, lambda page: {"page": page})
    with pytest.raises(HoldoutAlreadyEvaluatedError):
        evaluate_holdout_once("route", "config", ("p1",), ledger, lambda page: {"page": page})
