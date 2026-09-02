from __future__ import annotations

from pathlib import Path

import pytest

from kawaneen.phase15.reporting import (
    assert_text_free,
    metric_status_artifact,
    write_aggregate_artifact,
)


def test_reporting_writes_text_free_aggregate_and_optional_reason(tmp_path: Path) -> None:
    payload = metric_status_artifact(status="NOT_RUN", reason="DEV inputs unavailable")
    destination = write_aggregate_artifact(tmp_path, "metrics.json", payload)
    assert destination == tmp_path / "data/evaluation/metrics.json"
    assert destination.is_file()
    assert metric_status_artifact(status="READY") == {
        "status": "READY",
        "provenance": "PHASE15_DEV",
    }


def test_reporting_rejects_private_fields_and_invalid_filenames() -> None:
    with pytest.raises(ValueError, match="private text field"):
        assert_text_free({"nested": [{"query_text": "private"}]})
    with pytest.raises(ValueError, match="one JSON file name"):
        write_aggregate_artifact(Path("."), "nested/metrics.json", {})
