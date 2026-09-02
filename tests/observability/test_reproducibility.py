from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from kawaneen.observability.reproducibility import (
    DEFAULT_REPRODUCTION_CONFIG,
    ReproducibilityError,
    build_result_rows,
    generate_result_csv,
    load_reproduction_config,
    reproduce_results,
)

ROOT = Path(__file__).parents[2]


def test_reproduction_config_has_exactly_six_public_results() -> None:
    config = load_reproduction_config(ROOT / DEFAULT_REPRODUCTION_CONFIG)

    assert len(config.results) == 6
    assert all(not Path(item.source_artifact).is_absolute() for item in config.results)
    assert all(not item.source_artifact.startswith("artifacts/private") for item in config.results)


def test_public_results_reproduce_byte_for_byte() -> None:
    report = reproduce_results(ROOT)

    assert len(report.rows) == 6
    assert report.actual_csv == report.expected_csv
    assert hashlib.sha256(report.actual_csv).hexdigest() == report.table_sha256


def test_required_object_key_traversal_and_finite_numeric_validation(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text('{"outer":{"metric":3.25}}', encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "result_id": "one",
                        "metric": "metric",
                        "population": "test",
                        "provenance": "synthetic",
                        "source_artifact": "source.json",
                        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        "value_path": ["outer", "metric"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    loaded = load_reproduction_config(config)
    rows = build_result_rows(tmp_path, loaded)
    assert rows[0].value == "3.25"

    source.write_text('{"outer":{}}', encoding="utf-8")
    with pytest.raises(ReproducibilityError, match="source hash"):
        build_result_rows(tmp_path, loaded)


@pytest.mark.parametrize(
    ("payload", "message"),
    (({"outer": {}}, "value path"), ({"outer": {"metric": "not numeric"}}, "numeric")),
)
def test_invalid_result_definition_fails_clearly(
    tmp_path: Path, payload: dict[str, object], message: str
) -> None:
    source = tmp_path / "source.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "result_id": "one",
                        "metric": "metric",
                        "population": "test",
                        "provenance": "synthetic",
                        "source_artifact": "source.json",
                        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        "value_path": ["outer", "metric"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ReproducibilityError, match=message):
        build_result_rows(tmp_path, load_reproduction_config(config))


def test_reproduction_loader_rejects_invalid_configs(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(ReproducibilityError, match="unavailable or invalid"):
        load_reproduction_config(missing)

    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps({"results": "not-an-array"}), encoding="utf-8")
    with pytest.raises(ReproducibilityError, match="must be an array"):
        load_reproduction_config(invalid)

    malformed = tmp_path / "malformed.json"
    malformed.write_text(json.dumps({"results": [{"value_path": ["ok", 1]}]}), encoding="utf-8")
    with pytest.raises(ReproducibilityError, match="value_path"):
        load_reproduction_config(malformed)


def test_reproduction_reports_missing_source_and_expected_table(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "result_id": "one",
                        "metric": "metric",
                        "population": "test",
                        "provenance": "synthetic",
                        "source_artifact": "missing.json",
                        "source_sha256": "0" * 64,
                        "value_path": ["value"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ReproducibilityError, match="source artifact is missing"):
        reproduce_results(tmp_path, config_path=config, expected_path=tmp_path / "expected.csv")


def test_csv_generation_is_deterministic() -> None:
    config = load_reproduction_config(ROOT / DEFAULT_REPRODUCTION_CONFIG)
    rows = build_result_rows(ROOT, config)

    assert generate_result_csv(rows) == generate_result_csv(tuple(rows))


def test_optional_mlflow_logging_uses_only_public_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class RunContext:
        def __enter__(self) -> RunContext:
            return self

        def __exit__(self, exc_type: object, exc: BaseException | None, tb: object) -> bool:
            return False

    class FakeMlflow:
        __version__ = "3.15.2"

        def set_tracking_uri(self, uri: str) -> None:
            return None

        def set_experiment(self, name: str) -> None:
            return None

        def start_run(self, **kwargs: object) -> RunContext:
            return RunContext()

        def log_params(self, values: object) -> None:
            return None

        def log_metrics(self, values: object) -> None:
            return None

        def log_artifact(self, path: str, artifact_path: str) -> None:
            assert Path(path).name in {"phase16_reproduction_config.json", "reproduced.csv"}

    monkeypatch.setitem(sys.modules, "mlflow", FakeMlflow())
    output = tmp_path / "reproduced.csv"

    report = reproduce_results(ROOT, output_path=output, mlflow=True)

    assert output.read_bytes() == report.actual_csv
