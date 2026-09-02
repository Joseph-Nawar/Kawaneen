from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from kawaneen.observability.reproducibility import (
    DEFAULT_REPRODUCTION_CONFIG,
    ReproducibilityError,
    ReproductionConfig,
    ResultDefinition,
    build_result_rows,
    generate_result_csv,
    load_reproduction_config,
    reproduce_results,
)

ROOT = Path(__file__).parents[2]


def _single_result_config(source_artifact: str, source_sha256: str) -> ReproductionConfig:
    return ReproductionConfig(
        (
            ResultDefinition(
                result_id="one",
                metric="metric",
                population="test",
                provenance="synthetic",
                source_artifact=source_artifact,
                source_sha256=source_sha256,
                value_path=("outer", "metric"),
            ),
        )
    )


def test_reproduction_config_has_exactly_six_public_results() -> None:
    config = load_reproduction_config(ROOT / DEFAULT_REPRODUCTION_CONFIG)

    assert len(config.results) == 6
    assert all(not Path(item.source_artifact).is_absolute() for item in config.results)
    assert all(not item.source_artifact.startswith("artifacts/private") for item in config.results)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda raw: raw.update(schema_version="wrong"), "schema_version"),
        (lambda raw: raw["results"].pop(), "exactly six"),
        (
            lambda raw: raw["results"][1].update(result_id=raw["results"][0]["result_id"]),
            "unique",
        ),
        (lambda raw: raw["results"][0].update(source_sha256="A" * 64), "lowercase hex"),
        (lambda raw: raw["results"][0].update(source_artifact="../secret.json"), "relative"),
        (
            lambda raw: raw["results"][0].update(source_artifact="artifacts/private/raw.json"),
            "public",
        ),
    ),
)
def test_reproduction_config_validation_is_strict(
    tmp_path: Path, mutation: object, message: str
) -> None:
    raw = json.loads((ROOT / DEFAULT_REPRODUCTION_CONFIG).read_text(encoding="utf-8"))
    mutation(raw)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ReproducibilityError, match=message):
        load_reproduction_config(path)


def test_public_results_reproduce_byte_for_byte() -> None:
    report = reproduce_results(ROOT)

    assert len(report.rows) == 6
    assert report.actual_csv == report.expected_csv
    assert hashlib.sha256(report.actual_csv).hexdigest() == report.table_sha256
    assert report.unique_source_artifact_count == 5


def test_required_object_key_traversal_and_finite_numeric_validation(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text('{"outer":{"metric":3.25}}', encoding="utf-8")

    loaded = _single_result_config("source.json", hashlib.sha256(source.read_bytes()).hexdigest())
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

    loaded = _single_result_config("source.json", hashlib.sha256(source.read_bytes()).hexdigest())
    with pytest.raises(ReproducibilityError, match=message):
        build_result_rows(tmp_path, loaded)


def test_reproduction_loader_rejects_invalid_configs(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(ReproducibilityError, match="unavailable or invalid"):
        load_reproduction_config(missing)

    invalid = tmp_path / "invalid.json"
    invalid.write_text(
        json.dumps({"schema_version": "phase16-reproduction-config-v1", "results": "not-an-array"}),
        encoding="utf-8",
    )
    with pytest.raises(ReproducibilityError, match="must be an array"):
        load_reproduction_config(invalid)

    malformed = tmp_path / "malformed.json"
    malformed_value = json.loads((ROOT / DEFAULT_REPRODUCTION_CONFIG).read_text(encoding="utf-8"))
    malformed_value["results"][0]["value_path"] = ["ok", 1]
    malformed.write_text(json.dumps(malformed_value), encoding="utf-8")
    with pytest.raises(ReproducibilityError, match="value_path"):
        load_reproduction_config(malformed)


def test_reproduction_reports_missing_source_and_expected_table(tmp_path: Path) -> None:
    with pytest.raises(ReproducibilityError, match="source artifact is missing"):
        build_result_rows(tmp_path, _single_result_config("missing.json", "0" * 64))


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

        def __init__(self) -> None:
            self.logged_params: dict[str, object] = {}

        def set_tracking_uri(self, uri: str) -> None:
            return None

        def set_experiment(self, name: str) -> None:
            return None

        def start_run(self, **kwargs: object) -> RunContext:
            return RunContext()

        def log_params(self, values: object) -> None:
            self.logged_params = values  # type: ignore[assignment]

        def log_metrics(self, values: object) -> None:
            return None

        def log_artifact(self, path: str, artifact_path: str) -> None:
            assert Path(path).name in {"phase16_reproduction_config.json", "reproduced.csv"}

    fake_mlflow = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    output = tmp_path / "reproduced.csv"

    report = reproduce_results(ROOT, output_path=output, mlflow=True)

    assert output.read_bytes() == report.actual_csv
    assert fake_mlflow.logged_params["source_artifact_count"] == 5
