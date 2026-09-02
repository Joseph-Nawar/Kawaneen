"""Deterministic reconstruction of the public Phase 16 result table."""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
import math
import platform
import re
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, cast

DEFAULT_REPRODUCTION_CONFIG = Path("data/manifests/observability/phase16_reproduction_config.json")
DEFAULT_EXPECTED_TABLE = Path("data/evaluation/phase16_reported_results.csv")
REPRODUCTION_SCHEMA_VERSION = "phase16-reproduction-config-v1"
SOURCE_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
CSV_COLUMNS = (
    "result_id",
    "metric",
    "population",
    "provenance",
    "value",
    "source_artifact",
    "source_sha256",
    "value_path",
)


class ReproducibilityError(ValueError):
    """A public reproduction artifact is missing, changed, or malformed."""


@dataclass(frozen=True, slots=True)
class ResultDefinition:
    result_id: str
    metric: str
    population: str
    provenance: str
    source_artifact: str
    source_sha256: str
    value_path: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReproductionConfig:
    results: tuple[ResultDefinition, ...]


@dataclass(frozen=True, slots=True)
class ResultRow:
    result_id: str
    metric: str
    population: str
    provenance: str
    value: str
    source_artifact: str
    source_sha256: str
    value_path: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReproductionReport:
    rows: tuple[ResultRow, ...]
    actual_csv: bytes
    expected_csv: bytes
    table_sha256: str
    reproduction_config_sha256: str

    @property
    def unique_source_artifact_count(self) -> int:
        return len({row.source_artifact for row in self.rows})


def load_reproduction_config(path: Path) -> ReproductionConfig:
    try:
        raw_value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReproducibilityError(
            f"reproduction config is unavailable or invalid: {path}"
        ) from error
    raw = _mapping(raw_value, "reproduction config")
    if raw.get("schema_version") != REPRODUCTION_SCHEMA_VERSION:
        raise ReproducibilityError(f"schema_version must be {REPRODUCTION_SCHEMA_VERSION}")
    definitions = raw.get("results")
    if not isinstance(definitions, list):
        raise ReproducibilityError("reproduction config results must be an array")
    definitions = cast(list[object], definitions)
    if len(definitions) != 6:
        raise ReproducibilityError("reproduction config must contain exactly six results")
    result: list[ResultDefinition] = []
    result_ids: set[str] = set()
    for index, value in enumerate(definitions, start=1):
        item = _mapping(value, f"result definition {index}")
        result_id = _required_string(item.get("result_id"), "result_id")
        if result_id in result_ids:
            raise ReproducibilityError("result_id values must be unique")
        result_ids.add(result_id)
        path_value = item.get("value_path")
        if not isinstance(path_value, list):
            raise ReproducibilityError(f"result definition {index} value_path must be object keys")
        path_value = cast(list[object], path_value)
        if any(not isinstance(key, str) or not key for key in path_value):
            raise ReproducibilityError(f"result definition {index} value_path must be object keys")
        source_artifact = _required_string(item.get("source_artifact"), "source_artifact")
        source_path = Path(source_artifact.replace("\\", "/"))
        if source_path.is_absolute() or ".." in source_path.parts:
            raise ReproducibilityError(
                "source_artifact must be a relative public path without '..'"
            )
        if not source_artifact.startswith("data/evaluation/"):
            raise ReproducibilityError(
                "source_artifact must reference a tracked public data/evaluation path"
            )
        source_sha256 = _required_string(item.get("source_sha256"), "source_sha256")
        if SOURCE_SHA256_PATTERN.fullmatch(source_sha256) is None:
            raise ReproducibilityError("source_sha256 must be 64 lowercase hex characters")
        result.append(
            ResultDefinition(
                result_id=result_id,
                metric=_required_string(item.get("metric"), "metric"),
                population=_required_string(item.get("population"), "population"),
                provenance=_required_string(item.get("provenance"), "provenance"),
                source_artifact=source_artifact,
                source_sha256=source_sha256,
                value_path=tuple(cast(str, key) for key in path_value),
            )
        )
    return ReproductionConfig(tuple(result))


def build_result_rows(root: Path, config: ReproductionConfig) -> tuple[ResultRow, ...]:
    rows: list[ResultRow] = []
    for definition in config.results:
        source = root / definition.source_artifact
        if not source.is_file():
            raise ReproducibilityError(f"source artifact is missing: {definition.source_artifact}")
        source_bytes = source.read_bytes()
        actual_hash = hashlib.sha256(source_bytes).hexdigest()
        if actual_hash != definition.source_sha256:
            raise ReproducibilityError(
                f"source hash mismatch for {definition.source_artifact}: "
                f"expected {definition.source_sha256}, got {actual_hash}"
            )
        try:
            document = json.loads(source_bytes)
        except json.JSONDecodeError as error:
            raise ReproducibilityError(f"source artifact is invalid JSON: {source}") from error
        value: object = document
        for key in definition.value_path:
            if not isinstance(value, dict) or key not in value:
                raise ReproducibilityError(
                    f"value path {list(definition.value_path)!r} is missing key {key!r}"
                )
            value = cast(dict[str, object], value)[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ReproducibilityError(f"result {definition.result_id} value must be numeric")
        if not math.isfinite(float(value)):
            raise ReproducibilityError(f"result {definition.result_id} value must be finite")
        rows.append(
            ResultRow(
                result_id=definition.result_id,
                metric=definition.metric,
                population=definition.population,
                provenance=definition.provenance,
                value=_number_text(value),
                source_artifact=definition.source_artifact,
                source_sha256=definition.source_sha256,
                value_path=definition.value_path,
            )
        )
    return tuple(rows)


def generate_result_csv(rows: tuple[ResultRow, ...]) -> bytes:
    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                "result_id": row.result_id,
                "metric": row.metric,
                "population": row.population,
                "provenance": row.provenance,
                "value": row.value,
                "source_artifact": row.source_artifact,
                "source_sha256": row.source_sha256,
                "value_path": json.dumps(
                    list(row.value_path), ensure_ascii=False, separators=(",", ":")
                ),
            }
        )
    return output.getvalue().encode("utf-8")


def reproduce_results(
    root: Path,
    *,
    config_path: Path | None = None,
    expected_path: Path | None = None,
    output_path: Path | None = None,
    mlflow: bool = False,
) -> ReproductionReport:
    effective_config = config_path or root / DEFAULT_REPRODUCTION_CONFIG
    effective_expected = expected_path or root / DEFAULT_EXPECTED_TABLE
    config = load_reproduction_config(effective_config)
    rows = build_result_rows(root, config)
    actual = generate_result_csv(rows)
    try:
        expected = effective_expected.read_bytes()
    except OSError as error:
        raise ReproducibilityError(
            f"expected result table is unavailable: {effective_expected}"
        ) from error
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(actual)
    report = ReproductionReport(
        rows=rows,
        actual_csv=actual,
        expected_csv=expected,
        table_sha256=hashlib.sha256(actual).hexdigest(),
        reproduction_config_sha256=hashlib.sha256(effective_config.read_bytes()).hexdigest(),
    )
    if actual != expected:
        raise ReproducibilityError(
            f"reproduced result table differs from committed table: {effective_expected}"
        )
    if mlflow:
        _log_reproduction_run(root, report, effective_config, output_path)
    return report


def _log_reproduction_run(
    root: Path,
    report: ReproductionReport,
    config_path: Path,
    output_path: Path | None,
) -> None:
    try:
        mlflow = importlib.import_module("mlflow")
    except ImportError as error:
        raise ReproducibilityError(
            "MLflow logging requested; install it with `uv sync --group observability`"
        ) from error
    from kawaneen.core.config import Settings
    from kawaneen.observability.identity import ServingIdentity

    settings = Settings()
    identity = ServingIdentity.build(root / "data")
    mlflow_module: Any = mlflow
    mlflow_module.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow_module.set_experiment(settings.mlflow_repro_experiment)
    with mlflow_module.start_run(run_name="phase16-public-reproduction"):
        mlflow_module.log_params(
            {
                "serving_configuration_version": identity.configuration_version,
                "reproduction_config_sha256": report.reproduction_config_sha256,
                "result_table_sha256": report.table_sha256,
                "source_artifact_count": report.unique_source_artifact_count,
                "git_commit": _git_commit(root),
                "python_version": platform.python_version(),
                "mlflow_version": mlflow_module.__version__,
            }
        )
        mlflow_module.log_metrics({row.result_id: float(row.value) for row in report.rows})
        mlflow_module.log_artifact(str(config_path), artifact_path="phase16")
        if output_path is not None:
            mlflow_module.log_artifact(str(output_path), artifact_path="phase16")


def _git_commit(root: Path) -> str:
    import subprocess

    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ReproducibilityError(f"{label} must be an object")
    return cast(dict[str, object], value)


def _required_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReproducibilityError(f"{label} must be a non-empty string")
    return value


def _number_text(value: int | float) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


__all__ = [
    "DEFAULT_EXPECTED_TABLE",
    "DEFAULT_REPRODUCTION_CONFIG",
    "REPRODUCTION_SCHEMA_VERSION",
    "ReproducibilityError",
    "ReproductionConfig",
    "ReproductionReport",
    "ResultDefinition",
    "ResultRow",
    "build_result_rows",
    "generate_result_csv",
    "load_reproduction_config",
    "reproduce_results",
]
