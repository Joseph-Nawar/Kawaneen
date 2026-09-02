"""Small metadata-only observer boundary with lazy MLflow integration."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from typing import Any, Protocol, cast

from kawaneen.core.config import Settings
from kawaneen.observability.identity import ServingIdentity

LOGGER = logging.getLogger(__name__)
TRACE_SCHEMA_VERSION = "phase16-trace-v1"


class TraceSpan(AbstractContextManager["TraceSpan"], Protocol):
    def set_outputs(self, values: Mapping[str, object]) -> None: ...

    def set_attributes(self, values: Mapping[str, object]) -> None: ...

    def record_exception(self, error: BaseException) -> None: ...


class TraceObserver(Protocol):
    def root(self, name: str, attributes: Mapping[str, object]) -> TraceSpan: ...

    def span(self, name: str, span_type: str, attributes: Mapping[str, object]) -> TraceSpan: ...


class _NoOpSpan:
    def __enter__(self) -> _NoOpSpan:
        return self

    def __exit__(self, exc_type: object, exc: BaseException | None, tb: object) -> bool:
        return False

    def set_outputs(self, values: Mapping[str, object]) -> None:
        del values

    def set_attributes(self, values: Mapping[str, object]) -> None:
        del values

    def record_exception(self, error: BaseException) -> None:
        del error


class NoOpObserver:
    """Observer used when telemetry is disabled."""

    def root(self, name: str, attributes: Mapping[str, object]) -> _NoOpSpan:
        del name, attributes
        return _NoOpSpan()

    def span(self, name: str, span_type: str, attributes: Mapping[str, object]) -> _NoOpSpan:
        del name, span_type, attributes
        return _NoOpSpan()


class _MlflowSpan:
    def __init__(self, starter: Callable[[], Any]) -> None:
        self._starter = starter
        self._context: Any | None = None
        self._span: Any | None = None

    def __enter__(self) -> _MlflowSpan:
        try:
            context = self._starter()
            self._context = context
            self._span = context.__enter__()
        except Exception as error:
            self._warn("could not start MLflow span", error)
            self._context = None
            self._span = None
        return self

    def __exit__(self, exc_type: object, exc: BaseException | None, tb: object) -> bool:
        if exc is not None:
            self.record_exception(exc)
        context = cast(Any, self._context)
        if context is not None:
            try:
                context.__exit__(exc_type, exc, tb)
            except Exception as error:
                self._warn("could not finish MLflow span", error)
        return False

    def set_outputs(self, values: Mapping[str, object]) -> None:
        self._call("set_outputs", dict(values))

    def set_attributes(self, values: Mapping[str, object]) -> None:
        self._call("set_attributes", dict(values))

    def record_exception(self, error: BaseException) -> None:
        if self._span is None:
            return
        try:
            record = getattr(self._span, "record_exception", None)
            if callable(record):
                record(error)
            else:
                self._span.set_attributes({"error.type": type(error).__name__})
        except Exception as telemetry_error:
            self._warn("could not record MLflow span exception", telemetry_error)

    def _call(self, method_name: str, value: object) -> None:
        if self._span is None:
            return
        try:
            method = getattr(self._span, method_name)
            method(value)
        except Exception as error:
            self._warn(f"could not write MLflow span {method_name}", error)

    @staticmethod
    def _warn(message: str, error: BaseException) -> None:
        LOGGER.warning("%s: %s", message, error)


class MlflowObserver:
    """Manual MLflow observer. Construction performs the startup preflight."""

    def __init__(self, settings: Settings, identity: ServingIdentity) -> None:
        self.settings = settings
        self.identity = identity
        self._mlflow = _import_mlflow()
        self._configure()

    def root(self, name: str, attributes: Mapping[str, object]) -> TraceSpan:
        return self._make_span(name, "CHAIN", attributes)

    def span(self, name: str, span_type: str, attributes: Mapping[str, object]) -> TraceSpan:
        return self._make_span(name, span_type, attributes)

    def _configure(self) -> None:
        try:
            self._mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)
            client_type = getattr(self._mlflow, "MlflowClient", None)
            if client_type is None:
                tracking = importlib.import_module("mlflow.tracking")
                client_type = tracking.MlflowClient
            client = client_type(tracking_uri=self.settings.mlflow_tracking_uri)
            experiment = client.get_experiment_by_name(self.settings.mlflow_serving_experiment)
            if experiment is None:
                client.create_experiment(self.settings.mlflow_serving_experiment)
            self._mlflow.set_experiment(self.settings.mlflow_serving_experiment)
        except Exception as error:
            raise RuntimeError(
                "MLflow observability is enabled but the configured tracking server "
                f"{self.settings.mlflow_tracking_uri!r} is unreachable or invalid"
            ) from error

    def _make_span(
        self, name: str, span_type: str, attributes: Mapping[str, object]
    ) -> _MlflowSpan:
        mlflow = self._mlflow
        span_kind = _span_type(mlflow, span_type)
        safe_attributes = dict(attributes)
        return _MlflowSpan(
            lambda: mlflow.start_span(
                name=name,
                span_type=span_kind,
                attributes=safe_attributes,
            )
        )


def create_observer(settings: Settings, identity: ServingIdentity | None = None) -> TraceObserver:
    if not settings.observability_enabled:
        return NoOpObserver()
    if identity is None:
        identity = ServingIdentity.build(settings.data_directory)
    try:
        return MlflowObserver(settings, identity)
    except ImportError as error:
        raise RuntimeError(
            "MLflow observability is enabled but MLflow is not installed; "
            "install the observability dependency group with `uv sync --group observability`"
        ) from error


def root_attributes(
    identity: ServingIdentity,
    *,
    request_id: str,
    operation: str,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    values: dict[str, object] = {
        "kawaneen.trace_schema_version": TRACE_SCHEMA_VERSION,
        "kawaneen.request_id": request_id,
        "kawaneen.operation": operation,
        "kawaneen.configuration_version": identity.configuration_version,
        "kawaneen.corpus_version": identity.corpus_version,
        "kawaneen.embedding_model_id": identity.embedding.model_id,
        "kawaneen.embedding_model_revision": identity.embedding.revision,
        "kawaneen.retrieval_strategy": identity.retrieval.strategy,
        "kawaneen.reranker_model_id": identity.reranker.model_id,
        "kawaneen.reranker_model_revision": identity.reranker.revision,
    }
    if operation == "answer":
        values.update(
            {
                "kawaneen.generator_provider": identity.generator.provider,
                "kawaneen.generator_model": identity.generator.model,
                "kawaneen.generator_revision": identity.generator.revision,
                "kawaneen.prompt_template_version": identity.prompt.template_version,
                "kawaneen.prompt_version_hash": identity.prompt.version_hash,
                "kawaneen.answerability_policy_version": identity.answerability.version,
            }
        )
    if metadata:
        values.update(metadata)
    return values


def _span_type(mlflow: Any, name: str) -> object:
    try:
        span_types = mlflow.entities.SpanType
        return getattr(span_types, name, name)
    except AttributeError:
        return name


def _import_mlflow() -> Any:
    return cast(Any, importlib.import_module("mlflow"))


__all__ = [
    "TRACE_SCHEMA_VERSION",
    "MlflowObserver",
    "NoOpObserver",
    "TraceObserver",
    "TraceSpan",
    "create_observer",
    "root_attributes",
]
