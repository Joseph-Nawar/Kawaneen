"""Dependency-injected serving runtime and readiness snapshots."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from kawaneen.core.config import Settings
from kawaneen.extraction.serving import ServingExtractor


class ExpectedAssetUnavailable(RuntimeError):
    """An optional local model/corpus asset is absent at startup."""


@dataclass(frozen=True, slots=True)
class ComponentReadiness:
    name: str
    ready: bool
    required: bool
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class ModelSnapshot:
    capability: str
    provider: str
    model: str | None
    revision: str | None
    loaded: bool
    ready: bool


@dataclass(frozen=True, slots=True)
class HealthSnapshot:
    status: str
    components: tuple[ComponentReadiness, ...]


class ServiceContainer:
    """Own reusable services; loading is explicit and idempotent."""

    def __init__(
        self,
        *,
        retriever: object | None = None,
        answerer: object | None = None,
        extractor: object | None = None,
        corpus: object | None = None,
        components: Sequence[ComponentReadiness] | None = None,
        model_metadata: Sequence[ModelSnapshot] = (),
        initializer: Callable[[], None] | None = None,
        closer: Callable[[], None] | None = None,
        capabilities: Callable[[], object] | Sequence[ModelSnapshot] = (),
    ) -> None:
        self.retriever = retriever
        self.answerer = answerer
        self.extractor = extractor
        self.corpus = corpus
        self._components = tuple(
            components
            if components is not None
            else (
                ComponentReadiness("corpus", False, True, "canonical corpus is not configured"),
                ComponentReadiness(
                    "retrieval", False, True, "retrieval indexes are not configured"
                ),
                ComponentReadiness("answer", False, True, "answer model is not configured"),
                ComponentReadiness("extraction_deterministic", True, False),
                ComponentReadiness(
                    "extraction_hybrid", False, False, "hybrid model is not configured"
                ),
            )
        )
        self._models = tuple(model_metadata)
        self._initializer = initializer
        self._closer = closer
        self._capabilities = capabilities
        self._initialized = False
        self._closed = False
        self.initialization_count = 0
        self.cleanup_count = 0

    def initialize(self) -> None:
        if self._initialized:
            return
        if self._initializer is not None:
            try:
                self._initializer()
            except ExpectedAssetUnavailable as error:
                self._components = tuple(
                    ComponentReadiness(item.name, False, item.required, str(error))
                    if item.required
                    else item
                    for item in self._components
                )
        self._initialized = True
        self._closed = False
        self.initialization_count += 1

    def close(self) -> None:
        if not self._initialized or self._closed:
            return
        if self._closer is not None:
            self._closer()
        self._closed = True
        self.cleanup_count += 1

    def health(self) -> HealthSnapshot:
        components = tuple(self._components)
        ready = all(item.ready for item in components if item.required)
        return HealthSnapshot("ready" if ready else "degraded", components)

    def models(self) -> tuple[ModelSnapshot, ...]:
        # Metadata is captured at construction/startup; this method intentionally never loads.
        return self._models


def build_default_container(settings: Settings | None = None) -> ServiceContainer:
    """Build a safe default without opening local model files or running inference."""

    del settings
    return ServiceContainer(
        extractor=ServingExtractor(provider=None),
        corpus=None,
        model_metadata=(
            ModelSnapshot("retrieval", "unconfigured", None, None, False, False),
            ModelSnapshot("answer", "unconfigured", None, None, False, False),
            ModelSnapshot("extraction-hybrid", "unconfigured", None, None, False, False),
        ),
    )


__all__ = [
    "ComponentReadiness",
    "ExpectedAssetUnavailable",
    "HealthSnapshot",
    "ModelSnapshot",
    "ServiceContainer",
    "build_default_container",
]
