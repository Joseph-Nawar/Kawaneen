"""Dependency-injected serving runtime and readiness snapshots."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass

from kawaneen.core.config import Settings
from kawaneen.extraction.serving import ServingExtractor
from kawaneen.observability.identity import ServingIdentity
from kawaneen.observability.tracing import (
    NoOpObserver,
    TraceObserver,
    create_observer,
    root_attributes,
)


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


ComponentInitializer = Callable[[], Sequence[ModelSnapshot] | None]


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
        settings: Settings | None = None,
        component_initializers: Mapping[str, ComponentInitializer] | None = None,
        observer: TraceObserver | None = None,
        observability_identity: ServingIdentity | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.observer = observer or NoOpObserver()
        self.observability_identity = observability_identity
        self._observer_supplied = observer is not None
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
        self._component_initializers = dict(component_initializers or {})
        self._initialized = False
        self._closed = False
        self.initialization_count = 0
        self.cleanup_count = 0

    def initialize(self) -> None:
        if self._initialized:
            return
        if self.settings.observability_enabled and not self._observer_supplied:
            self.observability_identity = ServingIdentity.build(self.settings.data_directory)
            self.observer = create_observer(self.settings, self.observability_identity)
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
        for name, initializer in self._component_initializers.items():
            try:
                snapshots = initializer()
            except ExpectedAssetUnavailable as error:
                self._set_component(name, ready=False, detail=str(error))
                continue
            self._set_component(name, ready=True, detail=None)
            if snapshots:
                self._replace_models(snapshots)
        self._initialized = True
        self._closed = False
        self.initialization_count += 1

    def root_trace(
        self,
        operation: str,
        request_id: str,
        metadata: Mapping[str, object] | None = None,
    ):
        if self.observability_identity is None:
            return self.observer.root(f"kawaneen.{operation}", {})
        return self.observer.root(
            f"kawaneen.{operation}",
            root_attributes(
                self.observability_identity,
                request_id=request_id,
                operation=operation,
                metadata=metadata,
            ),
        )

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

    def component_ready(self, name: str) -> bool:
        matches = tuple(item for item in self._components if item.name == name)
        return all(item.ready for item in matches) if matches else True

    def _set_component(self, name: str, *, ready: bool, detail: str | None) -> None:
        self._components = tuple(
            ComponentReadiness(item.name, ready, item.required, detail)
            if item.name == name
            else item
            for item in self._components
        )

    def _replace_models(self, snapshots: Sequence[ModelSnapshot]) -> None:
        replacements = {item.capability: item for item in snapshots}
        self._models = tuple(replacements.get(item.capability, item) for item in self._models)


def build_default_container(settings: Settings | None = None) -> ServiceContainer:
    """Compose the real serving boundary, degrading only absent local assets."""

    effective_settings = settings or Settings()
    observability_identity = None
    observer: TraceObserver | None = None
    if effective_settings.observability_enabled:
        observability_identity = ServingIdentity.build(effective_settings.data_directory)
        observer = create_observer(effective_settings, observability_identity)
    from kawaneen.api.composition import (
        build_hybrid_extraction,
        build_serving_retrieval,
        build_stage_d_generation,
        load_frozen_serving_configuration,
    )
    from kawaneen.corpus.serving import CanonicalCorpusRepository
    from kawaneen.generation.answerability import (
        load_source_eligibility_registry,
        load_structural_roles,
    )
    from kawaneen.generation.policy import default_deployment_scope
    from kawaneen.generation.serving import ServingAnswerer
    from kawaneen.grounding.assembly import ContextAssembler
    from kawaneen.grounding.provenance import CanonicalCorpusResolver

    private = effective_settings.artifacts_directory / "private"
    canonical_path = (
        private / "phase6_evaluation" / "ai-reviewed-v1" / "corpus" / "canonical_units.json"
    )
    chunks_path = private / "phase7_retrieval" / "corpus" / "chunks.jsonl"
    corpus = None
    with suppress(FileNotFoundError):
        corpus = CanonicalCorpusRepository.from_json(canonical_path)

    configuration = None
    with suppress(ExpectedAssetUnavailable):
        configuration = load_frozen_serving_configuration(effective_settings.data_directory)

    retrieval_bundle = None
    if configuration is not None:
        with suppress(ExpectedAssetUnavailable, FileNotFoundError):
            if observer is None:
                retrieval_bundle = build_serving_retrieval(effective_settings, configuration)
            else:
                retrieval_bundle = build_serving_retrieval(
                    effective_settings, configuration, observer=observer
                )

    resolver = None
    if corpus is not None and retrieval_bundle is not None:
        try:
            resolver = CanonicalCorpusResolver.from_json(canonical_path, chunks_path)
        except FileNotFoundError:
            resolver = None

    generation_bundle = None
    if resolver is not None and retrieval_bundle is not None:
        with suppress(ExpectedAssetUnavailable):
            generation_bundle = build_stage_d_generation(effective_settings)

    answerer = None
    if resolver is not None and retrieval_bundle is not None and generation_bundle is not None:
        assert configuration is not None
        registry_path = effective_settings.data_directory / "manifests" / "source_registry.csv"
        scope_path = (
            effective_settings.data_directory
            / "manifests"
            / "generation"
            / "phase10_jurisdiction_scope.json"
        )
        source_registry = load_source_eligibility_registry(registry_path)
        structural_roles = load_structural_roles(canonical_path)
        scope = default_deployment_scope(scope_path)

        class _ServingTokenCounter:
            identity = "serving-codepoint-v1"

            def count(self, text: str) -> int:
                return len(text)

        assembler = ContextAssembler(
            resolver,
            _ServingTokenCounter(),
            max_context_tokens=2_944,
        )
        answerer = ServingAnswerer.from_phase9_10(
            retriever=retrieval_bundle.retriever,
            assembler=assembler,
            resolver=resolver,
            phase8_selection_sha256=configuration.phase8_selection_sha256,
            canonical_corpus_hash=configuration.corpus_hash,
            deployment_scope=scope,
            generator=lambda query, context: generation_bundle.generator(query, context),
            source_registry=source_registry,
            structural_roles=structural_roles,
            observer=observer,
            generator_provider=observability_identity.generator.provider
            if observability_identity is not None
            else None,
            generator_model=observability_identity.generator.model
            if observability_identity is not None
            else None,
            generator_revision=observability_identity.generator.revision
            if observability_identity is not None
            else None,
            prompt_template_version=observability_identity.prompt.template_version
            if observability_identity is not None
            else None,
            prompt_version_hash=observability_identity.prompt.version_hash
            if observability_identity is not None
            else None,
        )

    hybrid_bundle = None
    with suppress(ExpectedAssetUnavailable):
        hybrid_bundle = build_hybrid_extraction(effective_settings)

    components = (
        ComponentReadiness(
            "corpus",
            corpus is not None,
            True,
            None if corpus is not None else "canonical corpus is unavailable",
        ),
        ComponentReadiness(
            "retrieval",
            False,
            True,
            "retrieval models are not initialized"
            if retrieval_bundle is not None
            else "retrieval indexes are unavailable",
        ),
        ComponentReadiness(
            "answer",
            False,
            True,
            "answer model is not initialized"
            if answerer is not None
            else "answer service is unavailable",
        ),
        ComponentReadiness("extraction_deterministic", True, False),
        ComponentReadiness(
            "extraction_hybrid",
            False,
            False,
            "hybrid model is not initialized"
            if hybrid_bundle is not None
            else "hybrid model is unavailable",
        ),
    )

    model_metadata: list[ModelSnapshot] = []
    if retrieval_bundle is not None:
        model_metadata.extend(
            (
                ModelSnapshot(
                    "retrieval-dense",
                    "huggingface",
                    retrieval_bundle.dense_model_id,
                    retrieval_bundle.dense_revision,
                    False,
                    False,
                ),
                ModelSnapshot(
                    "retrieval-reranker",
                    "huggingface",
                    retrieval_bundle.reranker_model_id,
                    retrieval_bundle.reranker_revision,
                    False,
                    False,
                ),
            )
        )
    if generation_bundle is not None:
        model_metadata.append(
            ModelSnapshot(
                "answer-stage-d",
                generation_bundle.provider,
                generation_bundle.model,
                generation_bundle.revision,
                False,
                False,
            )
        )
    if hybrid_bundle is not None:
        model_metadata.append(
            ModelSnapshot(
                "extraction-hybrid",
                hybrid_bundle.provider_name,
                hybrid_bundle.model,
                hybrid_bundle.revision,
                False,
                False,
            )
        )

    initializers: dict[str, Callable[[], Sequence[ModelSnapshot] | None]] = {}
    if retrieval_bundle is not None:

        def initialize_retrieval() -> Sequence[ModelSnapshot]:
            retrieval_bundle.initialize()
            return tuple(
                ModelSnapshot(item.capability, item.provider, item.model, item.revision, True, True)
                for item in model_metadata
                if item.capability.startswith("retrieval-")
            )

        initializers["retrieval"] = initialize_retrieval
    if answerer is not None and generation_bundle is not None:

        def initialize_answer() -> Sequence[ModelSnapshot]:
            if not container.component_ready("retrieval"):
                raise ExpectedAssetUnavailable("retrieval is not ready for answer serving")
            generation_bundle.initialize()
            return tuple(
                ModelSnapshot(item.capability, item.provider, item.model, item.revision, True, True)
                for item in model_metadata
                if item.capability == "answer-stage-d"
            )

        initializers["answer"] = initialize_answer
    if hybrid_bundle is not None:

        def initialize_hybrid() -> Sequence[ModelSnapshot]:
            hybrid_bundle.initialize()
            return tuple(
                ModelSnapshot(item.capability, item.provider, item.model, item.revision, True, True)
                for item in model_metadata
                if item.capability == "extraction-hybrid"
            )

        initializers["extraction_hybrid"] = initialize_hybrid

    container = ServiceContainer(
        retriever=retrieval_bundle.retriever if retrieval_bundle is not None else None,
        answerer=answerer,
        extractor=ServingExtractor(
            provider=hybrid_bundle.provider if hybrid_bundle else None,
            observer=observer,
            provider_name=hybrid_bundle.provider_name if hybrid_bundle else None,
            model=hybrid_bundle.model if hybrid_bundle else None,
            revision=hybrid_bundle.revision if hybrid_bundle else None,
        ),
        corpus=corpus,
        components=components,
        model_metadata=tuple(model_metadata),
        settings=effective_settings,
        component_initializers=initializers,
        observer=observer,
        observability_identity=observability_identity,
    )
    return container


__all__ = [
    "ComponentReadiness",
    "ExpectedAssetUnavailable",
    "HealthSnapshot",
    "ModelSnapshot",
    "ServiceContainer",
    "build_default_container",
]
