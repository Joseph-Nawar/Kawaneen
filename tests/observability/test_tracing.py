from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from kawaneen.core.config import Settings
from kawaneen.generation.policy import PolicyOutcome
from kawaneen.grounding.contracts import GeneratedDraft, VerificationResult
from kawaneen.observability.identity import ServingIdentity
from kawaneen.observability.tracing import (
    NoOpObserver,
    create_observer,
    root_attributes,
)
from kawaneen.retrieval.hybrid.contracts import FusionConfig, SourceHit
from kawaneen.retrieval.models import RetrievalChunk
from kawaneen.retrieval.serving import (
    HybridServingRetriever,
    ServingEvidence,
    ServingRetrievalResult,
    ServingRetrievalSummary,
)

ROOT = Path(__file__).parents[2]


@dataclass
class RecordedSpan:
    name: str
    span_type: str
    attributes: dict[str, object]
    outputs: list[dict[str, object]] = field(default_factory=list)
    error: BaseException | None = None

    def __enter__(self) -> RecordedSpan:
        return self

    def __exit__(self, exc_type: object, exc: BaseException | None, tb: object) -> bool:
        if exc is not None:
            self.error = exc
        return False

    def set_outputs(self, values: dict[str, object]) -> None:
        self.outputs.append(values)

    def set_attributes(self, values: dict[str, object]) -> None:
        self.attributes.update(values)

    def record_exception(self, error: BaseException) -> None:
        self.error = error


class RecordingObserver:
    def __init__(self) -> None:
        self.spans: list[RecordedSpan] = []

    def root(self, name: str, attributes: dict[str, object]) -> RecordedSpan:
        span = RecordedSpan(name, "CHAIN", dict(attributes))
        self.spans.append(span)
        return span

    def span(self, name: str, span_type: str, attributes: dict[str, object]) -> RecordedSpan:
        span = RecordedSpan(name, span_type, dict(attributes))
        self.spans.append(span)
        return span


def _retriever(observer: Any) -> HybridServingRetriever:
    chunks = {
        f"chunk-{index}": RetrievalChunk(
            chunk_id=f"chunk-{index}",
            document_id=f"doc-{index}",
            source_id="fixture",
            unit_type="article",
            display_text=f"{index} legal evidence sentinel",
            search_text=f"{index} query sentinel",
            source_unit_ids=(f"unit-{index}",),
            chunk_policy_hash="a" * 64,
            normalization_policy_id="arabic-light-v1",
            normalization_policy_hash="b" * 64,
            token_count=2,
        )
        for index in range(1, 4)
    }
    return HybridServingRetriever(
        chunks=chunks,
        sparse_search=lambda query, top_k: tuple(
            SourceHit(f"chunk-{index}", 1.0 / index) for index in range(1, 4)
        ),
        dense_search=lambda query, top_k: tuple(
            SourceHit(f"chunk-{index}", 1.0 / index) for index in range(1, 4)
        ),
        reranker=lambda query, candidates: {
            candidate.chunk_id: float(4 - candidate.fused_rank) for candidate in candidates
        },
        fusion_config=FusionConfig(sparse_weight=1.0, dense_weight=0.25),
        observer=observer,
    )


def test_noop_observer_does_not_alter_retrieval_behavior() -> None:
    without = _retriever(None).search("query sentinel", limit=2)
    with_noop = _retriever(NoOpObserver()).search("query sentinel", limit=2)

    assert with_noop == without


def test_retrieval_spans_capture_ids_and_all_scores_without_text() -> None:
    observer = RecordingObserver()
    result = _retriever(observer).search("query sentinel", limit=2)

    assert [span.name for span in observer.spans] == ["retrieval.first_stage", "retrieval.rerank"]
    first, rerank = observer.spans
    assert first.span_type == "RETRIEVER"
    assert first.outputs == [{"ordered_fused_candidate_ids": ["chunk-1", "chunk-2", "chunk-3"]}]
    assert rerank.span_type == "RERANKER"
    assert len(rerank.outputs[0]["scores"]) == 3
    assert rerank.outputs[0]["returned_chunk_ids"] == [item.chunk_id for item in result.evidence]
    serialized = repr(observer.spans)
    assert "query sentinel" not in serialized
    assert "legal evidence sentinel" not in serialized


def test_root_attributes_use_existing_request_id_and_identity() -> None:
    identity = ServingIdentity.build(ROOT / "data")
    attributes = root_attributes(identity, request_id="client:req-1", operation="answer")

    assert attributes["kawaneen.request_id"] == "client:req-1"
    assert attributes["kawaneen.configuration_version"] == identity.configuration_version
    assert "query" not in repr(attributes)


def test_synchronous_search_root_uses_existing_request_id() -> None:
    from kawaneen.api.routers import _observed_search
    from kawaneen.api.runtime import ComponentReadiness, ServiceContainer

    observer = RecordingObserver()
    container = ServiceContainer(
        retriever=_retriever(observer),
        observer=observer,
        observability_identity=ServingIdentity.build(ROOT / "data"),
        components=(ComponentReadiness("retrieval", True, True),),
    )

    _observed_search(container, "middleware:req-7", "query sentinel", "SA", 2)

    assert observer.spans[-3].name == "kawaneen.search"
    assert observer.spans[-3].attributes["kawaneen.request_id"] == "middleware:req-7"


def test_disabled_import_path_does_not_require_mlflow(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "mlflow", raising=False)
    observer = create_observer(Settings(_env_file=None))

    assert isinstance(observer, NoOpObserver)
    assert "mlflow" not in sys.modules


def test_enabled_observability_reports_missing_mlflow_clearly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kawaneen.observability import tracing

    monkeypatch.setattr(tracing, "_import_mlflow", lambda: (_ for _ in ()).throw(ImportError()))

    with pytest.raises(RuntimeError, match="observability dependency group"):
        create_observer(
            Settings(
                _env_file=None,
                observability_enabled=True,
            ),
            ServingIdentity.build(ROOT / "data"),
        )


def test_mlflow_observer_uses_manual_spans_and_isolates_write_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kawaneen.observability import tracing

    class FakeLiveSpan:
        def __init__(self, *, fail_writes: bool = False) -> None:
            self.fail_writes = fail_writes
            self.attributes: dict[str, object] = {}

        def set_outputs(self, values: object) -> None:
            if self.fail_writes:
                raise OSError("telemetry write failed")
            self.attributes["outputs"] = values

        def set_attributes(self, values: object) -> None:
            if self.fail_writes:
                raise OSError("telemetry write failed")
            self.attributes.update(values)  # type: ignore[union-attr]

        def record_exception(self, error: BaseException) -> None:
            self.attributes["error"] = type(error).__name__

    class FakeContext:
        def __init__(self, span: FakeLiveSpan) -> None:
            self.span = span

        def __enter__(self) -> FakeLiveSpan:
            return self.span

        def __exit__(self, exc_type: object, exc: BaseException | None, tb: object) -> bool:
            return False

    class FakeClient:
        def get_experiment_by_name(self, name: str) -> None:
            return None

        def create_experiment(self, name: str) -> str:
            return "1"

    class FakeMlflow:
        class entities:
            class SpanType:
                CHAIN = "CHAIN"
                RETRIEVER = "RETRIEVER"

        __version__ = "3.15.2"

        def __init__(self) -> None:
            self.client = FakeClient()

        def set_tracking_uri(self, uri: str) -> None:
            assert uri == "http://fake"

        def set_experiment(self, name: str) -> None:
            assert name == "fake-serving"

        def MlflowClient(self, tracking_uri: str) -> FakeClient:
            return self.client

        def start_span(self, **kwargs: object) -> FakeContext:
            return FakeContext(FakeLiveSpan(fail_writes=kwargs["name"] == "failing"))

    monkeypatch.setattr(tracing, "_import_mlflow", lambda: FakeMlflow())
    settings = Settings(
        _env_file=None,
        observability_enabled=True,
        mlflow_tracking_uri="http://fake",
        mlflow_serving_experiment="fake-serving",
    )
    observer = create_observer(settings, ServingIdentity.build(ROOT / "data"))

    with observer.root("root", {"request_id": "request-id"}) as root:
        root.set_outputs({"status": "ok"})
        with observer.span("child", "RETRIEVER", {}) as child:
            child.set_attributes({"one": 1})
            child.set_outputs({"status": "ok"})
    with observer.span("failing", "CHAIN", {}) as failing:
        failing.set_outputs({"status": "still-returned"})

    assert True


def test_answer_span_records_policy_abstention_and_not_run_stages() -> None:
    from kawaneen.generation.serving import ServingAnswerer

    retrieval = ServingRetrievalResult(
        evidence=(
            ServingEvidence(
                chunk_id="chunk-1",
                rank=1,
                text="legal evidence sentinel",
                document_id="doc-1",
                document_title=None,
                article=None,
                page=None,
                source_url=None,
                score=1.0,
            ),
        ),
        summary=ServingRetrievalSummary(returned_count=1),
    )

    observer = RecordingObserver()
    answerer = ServingAnswerer(
        retriever=lambda query, limit=8: retrieval,
        context_builder=lambda query, retrieval: object(),
        policy_evaluator=lambda query, context: PolicyOutcome(allowed=False),
        generator=lambda query, context: pytest.fail("generator should not run"),
        verifier=lambda context, draft: pytest.fail("verifier should not run"),
        observer=observer,
    )

    result = answerer.answer("query sentinel")

    assert result.answerable is False
    assert observer.spans[-3].name == "answerability.policy"
    assert observer.spans[-2].outputs == [{"status": "not_run_policy_abstention"}]
    assert observer.spans[-1].outputs == [{"status": "not_run_policy_abstention"}]
    assert "query sentinel" not in repr(observer.spans)


def _answerer_for_trace(
    observer: RecordingObserver,
    *,
    generator: Any,
    verifier: Any,
    policy: PolicyOutcome | None = None,
    generator_metadata: dict[str, str] | None = None,
) -> Any:
    from kawaneen.generation.serving import ServingAnswerer

    retrieval = ServingRetrievalResult(
        evidence=(),
        summary=ServingRetrievalSummary(),
    )
    return ServingAnswerer(
        retriever=lambda query, limit=8: retrieval,
        context_builder=lambda query, result: object(),
        policy_evaluator=lambda query, context: policy or PolicyOutcome(allowed=True),
        generator=generator,
        verifier=verifier,
        observer=observer,
        **(generator_metadata or {}),
    )


def test_successful_answer_has_complete_stage_statuses() -> None:
    observer = RecordingObserver()
    result = _answerer_for_trace(
        observer,
        generator=lambda query, context: GeneratedDraft(answer_text="answer sentinel", claims=()),
        verifier=lambda context, draft: VerificationResult(
            valid_citations=(),
            invalid_citations=(),
            unsupported_claims=(),
            structurally_valid=True,
            should_abstain=False,
        ),
        generator_metadata={
            "generator_provider": "ollama",
            "generator_model": "qwen3:test",
            "generator_revision": "revision-1",
            "prompt_template_version": "prompt-v1",
            "prompt_version_hash": "hash-1",
        },
    ).answer("query sentinel")

    assert result.answerable is True
    assert observer.spans[-2].outputs == [
        {"status": "generated", "decision": "answer", "claim_count": 0}
    ]
    assert observer.spans[-1].outputs[0]["status"] == "passed"
    assert observer.spans[-2].attributes == {
        "provider": "ollama",
        "model": "qwen3:test",
        "revision": "revision-1",
        "prompt_template_version": "prompt-v1",
        "prompt_version_hash": "hash-1",
    }
    assert "answer sentinel" not in repr(observer.spans)


def test_unavailable_generation_is_marked_and_domain_error_is_preserved() -> None:
    from kawaneen.generation.serving import GenerationModelUnavailableError

    observer = RecordingObserver()
    error = GenerationModelUnavailableError("provider unavailable")
    answerer = _answerer_for_trace(
        observer,
        generator=lambda query, context: (_ for _ in ()).throw(error),
        verifier=lambda context, draft: pytest.fail("verifier should not run"),
    )

    with pytest.raises(GenerationModelUnavailableError) as raised:
        answerer.answer("query sentinel")

    assert raised.value is error
    assert observer.spans[-1].outputs == [{"status": "unavailable"}]


def test_model_abstention_marks_generation_and_verification_not_run() -> None:
    observer = RecordingObserver()
    result = _answerer_for_trace(
        observer,
        generator=lambda query, context: None,
        verifier=lambda context, draft: pytest.fail("verifier should not run"),
    ).answer("query sentinel")

    assert result.abstention_reason == "MODEL_ABSTENTION"
    assert observer.spans[-2].outputs == [{"status": "not_run_model_abstention"}]
    assert observer.spans[-1].outputs == [{"status": "not_run_model_abstention"}]


def test_invalid_generation_marks_generation_invalid_and_verification_not_run() -> None:
    observer = RecordingObserver()
    result = _answerer_for_trace(
        observer,
        generator=lambda query, context: (_ for _ in ()).throw(ValueError("invalid")),
        verifier=lambda context, draft: pytest.fail("verifier should not run"),
    ).answer("query sentinel")

    assert result.abstention_reason == "INVALID_GENERATION"
    assert observer.spans[-2].outputs == [{"status": "invalid_generation"}]
    assert observer.spans[-1].outputs == [{"status": "not_run_invalid_generation"}]


def test_citation_verification_failure_is_failed_closed() -> None:
    observer = RecordingObserver()
    result = _answerer_for_trace(
        observer,
        generator=lambda query, context: GeneratedDraft(answer_text="answer", claims=()),
        verifier=lambda context, draft: VerificationResult(
            valid_citations=(),
            invalid_citations=(),
            unsupported_claims=("claim",),
            structurally_valid=False,
            should_abstain=True,
        ),
    ).answer("query sentinel")

    assert result.answerable is False
    assert observer.spans[-1].outputs[0]["status"] == "failed_closed"


def test_extraction_trace_captures_mode_and_capability_without_text() -> None:
    from kawaneen.extraction.serving import ServingExtractor

    observer = RecordingObserver()
    response = ServingExtractor(observer=observer).extract(
        "extraction sentinel", mode="deterministic"
    )

    assert response.capability_status == "operational_candidates"
    assert observer.spans[0].name == "extraction"
    assert observer.spans[0].attributes["mode"] == "deterministic"
    assert observer.spans[0].outputs == [{"capability_status": "operational_candidates"}]
    assert "extraction sentinel" not in repr(observer.spans)


def test_domain_exception_is_re_raised_unchanged() -> None:
    from kawaneen.generation.serving import ServingAnswerer

    error = RuntimeError("domain sentinel failure")
    observer = RecordingObserver()
    answerer = ServingAnswerer(
        retriever=lambda query, limit=8: _retriever(observer).search(query),
        context_builder=lambda query, retrieval: (_ for _ in ()).throw(error),
        policy_evaluator=lambda query, context: PolicyOutcome(allowed=True),
        generator=lambda query, context: GeneratedDraft(answer_text="", claims=()),
        verifier=lambda context, draft: VerificationResult(
            valid_citations=(),
            invalid_citations=(),
            unsupported_claims=(),
            structurally_valid=False,
            should_abstain=True,
        ),
        observer=observer,
    )

    with pytest.raises(RuntimeError) as raised:
        answerer.answer("query sentinel")
    assert raised.value is error
    assert any(span.error is error for span in observer.spans)
