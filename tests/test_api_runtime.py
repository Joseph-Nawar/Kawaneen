from __future__ import annotations

import json
from pathlib import Path

import pytest

from kawaneen.core.config import Settings


def test_service_container_initializes_once_and_cleans_up_once() -> None:
    from kawaneen.api.runtime import ServiceContainer

    counts = {"init": 0, "close": 0, "loads": 0}

    def initialize() -> None:
        counts["init"] += 1

    def close() -> None:
        counts["close"] += 1

    container = ServiceContainer(
        initializer=initialize,
        closer=close,
        capabilities=lambda: (counts.__setitem__("loads", counts["loads"] + 1), ())[1],
    )
    container.initialize()
    container.initialize()
    assert counts["init"] == 1
    assert container.health().status == "degraded"

    container.close()
    container.close()
    assert counts["close"] == 1
    assert counts["loads"] == 0
    assert container.models() == ()


def test_ready_container_reports_required_capabilities_without_loading_models() -> None:
    from kawaneen.api.runtime import ComponentReadiness, ServiceContainer

    container = ServiceContainer(
        components=(ComponentReadiness("corpus", True, True),),
        capabilities=lambda: (),
    )
    container.initialize()

    assert container.health().status == "ready"
    assert container.models() == ()


def test_default_container_retains_settings_for_real_composition(tmp_path: Path) -> None:
    from kawaneen.api.runtime import build_default_container

    settings = Settings(
        data_directory=tmp_path / "data",
        artifacts_directory=tmp_path / "artifacts",
    )

    container = build_default_container(settings)

    assert container.settings is settings


def test_default_serving_configuration_loads_authoritative_phase8_selection() -> None:
    from kawaneen.api.composition import load_frozen_serving_configuration

    configuration = load_frozen_serving_configuration(Path("data"))

    assert configuration.fusion.sparse_weight == 1.0
    assert configuration.fusion.dense_weight == 0.25
    assert configuration.fusion.sparse_top_k == 50
    assert configuration.fusion.dense_top_k == 50
    assert configuration.fusion.candidate_k == 20
    assert configuration.reranker.serving_depth == 8
    assert configuration.reranker.scoring_contract == "raw-logit-v1"
    assert configuration.dense_model_revision == ("5617a9f61b028005a4858fdac845db406aefb181")


def test_component_initializers_degrade_only_the_failed_capability() -> None:
    from kawaneen.api.runtime import ComponentReadiness, ExpectedAssetUnavailable, ServiceContainer

    container = ServiceContainer(
        components=(
            ComponentReadiness("retrieval", False, True),
            ComponentReadiness("answer", False, True),
            ComponentReadiness("extraction_deterministic", True, False),
        ),
        component_initializers={
            "retrieval": lambda: (_ for _ in ()).throw(
                ExpectedAssetUnavailable("retrieval model absent")
            ),
            "answer": lambda: (),
        },
    )

    container.initialize()
    status = {item.name: item for item in container.health().components}

    assert status["retrieval"].ready is False
    assert status["answer"].ready is True
    assert status["extraction_deterministic"].ready is True


def test_default_container_composes_injected_serving_adapters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kawaneen.api import composition
    from kawaneen.api.runtime import build_default_container
    from kawaneen.generation.policy import JurisdictionScope

    private_corpus = tmp_path / "private" / "phase6_evaluation" / "ai-reviewed-v1" / "corpus"
    private_corpus.mkdir(parents=True)
    (private_corpus / "canonical_units.json").write_text(
        json.dumps(
            {
                "summary": {"corpus_hash": "a" * 64},
                "units": [
                    {
                        "unit_id": "u1",
                        "document_id": "d1",
                        "ordinal": 1,
                        "text": "النص القانوني",
                        "unit_type": "events",
                        "provenance": {
                            "source_id": "fixture",
                            "source_version": "v1",
                            "source_path": "fixture",
                            "source_row": 1,
                            "source_field": "text",
                            "split": "api",
                        },
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    retrieval_corpus = tmp_path / "private" / "phase7_retrieval" / "corpus"
    retrieval_corpus.mkdir(parents=True)
    (retrieval_corpus / "chunks.jsonl").write_text(
        json.dumps(
            {
                "chunk_id": "c1",
                "source_unit_ids": ["u1"],
                "source_spans": [{"unit_id": "u1", "start": 0, "end": 10}],
            }
        ),
        encoding="utf-8",
    )

    configuration = composition.load_frozen_serving_configuration(Path("data"))

    class FakeProvider:
        def propose(self, canonical_text: str, registry: object) -> object:
            return {}

    retrieval_bundle = composition.ServingRetrievalBundle(
        retriever=lambda query, limit=8: object(),
        initialize=lambda: None,
        dense_model_id="fixture/dense",
        dense_revision="dense-revision",
        reranker_model_id="fixture/reranker",
        reranker_revision="reranker-revision",
    )
    generation_bundle = composition.ServingGenerationBundle(
        generator=lambda query, context: None,
        initialize=lambda: None,
        provider="fixture",
        model="fixture-answer",
        revision="answer-revision",
    )
    extraction_bundle = composition.ServingExtractionBundle(
        provider=FakeProvider(),
        initialize=lambda: None,
        provider_name="fixture",
        model="fixture-extraction",
        revision="extraction-revision",
    )
    monkeypatch.setattr(composition, "load_frozen_serving_configuration", lambda _: configuration)
    monkeypatch.setattr(
        composition, "build_serving_retrieval", lambda settings, config: retrieval_bundle
    )
    monkeypatch.setattr(composition, "build_stage_d_generation", lambda settings: generation_bundle)
    monkeypatch.setattr(composition, "build_hybrid_extraction", lambda settings: extraction_bundle)
    monkeypatch.setattr(
        "kawaneen.generation.answerability.load_source_eligibility_registry",
        lambda path: {},
    )
    monkeypatch.setattr(
        "kawaneen.generation.answerability.load_structural_roles",
        lambda path: {},
    )
    monkeypatch.setattr(
        "kawaneen.generation.policy.default_deployment_scope",
        lambda path: JurisdictionScope(
            active_jurisdiction="SA",
            authoritative_jurisdiction="SA",
            allowed_jurisdictions=("SA",),
            mode="single",
            required=True,
        ),
    )

    container = build_default_container(
        Settings(data_directory=Path("data"), artifacts_directory=tmp_path)
    )
    container.initialize()

    assert container.health().status == "ready"
    assert {item.capability for item in container.models()} == {
        "retrieval-dense",
        "retrieval-reranker",
        "answer-stage-d",
        "extraction-hybrid",
    }
