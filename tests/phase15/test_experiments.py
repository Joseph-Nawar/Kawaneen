from __future__ import annotations

import pytest

from kawaneen.phase15.counterfactuals import (
    candidate_answer_counterfactual,
    citation_counterfactual,
    is_candidate_answer,
    score_gate_sensitivity,
)
from kawaneen.phase15.dialect import DialectVariant, validate_variants_before_outcomes
from kawaneen.phase15.embedding import EmbeddingRun, create_arabic_model_lock
from kawaneen.phase15.generation import validate_allam_preflight
from kawaneen.phase15.latency import measure_latency
from kawaneen.phase15.reranking import evaluate_reranking


def test_arabic_embedding_lock_and_identity_contract() -> None:
    lock = create_arabic_model_lock("899f6e1b765915a72d5e4ace6bb2b221715550d8")
    assert lock.model_id == "omarelshehy/Arabic-Retrieval-v1.0"
    assert lock.dimension == 768
    run = EmbeddingRun(
        system="arabic",
        normalization="arabic-raw-v1",
        query_ids=("q1", "q2"),
        qrel_ids=("r1", "r2"),
        metrics={"Recall@10": (1.0, 0.0)},
    )
    run.validate_identity(("q1", "q2"), ("r1", "r2"))
    with pytest.raises(ValueError):
        run.validate_identity(("q2", "q1"), ("r1", "r2"))


def test_allam_preflight_is_strictly_4_bit() -> None:
    with pytest.raises(ValueError, match="4-bit"):
        validate_allam_preflight(
            model_id="humain-ai/ALLaM-7B-Instruct-preview",
            revision="a28dd1e67420cde72d3629c8633a974cf7d9c366",
            quantization_bits=16,
            artifact_sha256="a" * 64,
            runtime="mlx",
            device="mps",
            bounded_smoke_passed=True,
        )


def test_counterfactual_and_gate_do_not_change_serving_policy() -> None:
    result = citation_counterfactual((1, 1, 0, 0), (0, 1, 0, 0))
    assert result["absolute_risk_reduction"] == 0.25
    sensitivity = score_gate_sensitivity((0.1, 0.2, 0.3, 0.4))
    assert sensitivity["method"] == "uncalibrated score-gate sensitivity analysis"
    assert sensitivity["gates"]["bottom50"]["derived_without_relevance_labels"] is True


def test_latency_requires_three_warmups() -> None:
    with pytest.raises(ValueError):
        measure_latency(lambda: None, warmups=2)
    clock_values = iter(range(100))
    summary = measure_latency(lambda: None, samples=2, warmups=3, clock=lambda: next(clock_values))
    assert summary.batch_size == 1
    assert summary.warmups == 3


def test_dialect_validator_runs_before_retrieval_results() -> None:
    base = {
        "q": {
            "legal_intent_fingerprint": "intent",
            "qrel_fingerprint": "qrel",
            "query_text": "ما هي المادة 12 في النظام؟",
            "article_identifiers": ("المادة 12",),
            "number_identifiers": (),
            "date_identifiers": (),
        }
    }
    variants = [
        DialectVariant(
            variant_id=f"{dialect}-{i}",
            base_intent_id="q",
            dialect=dialect,
            legal_intent_fingerprint="intent",
            qrel_fingerprint="qrel",
            article_identifiers=("المادة 12",),
            text=f"صياغة {dialect} مختلفة {i}",
        )
        for dialect in ("egyptian", "gulf_saudi", "levantine")
        for i in range(20)
    ]
    validate_variants_before_outcomes(base, variants)


def test_dialect_validator_rejects_changed_article_identifiers() -> None:
    base = {
        "q": {
            "legal_intent_fingerprint": "intent",
            "qrel_fingerprint": "qrel",
            "query_text": "ما هي المادة 12؟",
            "article_identifiers": ("المادة 12",),
            "number_identifiers": (),
            "date_identifiers": (),
        }
    }
    variants = _dialect_variants(base, article_identifiers=("المادة 13",))
    with pytest.raises(ValueError, match="article identifiers"):
        validate_variants_before_outcomes(base, variants)


def test_dialect_validator_rejects_duplicates_and_base_text() -> None:
    base = {
        "q": {
            "legal_intent_fingerprint": "intent",
            "qrel_fingerprint": "qrel",
            "query_text": "ما هي المادة 12؟",
            "article_identifiers": ("المادة 12",),
            "number_identifiers": (),
            "date_identifiers": (),
        }
    }
    with pytest.raises(ValueError, match="identical"):
        validate_variants_before_outcomes(base, _dialect_variants(base, text="ما هي المادة 12؟"))
    duplicate = list(_dialect_variants(base))
    duplicate[-1] = duplicate[-1].model_copy(update={"text": duplicate[0].text})
    with pytest.raises(ValueError, match="duplicates"):
        validate_variants_before_outcomes(base, duplicate)


def _dialect_variants(
    base: dict[str, dict[str, object]],
    *,
    text: str | None = None,
    article_identifiers: tuple[str, ...] | None = None,
) -> list[DialectVariant]:
    return [
        DialectVariant(
            variant_id=f"{dialect}-{i}",
            base_intent_id="q",
            dialect=dialect,
            legal_intent_fingerprint="intent",
            qrel_fingerprint="qrel",
            article_identifiers=article_identifiers or ("المادة 12",),
            text=text or f"صياغة {dialect} مختلفة {i}",
        )
        for dialect in ("egyptian", "gulf_saudi", "levantine")
        for i in range(20)
    ]


def test_candidate_counterfactual_excludes_non_answer_json() -> None:
    raw_abstention = {"raw_output": '{"decision":"abstain"}', "result": {"decision": "abstain"}}
    candidate = {"raw_output": '{"decision":"answer"}', "result": {"decision": "answer"}}
    assert is_candidate_answer(raw_abstention) is False
    assert is_candidate_answer(candidate) is True
    result = candidate_answer_counterfactual((1,), (0,), defect_counts={"quotation": 1})
    assert result["pre_unsafe_acceptance"] == 1.0
    assert result["population_definition"].startswith("schema-parsed")


def test_hard_query_rule_uses_each_enabled_criterion_and_disables_others() -> None:
    from kawaneen.phase15.reranking import HardQueryRule, select_hard_queries

    fields = (
        "multi_evidence",
        "exact_provision",
        "authority",
        "deadline",
        "cross_language",
        "long_query",
    )
    for field in fields:
        row = {"id": field, **{name: False for name in fields}, "pre_rerank_relevant_rank": 1}
        row[field] = True
        rule = HardQueryRule(
            **{name: name == field for name in fields},
            min_pre_rerank_relevant_rank_for_hard=20,
        )
        assert select_hard_queries((row,), rule=rule) == (field,)
    rank_row = {"id": "rank", **{name: False for name in fields}, "pre_rerank_relevant_rank": 20}
    assert select_hard_queries(
        (rank_row,), rule=HardQueryRule(**{name: False for name in fields})
    ) == ("rank",)


def test_reranking_returns_paired_effects() -> None:
    result = evaluate_reranking({"nDCG@10": (0.1, 0.2)}, {"nDCG@10": (0.2, 0.2)})
    assert result["metrics"]["nDCG@10"]["rank_biserial"] == 0.5
