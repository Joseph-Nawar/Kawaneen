from __future__ import annotations

import json

import pytest

from kawaneen.generation.artifacts import artifact_fingerprint, write_text_free_artifact
from kawaneen.generation.budgeting import BudgetedContext
from kawaneen.generation.checkpoints import CheckpointManifest, load_checkpoint, write_checkpoint
from kawaneen.generation.contracts import (
    AbstentionReason,
    GenerationDecision,
    GenerationResult,
)
from kawaneen.generation.evaluation import evaluate_budget_report, evaluate_generation_results
from kawaneen.generation.experiments import ExperimentSpec, prepare_experiment


def test_checkpoint_write_is_atomic_and_round_trips(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "checkpoint.json"
    manifest = CheckpointManifest(
        checkpoint_id="run-1",
        schema_version=1,
        generator_name="extractive",
        completed_query_ids=("q1",),
        artifact_fingerprint="a" * 64,
    )

    write_checkpoint(path, manifest)

    assert load_checkpoint(path) == manifest
    assert json.loads(path.read_text(encoding="utf-8"))["checkpoint_id"] == "run-1"


def test_text_free_artifact_rejects_source_text_and_quotes(tmp_path) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError):
        write_text_free_artifact(tmp_path / "bad.json", {"quoted_text": "source"})

    path = tmp_path / "good.json"
    write_text_free_artifact(path, {"count": 2, "status": "complete"})

    assert json.loads(path.read_text(encoding="utf-8")) == {"count": 2, "status": "complete"}


def test_artifact_fingerprint_is_stable_and_evaluation_is_text_free() -> None:
    assert artifact_fingerprint({"b": 2, "a": 1}) == artifact_fingerprint({"a": 1, "b": 2})

    report = evaluate_generation_results(
        (
            GenerationResult(
                decision=GenerationDecision.ABSTAIN, abstention_reason=AbstentionReason.NO_CONTEXT
            ),
            GenerationResult(
                decision=GenerationDecision.ABSTAIN,
                abstention_reason=AbstentionReason.INVALID_GENERATION,
            ),
        )
    )

    assert report == {
        "result_count": 2,
        "answer_count": 0,
        "abstain_count": 2,
        "abstention_reasons": {"INVALID_GENERATION": 1, "NO_CONTEXT": 1},
    }


def test_experiment_preparation_never_invokes_a_generator() -> None:
    spec = ExperimentSpec(
        experiment_id="stage-a-smoke",
        generator_names=("extractive", "qwen3-local"),
        prompt_version="phase10-prompt-template-v1",
        output_schema_version="phase10-output-schema-v1",
    )

    plan = prepare_experiment(spec)

    assert plan.experiment_id == "stage-a-smoke"
    assert plan.generator_names == spec.generator_names
    assert plan.execution_allowed is False


def test_budget_report_is_per_generator_and_text_free() -> None:
    report = evaluate_budget_report(
        {
            "extractive": BudgetedContext(
                context_pack=None,  # type: ignore[arg-type]
                tokenizer_identity="codepoint-v1",
                non_evidence_prompt_tokens=10,
                evidence_token_count=20,
                prompt_token_count=30,
                evidence_budget_tokens=25,
                omitted_unit_ids=("u2",),
                gold_evidence_retention=0.5,
                complete_gold_evidence_retention=0.0,
            )
        }
    )

    assert report["extractive"]["prompt_token_count"] == 30
    assert report["extractive"]["omitted_unit_count"] == 1
