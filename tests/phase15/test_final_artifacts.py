from __future__ import annotations

import json
from pathlib import Path

from kawaneen.phase15.contracts import ErrorCategory, ReviewOutcome
from kawaneen.phase15.figures import build_report_figures
from kawaneen.phase15.final_artifacts import (
    build_error_analysis,
    build_research_questions,
    write_final_aggregates,
)


def _audit_payload() -> dict[str, object]:
    return {
        "population_hash": "population",
        "audit_hash": "audit",
        "summary": {
            "initial_model_vs_rule_based_audit_category_agreement": {
                "comparable_category_count": 1,
                "agreement_count": 1,
                "agreement_rate": 1.0,
                "disagreement_counts": {},
            },
            "prior_ai_unavailable_count": 0,
        },
        "cases": [
            {
                "pipeline_stage": "retrieval",
                "language": "ar",
                "legal_category": "deadline",
                "adjudication": {
                    "outcome": ReviewOutcome.CONFIRMED_FAILURE.value,
                    "primary_category": ErrorCategory.LEXICAL_MISMATCH.value,
                    "failure_mode": None,
                    "confidence": 4,
                },
            },
            {
                "pipeline_stage": "normalization",
                "language": "en",
                "legal_category": "definition",
                "adjudication": {
                    "outcome": ReviewOutcome.BORDERLINE_NO_CONFIRMED_FAILURE.value,
                    "primary_category": None,
                    "failure_mode": None,
                    "confidence": 3,
                },
            },
        ],
    }


def test_final_aggregate_builders_are_text_free_and_have_seven_questions(tmp_path: Path) -> None:
    content = {
        "total_count": 60,
        "valid_count": 59,
        "invalid_count": 1,
        "valid_variant_ids_sha256": "valid",
    }
    (tmp_path / "data/evaluation").mkdir(parents=True)
    (tmp_path / "data/evaluation/phase15_dialect_content_validity.json").write_text(
        json.dumps(content), encoding="utf-8"
    )
    error = build_error_analysis(_audit_payload(), content_audit=content)
    assert error["confirmed_failure_taxonomy"] == {ErrorCategory.LEXICAL_MISMATCH.value: 1}
    assert error["borderline_count"] == 1
    questions = build_research_questions()
    assert questions["frozen_count"] == 7
    assert len(questions["research_questions"]) == 7
    error_path, question_path = write_final_aggregates(tmp_path, _audit_payload())
    assert error_path.is_file() and question_path.is_file()


def test_report_figures_use_only_aggregate_inputs(tmp_path: Path) -> None:
    evaluation = tmp_path / "data/evaluation"
    evaluation.mkdir(parents=True)
    (evaluation / "phase5_chunking_metrics.json").write_text(
        json.dumps(
            {
                "retrieval_metrics": {
                    "fixed-256-v1": {"ndcg_at_10": 0.1},
                    "legal-structure-v1": {"ndcg_at_10": 0.2},
                }
            }
        ),
        encoding="utf-8",
    )
    (evaluation / "phase15_latency_metrics.json").write_text(
        json.dumps(
            {
                "operations": {
                    "BM25": {"quality": {"nDCG@10": {"mean": 0.1}}},
                    "Arabic": {"quality": {"nDCG@10": {"mean": 0.2}}},
                }
            }
        ),
        encoding="utf-8",
    )
    (evaluation / "phase15_dialect_metrics.json").write_text(
        json.dumps(
            {
                "dialects": {
                    name: {"hybrid": {"dialect_minus_msa": {"Recall@10": {"delta": 0.0}}}}
                    for name in ("egyptian", "gulf_saudi", "levantine")
                },
                "pooled": {"hybrid": {"dialect_minus_msa": {"Recall@10": {"delta": 0.0}}}},
            }
        ),
        encoding="utf-8",
    )
    (evaluation / "phase15_generator_metrics.json").write_text(
        json.dumps(
            {
                "metrics": {
                    "SupportedAnswerCoverage": {"value": 0.0},
                    "invalid_generation_rate": {"value": 1.0},
                }
            }
        ),
        encoding="utf-8",
    )
    (evaluation / "phase15_error_analysis.json").write_text(
        json.dumps(
            {
                "confirmed_failure_taxonomy": {"semantic retrieval failure": 1},
                "non_taxonomy_failure_modes": {"INVALID_GENERATION_CONTRACT": 1},
                "borderline_count": 1,
            }
        ),
        encoding="utf-8",
    )
    (evaluation / "phase15_citation_counterfactual.json").write_text(
        json.dumps({"pre_defect_surface_rate": 0.7, "post_defect_surface_rate": 0.0}),
        encoding="utf-8",
    )
    paths = build_report_figures(tmp_path)
    assert len(paths) == 6
    assert all(path.suffix == ".svg" and path.stat().st_size > 0 for path in paths)
