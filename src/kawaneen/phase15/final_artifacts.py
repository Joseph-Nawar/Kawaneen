"""Reproducible text-free final Phase 15 aggregate builders."""

# Long evidence statements are serialized into tracked JSON verbatim.
# ruff: noqa: E501

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .contracts import ErrorCategory, ReviewOutcome
from .evidence import RESEARCH_QUESTIONS
from .reporting import write_aggregate_artifact


def build_error_analysis(
    audit_payload: Mapping[str, Any], *, content_audit: Mapping[str, Any]
) -> dict[str, Any]:
    records = tuple(audit_payload.get("cases", ()))
    confirmed = tuple(
        record
        for record in records
        if record.get("adjudication", {}).get("outcome") == ReviewOutcome.CONFIRMED_FAILURE.value
    )
    taxonomy = Counter(
        str(record["adjudication"]["primary_category"])
        for record in confirmed
        if record["adjudication"].get("primary_category")
        in {category.value for category in ErrorCategory}
    )
    failure_modes = Counter(
        str(record["adjudication"]["failure_mode"])
        for record in confirmed
        if record["adjudication"].get("failure_mode") is not None
    )

    def distribution(field: str) -> dict[str, int]:
        return dict(
            sorted(Counter(str(record.get(field, "unknown")) for record in records).items())
        )

    return {
        "schema_version": "phase15-error-analysis-v1",
        "methodology_label": "AUTOMATED_ADJUDICATION_DIAGNOSTIC",
        "provenance": "PHASE15_DEV",
        "population_hash": audit_payload["population_hash"],
        "audit_hash": audit_payload["audit_hash"],
        "audit_count": len(records),
        "outcome_distribution": dict(
            sorted(Counter(str(record["adjudication"]["outcome"]) for record in records).items())
        ),
        "confirmed_failure_count": len(confirmed),
        "confirmed_failure_taxonomy": dict(sorted(taxonomy.items())),
        "non_taxonomy_failure_modes": dict(sorted(failure_modes.items())),
        "borderline_count": sum(
            record["adjudication"]["outcome"] == ReviewOutcome.BORDERLINE_NO_CONFIRMED_FAILURE.value
            for record in records
        ),
        "uncertain_count": sum(
            record["adjudication"]["outcome"] == ReviewOutcome.UNCERTAIN.value for record in records
        ),
        "confidence_distribution": dict(
            sorted(Counter(str(record["adjudication"]["confidence"]) for record in records).items())
        ),
        "pipeline_stage_distribution": distribution("pipeline_stage"),
        "language_distribution": distribution("language"),
        "legal_category_distribution": distribution("legal_category"),
        "limitations": [
            "This is an operational diagnostic taxonomy, not causal proof or population prevalence.",
            "The 30-case subset is enriched/preselected and has no human or expert gold labels.",
            "Contract failures are reported separately from the 12 root-cause categories.",
        ],
        "prior_ai_comparison": audit_payload["summary"][
            "initial_model_vs_rule_based_audit_category_agreement"
        ],
        "prior_ai_unavailable_count": audit_payload["summary"]["prior_ai_unavailable_count"],
        "dialect_content_audit": {
            "total_count": content_audit["total_count"],
            "valid_count": content_audit["valid_count"],
            "invalid_count": content_audit["invalid_count"],
            "valid_variant_ids_sha256": content_audit["valid_variant_ids_sha256"],
        },
        "human_labels_present": False,
        "expert_gold_present": False,
    }


def build_research_questions() -> dict[str, Any]:
    refs = {
        1: ["data/evaluation/phase5_chunking_metrics.json"],
        2: [
            "data/evaluation/phase4_normalization_metrics.json",
            "data/evaluation/phase15_embedding_metrics.json",
        ],
        3: ["data/evaluation/phase8_dev_fusion_metrics.json"],
        4: ["data/evaluation/phase15_reranking_metrics.json"],
        5: [
            "data/evaluation/phase15_dialect_metrics.json",
            "data/evaluation/phase15_dialect_content_validity.json",
        ],
        6: [
            "data/evaluation/phase15_citation_counterfactual.json",
            "data/evaluation/phase10_qwen_stage_d_metrics.json",
        ],
        7: [
            "data/evaluation/phase15_generator_metrics.json",
            "data/evaluation/phase10_qwen_stage_d_metrics.json",
        ],
    }
    entries = [
        {
            "question": RESEARCH_QUESTIONS[0],
            "status": "SUPPORTED",
            "population": "Phase 5 150-query chunking challenge",
            "provenance": "HISTORICAL_FROZEN",
            "primary_evidence": "legal-structure-v1 exceeded fixed-256-v1 on Recall@10, MRR@10, nDCG@10 and citation precision.",
            "effect_and_ci": "No Phase 15 re-estimation; see frozen paired intervals in the Phase 5 artifact.",
            "limitations": "Historical challenge and tracked aggregate; not a production correctness guarantee.",
            "artifact_refs": refs[1],
        },
        {
            "question": RESEARCH_QUESTIONS[1],
            "status": "PARTIALLY_SUPPORTED",
            "population": "Phase 4 normalization challenge plus 141-query Phase 15 embedding DEV",
            "provenance": "HISTORICAL_FROZEN and PHASE15_DEV",
            "primary_evidence": "Phase 4 raw/light/aggressive aggregates are identical; Arabic-Retrieval raw Recall@10 is 0.1418 versus 0.1241 light and 0.1206 aggressive.",
            "effect_and_ci": "Arabic-Retrieval raw versus BGE-M3 raw Recall@10 delta +0.0461, 95% CI [-0.0106, 0.1028]; raw versus light/aggressive paired deltas are negative on Recall@10.",
            "limitations": "Effects differ by corpus/model and are DEV-only; no normalization policy promotion follows.",
            "artifact_refs": refs[2],
        },
        {
            "question": RESEARCH_QUESTIONS[2],
            "status": "INCONCLUSIVE",
            "population": "Phase 8 DEV fusion evaluation",
            "provenance": "HISTORICAL_FROZEN",
            "primary_evidence": "RRF improves aggregate MRR@10 and nDCG@10 over dense-only, but the frozen question requires a demonstrated Arabic-and-English claim and tracked aggregate does not establish that language-stratified comparison.",
            "effect_and_ci": "Aggregate RRF MRR@10 0.1410 versus dense 0.0620; no unsupported language-specific CI is asserted.",
            "limitations": "Language-stratified hybrid evidence is insufficient for the exact wording of the question.",
            "artifact_refs": refs[3],
        },
        {
            "question": RESEARCH_QUESTIONS[3],
            "status": "SUPPORTED",
            "population": "46-query corrected Phase 15 hard DEV slice",
            "provenance": "PHASE15_DEV",
            "primary_evidence": "Frozen hybrid plus BGE reranking has positive paired deltas on all four reported hard-slice metrics.",
            "effect_and_ci": "Recall@10 delta +0.0435, 95% CI [0.0000, 0.1087], 2 wins/44 ties/0 losses; MRR@10 delta +0.0101, CI [0.0000, 0.0275], 3/43/0.",
            "limitations": "Enriched deterministic hard slice; operational retrieval gain, not legal-answer correctness.",
            "artifact_refs": refs[4],
        },
        {
            "question": RESEARCH_QUESTIONS[4],
            "status": "INCONCLUSIVE",
            "population": "59 content-valid AI-generated dialect variants paired to MSA controls; 53 answerable metric pairs pooled",
            "provenance": "PHASE15_DEV",
            "primary_evidence": "Effects are small and mixed: pooled hybrid Recall@10 +0.0189, reranked +0.0377, while pooled BGE MRR@10 is -0.0031; dialect text is diagnostic, not human dialect gold.",
            "effect_and_ci": "See valid-pair bootstrap intervals by dialect/system; one malformed Gulf/Saudi concatenation was excluded before analysis.",
            "limitations": "AI-generated/AI-validated perturbations, unequal effective counts, and no human dialect validation.",
            "artifact_refs": refs[5],
        },
        {
            "question": RESEARCH_QUESTIONS[5],
            "status": "SUPPORTED",
            "population": "40 persisted Phase 10 DEV candidate answers",
            "provenance": "PHASE15_DEV",
            "primary_evidence": "The verifier exposed 29 independently evidenced contract defects before surfacing and zero after verification.",
            "effect_and_ci": "Pre 29/40=0.725; post 0/40=0.0; absolute paired reduction 0.725, 95% CI [0.575, 0.850], 29 discordant pairs.",
            "limitations": "Measures citation/verification contract defects only, not substantive legal correctness or verifier completeness.",
            "artifact_refs": refs[6],
        },
        {
            "question": RESEARCH_QUESTIONS[6],
            "status": "PARTIALLY_SUPPORTED",
            "population": "Historical Stage-D DEV plus enriched Phase 15 matched-80 generator subset",
            "provenance": "HISTORICAL_FROZEN and PHASE15_DEV",
            "primary_evidence": "Historical Stage-D produced 13 supported answers with 0.1368 supported precision and 0.9474 unanswerable abstention recall; the locked 1.5B Arabic fallback produced 80/80 invalid generations under the frozen contract.",
            "effect_and_ci": "Phase 15 matched population is 31 gold-present answerable, 30 gold-absent answerable, and 19 explicit unanswerable; no ALLaM score exists because it was blocked before scoring.",
            "limitations": "Enriched DEV subset, fallback contract failure, ALLaM blocked, and no claim of production model promotion.",
            "artifact_refs": refs[7],
        },
    ]
    return {
        "schema_version": "phase15-research-questions-v1",
        "methodology_label": "PHASE15_SYNTHESIS_WITH_AUTOMATED_ADJUDICATION_DIAGNOSTIC",
        "provenance": "HISTORICAL_FROZEN and PHASE15_DEV",
        "research_questions": entries,
        "frozen_count": len(entries),
        "human_gold_present": False,
    }


def write_final_aggregates(root: Path, audit_payload: Mapping[str, Any]) -> tuple[Path, Path]:
    content = json.loads(
        (root / "data/evaluation/phase15_dialect_content_validity.json").read_text(encoding="utf-8")
    )
    error = build_error_analysis(audit_payload, content_audit=content)
    error_path = write_aggregate_artifact(root, "phase15_error_analysis.json", error)
    rq_path = write_aggregate_artifact(
        root, "phase15_research_questions.json", build_research_questions()
    )
    return error_path, rq_path
