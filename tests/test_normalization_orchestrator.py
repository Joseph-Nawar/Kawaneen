from __future__ import annotations

# ruff: noqa: RUF001
from kawaneen.corpus.models import CanonicalUnit, SourceProvenance, UnitType
from kawaneen.normalization.diagnostics import diagnose_policy
from kawaneen.normalization.orchestrator import (
    SelectionDecision,
    SelectionEvidence,
    _gate_status,
    _materialize_policy,
    _normalized_schema,
    normalization_plan,
    select_policy,
)
from kawaneen.normalization.policies import get_policy


def _unit() -> CanonicalUnit:
    return CanonicalUnit(
        unit_id="unit-1",
        document_id="document-1",
        unit_type=UnitType.CASE_TEXT,
        text="  أ/١٢  ",
        provenance=SourceProvenance(
            source_id="synthetic",
            source_version="v1",
            source_path="fixture",
            source_row=1,
            source_field="text",
        ),
    )


def _evidence(light_gain: float, aggressive_gain: float = 0.0) -> SelectionEvidence:
    metrics = {
        "arabic-raw-v1": {"mrr_at_10": 0.50, "recall_at_10": 0.60, "ndcg_at_10": 0.50},
        "arabic-light-v1": {
            "mrr_at_10": 0.50 + light_gain,
            "recall_at_10": 0.60 + light_gain,
            "ndcg_at_10": 0.50 + light_gain,
        },
        "arabic-aggressive-v1": {
            "mrr_at_10": 0.50 + light_gain + aggressive_gain,
            "recall_at_10": 0.60 + light_gain + aggressive_gain,
            "ndcg_at_10": 0.50 + light_gain + aggressive_gain,
        },
    }
    intervals = {
        "arabic-raw-v1__vs__arabic-light-v1__mrr_at_10": {
            "estimate": -light_gain,
            "lower": -light_gain - 0.001,
            "upper": -light_gain + 0.001,
        },
        "arabic-light-v1__vs__arabic-aggressive-v1__mrr_at_10": {
            "estimate": -aggressive_gain,
            "lower": -aggressive_gain - 0.001,
            "upper": -aggressive_gain + 0.001,
        },
    }
    return SelectionEvidence(
        policy_metrics=metrics,
        slice_metrics={
            policy: {
                "unchanged_control": {"mrr_at_10": value["mrr_at_10"]},
                "collision_risk": {"mrr_at_10": value["mrr_at_10"]},
            }
            for policy, value in metrics.items()
        },
        paired_confidence_intervals=intervals,
        policy_gate_status={policy: {"eligible": True, "failures": []} for policy in metrics},
    )


def test_selection_prefers_raw_when_light_is_effectively_tied() -> None:
    decision = select_policy(_evidence(0.005))
    assert decision.selected_policy_id == "arabic-raw-v1"


def test_selection_chooses_light_only_for_meaningful_gain() -> None:
    decision = select_policy(_evidence(0.05))
    assert decision.selected_policy_id == "arabic-light-v1"


def test_selection_rejects_aggressive_when_its_gate_fails() -> None:
    evidence = _evidence(0.05, 0.05)
    evidence.policy_gate_status["arabic-aggressive-v1"] = {
        "eligible": False,
        "failures": ["identifier_safety"],
    }
    decision = select_policy(evidence)
    assert decision.selected_policy_id == "arabic-light-v1"


def test_normalization_plan_is_versioned_and_explicitly_scoped() -> None:
    plan = normalization_plan()
    assert plan["schema_version"] == 1
    assert plan["phase"] == "phase-4-arabic-normalization"
    assert "chunking" in plan["scope_exclusions"]
    assert {item["policy_id"] for item in plan["policies"]} == {
        "arabic-raw-v1",
        "arabic-light-v1",
        "arabic-aggressive-v1",
    }


def test_selection_decision_serializes_rejections_without_text() -> None:
    decision = SelectionDecision(
        selected_policy_id="arabic-raw-v1",
        rationale="synthetic",
        eligible_policies=("arabic-raw-v1",),
        rejected_policies={"arabic-light-v1": ("identifier_safety",)},
    )
    assert decision.to_sanitized_dict()["rejected_policies"] == {
        "arabic-light-v1": ["identifier_safety"]
    }


def test_private_materialization_preserves_contract_and_schema() -> None:
    records, safety = _materialize_policy((_unit(),), get_policy("arabic-light-v1"))
    assert len(records) == 1
    assert safety == {"identifier_safety_failures": 0}
    assert records[0].display_text == "  أ/١٢  "
    assert records[0].search_text == "ا/١٢"
    assert [field.name for field in _normalized_schema()][3:7] == [
        "display_text",
        "search_text",
        "policy_id",
        "policy_hash",
    ]


def test_gate_status_reports_clean_synthetic_policy() -> None:
    units = (_unit(),)
    policy = get_policy("arabic-raw-v1")
    diagnostics = diagnose_policy(units, policy)
    status = _gate_status(units, policy, diagnostics, 0)
    assert status["eligible"] is True
    assert status["preservation_checked"] == 1
    assert status["determinism_failures"] == 0
    assert status["idempotency_failures"] == 0
