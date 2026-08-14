from __future__ import annotations

from pathlib import Path

from kawaneen.evaluation.balance import build_source_balance_audit
from kawaneen.evaluation.candidates import _allocate_source_quotas, build_draft_candidates
from kawaneen.evaluation.corpus import freeze_evaluation_corpus, load_evaluation_units


def test_source_quotas_follow_available_opportunity_share() -> None:
    assert _allocate_source_quotas({"alarb": 90, "arabiccr": 10}, 10) == {
        "alarb": 9,
        "arabiccr": 1,
    }


def test_source_quotas_cover_each_available_source_without_forcing_equal_split() -> None:
    assert _allocate_source_quotas({"alarb": 100, "arabiccr": 100}, 3) == {
        "alarb": 2,
        "arabiccr": 1,
    }


def test_source_balance_audit_is_sanitized_and_records_the_corrected_flow(tmp_path: Path) -> None:
    corpus = freeze_evaluation_corpus(
        load_evaluation_units(Path("data/interim/canonical")),
        canonical_root=Path("data/interim/canonical"),
    )
    draft = build_draft_candidates(corpus, output_root=tmp_path)
    report = build_source_balance_audit(
        corpus, draft.base_candidates, draft.selected_base_candidates
    )

    assert report["retrieval_scores_used"] is False
    assert report["finding"] == {
        "cause": "source_sorted_candidate_cap",
        "corrected": True,
        "equal_split_forced": False,
        "selection_basis": "measured eligible opportunity share",
    }
    source_flow = report["source_flow"]
    assert source_flow["definition"]["arabiccr"]["eligible_opportunities"] == 859
    assert source_flow["definition"]["arabiccr"]["selected_base_intents"] == 16
    assert report["section_parity"]["facts_events"]["left"]["unit_count"] == 13341
    assert report["section_parity"]["facts_events"]["right"]["unit_count"] == 12806
