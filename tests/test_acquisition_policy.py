from __future__ import annotations

import pytest

from kawaneen.acquisition.models import AcquisitionOperation, AcquisitionPurpose
from kawaneen.acquisition.policy import authorize_source


def test_alarb_allows_only_inspection_purposes() -> None:
    assert authorize_source(
        "alarb", AcquisitionOperation.ACQUIRE, AcquisitionPurpose.EVALUATION
    ).allowed
    assert authorize_source(
        "alarb", AcquisitionOperation.AUDIT, AcquisitionPurpose.PRIVACY_INSPECTION
    ).allowed
    assert not authorize_source(
        "alarb", AcquisitionOperation.ACQUIRE, AcquisitionPurpose.LOCAL_RESEARCH
    ).allowed


def test_arabiccr_allows_local_research_but_not_training_or_publication() -> None:
    assert authorize_source(
        "arabiccr", AcquisitionOperation.IMPORT_LOCAL, AcquisitionPurpose.LOCAL_RESEARCH
    ).allowed
    assert not authorize_source(
        "arabiccr", AcquisitionOperation.ACQUIRE, AcquisitionPurpose.TRAINING
    ).allowed
    assert not authorize_source(
        "arabiccr", AcquisitionOperation.PUBLIC_DEMO, AcquisitionPurpose.PUBLIC_DEMO
    ).allowed


def test_local_parsing_is_separate_from_public_display() -> None:
    assert authorize_source(
        "alarb", AcquisitionOperation.PARSE, AcquisitionPurpose.LOCAL_PARSING
    ).allowed
    assert not authorize_source(
        "alarb", AcquisitionOperation.PUBLIC_DISPLAY, AcquisitionPurpose.PUBLIC_DISPLAY
    ).allowed


@pytest.mark.parametrize("source_id", ["alcd", "saudi-9699", "saudi-boe-portal", "uae-legislation"])
def test_other_sources_are_denied(source_id: str) -> None:
    decision = authorize_source(
        source_id, AcquisitionOperation.ACQUIRE, AcquisitionPurpose.EVALUATION
    )
    assert not decision.allowed
    assert decision.reason


def test_moj_derived_seed_is_local_research_only() -> None:
    assert authorize_source(
        "saudi-moj-derived", AcquisitionOperation.ACQUIRE, AcquisitionPurpose.LOCAL_RESEARCH
    ).allowed
    assert authorize_source(
        "saudi-moj-derived", AcquisitionOperation.PARSE, AcquisitionPurpose.LOCAL_PARSING
    ).allowed
    assert not authorize_source(
        "saudi-moj-derived", AcquisitionOperation.TRAIN, AcquisitionPurpose.TRAINING
    ).allowed
    assert not authorize_source(
        "saudi-moj-derived", AcquisitionOperation.PUBLIC_DISPLAY, AcquisitionPurpose.PUBLIC_DISPLAY
    ).allowed
