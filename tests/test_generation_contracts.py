from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from kawaneen.generation.abstention import invalid_generation_result
from kawaneen.generation.contracts import (
    AbstentionReason,
    GenerationDecision,
    GenerationSettings,
    ModelOutput,
    ModelOutputCitation,
    ModelOutputClaim,
    parse_model_output,
)


def valid_payload() -> dict[str, object]:
    return {
        "decision": "answer",
        "claims": [
            {
                "text": "النص يحدد المهلة.",
                "citations": [{"evidence_id": "E001", "quoted_text": "المهلة"}],
            }
        ],
    }


def test_model_output_accepts_only_strict_claim_and_citation_shape() -> None:
    output = parse_model_output(json.dumps(valid_payload(), ensure_ascii=False))

    assert output.decision is GenerationDecision.ANSWER
    assert output.claims[0].text == "النص يحدد المهلة."
    assert output.claims[0].citations[0].evidence_id == "E001"


@pytest.mark.parametrize(
    "extra",
    ("answer", "reasoning", "document_id", "page", "article", "jurisdiction", "title"),
)
def test_model_output_rejects_untrusted_top_level_or_citation_metadata(extra: str) -> None:
    payload = valid_payload()
    if extra in {"answer", "reasoning"}:
        payload[extra] = "forbidden"
    else:
        citation = payload["claims"][0]["citations"][0]  # type: ignore[index]
        citation[extra] = "forbidden"  # type: ignore[index]

    with pytest.raises(ValidationError):
        ModelOutput.model_validate(payload)


def test_model_output_rejects_empty_quotes_and_empty_claims() -> None:
    with pytest.raises(ValidationError):
        ModelOutputCitation(evidence_id="E001", quoted_text=" ")
    with pytest.raises(ValidationError):
        ModelOutputClaim(text="claim", citations=())


@pytest.mark.parametrize(
    "payload",
    (
        {"decision": "answer", "claims": []},
        {
            "decision": "abstain",
            "claims": [
                {
                    "text": "claim",
                    "citations": [{"evidence_id": "E001", "quoted_text": "quote"}],
                }
            ],
        },
        {
            "decision": "answer",
            "claims": [
                {
                    "text": "claim",
                    "citations": [{"evidence_id": "E001", "quoted_text": "quote"}],
                }
            ]
            * 4,
        },
    ),
)
def test_model_output_enforces_decision_and_claim_cardinality(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ModelOutput.model_validate(payload)


def test_invalid_json_fails_closed_to_invalid_generation() -> None:
    result = invalid_generation_result("not-json")

    assert result.decision is GenerationDecision.ABSTAIN
    assert result.claims == ()
    assert result.abstention_reason is AbstentionReason.INVALID_GENERATION


def test_generation_settings_have_controlled_defaults_and_are_frozen() -> None:
    settings = GenerationSettings()

    assert settings.temperature == 0
    assert settings.do_sample is False
    assert settings.max_new_tokens == 384
    assert settings.max_claims == 3
    with pytest.raises(ValidationError):
        settings.max_new_tokens = 10  # type: ignore[misc]
