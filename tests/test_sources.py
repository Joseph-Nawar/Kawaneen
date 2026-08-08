from __future__ import annotations

import csv
from pathlib import Path

import pytest
from pydantic import ValidationError

from kawaneen.sources.models import (
    Decision,
    Jurisdiction,
    PermissionState,
    SourceRecord,
    SourceRole,
)
from kawaneen.sources.registry import (
    RegistryValidationError,
    load_registry,
    summarize_registry,
)


def record_data(**overrides: str) -> dict[str, str]:
    data = {
        "source_id": "test-source",
        "source_name": "Test source",
        "jurisdiction": "Saudi Arabia",
        "source_type": "dataset",
        "description": "A metadata-only test source.",
        "source_url": "https://example.org/source",
        "evidence_url": "https://example.org/paper",
        "evidence_type": "official_paper",
        "evidence_summary": "The paper describes the source but does not grant data rights.",
        "provenance": "Original source provenance is documented in the evidence note.",
        "publisher": "Test publisher",
        "original_publisher": "Test original publisher",
        "task": "classification",
        "language": "Arabic",
        "size": "10",
        "size_unit": "records",
        "file_format": "CSV",
        "content_unit": "case",
        "citation": "https://doi.org/10.0000/example",
        "known_quality_issues": "Not independently inspected.",
        "contains_personal_data": "unknown",
        "access_status": "public",
        "requires_auth": "unknown",
        "attribution_required": "unknown",
        "licence_status": "missing",
        "licence_name": "",
        "licence_evidence_url": "",
        "permission_evidence_url": "",
        "terms_url": "https://example.org/terms",
        "access_method": "repository",
        "automated_access_permission": "unknown",
        "dataset_licence": "unknown",
        "public_display_permission": "unknown",
        "model_training_permission": "unknown",
        "public_demo_permission": "unknown",
        "commercial_use": "unknown",
        "derivatives": "unknown",
        "source_role": "reference",
        "authority_level": "academic",
        "privacy_risk": "unknown",
        "decision": "metadata_only",
        "verification_date": "2026-08-06",
        "conditions": "",
        "required_rights": "Confirm dataset licence and privacy controls.",
        "manual_action": "Confirm licence and privacy terms with the provider.",
        "original_source_rights": "unknown",
        "paper_licence": "unknown",
        "code_licence": "unknown",
        "notes": "",
    }
    data.update(overrides)
    return data


def test_source_record_normalizes_enum_values() -> None:
    record = SourceRecord.model_validate(record_data())
    assert record.dataset_licence is PermissionState.UNKNOWN
    assert record.decision is Decision.METADATA_ONLY
    assert record.source_role is SourceRole.REFERENCE


def test_positive_permission_requires_explicit_evidence() -> None:
    with pytest.raises(ValidationError, match="permission_evidence_url"):
        SourceRecord.model_validate(
            record_data(
                dataset_licence="yes",
                licence_status="confirmed",
                licence_evidence_url="https://example.org/licence",
            )
        )


def test_positive_permission_with_explicit_evidence_is_accepted() -> None:
    record = SourceRecord.model_validate(
        record_data(
            licence_status="confirmed",
            permission_evidence_url="https://example.org/licence",
            dataset_licence="yes",
            commercial_use="yes",
            derivatives="yes",
        )
    )
    assert record.dataset_licence is PermissionState.YES


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {
                "licence_status": "paper_only",
                "dataset_licence": "yes",
                "permission_evidence_url": "https://example.org/permission",
            },
            "paper-only",
        ),
        (
            {
                "privacy_risk": "high",
                "public_demo_permission": "yes",
                "permission_evidence_url": "https://example.org/permission",
            },
            "privacy",
        ),
        (
            {
                "automated_access_permission": "no",
                "decision": "approved",
                "licence_status": "confirmed",
                "licence_evidence_url": "https://example.org/licence",
                "dataset_licence": "no",
                "privacy_risk": "low",
            },
            "automated access",
        ),
        (
            {"dataset_licence": "conditional", "conditions": ""},
            "conditions",
        ),
        (
            {"decision": "blocked_pending_review", "manual_action": ""},
            "manual_action",
        ),
    ],
)
def test_fail_closed_cross_field_rules(overrides: dict[str, str], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        SourceRecord.model_validate(record_data(**overrides))


def test_registry_rejects_duplicate_ids(tmp_path: Path) -> None:
    rows = [record_data(), record_data(source_name="Second source")]
    path = tmp_path / "registry.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(RegistryValidationError, match="duplicate source_id"):
        load_registry(path)


def test_registry_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(RegistryValidationError, match="does not exist"):
        load_registry(tmp_path / "missing.csv")


def test_real_registry_contains_requested_sources() -> None:
    records = load_registry()
    assert len(records) == 13
    assert {record.source_id for record in records} >= {
        "alarb",
        "arablegaleval",
        "saudi-9699",
        "alcd",
        "arabiccr",
        "saudi-moj-open",
        "saudi-moj-portal",
        "saudi-bog-open",
        "saudi-boe-portal",
        "egypt-court-cassation",
        "egypt-official-legislation-candidates",
        "uae-legislation",
        "saudi-moj-derived",
    }


def test_summary_counts_decisions() -> None:
    records = [
        SourceRecord.model_validate(record_data(source_id="one")),
        SourceRecord.model_validate(
            record_data(
                source_id="two",
                decision="blocked_pending_review",
                manual_action="Obtain terms.",
            )
        ),
    ]
    summary = summarize_registry(records)
    assert summary["source_count"] == 2
    assert summary["decisions"] == {"blocked_pending_review": 1, "metadata_only": 1}


def test_required_technical_fields_cannot_be_omitted() -> None:
    data = record_data()
    data.pop("file_format")
    with pytest.raises(ValidationError, match="file_format"):
        SourceRecord.model_validate(data)


def test_controlled_jurisdiction_rejects_free_text() -> None:
    with pytest.raises(ValidationError, match="jurisdiction"):
        SourceRecord.model_validate(record_data(jurisdiction="Arabic legal domain"))


def test_search_result_urls_are_not_canonical_evidence() -> None:
    with pytest.raises(ValidationError, match="canonical"):
        SourceRecord.model_validate(record_data(source_url="https://example.org/search?q=cases"))


def test_open_dataset_licence_does_not_make_public_demo_safe() -> None:
    record = SourceRecord.model_validate(
        record_data(
            licence_status="confirmed",
            licence_name="Apache-2.0",
            licence_evidence_url="https://example.org/licence",
            permission_evidence_url="https://example.org/licence",
            dataset_licence="yes",
            public_demo_permission="unknown",
        )
    )
    assert record.public_demo_permission is PermissionState.UNKNOWN


def test_evaluation_and_local_research_decisions_are_distinct() -> None:
    evaluation = SourceRecord.model_validate(record_data(decision="evaluation_only"))
    local = SourceRecord.model_validate(
        record_data(decision="local_research_only", source_role="primary_corpus")
    )
    assert evaluation.decision is Decision.EVALUATION_ONLY
    assert local.decision is Decision.LOCAL_RESEARCH_ONLY


def test_real_registry_has_required_fields_and_controlled_jurisdictions() -> None:
    records = load_registry()
    assert all(record.publisher and record.file_format and record.citation for record in records)
    assert all(record.jurisdiction in set(Jurisdiction) for record in records)


def test_real_registry_records_corrected_release_facts() -> None:
    records = {record.source_id: record for record in load_registry()}
    assert records["alarb"].split_info == "Train: 12012; test: 1329."
    assert records["saudi-9699"].split_info == (
        "Train: 7759; test: 1940; categories: Administrative 6727; Commercial 2035; Criminal 937."
    )
    assert records["arabiccr"].source_url == "https://data.mendeley.com/datasets/np538c95yy/3"
    assert records["arablegaleval"].source_url == (
        "https://huggingface.co/datasets/THIQAH-RD/ArabLegalEval"
    )
    assert records["arablegaleval"].size == "27032"
    assert records["saudi-moj-derived"].decision.value == "local_research_only"
    assert records["saudi-moj-derived"].dataset_licence.value == "yes"
    assert records["saudi-moj-derived"].original_source_rights.value == "unknown"


def test_missing_licence_cannot_assert_licence_derived_attribution() -> None:
    with pytest.raises(ValidationError, match="attribution"):
        SourceRecord.model_validate(
            record_data(
                attribution_required="yes",
                permission_evidence_url="https://example.org/permission",
                terms_url="",
            )
        )


def test_attribution_can_be_supported_by_separate_terms_evidence() -> None:
    record = SourceRecord.model_validate(
        record_data(
            attribution_required="yes",
            permission_evidence_url="https://example.org/permission",
            terms_url="https://example.org/terms",
        )
    )
    assert record.attribution_required is PermissionState.YES
