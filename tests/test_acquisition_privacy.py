from __future__ import annotations

from pathlib import Path

from kawaneen.acquisition.models import FileExpectation, SourceSpecification
from kawaneen.acquisition.privacy import screen_privacy, summarize_privacy


def test_privacy_findings_are_masked_and_not_clearance(tmp_path: Path) -> None:
    path = tmp_path / "fiction.csv"
    path.write_text(
        "email,national_id,address\nfake.person@example.test,1234567890,Imaginary Street\n",
        encoding="utf-8",
    )
    spec = SourceSpecification(
        schema_version=1,
        source_id="arabiccr",
        version="test",
        revision="test",
        provider="fixture",
        identifier="fixture",
        licence="test",
        expected_records=1,
        files=(FileExpectation(path="fiction.csv", format="csv", expected_records=1),),
    )
    result = screen_privacy(spec, tmp_path)
    assert result.finding_count >= 3
    assert result.legal_clearance is False
    assert all("fake.person" not in finding.masked_value for finding in result.findings)
    assert all("Imaginary" not in finding.masked_value for finding in result.findings)
    assert all(finding.masked_value == "[REDACTED]" for finding in result.findings)


def test_privacy_summary_is_aggregated_and_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "fiction.csv"
    path.write_text(
        "email,address\nfake.person@example.test,Imaginary Street\n",
        encoding="utf-8",
    )
    spec = SourceSpecification(
        schema_version=1,
        source_id="arabiccr",
        version="test",
        revision="test",
        provider="fixture",
        identifier="fixture",
        licence="test",
        expected_records=1,
        files=(FileExpectation(path="fiction.csv", format="csv", expected_records=1),),
    )
    summary = summarize_privacy(screen_privacy(spec, tmp_path), sample_cap=1)
    assert summary.affected_record_count == 1
    assert summary.deterministic_review_sample_size == 1
    assert summary.confirmed_pii_count is None
    assert summary.likely_false_positive_count is None
    assert summary.findings_by_detector == {"email": 1, "identifier_or_address_column": 1}
    assert summary.findings_by_column == {"address": 1, "email": 1}
