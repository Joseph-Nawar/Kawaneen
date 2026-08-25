"""Governed eligibility policy for regulatory extraction inputs."""

from __future__ import annotations

from kawaneen.corpus.models import UnitType
from kawaneen.sources.models import Decision, Jurisdiction, SourceRecord, SourceRole, SourceType


def eligible_source(record: SourceRecord) -> bool:
    """Allow only Saudi primary statutory corpus records with local-use permission."""

    return (
        record.jurisdiction is Jurisdiction.SAUDI_ARABIA
        and record.source_role is SourceRole.PRIMARY_CORPUS
        and record.source_type is SourceType.DATASET
        and record.decision
        in {Decision.APPROVED, Decision.CONDITIONAL, Decision.LOCAL_RESEARCH_ONLY}
    )


def eligible_regulatory_unit(record: SourceRecord, unit_type: UnitType) -> bool:
    """Use canonical statutory unit typing to exclude case-law units."""

    return eligible_source(record) and unit_type in {UnitType.ARTICLE, UnitType.ARTICLE_FRAGMENT}
