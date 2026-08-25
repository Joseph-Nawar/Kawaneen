from pathlib import Path

from kawaneen.corpus.models import UnitType
from kawaneen.extraction.source_policy import eligible_regulatory_unit, eligible_source
from kawaneen.sources.registry import load_registry


def test_statutory_source_is_eligible_and_case_law_unit_is_not() -> None:
    records = {record.source_id: record for record in load_registry()}
    statutory = records["saudi-moj-derived"]
    case_law = records["alarb"]
    assert eligible_source(statutory)
    assert not eligible_source(case_law)
    assert eligible_regulatory_unit(statutory, UnitType.ARTICLE)
    assert not eligible_regulatory_unit(case_law, UnitType.FACTS)


def test_policy_does_not_require_a_source_name_hack() -> None:
    source_policy = Path("src/kawaneen/extraction/source_policy.py").read_text(encoding="utf-8")
    assert '"saudi-moj-derived"' not in source_policy
