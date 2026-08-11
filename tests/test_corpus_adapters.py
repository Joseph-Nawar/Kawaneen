from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from kawaneen.acquisition.models import FileExpectation, SourceSpecification
from kawaneen.corpus import adapters
from kawaneen.corpus.adapters import build_source


def _spec(
    source_id: str, version: str, files: tuple[FileExpectation, ...], records: int
) -> SourceSpecification:
    return SourceSpecification(
        schema_version=1,
        source_id=source_id,
        version=version,
        revision=version,
        provider="fixture",
        identifier=source_id,
        licence="fixture",
        expected_records=records,
        files=files,
    )


def test_all_source_adapters_preserve_fixture_text_and_account_rows(
    tmp_path: Path, monkeypatch
) -> None:
    specs = {
        "alarb": _spec(
            "alarb",
            "v1",
            (
                FileExpectation(
                    path="train.parquet",
                    format="parquet",
                    expected_records=1,
                    expected_columns=(
                        "case_facts",
                        "court_reasoning",
                        "applicable_laws",
                        "verdict",
                    ),
                    split="train",
                ),
            ),
            1,
        ),
        "arabiccr": _spec(
            "arabiccr",
            "v1",
            (FileExpectation(path="cases.csv", format="csv", expected_records=1),),
            1,
        ),
        "saudi-moj-derived": _spec(
            "saudi-moj-derived",
            "v1",
            (
                FileExpectation(
                    path="statutes.parquet",
                    format="parquet",
                    expected_records=2,
                    expected_columns=("text", "article_number", "law_name", "law_type"),
                ),
            ),
            2,
        ),
    }
    monkeypatch.setattr(adapters, "load_specifications", lambda: specs)
    (tmp_path / "raw/alarb/v1").mkdir(parents=True)
    (tmp_path / "raw/arabiccr/v1").mkdir(parents=True)
    (tmp_path / "raw/saudi-moj-derived/v1").mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "case_facts": ["fiction A"],
                "court_reasoning": ["reason"],
                "applicable_laws": ["law"],
                "verdict": ["outcome"],
            }
        ),
        tmp_path / "raw/alarb/v1/train.parquet",
    )
    (tmp_path / "raw/arabiccr/v1/cases.csv").write_text(
        "case_number,judgment_number,court_name,case_type,judgment_date,year,city,details_url,"
        "case_text,EVENTS,REASONING,RULING\n"
        "C1,J1,Court,Commercial,2025,2025,Riyadh,https://example.test,"
        "exact case,events,reasoning,ruling\n",
        encoding="utf-8",
    )
    pq.write_table(
        pa.table(
            {
                "text": ["fragment one", "fragment two"],
                "article_number": ["1", "1"],
                "law_name": ["Law", "Law"],
                "law_type": ["Regulation", "Regulation"],
            }
        ),
        tmp_path / "raw/saudi-moj-derived/v1/statutes.parquet",
    )
    results = [build_source(source, tmp_path / "raw", tmp_path / "canonical") for source in specs]
    assert [result.accounted_records for result in results] == [1, 1, 2]
    assert (tmp_path / "canonical/arabiccr/v1/units.parquet").is_file()
    assert (tmp_path / "canonical/saudi-moj-derived/v1/fragments.parquet").is_file()
