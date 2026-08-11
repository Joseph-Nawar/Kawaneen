from pathlib import Path

import pyarrow as pa

from kawaneen.corpus.models import RawAccounting
from kawaneen.corpus.serialization import canonical_root, write_parquet


def test_canonical_parquet_write_is_stable_and_path_scoped(tmp_path: Path) -> None:
    root = canonical_root(tmp_path / "canonical", "fixture", "v1")
    schema = pa.schema([("id", pa.string()), ("text", pa.string())])
    first = write_parquet([{"id": "a", "text": "fiction"}], root / "units.parquet", schema)
    second = write_parquet([{"id": "a", "text": "fiction"}], root / "units.parquet", schema)
    assert first["sha256"] == second["sha256"]
    assert (
        RawAccounting(
            source_id="fixture",
            expected_records=1,
            accounted_records=1,
            canonical_documents=0,
            canonical_units=1,
            excluded_records=0,
            error_records=0,
        ).accounted_records
        == 1
    )


def test_canonical_root_rejects_raw_storage(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(ValueError, match="data/raw"):
        canonical_root(tmp_path / "data" / "raw", "fixture", "v1")
