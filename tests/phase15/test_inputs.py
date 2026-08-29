from __future__ import annotations

import json
from pathlib import Path

import pytest

from kawaneen.phase15.inputs import (
    Phase15InputRoots,
    load_dev_query_records,
    load_dev_rankings,
)


def test_dev_input_root_keeps_historical_reads_separate_from_outputs(tmp_path: Path) -> None:
    roots = Phase15InputRoots(historical_private_root=tmp_path / "historical", output_root=tmp_path)
    assert roots.private_path("phase7_retrieval/dev/rankings/r.json") == (
        tmp_path / "historical/phase7_retrieval/dev/rankings/r.json"
    )
    assert (
        roots.output_path("raw/results.json")
        == tmp_path / "artifacts/private/phase15_evaluation/raw/results.json"
    )


def test_dev_loaders_filter_only_dev_and_reject_holdout_paths(tmp_path: Path) -> None:
    query_path = tmp_path / "phase6_evaluation/ai-reviewed-v1/draft/selected_and_variants.jsonl"
    query_path.parent.mkdir(parents=True)
    query_path.write_text(
        "\n".join(
            json.dumps({"query_id": f"q{i}", "split": split})
            for i, split in enumerate(("dev", "holdout"))
        )
        + "\n"
    )
    roots = Phase15InputRoots(historical_private_root=tmp_path, output_root=tmp_path / "out")
    assert [row["query_id"] for row in load_dev_query_records(roots)] == ["q0"]
    with pytest.raises(ValueError, match="HOLDOUT"):
        load_dev_rankings(roots, Path("phase8_retrieval/holdout/rankings/r.json"), ("q0",))
