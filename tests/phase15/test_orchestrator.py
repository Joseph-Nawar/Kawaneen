from __future__ import annotations

import json
from pathlib import Path

from kawaneen.phase15.orchestrator import (
    phase15_freeze,
    phase15_plan,
    write_phase15_status_artifacts,
)


def test_plan_and_freeze_write_only_governance_artifacts(tmp_path: Path) -> None:
    for phase in (*range(3, 12), 14):
        path = tmp_path / "data" / "evaluation" / f"phase{phase}_frozen.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")

    plan = phase15_plan(tmp_path)
    assert plan["seed"] == 20260826
    result = phase15_freeze(tmp_path)
    assert result["registry_entries"] == 10
    assert (tmp_path / "data/manifests/evaluation/phase15_experiment_plan.json").is_file()
    assert (tmp_path / "data/manifests/evaluation/phase15_evidence_registry.json").is_file()
    assert not (tmp_path / "data/evaluation/phase15_research_questions.json").exists()
    stored = json.loads(
        (tmp_path / "data/manifests/evaluation/phase15_experiment_plan.json").read_text()
    )
    assert stored["base_sha"] == "03f58284426c84c6c813be2b1e1bbbbbfd1c9a2d"


def test_status_artifacts_are_aggregate_and_do_not_claim_results(tmp_path: Path) -> None:
    result = write_phase15_status_artifacts(tmp_path)
    assert len(result) == 9
    for path in result:
        payload = json.loads(Path(path).read_text())
        assert payload["status"] == "NOT_RUN"
        assert payload["provenance"] == "PHASE15_DEV"
