from __future__ import annotations

import json
import subprocess
import tomllib
from pathlib import Path

from kawaneen.chunking.policies import chunk_policy_configurations

ROOT = Path(__file__).parents[1]


def test_chunking_toml_mirrors_versioned_code_policies() -> None:
    config = tomllib.loads((ROOT / "configs/chunking/policies.toml").read_text(encoding="utf-8"))
    file_policies = {item["policy_id"]: item for item in config["policies"]}
    code_policies = {item["policy_id"]: item for item in chunk_policy_configurations()}
    assert set(file_policies) == set(code_policies)
    for policy_id, code_policy in code_policies.items():
        assert file_policies[policy_id] == code_policy


def test_phase5_private_root_is_ignored_and_not_tracked() -> None:
    assert (
        subprocess.run(
            ["git", "check-ignore", "-q", "artifacts/private/phase5_chunking"],
            cwd=ROOT,
            check=False,
        ).returncode
        == 0
    )
    tracked = subprocess.run(
        ["git", "ls-files", "artifacts/private/phase5_chunking"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert tracked.stdout == ""


def test_sanitized_phase5_metrics_contain_no_text_fields_if_materialized() -> None:
    path = ROOT / "data/evaluation/phase5_chunking_metrics.json"
    if not path.is_file():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))

    def walk(value: object) -> list[str]:
        if isinstance(value, dict):
            keys = [str(key) for key in value]
            return keys + [key for child in value.values() for key in walk(child)]
        if isinstance(value, list):
            return [key for child in value for key in walk(child)]
        return []

    assert not {
        "display_text",
        "search_text",
        "query_text",
        "source_text",
    }.intersection(walk(payload))
