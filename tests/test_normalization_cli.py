from __future__ import annotations

import json

from kawaneen.cli import main


def test_normalization_plan_is_sanitized(capsys) -> None:
    assert main(["normalization", "plan"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [policy["policy_id"] for policy in payload["policies"]] == [
        "arabic-raw-v1",
        "arabic-light-v1",
        "arabic-aggressive-v1",
    ]
    assert payload["private_root"] == "artifacts/private/phase4_normalization"
