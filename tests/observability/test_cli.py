from __future__ import annotations

import json

from kawaneen.cli import build_parser, main


def test_phase16_cli_verifies_identity_and_reproduction(capsys) -> None:
    assert main(["phase16", "identity"]) == 0
    identity = json.loads(capsys.readouterr().out)
    assert identity["configuration_version"]

    assert main(["phase16", "reproduce"]) == 0
    output = capsys.readouterr().out
    assert "rows reproduced 6/6" in output
    assert "PASS" in output

    assert main(["phase16", "verify"]) == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified["status"] == "PASS"


def test_phase16_reproduce_accepts_optional_mlflow_flag() -> None:
    args = build_parser().parse_args(["phase16", "reproduce", "--mlflow"])

    assert args.mlflow is True
