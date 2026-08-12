from __future__ import annotations

from kawaneen.cli import build_parser, main


def test_chunking_cli_exposes_phase5_commands() -> None:
    parser = build_parser()
    assert parser.parse_args(["chunking", "plan"]).chunking_command == "plan"
    assert parser.parse_args(["chunking", "build"]).chunking_command == "build"
    assert parser.parse_args(["chunking", "experiment"]).chunking_command == "experiment"
    assert parser.parse_args(["chunking", "validate"]).chunking_command == "validate"


def test_chunking_cli_plan_and_validate_execute(capsys) -> None:
    assert main(["chunking", "plan"]) == 0
    assert '"phase": "phase-05-legal-structure-and-chunking"' in capsys.readouterr().out
    assert main(["chunking", "validate"]) == 0
    assert '"valid": true' in capsys.readouterr().out
