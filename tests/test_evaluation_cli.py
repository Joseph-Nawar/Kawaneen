from __future__ import annotations

from kawaneen.cli import build_parser, main


def test_evaluation_cli_exposes_phase6_commands() -> None:
    parser = build_parser()
    for command in (
        "plan",
        "build-draft",
        "balance-audit",
        "export-review",
        "import-review",
        "validate",
        "freeze",
        "freeze-ai-reviewed",
        "stats",
    ):
        args = ["evaluation", command]
        if command == "import-review":
            args.extend(["--file", "review.jsonl"])
        assert parser.parse_args(args).evaluation_command == command


def test_evaluation_plan_is_sanitized(capsys) -> None:
    assert main(["evaluation", "plan"]) == 0
    output = capsys.readouterr().out
    assert "phase-06-retrieval-evaluation-dataset" in output
    assert "query_text" not in output
