import kawaneen.cli as cli
from kawaneen.cli import build_parser


def test_retrieval_cli_exposes_phase7_stages() -> None:
    parser = build_parser()
    commands = (
        "plan",
        "build-corpus",
        "smoke",
        "encode-corpus",
        "cache-status",
        "evaluate-dev",
        "freeze-dev-selection",
        "evaluate-holdout",
        "recover-holdout-artifacts",
        "report",
    )
    for command in commands:
        args = ["retrieval", command]
        if command == "evaluate-holdout":
            args.append("--allow-holdout")
        elif command == "encode-corpus":
            args.extend(["--model", "bge-m3", "--resume"])
        elif command == "cache-status":
            args.extend(["--model", "bge-m3"])
        parsed = parser.parse_args(args)
        assert parsed.retrieval_command == command


def test_retrieval_cli_dispatches_all_phase7_operations(monkeypatch, capsys) -> None:
    operations = {
        "plan": "retrieval_plan",
        "build-corpus": "build_retrieval_corpus",
        "smoke": "retrieval_smoke",
        "encode-corpus": "encode_corpus",
        "cache-status": "cache_status",
        "real-model-smoke": "real_model_smoke",
        "evaluate-dev": "evaluate_dev",
        "dense-sanity-audit": "dense_sanity_audit",
        "freeze-dev-selection": "freeze_dev_selection",
        "evaluate-holdout": "evaluate_holdout",
        "recover-holdout-artifacts": "recover_holdout_artifacts",
        "verify-holdout-readiness": "verify_holdout_readiness",
        "report": "retrieval_report",
        "final-report": "build_final_report",
    }
    for command, function_name in operations.items():
        monkeypatch.setattr(
            cli,
            function_name,
            lambda command=command, **_kwargs: {"command": command},
        )
        args = ["retrieval", command]
        if command == "encode-corpus":
            args.extend(["--model", "bge-m3", "--resume"])
        elif command == "cache-status":
            args.extend(["--model", "bge-m3"])
        elif command in {"evaluate-holdout", "recover-holdout-artifacts"}:
            args.append("--allow-holdout")
        assert cli.main(args) == 0
        assert f'"command": "{command}"' in capsys.readouterr().out
