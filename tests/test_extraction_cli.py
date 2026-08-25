from kawaneen.cli import build_parser


def test_extraction_cli_has_protected_commands() -> None:
    args = build_parser().parse_args(["extraction", "validate-annotations", "--split", "dev"])
    assert args.extraction_command == "validate-annotations"
    assert args.allow_holdout is False

    args = build_parser().parse_args(["extraction", "run-hybrid", "--split", "dev", "--resume"])
    assert args.extraction_command == "run-hybrid"
    assert args.resume is True

    args = build_parser().parse_args(
        ["extraction", "run-hybrid", "--split", "dev", "--resume", "--retry-timeouts"]
    )
    assert args.retry_timeouts is True

    args = build_parser().parse_args(
        ["extraction", "run-hybrid", "--split", "dev", "--stage", "b2", "--preflight-only"]
    )
    assert args.stage == "b2"
    assert args.preflight_only is True

    args = build_parser().parse_args(["extraction", "annotate-dev", "--next", "--interactive"])
    assert args.extraction_command == "annotate-dev"
    assert args.next is True
    assert args.interactive is True

    args = build_parser().parse_args(
        ["extraction", "import-reviewed-dev", "--file", "private-reviewed.json", "--partial"]
    )
    assert args.extraction_command == "import-reviewed-dev"
    assert args.partial is True

    args = build_parser().parse_args(["extraction", "export-dev-annotation-batch"])
    assert args.extraction_command == "export-dev-annotation-batch"


def test_holdout_evaluation_requires_an_explicit_flag_in_parser() -> None:
    args = build_parser().parse_args(
        ["extraction", "evaluate", "--extractor", "deterministic-v1", "--split", "holdout"]
    )
    assert args.allow_holdout is False
