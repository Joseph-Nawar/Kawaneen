"""Command-line entry point for foundation diagnostics."""

from __future__ import annotations

import argparse
import json
import sys

from kawaneen import __version__
from kawaneen.sources.registry import (
    RegistryValidationError,
    format_summary,
    load_registry,
    summarize_registry,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kawaneen", description="Kawaneen foundation tools")
    parser.add_argument("--version", action="version", version=f"kawaneen {__version__}")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("doctor", help="check that the foundation is available")
    sources_parser = subparsers.add_parser(
        "sources", help="validate and summarize source governance"
    )
    sources_subparsers = sources_parser.add_subparsers(dest="sources_command", required=True)
    sources_subparsers.add_parser("validate", help="validate the source registry")
    summary_parser = sources_subparsers.add_parser("summary", help="summarize source decisions")
    summary_parser.add_argument("--format", choices=("text", "json"), default="text")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        print("Kawaneen foundation: ready")
    elif args.command == "sources":
        try:
            records = load_registry()
        except RegistryValidationError as exc:
            print(f"Source registry invalid: {exc}", file=sys.stderr)
            return 1
        if args.sources_command == "validate":
            print(f"Source registry valid: {len(records)} records")
        else:
            summary = summarize_registry(records)
            if args.format == "json":
                print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
            else:
                print(format_summary(summary))
    else:
        build_parser().print_help()
    return 0
