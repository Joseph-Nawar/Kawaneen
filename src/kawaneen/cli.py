"""Command-line entry point for foundation diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from kawaneen import __version__
from kawaneen.acquisition.models import AcquisitionPurpose
from kawaneen.acquisition.orchestrator import (
    acquire_source,
    audit_source,
    audit_statutory,
    build_manifest,
    import_local,
    plan,
    rebuild_auto,
    status,
    validate_manifest,
    verify_source,
)
from kawaneen.acquisition.specs import load_specifications
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
    data_parser = subparsers.add_parser("data", help="gated acquisition and local inspection")
    data_subparsers = data_parser.add_subparsers(dest="data_command", required=True)
    data_subparsers.add_parser("plan", help="show source-specific acquisition policy")
    purpose_choices = [purpose.value for purpose in AcquisitionPurpose]
    acquire_parser = data_subparsers.add_parser("acquire", help="acquire an authorized source")
    acquire_parser.add_argument("source")
    acquire_parser.add_argument("--purpose", choices=purpose_choices, required=True)
    import_parser = data_subparsers.add_parser("import-local", help="import one local source file")
    import_parser.add_argument("source")
    import_parser.add_argument("--file", type=Path, required=True)
    import_parser.add_argument("--purpose", choices=purpose_choices, required=True)
    for command in ("verify", "audit", "manifest", "status", "rebuild"):
        command_parser = data_subparsers.add_parser(
            command, help=f"{command} local acquisition state"
        )
        if command in {"verify", "audit"}:
            command_parser.add_argument("--source")
        if command == "manifest":
            manifest_subparsers = command_parser.add_subparsers(
                dest="manifest_command", required=True
            )
            manifest_build = manifest_subparsers.add_parser("build")
            manifest_build.add_argument("source")
            manifest_subparsers.add_parser("validate")
        if command == "rebuild":
            command_parser.add_argument("--auto", action="store_true")
    statutory_parser = data_subparsers.add_parser(
        "audit-statutory", help="run a sanitized statutory quality audit"
    )
    statutory_parser.add_argument("source")
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
    elif args.command == "data":
        try:
            if args.data_command == "plan":
                print(json.dumps(plan(), ensure_ascii=False, indent=2, sort_keys=True))
            elif args.data_command == "acquire":
                result = acquire_source(args.source, AcquisitionPurpose(args.purpose))
                print(json.dumps([item.model_dump() for item in result], sort_keys=True))
            elif args.data_command == "import-local":
                result = import_local(args.source, args.file, AcquisitionPurpose(args.purpose))
                print(json.dumps([item.model_dump() for item in result], sort_keys=True))
            elif args.data_command == "verify":
                sources = [args.source] if args.source else sorted(load_specifications())
                availability = {item["source_id"]: item["raw_present"] for item in status()}
                print(
                    json.dumps(
                        [
                            verify_source(source).model_dump()
                            if availability[source]
                            else {"source_id": source, "status": "not_acquired"}
                            for source in sources
                        ],
                        sort_keys=True,
                    )
                )
            elif args.data_command == "audit":
                sources = [args.source] if args.source else sorted(load_specifications())
                availability = {item["source_id"]: item["raw_present"] for item in status()}
                print(
                    json.dumps(
                        [
                            audit_source(source).model_dump(exclude={"findings"})
                            if availability[source]
                            else {"source_id": source, "status": "not_acquired"}
                            for source in sources
                        ],
                        sort_keys=True,
                    )
                )
            elif args.data_command == "audit-statutory":
                print(json.dumps(audit_statutory(args.source).model_dump(), sort_keys=True))
            elif args.data_command == "manifest":
                if args.manifest_command == "build":
                    build_manifest(args.source)
                    print("Acquisition manifests built")
                else:
                    validate_manifest()
                    print("Acquisition manifests valid")
            elif args.data_command == "status":
                print(json.dumps(status(), sort_keys=True))
            elif args.data_command == "rebuild":
                if not args.auto:
                    raise ValueError("rebuild requires --auto; no generic rebuild bypass exists")
                print(json.dumps(rebuild_auto(), sort_keys=True))
        except (OSError, PermissionError, ValueError, RuntimeError) as exc:
            print(f"Data operation denied or failed: {exc}", file=sys.stderr)
            return 1
    else:
        build_parser().print_help()
    return 0
