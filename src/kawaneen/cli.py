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
from kawaneen.corpus.orchestrator import build as build_corpus
from kawaneen.corpus.orchestrator import (
    gaps as corpus_gaps,
)
from kawaneen.corpus.orchestrator import (
    inventory as corpus_inventory,
)
from kawaneen.corpus.orchestrator import (
    plan as corpus_plan,
)
from kawaneen.corpus.orchestrator import (
    statutory_status as corpus_statutory_status,
)
from kawaneen.corpus.orchestrator import (
    validate as validate_corpus,
)
from kawaneen.normalization.orchestrator import (
    normalization_plan,
    run_phase4_experiment,
)
from kawaneen.normalization.sensitivity import run_sensitivity_validation
from kawaneen.parsing.benchmark import preflight_pdfs, qualification_status
from kawaneen.parsing.diagnostics import diagnose_docling
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
    corpus_parser = subparsers.add_parser("corpus", help="canonical corpus construction")
    corpus_subparsers = corpus_parser.add_subparsers(dest="corpus_command", required=True)
    corpus_subparsers.add_parser("plan", help="show canonicalization plan")
    build_parser_command = corpus_subparsers.add_parser(
        "build", help="build ignored canonical outputs"
    )
    build_parser_command.add_argument("--source", action="append")
    corpus_subparsers.add_parser("validate", help="validate canonical outputs")
    corpus_subparsers.add_parser("inventory", help="show sanitized canonical inventory")
    corpus_subparsers.add_parser("statutory-status", help="show statutory reconstruction status")
    corpus_subparsers.add_parser(
        "duplicate-diagnostics", help="show sanitized collision diagnostics"
    )
    corpus_subparsers.add_parser("gaps", help="show statutory acquisition gaps")
    parsing_parser = subparsers.add_parser("parsing", help="private parser qualification tools")
    parsing_subparsers = parsing_parser.add_subparsers(dest="parsing_command", required=True)
    preflight_parser = parsing_subparsers.add_parser(
        "preflight", help="inspect private PDF metadata without extracting text"
    )
    preflight_parser.add_argument(
        "--path", type=Path, default=Path("artifacts/private/parsing_benchmark/source_pdfs")
    )
    parsing_subparsers.add_parser("benchmark", help="show the sanitized parser benchmark gate")
    diagnose_parser = parsing_subparsers.add_parser(
        "diagnose", help="diagnose one-page Docling execution in a subprocess"
    )
    diagnose_parser.add_argument("--path", type=Path, required=True)
    diagnose_parser.add_argument("--device", choices=("cpu", "auto"), default="cpu")
    normalization_parser = subparsers.add_parser(
        "normalization", help="private Phase 4 Arabic normalization experiments"
    )
    normalization_subparsers = normalization_parser.add_subparsers(
        dest="normalization_command", required=True
    )
    normalization_subparsers.add_parser("plan", help="show versioned normalization policies")
    normalization_subparsers.add_parser("run", help="run the private Phase 4 ablation")
    normalization_subparsers.add_parser("validate", help="validate sanitized Phase 4 artifacts")
    normalization_subparsers.add_parser(
        "sensitivity", help="run bounded Phase 4 sensitivity validation"
    )
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
    elif args.command == "corpus":
        try:
            if args.corpus_command == "plan":
                print(json.dumps(corpus_plan(), ensure_ascii=False, indent=2, sort_keys=True))
            elif args.corpus_command == "build":
                print(json.dumps(build_corpus(args.source), ensure_ascii=False, sort_keys=True))
            elif args.corpus_command == "validate":
                print(json.dumps(validate_corpus(), ensure_ascii=False, sort_keys=True))
            elif args.corpus_command == "inventory":
                print(json.dumps(corpus_inventory(), ensure_ascii=False, sort_keys=True))
            elif args.corpus_command == "statutory-status":
                print(json.dumps(corpus_statutory_status(), ensure_ascii=False, sort_keys=True))
            elif args.corpus_command == "duplicate-diagnostics":
                path = Path("data/manifests/canonical/duplicate_diagnostics.json")
                print(path.read_text(encoding="utf-8"))
            elif args.corpus_command == "gaps":
                print(json.dumps(corpus_gaps(), ensure_ascii=False, sort_keys=True))
        except (OSError, PermissionError, ValueError, RuntimeError) as exc:
            print(f"Corpus operation failed: {exc}", file=sys.stderr)
            return 1
    elif args.command == "parsing":
        if args.parsing_command == "preflight":
            print(json.dumps(preflight_pdfs(args.path), sort_keys=True))
        elif args.parsing_command == "benchmark":
            print(json.dumps(qualification_status(), sort_keys=True))
        elif args.parsing_command == "diagnose":
            print(json.dumps(diagnose_docling(args.path, device=args.device), sort_keys=True))
    elif args.command == "normalization":
        try:
            if args.normalization_command == "plan":
                print(json.dumps(normalization_plan(), ensure_ascii=False, sort_keys=True))
            elif args.normalization_command == "run":
                result = run_phase4_experiment()
                print(json.dumps(result, ensure_ascii=False, sort_keys=True))
            elif args.normalization_command == "validate":
                manifest = Path("data/manifests/normalization/phase4_manifest.json")
                metrics = Path("data/evaluation/phase4_normalization_metrics.json")
                if not manifest.is_file() or not metrics.is_file():
                    raise ValueError("Phase 4 sanitized artifacts are missing")
                print(
                    json.dumps(
                        {
                            "valid": True,
                            "manifest": manifest.as_posix(),
                            "metrics": metrics.as_posix(),
                        }
                    )
                )
            elif args.normalization_command == "sensitivity":
                print(json.dumps(run_sensitivity_validation(), ensure_ascii=False, sort_keys=True))
        except (OSError, PermissionError, ValueError, RuntimeError) as exc:
            print(f"Normalization operation failed: {exc}", file=sys.stderr)
            return 1
    else:
        build_parser().print_help()
    return 0
