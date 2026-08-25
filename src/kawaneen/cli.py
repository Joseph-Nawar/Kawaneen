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
from kawaneen.chunking.orchestrator import (
    chunking_plan,
    run_phase5_chunking,
    validate_phase5_chunking,
)
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
from kawaneen.evaluation.orchestrator import (
    evaluation_plan,
    evaluation_stats,
    export_review,
    freeze_ai_reviewed_release,
    freeze_evaluation,
    import_review,
    run_build_draft,
    run_build_draft_v3,
    run_build_draft_v4,
    run_build_draft_v5,
    run_build_final_candidate,
    run_source_balance_audit,
    validate_evaluation,
)
from kawaneen.extraction.interactive import run_interactive_dev_annotation
from kawaneen.extraction.orchestration import (
    annotation_progress,
    export_dev_annotation_batch,
    export_dev_annotation_batch_v2,
    export_holdout_annotation_batch,
    extraction_status,
    freeze_holdout_annotation_release,
    freeze_stage_b2_configuration,
    import_adjudicated_holdout,
    import_reviewed_dev,
    import_reviewed_holdout,
    next_dev_annotation,
    prepare_annotations,
    run_deterministic_split,
    run_hybrid_split,
    save_dev_annotation,
    validate_annotations,
    write_dev_candidate_audit_v2,
)
from kawaneen.extraction.orchestration import (
    evaluate_split as extraction_evaluate,
)
from kawaneen.generation.ollama import (
    LOCAL_OLLAMA_LOCK_PATH,
    UrllibOllamaTransport,
    inspect_ollama_model,
    write_local_model_lock,
)
from kawaneen.generation.orchestration import (
    generation_readiness,
    generation_status,
    run_dev_generation,
)
from kawaneen.generation.registry import default_model_registry
from kawaneen.generation.timeout_diagnostic import (
    evaluate_persisted_timeout_diagnostic,
    evaluate_persisted_timeout_diagnostic_v2,
    run_stage_b_timeout_diagnostic,
    run_stage_b_timeout_diagnostic_v2,
    timeout_diagnostic_status,
    timeout_diagnostic_v2_status,
)
from kawaneen.grounding.dev import assemble_dev as assemble_grounding_dev
from kawaneen.grounding.dev import audit_dev as audit_grounding_dev
from kawaneen.normalization.orchestrator import (
    normalization_plan,
    run_phase4_experiment,
)
from kawaneen.normalization.sensitivity import run_sensitivity_validation
from kawaneen.parsing.benchmark import preflight_pdfs, qualification_status
from kawaneen.parsing.diagnostics import diagnose_docling
from kawaneen.phase15.orchestrator import (
    phase15_finalize,
    phase15_freeze,
    phase15_model_lock,
    phase15_plan,
    phase15_review_prepare,
    phase15_review_status,
    phase15_synthesize,
    phase15_unavailable_experiment,
)
from kawaneen.retrieval.hybrid.finalization import finalize_phase8_holdout
from kawaneen.retrieval.hybrid.orchestration import (
    finalize_phase8_dev_selection,
    phase8_holdout,
    phase8_status,
    rerank_dev,
    run_dev_fusion,
)
from kawaneen.retrieval.orchestrator import (
    build_final_report,
    build_retrieval_corpus,
    cache_status,
    dense_sanity_audit,
    encode_corpus,
    evaluate_dev,
    evaluate_holdout,
    freeze_dev_selection,
    real_model_smoke,
    recover_holdout_artifacts,
    retrieval_plan,
    retrieval_report,
    retrieval_smoke,
    verify_holdout_readiness,
)
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
    api_parser = subparsers.add_parser("api", help="production API serving")
    api_subparsers = api_parser.add_subparsers(dest="api_command", required=True)
    serve_parser = api_subparsers.add_parser("serve", help="serve the versioned FastAPI API")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
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
    chunking_parser = subparsers.add_parser(
        "chunking", help="private Phase 5 legal structure and chunking experiments"
    )
    chunking_subparsers = chunking_parser.add_subparsers(dest="chunking_command", required=True)
    chunking_subparsers.add_parser("plan", help="show versioned chunking policies")
    chunking_subparsers.add_parser("build", help="build private chunk views")
    chunking_subparsers.add_parser("experiment", help="run the private chunking ablation")
    chunking_subparsers.add_parser("validate", help="validate sanitized Phase 5 artifacts")
    evaluation_parser = subparsers.add_parser(
        "evaluation", help="private Phase 6 retrieval evaluation dataset workflow"
    )
    evaluation_subparsers = evaluation_parser.add_subparsers(
        dest="evaluation_command", required=True
    )
    evaluation_subparsers.add_parser("plan", help="show Phase 6 scope and review gates")
    evaluation_subparsers.add_parser("build-draft", help="build the private draft candidate pool")
    v3_parser = evaluation_subparsers.add_parser(
        "build-draft-v3", help="regenerate the private semantic-target draft-v3"
    )
    v3_parser.add_argument("--review-file", type=Path, required=True)
    v4_parser = evaluation_subparsers.add_parser(
        "build-draft-v4", help="apply the bounded external review to private draft-v3"
    )
    v4_parser.add_argument("--review-file", type=Path, required=True)
    v5_parser = evaluation_subparsers.add_parser(
        "build-draft-v5", help="apply the final bounded external review to private draft-v4"
    )
    v5_parser.add_argument("--review-file", type=Path, required=True)
    final_parser = evaluation_subparsers.add_parser(
        "build-final-candidate", help="apply the final literal external patch to private v5"
    )
    final_parser.add_argument("--patch-file", type=Path, required=True)
    evaluation_subparsers.add_parser(
        "balance-audit", help="run the bounded pre-review source-balance audit"
    )
    evaluation_subparsers.add_parser("export-review", help="export the private review packet")
    import_parser = evaluation_subparsers.add_parser(
        "import-review", help="import private review decisions"
    )
    import_parser.add_argument("--file", type=Path, required=True)
    evaluation_subparsers.add_parser("validate", help="run non-retrieval evaluation validation")
    evaluation_subparsers.add_parser("freeze", help="freeze v1 only after human review gates pass")
    evaluation_subparsers.add_parser(
        "freeze-ai-reviewed",
        help="freeze the externally AI-reviewed engineering release without human attestations",
    )
    evaluation_subparsers.add_parser("stats", help="show sanitized draft/review statistics")
    retrieval_parser = subparsers.add_parser(
        "retrieval", help="Phase 7 reproducible retrieval baselines"
    )
    retrieval_subparsers = retrieval_parser.add_subparsers(dest="retrieval_command", required=True)
    retrieval_subparsers.add_parser("plan", help="show the frozen Phase 7 contract")
    retrieval_subparsers.add_parser("build-corpus", help="verify and manifest the retrieval corpus")
    retrieval_subparsers.add_parser("smoke", help="run offline retrieval pipeline smoke checks")
    encode_parser = retrieval_subparsers.add_parser(
        "encode-corpus", help="resume checkpointed dense corpus encoding"
    )
    encode_parser.add_argument("--model", choices=("bge-m3",), required=True)
    encode_parser.add_argument("--policy", default="arabic-raw-v1")
    encode_parser.add_argument("--device", default="cpu")
    encode_parser.add_argument("--block-size", type=int, default=1024)
    encode_parser.add_argument("--resume", action="store_true")
    status_parser = retrieval_subparsers.add_parser(
        "cache-status", help="show text-free dense checkpoint progress"
    )
    status_parser.add_argument("--model", choices=("bge-m3",), required=True)
    status_parser.add_argument("--policy", default="arabic-raw-v1")
    retrieval_subparsers.add_parser(
        "real-model-smoke", help="load locked Hugging Face models for a local smoke check"
    )
    retrieval_subparsers.add_parser("evaluate-dev", help="evaluate all dev baselines")
    retrieval_subparsers.add_parser(
        "dense-sanity-audit", help="audit selected dense retrieval on deterministic DEV samples"
    )
    retrieval_subparsers.add_parser(
        "freeze-dev-selection", help="freeze dev normalization/model selection"
    )
    holdout_parser = retrieval_subparsers.add_parser(
        "evaluate-holdout", help="evaluate frozen baselines on holdout"
    )
    holdout_parser.add_argument("--allow-holdout", action="store_true")
    recovery_parser = retrieval_subparsers.add_parser(
        "recover-holdout-artifacts", help="replay holdout once to recover private observability"
    )
    recovery_parser.add_argument("--allow-holdout", action="store_true")
    retrieval_subparsers.add_parser(
        "verify-holdout-readiness", help="check protected holdout gates without consuming holdout"
    )
    retrieval_subparsers.add_parser(
        "phase8-dev-fusion", help="run cheap Phase 8 DEV fusion over frozen Phase 7 artifacts"
    )
    retrieval_subparsers.add_parser(
        "phase8-finalize-dev", help="validate completed Phase 8 reranking and freeze DEV selection"
    )
    retrieval_subparsers.add_parser(
        "phase8-final-report",
        help="evaluate persisted Phase 8 DEV/holdout artifacts and freeze final reporting",
    )
    rerank_dev_parser = retrieval_subparsers.add_parser(
        "phase8-rerank-dev", help="run manually authorized Phase 8 DEV reranking"
    )
    rerank_dev_parser.add_argument("--resume", action="store_true")
    rerank_dev_parser.add_argument("--device", default="cpu")
    holdout_parser = retrieval_subparsers.add_parser(
        "phase8-holdout", help="run the one-shot Phase 8 holdout with private per-query capture"
    )
    holdout_parser.add_argument("--allow-holdout", action="store_true")
    holdout_parser.add_argument("--resume", action="store_true")
    holdout_parser.add_argument("--device", default="cpu")
    retrieval_subparsers.add_parser(
        "phase8-rerank-status",
        help="show Phase 8 checkpoint manifest status without loading a model",
    )
    retrieval_subparsers.add_parser("report", help="show the Phase 7 report")
    retrieval_subparsers.add_parser("final-report", help="write the final Phase 7 report")
    grounding_parser = subparsers.add_parser(
        "grounding", help="deterministic Phase 9 context and citation grounding"
    )
    grounding_subparsers = grounding_parser.add_subparsers(dest="grounding_command", required=True)
    for command in ("assemble-dev", "audit-dev"):
        command_parser = grounding_subparsers.add_parser(command)
        command_parser.add_argument("--max-context-tokens", type=int, default=4096)
    generation_parser = subparsers.add_parser(
        "generation", help="Phase 10 Stage-A local generation and abstention tools"
    )
    generation_subparsers = generation_parser.add_subparsers(
        dest="generation_command", required=True
    )
    generation_subparsers.add_parser("registry", help="show model candidates without loading them")
    lock_ollama_parser = generation_subparsers.add_parser(
        "lock-ollama", help="inspect and lock a manually obtained local Ollama model"
    )
    lock_ollama_parser.add_argument("--model", required=True)
    lock_ollama_parser.add_argument("--endpoint", default="http://localhost:11434")
    lock_ollama_parser.add_argument("--lock-path", type=Path, default=LOCAL_OLLAMA_LOCK_PATH)
    status_parser = generation_subparsers.add_parser(
        "status", help="show text-free resumable Qwen DEV checkpoint status"
    )
    status_parser.add_argument(
        "--generator",
        choices=(
            "qwen-ollama",
            "qwen-ollama-stage-b",
            "qwen-ollama-stage-c",
            "qwen-ollama-stage-d",
        ),
        required=True,
    )
    readiness_parser = generation_subparsers.add_parser(
        "readiness", help="assemble and audit generator contexts without generation"
    )
    readiness_parser.add_argument(
        "--generator",
        choices=(
            "qwen-ollama",
            "qwen-ollama-stage-b",
            "qwen-ollama-stage-c",
            "qwen-ollama-stage-d",
        ),
        required=True,
    )
    run_parser = generation_subparsers.add_parser(
        "run-dev", help="run resumable Qwen DEV generation"
    )
    run_parser.add_argument(
        "--generator",
        choices=(
            "qwen-ollama",
            "qwen-ollama-stage-b",
            "qwen-ollama-stage-c",
            "qwen-ollama-stage-d",
        ),
        required=True,
    )
    run_parser.add_argument("--resume", action="store_true")
    diagnose_timeout_parser = generation_subparsers.add_parser(
        "diagnose-stage-b-timeouts",
        help="replay only the frozen Stage-B timeout cohort for private diagnostics",
    )
    diagnose_timeout_parser.add_argument("--resume", action="store_true")
    generation_subparsers.add_parser(
        "timeout-diagnostic-status",
        help="show private Stage-B timeout diagnostic progress",
    )
    generation_subparsers.add_parser(
        "evaluate-timeout-diagnostic",
        help="evaluate persisted Stage-B timeout envelopes offline",
    )
    generation_subparsers.add_parser(
        "timeout-diagnostic-v2-status",
        help="show private Stage-B timeout diagnostic v2 progress",
    )
    diagnose_timeout_v2_parser = generation_subparsers.add_parser(
        "diagnose-stage-b-timeouts-v2",
        help="replay the frozen Stage-B timeout cohort into the v2 namespace",
    )
    diagnose_timeout_v2_parser.add_argument("--resume", action="store_true")
    generation_subparsers.add_parser(
        "evaluate-timeout-diagnostic-v2",
        help="evaluate persisted Stage-B timeout v2 envelopes offline",
    )
    extraction_parser = subparsers.add_parser(
        "extraction", help="Phase 11 structured regulatory extraction"
    )
    extraction_subparsers = extraction_parser.add_subparsers(
        dest="extraction_command", required=True
    )
    extraction_subparsers.add_parser("status", help="show text-free readiness status")
    extraction_subparsers.add_parser(
        "prepare-annotations", help="prepare the private annotation pack"
    )
    extraction_subparsers.add_parser(
        "export-dev-annotation-batch", help="export the private DEV review batch"
    )
    extraction_subparsers.add_parser(
        "export-dev-annotation-batch-v2", help="export the fresh private Phase-11 v2 DEV batch"
    )
    extraction_subparsers.add_parser(
        "export-holdout-annotation-batch",
        help="export the sealed source-only HOLDOUT annotation batch",
    )
    extraction_subparsers.add_parser(
        "freeze-stage-b2", help="freeze the selected B2 DEV configuration metadata"
    )
    extraction_subparsers.add_parser(
        "audit-dev-candidates-v2", help="write a private deterministic v2 candidate audit"
    )
    import_reviewed_parser = extraction_subparsers.add_parser(
        "import-reviewed-dev", help="import an explicit independent-AI DEV review"
    )
    import_reviewed_parser.add_argument("--file", type=Path, required=True)
    import_reviewed_parser.add_argument("--partial", action="store_true")
    import_reviewed_holdout_parser = extraction_subparsers.add_parser(
        "import-reviewed-holdout", help="import one explicit independent-AI HOLDOUT review"
    )
    import_reviewed_holdout_parser.add_argument("--file", type=Path, required=True)
    extraction_subparsers.add_parser(
        "freeze-holdout-annotations", help="freeze the final protected HOLDOUT reference release"
    )
    import_adjudicated_holdout_parser = extraction_subparsers.add_parser(
        "import-holdout-adjudication", help="apply the sealed HOLDOUT AI adjudication"
    )
    import_adjudicated_holdout_parser.add_argument("--file", type=Path, required=True)
    annotate_extraction_parser = extraction_subparsers.add_parser(
        "annotate-dev", help="inspect or save one private DEV annotation"
    )
    annotate_mode = annotate_extraction_parser.add_mutually_exclusive_group(required=True)
    annotate_mode.add_argument("--next", action="store_true")
    annotate_mode.add_argument("--save", action="store_true")
    annotate_extraction_parser.add_argument("--interactive", action="store_true")
    annotate_extraction_parser.add_argument("--record-id")
    annotate_extraction_parser.add_argument("--annotation-file", type=Path)
    progress_extraction_parser = extraction_subparsers.add_parser(
        "annotation-progress", help="report private DEV annotation progress"
    )
    progress_extraction_parser.add_argument("--split", choices=("dev",), required=True)
    validate_extraction_parser = extraction_subparsers.add_parser(
        "validate-annotations", help="validate private annotation records"
    )
    validate_extraction_parser.add_argument(
        "--split", choices=("dev", "smoke", "holdout"), required=True
    )
    validate_extraction_parser.add_argument("--allow-holdout", action="store_true")
    deterministic_extraction_parser = extraction_subparsers.add_parser(
        "run-deterministic", help="run deterministic extraction over a private split"
    )
    deterministic_extraction_parser.add_argument(
        "--split", choices=("dev", "smoke", "holdout"), required=True
    )
    deterministic_extraction_parser.add_argument("--allow-holdout", action="store_true")
    hybrid_extraction_parser = extraction_subparsers.add_parser(
        "run-hybrid", help="run the locked Phase 11B hybrid extractor over DEV"
    )
    hybrid_extraction_parser.add_argument(
        "--split", choices=("dev", "smoke", "holdout"), required=True
    )
    hybrid_extraction_parser.add_argument("--resume", action="store_true")
    hybrid_extraction_parser.add_argument(
        "--stage",
        choices=("b1-clean", "b2"),
        default="b1-clean",
        help="select the isolated DEV experiment stage",
    )
    hybrid_extraction_parser.add_argument(
        "--retry-timeouts",
        action="store_true",
        help="explicitly retry only first-attempt MODEL_TIMEOUT failures with --resume",
    )
    hybrid_extraction_parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate the clean Stage B2 namespace without making provider calls",
    )
    hybrid_extraction_parser.add_argument("--allow-holdout", action="store_true")
    evaluate_extraction_parser = extraction_subparsers.add_parser(
        "evaluate", help="evaluate only reviewed private annotations"
    )
    evaluate_extraction_parser.add_argument(
        "--extractor", choices=("deterministic-v1", "hybrid-qwen-v1"), required=True
    )
    evaluate_extraction_parser.add_argument(
        "--split", choices=("dev", "smoke", "holdout"), required=True
    )
    evaluate_extraction_parser.add_argument("--allow-holdout", action="store_true")
    phase15_parser = subparsers.add_parser(
        "phase15", help="Phase 15 frozen-evidence and DEV-only evaluation workflow"
    )
    phase15_subparsers = phase15_parser.add_subparsers(dest="phase15_command", required=True)
    phase15_commands = (
        "plan",
        "freeze",
        "synthesize",
        "embedding",
        "dialect-prepare",
        "dialect-evaluate",
        "reranking",
        "generation-preflight",
        "generation-run",
        "counterfactuals",
        "latency",
        "review-prepare",
        "review-status",
        "finalize",
    )
    phase15_help = {
        "plan": "show the frozen Phase 15 research plan",
        "freeze": "freeze the plan, evidence registry, and model identities",
        "synthesize": "verify historical frozen evidence without creating a report",
        "embedding": "run the registered Arabic embedding DEV comparison",
        "dialect-prepare": "prepare and validate the private dialect packet",
        "dialect-evaluate": "evaluate the frozen dialect packet on DEV",
        "reranking": "analyze the frozen hard-query reranking slice",
        "generation-preflight": "run the bounded local ALLaM 4-bit preflight",
        "generation-run": "run the matched-80 DEV generator comparison",
        "counterfactuals": "run offline citation and abstention counterfactuals",
        "latency": "run the fixed batch-1 latency-quality protocol",
        "review-prepare": "prepare the private 120-case review packet",
        "review-status": "show private human-review progress",
        "finalize": "finalize only after the human review gate",
    }
    for command in phase15_commands:
        phase15_subparsers.add_parser(command, help=phase15_help[command])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        print("Kawaneen foundation: ready")
    elif args.command == "api":
        if args.api_command == "serve":
            import uvicorn

            uvicorn.run(
                "kawaneen.api.app:create_app",
                factory=True,
                host=args.host,
                port=args.port,
            )
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
    elif args.command == "chunking":
        try:
            if args.chunking_command == "plan":
                print(json.dumps(chunking_plan(), ensure_ascii=False, sort_keys=True))
            elif args.chunking_command in {"build", "experiment"}:
                print(json.dumps(run_phase5_chunking(), ensure_ascii=False, sort_keys=True))
            elif args.chunking_command == "validate":
                print(json.dumps(validate_phase5_chunking(), ensure_ascii=False, sort_keys=True))
        except (OSError, PermissionError, ValueError, RuntimeError) as exc:
            print(f"Chunking operation failed: {exc}", file=sys.stderr)
            return 1
    elif args.command == "evaluation":
        try:
            if args.evaluation_command == "plan":
                print(json.dumps(evaluation_plan(), ensure_ascii=False, sort_keys=True))
            elif args.evaluation_command == "build-draft":
                print(json.dumps(run_build_draft(), ensure_ascii=False, sort_keys=True))
            elif args.evaluation_command == "build-draft-v3":
                print(
                    json.dumps(
                        run_build_draft_v3(review_file=args.review_file),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
            elif args.evaluation_command == "build-draft-v4":
                print(
                    json.dumps(
                        run_build_draft_v4(review_file=args.review_file),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
            elif args.evaluation_command == "build-draft-v5":
                print(
                    json.dumps(
                        run_build_draft_v5(review_file=args.review_file),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
            elif args.evaluation_command == "build-final-candidate":
                print(
                    json.dumps(
                        run_build_final_candidate(patch_file=args.patch_file),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
            elif args.evaluation_command == "balance-audit":
                print(json.dumps(run_source_balance_audit(), ensure_ascii=False, sort_keys=True))
            elif args.evaluation_command == "export-review":
                print(json.dumps(export_review(), ensure_ascii=False, sort_keys=True))
            elif args.evaluation_command == "import-review":
                print(json.dumps(import_review(args.file), ensure_ascii=False, sort_keys=True))
            elif args.evaluation_command == "validate":
                print(json.dumps(validate_evaluation(), ensure_ascii=False, sort_keys=True))
            elif args.evaluation_command == "freeze":
                print(json.dumps(freeze_evaluation(), ensure_ascii=False, sort_keys=True))
            elif args.evaluation_command == "freeze-ai-reviewed":
                print(json.dumps(freeze_ai_reviewed_release(), ensure_ascii=False, sort_keys=True))
            elif args.evaluation_command == "stats":
                print(json.dumps(evaluation_stats(), ensure_ascii=False, sort_keys=True))
        except (OSError, PermissionError, ValueError, RuntimeError) as exc:
            print(f"Evaluation operation failed: {exc}", file=sys.stderr)
            return 1
    elif args.command == "phase15":
        try:
            if args.phase15_command == "plan":
                result = phase15_plan()
            elif args.phase15_command == "freeze":
                result = phase15_freeze()
            elif args.phase15_command == "synthesize":
                result = phase15_synthesize()
            elif args.phase15_command == "review-prepare":
                result = phase15_review_prepare()
            elif args.phase15_command == "review-status":
                result = phase15_review_status()
            elif args.phase15_command == "finalize":
                result = phase15_finalize()
            elif args.phase15_command == "generation-preflight":
                result = phase15_model_lock()
            else:
                result = phase15_unavailable_experiment(Path("."), args.phase15_command)
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        except (OSError, PermissionError, ValueError, RuntimeError) as exc:
            print(f"Phase 15 operation failed: {exc}", file=sys.stderr)
            return 1
    elif args.command == "retrieval":
        try:
            if args.retrieval_command == "plan":
                print(json.dumps(retrieval_plan(), ensure_ascii=False, sort_keys=True))
            elif args.retrieval_command == "build-corpus":
                print(json.dumps(build_retrieval_corpus(), ensure_ascii=False, sort_keys=True))
            elif args.retrieval_command == "smoke":
                print(json.dumps(retrieval_smoke(), ensure_ascii=False, sort_keys=True))
            elif args.retrieval_command == "encode-corpus":
                print(
                    json.dumps(
                        encode_corpus(
                            model=args.model,
                            policy_id=args.policy,
                            device=args.device,
                            block_size=args.block_size,
                            resume=args.resume,
                        ),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
            elif args.retrieval_command == "cache-status":
                print(
                    json.dumps(
                        cache_status(model=args.model, policy_id=args.policy),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
            elif args.retrieval_command == "real-model-smoke":
                print(json.dumps(real_model_smoke(), ensure_ascii=False, sort_keys=True))
            elif args.retrieval_command == "evaluate-dev":
                print(json.dumps(evaluate_dev(), ensure_ascii=False, sort_keys=True))
            elif args.retrieval_command == "dense-sanity-audit":
                print(json.dumps(dense_sanity_audit(), ensure_ascii=False, sort_keys=True))
            elif args.retrieval_command == "freeze-dev-selection":
                print(json.dumps(freeze_dev_selection(), ensure_ascii=False, sort_keys=True))
            elif args.retrieval_command == "evaluate-holdout":
                print(
                    json.dumps(
                        evaluate_holdout(allow_holdout=args.allow_holdout),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
            elif args.retrieval_command == "recover-holdout-artifacts":
                print(
                    json.dumps(
                        recover_holdout_artifacts(allow_holdout=args.allow_holdout),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
            elif args.retrieval_command == "verify-holdout-readiness":
                print(json.dumps(verify_holdout_readiness(), ensure_ascii=False, sort_keys=True))
            elif args.retrieval_command == "phase8-dev-fusion":
                print(json.dumps(run_dev_fusion(), ensure_ascii=False, sort_keys=True))
            elif args.retrieval_command == "phase8-finalize-dev":
                print(
                    json.dumps(finalize_phase8_dev_selection(), ensure_ascii=False, sort_keys=True)
                )
            elif args.retrieval_command == "phase8-final-report":
                print(json.dumps(finalize_phase8_holdout(), ensure_ascii=False, sort_keys=True))
            elif args.retrieval_command == "phase8-rerank-dev":
                print(
                    json.dumps(
                        rerank_dev(resume=args.resume, device=args.device),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
            elif args.retrieval_command == "phase8-rerank-status":
                print(json.dumps(phase8_status(), ensure_ascii=False, sort_keys=True))
            elif args.retrieval_command == "phase8-holdout":
                print(
                    json.dumps(
                        phase8_holdout(
                            allow_holdout=args.allow_holdout,
                            resume=args.resume,
                            device=args.device,
                        ),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
            elif args.retrieval_command == "report":
                print(json.dumps(retrieval_report(), ensure_ascii=False, sort_keys=True))
            elif args.retrieval_command == "final-report":
                print(json.dumps(build_final_report(), ensure_ascii=False, sort_keys=True))
        except (OSError, PermissionError, ValueError, RuntimeError) as exc:
            print(f"Retrieval operation failed: {exc}", file=sys.stderr)
            return 1
    elif args.command == "grounding":
        try:
            if args.grounding_command == "assemble-dev":
                print(
                    json.dumps(
                        assemble_grounding_dev(max_context_tokens=args.max_context_tokens),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
            elif args.grounding_command == "audit-dev":
                print(
                    json.dumps(
                        audit_grounding_dev(max_context_tokens=args.max_context_tokens),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
        except (OSError, PermissionError, ValueError, RuntimeError) as exc:
            print(f"Grounding operation failed: {exc}", file=sys.stderr)
            return 1
    elif args.command == "generation":
        try:
            if args.generation_command == "registry":
                print(
                    json.dumps(
                        [
                            candidate.model_dump(mode="json")
                            for candidate in default_model_registry()
                        ],
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
            elif args.generation_command == "lock-ollama":
                candidate = next(
                    (item for item in default_model_registry() if item.ollama_model == args.model),
                    None,
                )
                if candidate is None:
                    raise ValueError(f"unknown registered Ollama model: {args.model}")
                identity = inspect_ollama_model(
                    args.endpoint,
                    args.model,
                    UrllibOllamaTransport(),
                )
                write_local_model_lock(args.lock_path, identity)
                print(
                    json.dumps(
                        {
                            "model": candidate.ollama_model,
                            "digest": identity.digest,
                            "lock_path": args.lock_path.as_posix(),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
            elif args.generation_command == "status":
                print(
                    json.dumps(
                        generation_status(args.generator),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
            elif args.generation_command == "readiness":
                print(
                    json.dumps(
                        generation_readiness(args.generator),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
            elif args.generation_command == "run-dev":
                print(
                    json.dumps(
                        run_dev_generation(
                            generator_name=args.generator,
                            resume=args.resume,
                        ),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
            elif args.generation_command == "diagnose-stage-b-timeouts":
                print(
                    json.dumps(
                        run_stage_b_timeout_diagnostic(resume=args.resume),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
            elif args.generation_command == "timeout-diagnostic-status":
                print(json.dumps(timeout_diagnostic_status(), ensure_ascii=False, sort_keys=True))
            elif args.generation_command == "evaluate-timeout-diagnostic":
                print(
                    json.dumps(
                        evaluate_persisted_timeout_diagnostic(),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
            elif args.generation_command == "timeout-diagnostic-v2-status":
                print(
                    json.dumps(
                        timeout_diagnostic_v2_status(),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
            elif args.generation_command == "diagnose-stage-b-timeouts-v2":
                print(
                    json.dumps(
                        run_stage_b_timeout_diagnostic_v2(resume=args.resume),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
            elif args.generation_command == "evaluate-timeout-diagnostic-v2":
                print(
                    json.dumps(
                        evaluate_persisted_timeout_diagnostic_v2(),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
        except (OSError, PermissionError, ValueError, RuntimeError) as exc:
            print(f"Generation operation failed: {exc}", file=sys.stderr)
            return 1
    elif args.command == "extraction":
        try:
            if args.extraction_command == "status":
                print(json.dumps(extraction_status(), ensure_ascii=False, sort_keys=True))
            elif args.extraction_command == "prepare-annotations":
                print(json.dumps(prepare_annotations(), ensure_ascii=False, sort_keys=True))
            elif args.extraction_command == "export-dev-annotation-batch":
                print(json.dumps(export_dev_annotation_batch(), ensure_ascii=False, sort_keys=True))
            elif args.extraction_command == "export-dev-annotation-batch-v2":
                print(
                    json.dumps(export_dev_annotation_batch_v2(), ensure_ascii=False, sort_keys=True)
                )
            elif args.extraction_command == "export-holdout-annotation-batch":
                print(
                    json.dumps(
                        export_holdout_annotation_batch(), ensure_ascii=False, sort_keys=True
                    )
                )
            elif args.extraction_command == "freeze-stage-b2":
                print(
                    json.dumps(freeze_stage_b2_configuration(), ensure_ascii=False, sort_keys=True)
                )
            elif args.extraction_command == "audit-dev-candidates-v2":
                print(
                    json.dumps(write_dev_candidate_audit_v2(), ensure_ascii=False, sort_keys=True)
                )
            elif args.extraction_command == "import-reviewed-dev":
                print(
                    json.dumps(
                        import_reviewed_dev(args.file, partial=args.partial),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
            elif args.extraction_command == "import-reviewed-holdout":
                print(
                    json.dumps(
                        import_reviewed_holdout(args.file),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
            elif args.extraction_command == "import-holdout-adjudication":
                print(
                    json.dumps(
                        import_adjudicated_holdout(args.file),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
            elif args.extraction_command == "freeze-holdout-annotations":
                print(
                    json.dumps(
                        freeze_holdout_annotation_release(),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
            elif args.extraction_command == "annotate-dev":
                if args.next:
                    if args.interactive:
                        print(
                            json.dumps(
                                run_interactive_dev_annotation(),
                                ensure_ascii=False,
                                sort_keys=True,
                            )
                        )
                    else:
                        print(json.dumps(next_dev_annotation(), ensure_ascii=False, sort_keys=True))
                else:
                    if args.interactive:
                        raise ValueError("--interactive requires --next")
                    if args.record_id is None or args.annotation_file is None:
                        raise ValueError("--save requires --record-id and --annotation-file")
                    print(
                        json.dumps(
                            save_dev_annotation(args.record_id, args.annotation_file),
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                    )
            elif args.extraction_command == "annotation-progress":
                print(
                    json.dumps(annotation_progress(args.split), ensure_ascii=False, sort_keys=True)
                )
            elif args.extraction_command == "validate-annotations":
                print(
                    json.dumps(
                        validate_annotations(args.split, allow_holdout=args.allow_holdout),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
            elif args.extraction_command == "run-deterministic":
                print(
                    json.dumps(
                        run_deterministic_split(args.split, allow_holdout=args.allow_holdout),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
            elif args.extraction_command == "run-hybrid":
                print(
                    json.dumps(
                        run_hybrid_split(
                            args.split,
                            stage=args.stage,
                            resume=args.resume,
                            retry_timeouts=args.retry_timeouts,
                            allow_holdout=args.allow_holdout,
                            preflight_only=args.preflight_only,
                        ),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
            elif args.extraction_command == "evaluate":
                print(
                    json.dumps(
                        extraction_evaluate(
                            args.extractor,
                            args.split,
                            allow_holdout=args.allow_holdout,
                        ),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
        except (OSError, PermissionError, ValueError, RuntimeError) as exc:
            print(f"Extraction operation failed: {exc}", file=sys.stderr)
            return 1
    else:
        build_parser().print_help()
    return 0
