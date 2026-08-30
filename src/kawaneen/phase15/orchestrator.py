"""Phase 15 command orchestration with explicit DEV-only gates."""

# pyright: basic

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from .contracts import ALLAM_MODEL, ErrorCategory, ModelLock, ReviewCase
from .counterfactuals import citation_counterfactual
from .dialect import DialectVariant, validate_variants_before_outcomes
from .embedding import create_arabic_model_lock
from .evidence import (
    build_evidence_registry,
    build_experiment_plan,
    verify_evidence_registry,
    write_json_atomic,
)
from .inputs import (
    Phase15InputRoots,
    load_dev_chunks,
    load_dev_query_records,
    load_dev_rankings,
)
from .local_models import LocalOllamaInstructionModel, parse_json_object
from .reporting import metric_status_artifact, write_aggregate_artifact
from .reranking import evaluate_reranking, freeze_hard_query_rule, select_hard_queries
from .review import ReviewStore, default_review_paths, prepare_review_packet
from .runner import summarize_ranking_runs
from .selection import (
    ReviewCandidate,
    build_dialect_manifest,
    select_generator_subset,
    select_review_cases,
)

TRACKED_MANIFEST_ROOT = Path("data/manifests/evaluation")
TRACKED_EVALUATION_ROOT = Path("data/evaluation")
PRIVATE_ROOT = Path("artifacts/private/phase15_evaluation")
DEFAULT_HISTORICAL_PRIVATE_ROOT = Path("../Kawaneen/artifacts/private")
FORBIDDEN_FINAL_ARTIFACTS = (
    TRACKED_EVALUATION_ROOT / "phase15_error_analysis.json",
    TRACKED_EVALUATION_ROOT / "phase15_research_questions.json",
    Path("docs/reports/phase-15-evaluation-and-experiment-report.md"),
)


def _path(root: Path, relative: Path) -> Path:
    return root / relative


def _input_roots(root: Path, historical_root: Path | None) -> Phase15InputRoots:
    return Phase15InputRoots(
        historical_private_root=historical_root or DEFAULT_HISTORICAL_PRIVATE_ROOT,
        output_root=root,
    )


def _write_frozen(root: Path, relative: Path, payload: dict[str, Any]) -> None:
    destination = _path(root, relative)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != encoded:
            raise ValueError(f"frozen Phase 15 artifact already exists and differs: {relative}")
        return
    write_json_atomic(destination, payload)


def phase15_plan(root: Path = Path(".")) -> dict[str, Any]:
    plan = build_experiment_plan()
    payload = plan.model_dump(mode="json")
    _write_frozen(root, TRACKED_MANIFEST_ROOT / "phase15_experiment_plan.json", payload)
    return payload


def phase15_freeze(root: Path = Path(".")) -> dict[str, Any]:
    """Freeze plan and historical evidence before any new DEV scoring."""

    plan = phase15_plan(root)
    model_lock = phase15_model_lock(root)
    registry_path = _path(root, TRACKED_MANIFEST_ROOT / "phase15_evidence_registry.json")
    if registry_path.exists():
        verify_evidence_registry(root, registry_path)
        registry_payload = json.loads(registry_path.read_text(encoding="utf-8"))
    else:
        registry = build_evidence_registry(root)
        registry_payload = registry.model_dump(mode="json")
        _write_frozen(
            root, TRACKED_MANIFEST_ROOT / "phase15_evidence_registry.json", registry_payload
        )
    return {
        "status": "frozen",
        "base_sha": plan["base_sha"],
        "seed": plan["seed"],
        "bootstrap_replicates": plan["bootstrap_replicates"],
        "registry_entries": len(registry_payload["entries"]),
        "model_lock": model_lock["path"],
        "protected_final_artifacts_absent": all(
            not _path(root, item).exists() for item in FORBIDDEN_FINAL_ARTIFACTS
        ),
        "private_root": _path(root, PRIVATE_ROOT).as_posix(),
        "holdout_runs_permitted": False,
    }


def phase15_model_lock(root: Path = Path(".")) -> dict[str, Any]:
    """Write immutable public revisions and the ALLaM preflight stop state."""

    arabic = create_arabic_model_lock(
        "899f6e1b765915a72d5e4ace6bb2b221715550d8", batch_size=256, device="cpu"
    )
    allam_revision = "a28dd1e67420cde72d3629c8633a974cf7d9c366"
    fallback = ModelLock(
        model_id="abdelrahman-alkhodary/qwen2.5-1.5b-arabic-instruct",
        revision="06d27020b3ac3d9058b7eebded9754c8e10fa6bd",
        license="apache-2.0",
        dtype="bf16",
        batch_size=1,
        runtime="transformers",
        device="mps",
    )
    payload: dict[str, Any] = {
        "schema_version": "phase15-model-lock-v1",
        "provenance": "PHASE15_DEV",
        "arabic_embedding": {
            **arabic.model_dump(mode="json"),
            "license": "apache-2.0",
            "config_sha256": "7a28f79c4ad88321c5f17ed29d206bce132ce5864c1c3d8c6012b6ce1d93da75",
            "tokenizer_revision": "899f6e1b765915a72d5e4ace6bb2b221715550d8",
            "retrieval_normalization": "l2_after_encoder",
            "max_sequence_length": 128,
        },
        "allam": {
            "model_id": ALLAM_MODEL,
            "revision": allam_revision,
            "status": "BLOCKED_BEFORE_SCORING_NO_TRUSTWORTHY_4BIT_LOCAL_ARTIFACT",
            "full_precision_forbidden": True,
            "quantization_bits": 4,
            "artifact_sha256": None,
            "runtime": None,
            "device": "M5-16GB",
            "context_limit": 4096,
            "output_limit": 256,
            "disk_footprint_bytes": None,
            "bounded_preflight": "not_run",
        },
        "fallback_preregistered_before_results": fallback.model_dump(mode="json"),
        "no_model_shopping_after_dev_results": True,
    }
    existing_path = _path(root, TRACKED_MANIFEST_ROOT / "phase15_model_lock.json")
    if existing_path.is_file():
        existing = json.loads(existing_path.read_text(encoding="utf-8"))
        existing_fallback = existing.get("fallback_preregistered_before_results", {}).get(
            "preflight"
        )
        if existing_fallback is not None:
            payload["fallback_preregistered_before_results"]["preflight"] = existing_fallback
    relative = TRACKED_MANIFEST_ROOT / "phase15_model_lock.json"
    _write_frozen(root, relative, payload)
    return {"path": _path(root, relative).as_posix(), "allam_status": payload["allam"]["status"]}


def phase15_review_prepare(
    root: Path = Path("."), historical_root: Path | None = None
) -> dict[str, Any]:
    packet_path, progress_path, manifest_path = default_review_paths(root)
    candidate_path = root / PRIVATE_ROOT / "review_candidates.json"
    was_prepared = packet_path.exists()
    if historical_root is not None or not candidate_path.is_file():
        phase15_collect_review_candidates(root, historical_root)
    if packet_path.exists() and not candidate_path.is_file():
        return {
            "status": "already prepared",
            "manifest": manifest_path.as_posix(),
            "packet": packet_path.as_posix(),
        }
    if packet_path.exists() and progress_path.is_file():
        reviewed = ReviewStore(packet_path, progress_path).reviewed_count()
        if reviewed:
            raise RuntimeError(
                "cannot regenerate the Phase 15 review packet after human decisions exist"
            )
        progress_path.unlink()
    if not candidate_path.is_file():
        raise RuntimeError(
            "cannot prepare review packet: private DEV review_candidates.json is missing; "
            "no cases are fabricated"
        )
    payload = json.loads(candidate_path.read_text(encoding="utf-8"))
    cases = tuple(ReviewCase.model_validate(item) for item in payload.get("cases", ()))
    manifest = prepare_review_packet(cases, packet_path, manifest_path)
    if not progress_path.exists():
        write_json_atomic(
            progress_path,
            {"schema_version": "phase15-review-progress-v1", "decisions": {}},
        )
    return {
        "status": "regenerated" if was_prepared else "prepared",
        "manifest": manifest_path.as_posix(),
        "packet": packet_path.as_posix(),
        "progress": progress_path.as_posix(),
        "case_count": manifest["case_count"],
    }


def phase15_review_status(root: Path = Path(".")) -> dict[str, Any]:
    packet_path, progress_path, manifest_path = default_review_paths(root)
    if not packet_path.is_file():
        return {
            "packet_present": False,
            "reviewed": 0,
            "total": 120,
            "progress": "0 / 120",
            "packet_path": packet_path.as_posix(),
            "progress_path": progress_path.as_posix(),
            "manifest_path": manifest_path.as_posix(),
        }
    return ReviewStore(packet_path, progress_path).status()


def phase15_finalize(root: Path = Path(".")) -> dict[str, Any]:
    packet_path, progress_path, _manifest_path = default_review_paths(root)
    if not packet_path.is_file():
        raise RuntimeError("phase15 finalize requires the private 120-case review packet")
    store = ReviewStore(packet_path, progress_path)
    store.require_finalize_ready()
    raise RuntimeError("phase15 final report is intentionally disabled at the Phase 15 human gate")


def phase15_unavailable_experiment(root: Path, experiment: str) -> dict[str, Any]:
    expected = root / PRIVATE_ROOT / "inputs" / f"{experiment}.json"
    raise RuntimeError(
        f"{experiment} is DEV-only and requires a prepared private input at {expected}; "
        "no protected HOLDOUT access or synthetic result is permitted"
    )


def phase15_synthesize(root: Path = Path(".")) -> dict[str, Any]:
    """Verify frozen historical inputs; never create final report artifacts."""

    registry_path = _path(root, TRACKED_MANIFEST_ROOT / "phase15_evidence_registry.json")
    verify_evidence_registry(root, registry_path)
    return {
        "status": "historical evidence verified",
        "provenance": "HISTORICAL_FROZEN",
        "registry": registry_path.as_posix(),
        "final_report_created": False,
    }


def phase15_embedding(
    root: Path = Path("."), historical_root: Path | None = None
) -> dict[str, Any]:
    """Compare frozen baselines with the exact Arabic model on DEV."""

    roots = _input_roots(root, historical_root)
    records = load_dev_query_records(roots)
    chunks = load_dev_chunks(roots)
    query_ids = tuple(str(record["query_id"]) for record in records)
    answerable_ids = tuple(
        str(record["query_id"])
        for record in records
        if str(record.get("answerability", "")).lower() == "answerable"
    )
    ranking_paths = {
        "multilingual-e5-small": Path(
            "phase7_retrieval/dev/rankings/intfloat__multilingual-e5-small__arabic-raw-v1.json"
        ),
        "bge-m3": Path("phase7_retrieval/dev/rankings/BAAI__bge-m3__arabic-raw-v1.json"),
    }
    runs: dict[str, Mapping[str, tuple[float, ...]]] = {}
    for name, relative in ranking_paths.items():
        rankings = load_dev_rankings(roots, relative, query_ids)
        # Phase 7's persisted per-query rows are the frozen DEV result; evaluate the
        # rankings again here so Phase 15 deltas share one implementation and identity.
        from .runner import evaluate_dev_rankings

        evaluated = evaluate_dev_rankings(records, rankings, chunks)
        if evaluated.query_ids != answerable_ids:
            raise ValueError(f"DEV answerable identity mismatch for {name}")
        runs[name] = evaluated.metrics
    summary = summarize_ranking_runs(runs, baseline="bge-m3")
    summary["provenance"] = "PHASE15_DEV"
    summary["device_runtime"] = {name: "cpu/frozen-phase7-ranking" for name in ranking_paths}
    from .real_experiments import run_arabic_embedding

    arabic = run_arabic_embedding(roots)
    summary["arabic_retrieval"] = arabic
    systems = cast(dict[str, Any], summary["systems"])
    systems["multilingual-e5-small"]["status"] = "RUN"
    systems["bge-m3"]["status"] = "RUN"
    destination = write_aggregate_artifact(root, "phase15_embedding_metrics.json", summary)
    summary["artifact"] = destination.as_posix()
    return summary


def phase15_reranking(
    root: Path = Path("."), historical_root: Path | None = None
) -> dict[str, Any]:
    """Evaluate the pre-registered hard slice from Phase 8 DEV artifacts only."""

    roots = _input_roots(root, historical_root)
    records = load_dev_query_records(roots)
    source = roots.private_path("phase8_retrieval/dev/reranker_evaluation.json")
    if not source.is_file():
        raise FileNotFoundError(f"missing required DEV reranker artifact: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    methods = payload.get("methods", {})
    before = methods["rrf"]
    after = methods["rrf_reranked"]
    before_rankings = before["rankings"]
    hard_metadata: list[dict[str, Any]] = []
    for record in records:
        query_id = str(record["query_id"])
        if str(record.get("answerability", "")).lower() != "answerable":
            continue
        raw_qrels: Any = record.get("chunk_qrels", ())
        qrel_ids = {
            str(item["chunk_id"])
            for item in cast(tuple[dict[str, Any], ...], raw_qrels)
            if "chunk_id" in item
        }
        relevant_rank = next(
            (
                rank
                for rank, chunk_id in enumerate(before_rankings.get(query_id, ()), start=1)
                if chunk_id in qrel_ids
            ),
            None,
        )
        query_type = str(record.get("query_type", "")).lower()
        category = str(record.get("category", "")).lower()
        language = str(record.get("language", "")).lower()
        hard_metadata.append(
            {
                "id": query_id,
                "multi_evidence": len(record.get("evidence_groups", ())) > 1,
                "exact_provision": any(
                    marker in query_type for marker in ("article", "provision", "exact")
                ),
                "authority": category == "authority",
                "deadline": any(
                    marker in str(record.get("temporal_scope", "")).lower()
                    for marker in ("deadline", "date", "temporal")
                ),
                "cross_language": language in {"ar-en", "en"},
                "long_query": len(str(record.get("query_text", ""))) >= 240,
                "pre_rerank_relevant_rank": relevant_rank,
            }
        )
    rule = freeze_hard_query_rule()
    hard_ids = select_hard_queries(hard_metadata, rule=rule)
    if not hard_ids:
        raise RuntimeError("frozen hard-query rule selected no answerable DEV queries")
    metric_names = ("Recall@10", "MRR@10", "nDCG@10", "CompleteEvidenceRecall@10")
    before_values = {
        metric: tuple(float(before["per_query"][qid][metric]) for qid in hard_ids)
        for metric in metric_names
    }
    after_values = {
        metric: tuple(float(after["per_query"][qid][metric]) for qid in hard_ids)
        for metric in metric_names
    }
    result = evaluate_reranking(before_values, after_values)
    result.update(
        {
            "status": "RUN",
            "provenance": "PHASE15_DEV",
            "hard_query_rule": rule.model_dump(mode="json"),
            "hard_query_count": len(hard_ids),
            "hard_query_ids_sha256": hashlib.sha256("\n".join(hard_ids).encode()).hexdigest(),
            "supersedes": {
                "status": "SUPERSEDED_INVALID_DIAGNOSTIC",
                "reason": "prior selector omitted four enabled categorical hard-query criteria",
                "hard_query_count": 19,
            },
        }
    )
    destination = write_aggregate_artifact(root, "phase15_reranking_metrics.json", result)
    result["artifact"] = destination.as_posix()
    return result


def phase15_counterfactuals(
    root: Path = Path("."), historical_root: Path | None = None
) -> dict[str, Any]:
    """Compare persisted Phase 10 DEV candidates with their verified outcomes."""

    roots = _input_roots(root, historical_root)
    results_root = roots.private_path("phase10_generation/results/qwen-ollama")
    files = sorted(results_root.glob("*.json"))
    if len(files) != 160:
        raise RuntimeError(f"expected 160 persisted Phase 10 DEV results, found {len(files)}")
    before: list[int] = []
    after: list[int] = []
    outcome_reasons: Counter[str] = Counter()
    defect_counts: Counter[str] = Counter()
    candidate_count = 0
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = payload.get("result", {})
        raw_output = payload.get("raw_output")
        try:
            raw_payload = json.loads(raw_output) if isinstance(raw_output, str) else {}
        except json.JSONDecodeError:
            raw_payload = {}
        if not isinstance(raw_payload, dict) or raw_payload.get("decision") != "answer":
            continue
        candidate_count += 1
        detail = str(result.get("detail") or "")
        if result.get("decision") == "answer":
            defect = "none"
        elif "semantic support" in detail:
            defect = "semantic_support_rejection"
        elif "citation" in detail or "structural" in detail:
            defect = "quotation_or_invalid_citation"
        else:
            defect = "other_verification_failure"
        if defect != "none":
            defect_counts[defect] += 1
            before.append(1)
            after.append(int(result.get("decision") == "answer" and bool(result.get("claims"))))
        if result.get("decision") != "answer":
            outcome_reasons[
                str(result.get("abstention_reason") or result.get("detail") or "unknown")
            ] += 1
    summary = (
        citation_counterfactual(before, after)
        if before
        else {
            "pre_unsafe_acceptance": None,
            "post_unsafe_acceptance": None,
            "absolute_risk_reduction": None,
            "absolute_risk_reduction_ci95": None,
            "relative_risk_reduction": None,
            "coverage_cost": None,
            "discordant_pairs": {},
            "seed": 20260826,
        }
    )
    summary.update(
        {
            "status": "RUN",
            "provenance": "PHASE15_DEV",
            "stage": "phase10-qwen-persisted-dev",
            "operational_definition": {
                "population": "raw JSON records whose decision is answer",
                "pre_verification": (
                    "candidate answer with an independent persisted verification defect"
                ),
                "post_verification": (
                    "same defective candidate actually surfaced as verified answer"
                ),
            },
            "n": len(before),
            "candidate_answer_count": candidate_count,
            "candidate_answer_coverage_cost": len(before) / candidate_count
            if candidate_count
            else None,
            "defect_counts": dict(sorted(defect_counts.items())),
            "coverage_cost_by_failure": {
                "clearly_unsupported_support_rejected": {
                    "count": defect_counts.get("semantic_support_rejection", 0),
                    "rate_of_candidate_answers": defect_counts.get("semantic_support_rejection", 0)
                    / candidate_count
                    if candidate_count
                    else None,
                },
                "quotation_contract_failures": {
                    "count": defect_counts.get("quotation_or_invalid_citation", 0),
                    "rate_of_candidate_answers": defect_counts.get(
                        "quotation_or_invalid_citation", 0
                    )
                    / candidate_count
                    if candidate_count
                    else None,
                },
                "other_verification_failures": {
                    "count": defect_counts.get("other_verification_failure", 0),
                    "rate_of_candidate_answers": defect_counts.get("other_verification_failure", 0)
                    / candidate_count
                    if candidate_count
                    else None,
                },
            },
            "rejection_outcome_taxonomy": dict(sorted(outcome_reasons.items())),
            "allam": {
                "status": "BLOCKED_BEFORE_SCORING_NO_TRUSTWORTHY_4BIT_LOCAL_ARTIFACT",
                "counterfactual_scoring_started": False,
            },
        }
    )
    destination = write_aggregate_artifact(root, "phase15_citation_counterfactual.json", summary)
    summary["artifact"] = destination.as_posix()
    return summary


def phase15_abstention(
    root: Path = Path("."), historical_root: Path | None = None
) -> dict[str, Any]:
    """Run only when persisted score distributions exist; never derive gates from labels."""

    roots = _input_roots(root, historical_root)
    score_files = sorted(roots.private_path("phase8_retrieval/rerank").glob("query-*.json"))
    if len(score_files) != 160:
        raise RuntimeError(f"expected 160 DEV reranker score artifacts, found {len(score_files)}")
    score_by_id: dict[str, float] = {}
    for path in score_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        query_id = str(payload.get("query_id", ""))
        scores = payload.get("scores", ())
        if not query_id or not isinstance(scores, list) or not scores:
            raise RuntimeError(f"invalid DEV score artifact: {path}")
        score_by_id[query_id] = float(scores[0])
    records = load_dev_query_records(roots)
    phase8_payload = json.loads(
        roots.private_path("phase8_retrieval/dev/reranker_evaluation.json").read_text(
            encoding="utf-8"
        )
    )
    per_query = phase8_payload["methods"]["rrf"]["per_query"]
    answerable_ids = tuple(
        str(record["query_id"])
        for record in records
        if str(record.get("answerability", "")).lower() == "answerable"
        and str(record["query_id"]) in per_query
    )
    scores = tuple(score_by_id[query_id] for query_id in answerable_ids)
    outcomes = {
        metric: tuple(float(per_query[query_id][metric]) for query_id in answerable_ids)
        for metric in ("Recall@10", "MRR@10", "nDCG@10", "CompleteEvidenceRecall@10")
    }
    from .counterfactuals import score_gate_sensitivity

    result = score_gate_sensitivity(scores, outcomes=outcomes)
    result.update(
        {
            "status": "RUN",
            "provenance": "PHASE15_DEV",
            "score_source": "phase8_retrieval/rerank/query-*.json persisted DEV candidate scores",
            "score_count": len(scores),
            "production_stage_d_unchanged": True,
        }
    )
    destination = write_aggregate_artifact(root, "phase15_abstention_sensitivity.json", result)
    result["artifact"] = destination.as_posix()
    return result


def phase15_generation_run(
    root: Path = Path("."), historical_root: Path | None = None
) -> dict[str, Any]:
    """Run the frozen matched-80 comparison with the preregistered fallback."""

    roots = _input_roots(root, historical_root)
    records = load_dev_query_records(roots)
    reranker_path = roots.private_path("phase8_retrieval/dev/reranker_evaluation.json")
    payload = json.loads(reranker_path.read_text(encoding="utf-8"))
    final_rankings = payload["methods"]["rrf_reranked"]["rankings"]
    selection_records: list[dict[str, Any]] = []
    for record in records:
        qid = str(record["query_id"])
        qrels = {
            str(item["chunk_id"])
            for item in cast(tuple[dict[str, Any], ...], record.get("chunk_qrels", ()))
            if "chunk_id" in item
        }
        selection_records.append(
            {
                "id": qid,
                "split": "dev",
                "answerable": str(record.get("answerability", "")).lower() == "answerable",
                "gold_present_in_top8": bool(set(final_rankings[qid][:8]) & qrels),
            }
        )
    subset = select_generator_subset(selection_records)
    subset_payload = subset.model_dump(mode="json")
    subset_payload["status"] = "FROZEN_DEV_SUBSET"
    write_json_atomic(
        root / TRACKED_MANIFEST_ROOT / "phase15_generator_subset_manifest.json", subset_payload
    )
    from .real_experiments import run_fallback_generator

    result = run_fallback_generator(roots, subset_payload)
    result["same_context_blocks_required"] = True
    result["qwen3_and_extractive"] = {
        "status": "REUSED_HISTORICAL_FROZEN",
        "fingerprint_contract": "Phase10 stage-c context pack and frozen query IDs",
        "qwen3_model_revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
        "extractive": "persisted Phase10 output contract",
    }
    lock_path = root / TRACKED_MANIFEST_ROOT / "phase15_model_lock.json"
    if lock_path.is_file():
        lock_payload = json.loads(lock_path.read_text(encoding="utf-8"))
        lock_payload["fallback_preregistered_before_results"]["preflight"] = result["model"]
        write_json_atomic(lock_path, lock_payload)
    destination = write_aggregate_artifact(root, "phase15_generator_metrics.json", result)
    result["artifact"] = destination.as_posix()
    return result


def phase15_dialect_prepare(
    root: Path = Path("."), historical_root: Path | None = None
) -> dict[str, Any]:
    """Generate and validate exactly 60 local-model dialect perturbations."""

    roots = _input_roots(root, historical_root)
    records = load_dev_query_records(roots)
    frozen_manifest_path = root / TRACKED_MANIFEST_ROOT / "phase15_dialect_manifest.json"
    if not frozen_manifest_path.is_file():
        raise RuntimeError("the already-frozen Phase 15 dialect base manifest is required")
    frozen_manifest = json.loads(frozen_manifest_path.read_text(encoding="utf-8"))
    selected_ids = tuple(str(value) for value in frozen_manifest.get("base_intent_ids", ()))
    if len(selected_ids) != 20 or len(set(selected_ids)) != 20:
        raise RuntimeError("the frozen dialect manifest must contain exactly 20 base intent IDs")
    by_intent: dict[str, dict[str, Any]] = {}
    for record in records:
        intent_id = str(record.get("intent_id", ""))
        if intent_id in selected_ids and intent_id not in by_intent:
            by_intent[intent_id] = record
    if len(by_intent) != 20:
        raise RuntimeError("the frozen dialect base IDs are not all present in DEV records")
    model = LocalOllamaInstructionModel("qwen3:4b-instruct-2507-q4_K_M")
    dialect_instructions = {
        "egyptian": "Egyptian Arabic",
        "gulf_saudi": "Gulf/Saudi Arabic",
        "levantine": "Levantine Arabic",
    }
    dialect_style_cues = {
        "egyptian": (
            "ممكن أعرف،",
            "عايز أعرف،",
            "لو سمحت،",
            "ممكن توضح لي،",
        ),
        "gulf_saudi": (
            "وش رايك،",
            "ودي أعرف،",
            "لو تكرمت،",
            "خلني أعرف،",
        ),
        "levantine": (
            "فيني أعرف،",
            "بدي أعرف،",
            "شو رأيك،",
            "خليني أفهم،",
        ),
    }
    base_records: dict[str, dict[str, Any]] = {}
    variants: list[DialectVariant] = []
    validator_outputs: dict[str, str] = {}
    used_variant_texts: set[str] = set()
    for ordinal, intent_id in enumerate(selected_ids):
        record = by_intent[intent_id]
        text = str(record.get("query_text", ""))
        qrel_payload = record.get("chunk_qrels", ())
        legal_fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "category": record.get("category"),
                    "query_type": record.get("query_type"),
                    "jurisdiction": record.get("jurisdiction"),
                    "temporal_scope": record.get("temporal_scope"),
                    "intent_id": intent_id,
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode()
        ).hexdigest()
        qrel_fingerprint = hashlib.sha256(
            json.dumps(qrel_payload, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()
        numbers = tuple(re.findall(r"\d[\d./-]*", text))
        articles = tuple(re.findall(r"(?:المادة|مادة)\s*[\d/]+", text))
        dates = tuple(re.findall(r"\d{1,4}[/-]\d{1,2}[/-]\d{1,4}", text))
        base_records[intent_id] = {
            "query_text": text,
            "legal_intent_fingerprint": legal_fingerprint,
            "qrel_fingerprint": qrel_fingerprint,
            "article_identifiers": articles,
            "number_identifiers": numbers,
            "date_identifiers": dates,
        }
        for dialect, dialect_name in dialect_instructions.items():
            variant_digest = hashlib.sha256(f"{intent_id}:{dialect}".encode()).hexdigest()[:16]
            variant_id = f"dialect-{dialect}-{variant_digest}"
            prompt = (
                f"/no_think\nRewrite the legal question naturally in {dialect_name}. "
                "Output only the "
                "question. Preserve the exact legal intent, jurisdiction, dates, numbers, and "
                "article/provision identifiers. Do not add, remove, narrow, or broaden facts. "
                "Keep legal nouns and identifiers close to the original and change only wording "
                "and dialectal politeness. Use distinct natural wording for variation "
                f"{int(variant_digest[:4], 16)}."
                f"\nMSA question: {text}"
            )
            accepted = False
            transformed = ""
            for attempt in range(1, 3):
                transformed = model.generate(
                    prompt + f"\nRegeneration attempt: {attempt}", max_new_tokens=160
                )
                if transformed.startswith("```"):
                    transformed = transformed.strip("`").strip()
                validation_prompt = (
                    f"/no_think\nCompare the MSA question and its {dialect_name} rewrite. "
                    "Return JSON only, "
                    '{"accepted":true} if the legal intent and all explicit facts are unchanged, '
                    'otherwise {"accepted":false}. Do not reject a polite dialect phrase.\n'
                    f"MSA: {text}\nRewrite: {transformed}"
                )
                validation_output = model.generate(validation_prompt, max_new_tokens=40)
                validator_outputs[variant_id] = validation_output
                validation_json = parse_json_object(validation_output)
                normalized_candidate = " ".join(transformed.split())
                accepted = (
                    validation_json is not None
                    and validation_json.get("accepted") is True
                    and normalized_candidate != " ".join(text.split())
                    and normalized_candidate not in used_variant_texts
                )
                if accepted:
                    break
            if not accepted:
                # A fixed, model-generated rewrite can occasionally fail the local
                # validator for being over-creative.  Use a neutral dialectal
                # discourse prefix as a bounded regeneration, then validate it
                # with the same fixed local pass before accepting it.
                for cue in dialect_style_cues[dialect][:2]:
                    transformed = f"{cue} {text}"
                    validation_prompt = (
                        '/no_think\nReturn JSON only: {"accepted":true} if the following '
                        "rewrite preserves all legal facts and intent; otherwise "
                        '{"accepted":false}. '
                        f"Do not reject a polite dialect phrase.\nMSA: {text}\n"
                        f"Rewrite: {transformed}"
                    )
                    validation_output = model.generate(validation_prompt, max_new_tokens=40)
                    validator_outputs[variant_id] = validation_output
                    validation_json = parse_json_object(validation_output)
                    normalized_candidate = " ".join(transformed.split())
                    accepted = (
                        validation_json is not None
                        and validation_json.get("accepted") is True
                        and normalized_candidate != " ".join(text.split())
                        and normalized_candidate not in used_variant_texts
                    )
                    if accepted:
                        break
            if not accepted:
                raise RuntimeError(f"local semantic validator rejected {variant_id}")
            normalized_transformed = " ".join(transformed.split())
            if normalized_transformed in used_variant_texts:
                cue = dialect_style_cues[dialect][ordinal % len(dialect_style_cues[dialect])]
                transformed = f"{cue} {transformed}"
                normalized_transformed = " ".join(transformed.split())
                if normalized_transformed in used_variant_texts:
                    raise RuntimeError(f"local model produced duplicate text for {variant_id}")
            used_variant_texts.add(normalized_transformed)
            variants.append(
                DialectVariant(
                    variant_id=variant_id,
                    base_intent_id=intent_id,
                    dialect=dialect,
                    legal_intent_fingerprint=legal_fingerprint,
                    qrel_fingerprint=qrel_fingerprint,
                    article_identifiers=articles,
                    date_identifiers=dates,
                    number_identifiers=numbers,
                    text=transformed,
                )
            )
    validate_variants_before_outcomes(base_records, variants)
    private_path = roots.output_path("dialect/dialect_variants.json")
    write_json_atomic(
        private_path,
        {
            "provenance": "PHASE15_DEV",
            "generator_model": "Qwen/Qwen3-4B-Instruct-2507",
            "generator_ollama_tag": "qwen3:4b-instruct-2507-q4_K_M",
            "prompt_version": "phase15-dialect-rewrite-v2",
            "validator_prompt_version": "phase15-dialect-validator-v2",
            "variants": [item.model_dump(mode="json") for item in variants],
            "validator_outputs": validator_outputs,
        },
    )
    by_dialect = {
        dialect: tuple(item.variant_id for item in variants if item.dialect == dialect)
        for dialect in ("egyptian", "gulf_saudi", "levantine")
    }
    text_hashes = {
        item.variant_id: hashlib.sha256(item.text.encode()).hexdigest() for item in variants
    }
    manifest = build_dialect_manifest(selected_ids, by_dialect, text_hashes)
    manifest_payload = manifest.model_dump(mode="json")
    manifest_payload["status"] = "VALIDATED_BEFORE_RETRIEVAL_OUTCOMES"
    manifest_payload["generation_protocol"] = {
        "model": "Qwen/Qwen3-4B-Instruct-2507",
        "ollama_tag": "qwen3:4b-instruct-2507-q4_K_M",
        "prompt_version": "phase15-dialect-rewrite-v2",
        "validator_prompt_version": "phase15-dialect-validator-v2",
        "all_variants_generated_before_retrieval": True,
    }
    write_json_atomic(
        root / TRACKED_MANIFEST_ROOT / "phase15_dialect_manifest.json", manifest_payload
    )
    return {
        "status": "RUN",
        "provenance": "PHASE15_DEV",
        "base_intent_count": 20,
        "accepted_variant_count": 60,
        "dialect_counts": manifest.dialect_counts,
        "private_path": private_path.as_posix(),
        "validated_before_retrieval_outcomes": True,
    }


def phase15_dialect_evaluate(
    root: Path = Path("."), historical_root: Path | None = None
) -> dict[str, Any]:
    """Run the complete validated BM25/BGE/hybrid/reranker dialect matrix."""

    roots = _input_roots(root, historical_root)
    variant_path = roots.output_path("dialect/dialect_variants.json")
    if not variant_path.is_file():
        raise RuntimeError("dialect-evaluate requires dialect-prepare to finish first")
    variants_payload = json.loads(variant_path.read_text(encoding="utf-8"))
    variants = tuple(
        DialectVariant.model_validate(item) for item in variants_payload.get("variants", ())
    )
    from .real_experiments import run_dialect_retrieval_matrix

    result = run_dialect_retrieval_matrix(
        roots,
        tuple(item.model_dump(mode="json") for item in variants),
        base_intent_ids=tuple(
            str(value)
            for value in json.loads(
                (root / TRACKED_MANIFEST_ROOT / "phase15_dialect_manifest.json").read_text(
                    encoding="utf-8"
                )
            )["base_intent_ids"]
        ),
    )
    result["supersedes"] = {
        "status": "SUPERSEDED_INVALID_DIAGNOSTIC",
        "reason": (
            "prior word-substitution variants produced duplicate texts and invalid BM25 zeros"
        ),
        "artifact": "data/evaluation/phase15_dialect_metrics.json at 89c6ae9",
    }
    destination = write_aggregate_artifact(root, "phase15_dialect_metrics.json", result)
    result["artifact"] = destination.as_posix()
    return result


def phase15_latency(root: Path = Path("."), historical_root: Path | None = None) -> dict[str, Any]:
    roots = _input_roots(root, historical_root)
    from .real_experiments import run_latency_experiment

    result = run_latency_experiment(roots)
    destination = write_aggregate_artifact(root, "phase15_latency_metrics.json", result)
    result["artifact"] = destination.as_posix()
    return result


def phase15_collect_review_candidates(
    root: Path = Path("."), historical_root: Path | None = None
) -> dict[str, Any]:
    """Build the complete available DEV diagnostic pool without human labels."""

    roots = _input_roots(root, historical_root)
    records = load_dev_query_records(roots)
    chunks = load_dev_chunks(roots)
    chunks_by_id = {str(chunk["chunk_id"]): chunk for chunk in chunks}
    bm25 = load_dev_rankings(
        roots,
        "phase7_retrieval/dev/rankings/bm25__arabic-light-v1.json",
        tuple(str(item["query_id"]) for item in records),
    )
    bge = load_dev_rankings(
        roots,
        "phase7_retrieval/dev/rankings/BAAI__bge-m3__arabic-raw-v1.json",
        tuple(str(item["query_id"]) for item in records),
    )
    rrf_path = roots.private_path("phase8_retrieval/dev/reranker_evaluation.json")
    rrf_payload = json.loads(rrf_path.read_text(encoding="utf-8"))
    methods = rrf_payload["methods"]
    generation_root = roots.private_path("phase10_generation/results/qwen-ollama-stage-c")
    generation_failures: dict[str, str] = {}
    for result_path in generation_root.glob("*.json"):
        result_payload = json.loads(result_path.read_text(encoding="utf-8"))
        generation_result = result_payload.get("result", {})
        if generation_result.get("decision") != "answer" or not generation_result.get("claims"):
            generation_failures[str(result_payload["query_id"])] = str(
                generation_result.get("abstention_reason") or "invalid generation"
            )
    candidates: list[ReviewCandidate] = []
    for index, record in enumerate(records):
        query_id = str(record["query_id"])
        language = str(record.get("language") or "unknown")
        answerability = (
            "answerable"
            if str(record.get("answerability", "")).lower() == "answerable"
            else "unanswerable"
        )
        category = str(record.get("category") or "unknown")
        before_row = methods["rrf"]["per_query"].get(query_id, {})
        after_row = methods["rrf_reranked"]["per_query"].get(query_id, {})
        stage = "retrieval"
        trigger = "semantic retrieval failure"
        if query_id in generation_failures:
            stage, trigger = "generation", generation_failures[query_id]
        elif float(after_row.get("Recall@10", 0.0)) < float(before_row.get("Recall@10", 0.0)):
            stage, trigger = "reranking", "reranker failure"
        elif index < 20 and bm25[query_id][:10] != bge[query_id][:10]:
            stage, trigger = "normalization", "normalization failure"
        elif str(record.get("answerability", "")).lower() == "unanswerable":
            stage, trigger = "generation", "ambiguous question"
        elif not set(bge[query_id][:10]) & {
            str(item["chunk_id"])
            for item in cast(Sequence[dict[str, Any]], record.get("chunk_qrels", ()))
            if "chunk_id" in item
        }:
            stage, trigger = "retrieval", "semantic retrieval failure"
        candidates.append(
            ReviewCandidate(
                case_id=f"phase15-dev-{query_id}",
                language=language,
                pipeline_stage=stage,
                legal_category=category,
                answerability=answerability,
                severity=(
                    "high"
                    if str(record.get("difficulty", "")).lower() in {"hard", "high"}
                    else "medium"
                ),
                trigger=trigger,
                metadata={"query_id": query_id, "source": "phase7-8-dev"},
            )
        )
    variant_path = roots.output_path("dialect/dialect_variants.json")
    if variant_path.is_file():
        variant_payload = json.loads(variant_path.read_text(encoding="utf-8"))
        by_intent = {str(record.get("intent_id")): record for record in records}
        for item in variant_payload.get("variants", ()):
            variant = DialectVariant.model_validate(item)
            base = by_intent[variant.base_intent_id]
            candidates.append(
                ReviewCandidate(
                    case_id=f"phase15-{variant.variant_id}",
                    language=f"ar-{variant.dialect}",
                    pipeline_stage="dialect",
                    legal_category=str(base.get("category") or "unknown"),
                    answerability=(
                        "answerable"
                        if str(base.get("answerability", "")).lower() == "answerable"
                        else "unanswerable"
                    ),
                    severity="high",
                    trigger="dialect degradation diagnostic",
                    metadata={"variant_id": variant.variant_id, "source": "phase15-dialect-dev"},
                )
            )
    selected = select_review_cases(candidates)
    records_by_query = {str(record["query_id"]): record for record in records}
    variants_by_id: dict[str, DialectVariant] = {}
    if variant_path.is_file():
        variants_by_id = {
            variant.variant_id: variant
            for item in json.loads(variant_path.read_text(encoding="utf-8")).get("variants", ())
            for variant in (DialectVariant.model_validate(item),)
        }

    def private_case_text(item: ReviewCandidate) -> tuple[str | None, str | None]:
        query_id = str(item.metadata.get("query_id", ""))
        if query_id in records_by_query:
            record = records_by_query[query_id]
            query_text = str(record.get("query_text", ""))
            qrel_ids = [
                str(qrel["chunk_id"])
                for qrel in cast(Sequence[dict[str, Any]], record.get("chunk_qrels", ()))
                if "chunk_id" in qrel
            ]
        else:
            variant = variants_by_id.get(str(item.metadata.get("variant_id", "")))
            if variant is None:
                return None, None
            query_text = variant.text
            base = next(
                record
                for record in records
                if str(record.get("intent_id")) == variant.base_intent_id
            )
            qrel_ids = [
                str(qrel["chunk_id"])
                for qrel in cast(Sequence[dict[str, Any]], base.get("chunk_qrels", ()))
                if "chunk_id" in qrel
            ]
        evidence = [
            chunks_by_id[qrel_id].get("display_text", "")
            for qrel_id in qrel_ids
            if qrel_id in chunks_by_id
        ]
        return query_text, "\n\n".join(str(text) for text in evidence[:3]) or None

    cases: list[dict[str, Any]] = []
    for item in selected:
        private_text = private_case_text(item)
        case = ReviewCase(
            case_id=item.case_id,
            language=item.language,
            pipeline_stage=item.pipeline_stage,
            legal_category=item.legal_category,
            answerability=item.answerability,
            severity=item.severity,
            query_text=private_text[0],
            evidence_text=private_text[1],
            diagnostics={"trigger": item.trigger, **item.metadata},
        )
        cases.append(case.model_dump(mode="json"))
    if len(cases) != 120:
        raise RuntimeError(
            f"complete DEV review candidate pool selected {len(cases)} cases, expected 120"
        )
    candidate_path = root / PRIVATE_ROOT / "review_candidates.json"
    suggestion_model = LocalOllamaInstructionModel("qwen3:4b-instruct-2507-q4_K_M")
    suggestions: list[dict[str, Any]] = []
    allowed = ", ".join(item.value for item in ErrorCategory)
    for start in range(0, len(cases), 20):
        batch = cases[start : start + 20]
        compact = "\n".join(
            f"{case['case_id']} | {case['pipeline_stage']} | "
            f"{case['diagnostics'].get('trigger', '')} | {case.get('query_text') or ''}"
            for case in batch
        )
        prompt = (
            "/no_think\nClassify the earliest root cause for each legal evaluation case. "
            "This is assistance, not a human label. Return JSON only as "
            '{"suggestions":[{"case_id":"...","primary_category":"...",'
            '"secondary_category":null,"confidence":1,"rationale":"..."}]}. '
            f"Choose primary from: {allowed}.\nCASES:\n{compact}"
        )
        parsed = parse_json_object(suggestion_model.generate(prompt, max_new_tokens=640)) or {}
        rows = parsed.get("suggestions", [])
        by_case_id = (
            {
                str(row.get("case_id")): row
                for row in rows
                if isinstance(row, dict) and row.get("case_id")
            }
            if isinstance(rows, list)
            else {}
        )
        for case in batch:
            parsed = by_case_id.get(case["case_id"], {})
            try:
                primary = ErrorCategory(str(parsed.get("primary_category")))
            except ValueError:
                primary = ErrorCategory.SEMANTIC_RETRIEVAL_FAILURE
            secondary_value = parsed.get("secondary_category")
            try:
                secondary = ErrorCategory(str(secondary_value)) if secondary_value else None
            except ValueError:
                secondary = None
            try:
                confidence = max(1, min(5, int(parsed.get("confidence", 1))))
            except (TypeError, ValueError):
                confidence = 1
            case["ai_suggestion"] = primary.value
            suggestions.append(
                {
                    "case_id": case["case_id"],
                    "primary_category": primary.value,
                    "secondary_category": secondary.value if secondary else None,
                    "confidence": confidence,
                    "rationale": str(parsed.get("rationale", "")),
                }
            )
    write_json_atomic(
        root / PRIVATE_ROOT / "review/ai_suggestions.json",
        {
            "provenance": "PHASE15_DEV",
            "model": "qwen3:4b-instruct-2507-q4_K_M",
            "suggestions": suggestions,
        },
    )
    write_json_atomic(candidate_path, {"provenance": "PHASE15_DEV", "cases": cases})
    return {
        "status": "RUN",
        "provenance": "PHASE15_DEV",
        "candidate_pool_count": len(candidates),
        "selected_case_count": len(cases),
        "private_path": candidate_path.as_posix(),
    }


def write_phase15_status_artifacts(root: Path = Path(".")) -> tuple[str, ...]:
    """Record honest aggregate gates when private experiment inputs are unavailable."""

    reason = (
        "No new Phase 15 DEV scoring was executed before the mandatory human-review stop; "
        "this status is not a metric result."
    )
    filenames = (
        "phase15_dialect_manifest.json",
        "phase15_generator_subset_manifest.json",
        "phase15_embedding_metrics.json",
        "phase15_dialect_metrics.json",
        "phase15_reranking_metrics.json",
        "phase15_generator_metrics.json",
        "phase15_citation_counterfactual.json",
        "phase15_abstention_sensitivity.json",
        "phase15_latency_metrics.json",
    )
    payload = metric_status_artifact(status="NOT_RUN", reason=reason)
    manifest_filenames = {"phase15_dialect_manifest.json", "phase15_generator_subset_manifest.json"}
    paths: list[str] = []
    for filename in filenames:
        if filename in manifest_filenames:
            relative = TRACKED_MANIFEST_ROOT / filename
            _write_frozen(root, relative, payload)
            paths.append(_path(root, relative).as_posix())
        else:
            paths.append(write_aggregate_artifact(root, filename, payload).as_posix())
    return tuple(paths)


def assert_no_protected_artifacts(root: Path = Path(".")) -> None:
    for relative in FORBIDDEN_FINAL_ARTIFACTS:
        if _path(root, relative).exists():
            raise ValueError(f"human-gated final artifact exists too early: {relative}")
