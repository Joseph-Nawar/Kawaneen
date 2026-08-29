"""Phase 15 command orchestration with explicit DEV-only gates."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from kawaneen.retrieval.bm25 import BM25Index
from kawaneen.retrieval.models import RetrievalChunk

from .contracts import ALLAM_MODEL, ModelLock, ReviewCase
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
from .statistics import paired_bootstrap_delta

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

    arabic = create_arabic_model_lock("899f6e1b765915a72d5e4ace6bb2b221715550d8")
    allam_revision = "a28dd1e67420cde72d3629c8633a974cf7d9c366"
    fallback = ModelLock(
        model_id="abdelrahman-alkhodary/qwen2.5-1.5b-arabic-instruct",
        revision="06d27020b3ac3d9058b7eebded9754c8e10fa6bd",
        license="apache-2.0",
        dtype="bf16",
        batch_size=1,
        runtime="transformers",
        device="cpu-or-mps",
    )
    payload: dict[str, Any] = {
        "schema_version": "phase15-model-lock-v1",
        "provenance": "PHASE15_DEV",
        "arabic_embedding": arabic.model_dump(mode="json"),
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
    """Score the frozen DEV ranking outputs and record an honest Arabic-model gate."""

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
    summary["arabic_retrieval"] = {
        "status": "BLOCKED",
        "reason": (
            "Exact locked Arabic-Retrieval-v1.0 revision is not present locally; bounded model "
            "load attempted at the locked revision and was interrupted during weight download."
        ),
        "model_id": "omarelshehy/Arabic-Retrieval-v1.0",
        "revision": "899f6e1b765915a72d5e4ace6bb2b221715550d8",
        "normalizations": ["raw", "light", "aggressive"],
        "scoring_started": False,
    }
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
    results_root = roots.private_path("phase10_generation/results/qwen-ollama-stage-c")
    files = sorted(results_root.glob("*.json"))
    if len(files) != 160:
        raise RuntimeError(f"expected 160 persisted Phase 10 DEV results, found {len(files)}")
    before: list[int] = []
    after: list[int] = []
    outcome_reasons: Counter[str] = Counter()
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = payload.get("result", {})
        raw_output = payload.get("raw_output")
        claims = result.get("claims", ())
        before.append(int(bool(raw_output)))
        after.append(int(result.get("decision") == "answer" and bool(claims)))
        if not after[-1]:
            outcome_reasons[
                str(result.get("abstention_reason") or result.get("detail") or "unknown")
            ] += 1
    summary = citation_counterfactual(before, after)
    summary.update(
        {
            "status": "RUN",
            "provenance": "PHASE15_DEV",
            "stage": "phase10-qwen-stage-c",
            "operational_definition": {
                "pre_verification": "non-empty persisted candidate response",
                "post_verification": "persisted answer decision with at least one verified claim",
            },
            "n": len(files),
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
    score_path = roots.private_path("phase8_retrieval/dev/scores.json")
    if not score_path.is_file():
        result: dict[str, Any] = {
            "status": "BLOCKED",
            "provenance": "PHASE15_DEV",
            "reason": f"required score-only DEV artifact is absent: {score_path}",
            "gates": {
                name: {"status": "BLOCKED"} for name in ("none", "bottom10", "bottom25", "bottom50")
            },
            "production_stage_d_unchanged": True,
        }
    else:
        raise RuntimeError(
            "score artifact exists but this checkout lacks the registered score loader; "
            "refusing an unreviewed gate"
        )
    destination = write_aggregate_artifact(root, "phase15_abstention_sensitivity.json", result)
    result["artifact"] = destination.as_posix()
    return result


def phase15_generation_run(
    root: Path = Path("."), historical_root: Path | None = None
) -> dict[str, Any]:
    """Freeze the matched-80 population and stop before an unavailable fallback is scored."""

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
    fallback_cache = Path.home() / (
        ".cache/huggingface/hub/models--abdelrahman-alkhodary--qwen2.5-1.5b-arabic-instruct"
    )
    result: dict[str, Any] = {
        "status": "BLOCKED",
        "provenance": "PHASE15_DEV",
        "reason": (
            f"fallback weights are not locally available for offline scoring: {fallback_cache}"
        ),
        "matched_80": {
            "answerable_gold_present": 31,
            "answerable_gold_absent": 30,
            "explicit_unanswerable": 19,
        },
        "same_context_blocks_required": True,
        "qwen3_and_extractive": (
            "not scored by this Phase 15 runner; persisted outputs require fingerprint match"
        ),
        "fallback": {
            "model_id": "abdelrahman-alkhodary/qwen2.5-1.5b-arabic-instruct",
            "revision": "06d27020b3ac3d9058b7eebded9754c8e10fa6bd",
            "license": "apache-2.0",
            "scoring_started": False,
        },
    }
    destination = write_aggregate_artifact(root, "phase15_generator_metrics.json", result)
    result["artifact"] = destination.as_posix()
    return result


def phase15_dialect_prepare(
    root: Path = Path("."), historical_root: Path | None = None
) -> dict[str, Any]:
    """Create and validate exactly 60 private diagnostic dialect perturbations."""

    roots = _input_roots(root, historical_root)
    records = load_dev_query_records(roots)
    by_intent: dict[str, dict[str, Any]] = {}
    for record in records:
        if str(record.get("language", "")).lower() not in {"ar", "ar-en"}:
            continue
        intent_id = str(record.get("intent_id", ""))
        if intent_id and intent_id not in by_intent:
            by_intent[intent_id] = record
    selected_ids = sorted(
        by_intent,
        key=lambda identifier: hashlib.sha256(f"20260826:{identifier}".encode()).hexdigest(),
    )[:20]
    if len(selected_ids) != 20:
        raise RuntimeError(f"could not freeze 20 MSA DEV base intents; found {len(selected_ids)}")
    substitutions = {
        "egyptian": {"ماذا": "إيه", "كيف": "إزاي", "هل": "هو", "يجب": "لازم"},
        "gulf_saudi": {"ماذا": "وش", "كيف": "شلون", "هل": "هل", "يجب": "لازم"},
        "levantine": {"ماذا": "شو", "كيف": "كيف", "هل": "هل", "يجب": "لازم"},
    }
    base_records: dict[str, dict[str, Any]] = {}
    variants: list[DialectVariant] = []
    for intent_id in selected_ids:
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
            "legal_intent_fingerprint": legal_fingerprint,
            "qrel_fingerprint": qrel_fingerprint,
            "number_identifiers": numbers,
            "date_identifiers": dates,
        }
        for dialect, mapping in substitutions.items():
            transformed = text
            for source, target in mapping.items():
                transformed = transformed.replace(source, target)
            variant_digest = hashlib.sha256(f"{intent_id}:{dialect}".encode()).hexdigest()[:16]
            variant_id = f"dialect-{dialect}-{variant_digest}"
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
        private_path, {"variants": [item.model_dump(mode="json") for item in variants]}
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
    """Run the available local BM25 dialect comparison after validation."""

    roots = _input_roots(root, historical_root)
    variant_path = roots.output_path("dialect/dialect_variants.json")
    if not variant_path.is_file():
        raise RuntimeError("dialect-evaluate requires dialect-prepare to finish first")
    variants_payload = json.loads(variant_path.read_text(encoding="utf-8"))
    variants = tuple(
        DialectVariant.model_validate(item) for item in variants_payload.get("variants", ())
    )
    records = load_dev_query_records(roots)
    by_intent = {str(record.get("intent_id")): record for record in records}
    base_records = [by_intent[item.base_intent_id] for item in variants[::3]]
    chunk_rows = load_dev_chunks(roots)
    retrieval_chunks = tuple(
        RetrievalChunk(
            chunk_id=str(chunk["chunk_id"]),
            document_id=str(chunk.get("document_id", "")),
            source_id=str(chunk.get("source_id", "")),
            unit_type=str(chunk.get("unit_type", "")),
            display_text=str(chunk.get("display_text", "")),
            search_text=str(chunk.get("search_text", "")),
            source_unit_ids=tuple(
                str(item) for item in cast(Sequence[Any], chunk.get("source_unit_ids", ()))
            ),
            chunk_policy_hash=str(chunk.get("chunk_policy_hash", "")),
            normalization_policy_id=str(chunk.get("normalization_policy_id", "")),
            normalization_policy_hash=str(chunk.get("normalization_policy_hash", "")),
            token_count=int(chunk.get("token_count", 0)),
            source_spans=tuple(
                (int(span.get("start", 0)), int(span.get("end", 0)))
                for span in cast(Sequence[dict[str, Any]], chunk.get("source_spans", ()))
            ),
        )
        for chunk in chunk_rows
    )
    index = BM25Index.build(retrieval_chunks, "arabic-light-v1", k1=1.2, b=0.75)
    variant_runs: dict[str, Mapping[str, Sequence[float]]] = {}
    from .runner import evaluate_dev_rankings

    for dialect in ("egyptian", "gulf_saudi", "levantine"):
        dialect_variants = tuple(item for item in variants if item.dialect == dialect)
        dialect_records: list[dict[str, Any]] = []
        rankings: dict[str, tuple[str, ...]] = {}
        for variant in dialect_variants:
            base = by_intent[variant.base_intent_id]
            dialect_record = dict(base)
            dialect_record["query_id"] = variant.variant_id
            dialect_record["query_text"] = variant.text
            dialect_records.append(dialect_record)
            rankings[variant.variant_id] = tuple(
                hit.chunk_id for hit in index.search(variant.text, top_k=10)
            )
        evaluated = evaluate_dev_rankings(dialect_records, rankings, chunk_rows)
        variant_runs[dialect] = evaluated.metrics
    msa_query_ids = tuple(str(item["query_id"]) for item in base_records)
    msa_rankings = load_dev_rankings(
        roots,
        "phase7_retrieval/dev/rankings/bm25__arabic-light-v1.json",
        msa_query_ids,
    )
    msa_evaluated = evaluate_dev_rankings(base_records, msa_rankings, chunk_rows)
    metrics = {
        metric: tuple(values)
        for metric, values in msa_evaluated.metrics.items()
        if metric in {"Recall@10", "MRR@10", "nDCG@10", "CompleteEvidenceRecall@10"}
    }
    dialect_deltas: dict[str, object] = {}
    for dialect, values_by_metric in variant_runs.items():
        dialect_deltas[dialect] = {
            metric: paired_bootstrap_delta(values, metrics[metric]).__dict__
            for metric, values in values_by_metric.items()
            if metric in metrics
        }
    result = {
        "status": "RUN_WITH_BLOCKED_SYSTEMS",
        "provenance": "PHASE15_DEV",
        "systems": {
            "bm25": "RUN",
            "bge-m3": "BLOCKED",
            "hybrid": "BLOCKED",
            "hybrid-reranker": "BLOCKED",
        },
        "bm25": {"msa": metrics, "dialect_minus_msa": dialect_deltas},
        "blocked_reason": (
            "No local Phase 15 dense/hybrid variant scorer is available without downloading "
            "a new corpus index/model"
        ),
    }
    destination = write_aggregate_artifact(root, "phase15_dialect_metrics.json", result)
    result["artifact"] = destination.as_posix()
    return result


def phase15_latency(root: Path = Path("."), historical_root: Path | None = None) -> dict[str, Any]:
    del historical_root
    result: dict[str, Any] = {
        "status": "BLOCKED",
        "provenance": "PHASE15_DEV",
        "reason": (
            "Phase 7-10 private DEV artifacts contain no per-operation latency samples; "
            "no cross-device proxy is reported"
        ),
        "protocol": {
            "batch_size": 1,
            "warmups": 3,
            "fixed_subset": "20 DEV query IDs",
            "device": "unavailable",
        },
        "available_generator_telemetry": "not reported until matched generator scoring runs",
    }
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
            if variant.dialect != "egyptian":
                continue
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

    cases = [
        (
            lambda private_text: ReviewCase(
                case_id=item.case_id,
                language=item.language,
                pipeline_stage=item.pipeline_stage,
                legal_category=item.legal_category,
                answerability=item.answerability,
                severity=item.severity,
                query_text=private_text[0],
                evidence_text=private_text[1],
                diagnostics={"trigger": item.trigger, **item.metadata},
            ).model_dump(mode="json")
        )(private_case_text(item))
        for item in selected
    ]
    if len(cases) != 120:
        raise RuntimeError(
            f"complete DEV review candidate pool selected {len(cases)} cases, expected 120"
        )
    candidate_path = root / PRIVATE_ROOT / "review_candidates.json"
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
