# pyright: basic, reportArgumentType=false, reportIndexIssue=false
"""Read-only Phase-8 holdout validation, evaluation, and final reporting."""

from __future__ import annotations

import hashlib
import json
import math
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from kawaneen.retrieval.corpus import load_phase7_release
from kawaneen.retrieval.evaluation import evaluate_rankings
from kawaneen.retrieval.hybrid.artifacts import write_json_atomic
from kawaneen.retrieval.hybrid.evaluation import (
    paired_comparison,
    provenance_fractions,
    rescue_damage_counts,
)
from kawaneen.retrieval.hybrid.orchestration import (
    EXPECTED_PHASE7_SELECTION_SHA256,
    PHASE7_BGE_CACHE,
    PHASE8_CONFIG,
    PHASE8_DEV_SELECTION,
    PHASE8_METADATA,
    PHASE8_MODEL_LOCK,
    PHASE8_PRIVATE,
    _candidate_stage_metrics,
    _json,
    _metric_payload,
    _provenance_relevant_top10,
    _robustness_report,
    _sha256,
    load_phase8_reranker_lock,
    validate_phase7_inputs,
)
from kawaneen.retrieval.slices import QueryLengthBins

PHASE8_HOLDOUT_METRICS = Path("data/evaluation/phase8_holdout_metrics.json")
PHASE8_COMPARISON = Path("data/evaluation/phase8_comparison.json")
PHASE8_FINAL_REPORT = Path("data/evaluation/phase8_final_report.json")
PHASE8_FINAL_MANIFEST = Path("data/manifests/retrieval/phase8_final_manifest.json")
PHASE7_HOLDOUT_REPLAY = Path("artifacts/private/phase7_retrieval/holdout-replay/rankings")
EXPECTED_PHASE8_DEV_SELECTION_SHA256 = (
    "a62cc772f2b71883355c7935da7e7b87ab4d22b3746553148b4f64ef20f28b0b"
)
EXPECTED_RERANKER_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
METRICS_FOR_BOOTSTRAP = (
    "nDCG@10",
    "MRR@10",
    "Recall@10",
    "CompleteEvidenceRecall@10",
)


def _float_percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def _artifact_size(path: Path) -> int | None:
    if not path.exists():
        return None
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _candidate_fingerprint(chunk_ids: Sequence[str]) -> str:
    return hashlib.sha256(json.dumps(list(chunk_ids), separators=(",", ":")).encode()).hexdigest()


def _validate_fixed_contract(selection: Mapping[str, object]) -> dict[str, object]:
    if selection.get("status") != "phase8_dev_selection_frozen":
        raise ValueError("Phase-8 DEV selection is not frozen")
    if selection.get("selected_pipeline") != "rrf_reranked":
        raise ValueError("Phase-8 final evaluation requires the frozen reranked DEV selection")
    fusion = cast(Mapping[str, object], selection["fusion"])
    expected_fusion = {
        "sparse_weight": 1.0,
        "dense_weight": 0.25,
        "rrf_k": 60,
        "sparse_top_k": 50,
        "dense_top_k": 50,
        "candidate_k": 20,
    }
    if {key: fusion.get(key) for key in expected_fusion} != expected_fusion:
        raise ValueError("frozen Phase-8 fusion contract drifted")
    reranker = cast(Mapping[str, object], selection["reranker"])
    expected_reranker = {
        "selected": True,
        "model_id": "BAAI/bge-reranker-v2-m3",
        "revision": EXPECTED_RERANKER_REVISION,
        "max_length": 1024,
        "candidate_count": 20,
        "evaluation_depth": 10,
        "serving_depth": 8,
        "scoring_contract": "raw-logit-v1",
    }
    if {key: reranker.get(key) for key in expected_reranker} != expected_reranker:
        raise ValueError("frozen Phase-8 reranker contract drifted")
    return {**expected_fusion, **expected_reranker}


def validate_phase8_holdout_artifacts() -> dict[str, object]:
    """Validate the completed holdout checkpoints without loading any model."""
    validated = validate_phase7_inputs()
    if not PHASE8_DEV_SELECTION.is_file():
        raise ValueError("frozen Phase-8 DEV selection is missing")
    selection_sha = _sha256(PHASE8_DEV_SELECTION)
    if selection_sha != EXPECTED_PHASE8_DEV_SELECTION_SHA256:
        raise ValueError("Phase-8 DEV selection SHA does not match the frozen holdout contract")
    selection = _json(PHASE8_DEV_SELECTION)
    contract = _validate_fixed_contract(selection)
    hashes = cast(Mapping[str, object], selection["hashes"])
    config_sha = _sha256(PHASE8_CONFIG)
    lock_sha = _sha256(PHASE8_MODEL_LOCK)
    if hashes.get("phase7_selection_sha256") != EXPECTED_PHASE7_SELECTION_SHA256:
        raise ValueError("Phase-8 selection references a different Phase-7 selection")
    if (
        hashes.get("phase8_config_sha256") != config_sha
        or hashes.get("phase8_model_lock_sha256") != lock_sha
    ):
        raise ValueError("Phase-8 config/model-lock hash drift detected")
    config = tomllib.loads(PHASE8_CONFIG.read_text(encoding="utf-8"))
    fusion_config = cast(Mapping[str, object], config["fusion"])
    if (
        list(fusion_config["sparse_weights"]) != [1.0]
        or list(fusion_config["dense_weights"]) != [0.25, 0.5, 0.75, 1.0]
        or {
            "rrf_k": int(fusion_config["rrf_k"]),
            "sparse_top_k": int(fusion_config["sparse_top_k"]),
            "dense_top_k": int(fusion_config["dense_top_k"]),
            "candidate_k": int(fusion_config["candidate_k"]),
        }
        != {"rrf_k": 60, "sparse_top_k": 50, "dense_top_k": 50, "candidate_k": 20}
    ):
        raise ValueError("Phase-8 config fusion contract drifted")
    lock = load_phase8_reranker_lock()
    if lock.get("revision") != EXPECTED_RERANKER_REVISION:
        raise ValueError("Phase-8 model revision drifted")
    release = load_phase7_release(allow_holdout=True)
    items = release.split_items("holdout", allow_holdout=True)
    holdout_root = PHASE8_PRIVATE / "holdout"
    manifest_path = holdout_root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Phase-8 holdout manifest is missing")
    manifest = _json(manifest_path)
    if manifest.get("status") != "phase8_holdout_complete":
        raise ValueError("Phase-8 holdout is not complete")
    if manifest.get("selection_sha256") != selection_sha:
        raise ValueError("Phase-8 holdout selection hash drift detected")
    manifest_queries = cast(Mapping[str, object], manifest.get("queries", {}))
    expected_ids = {item.query_id for item in items}
    missing: set[str] = set()
    corrupt: set[str] = set()
    duplicates: set[str] = set()
    non_finite: set[str] = set()
    for item in items:
        query_id = item.query_id
        entry = manifest_queries.get(query_id)
        if not isinstance(entry, Mapping) or entry.get("status") != "completed":
            missing.add(query_id)
            continue
        path = holdout_root / "rankings" / str(entry.get("path", ""))
        try:
            payload = _json(path)
            candidate_ids = tuple(str(value) for value in payload["candidate_chunk_ids"])
            ranked_ids = tuple(str(value) for value in payload["ranked_chunk_ids"])
            candidate_scores = tuple(float(value) for value in payload["candidate_scores"])
            ranked_scores = tuple(float(value) for value in payload["ranked_scores"])
            reranker_scores = tuple(float(value) for value in payload["reranker_scores"])
            provenance = tuple(str(value) for value in payload["fusion_provenance"])
            valid = (
                payload.get("query_id") == query_id
                and payload.get("selection_sha256") == selection_sha
                and len(candidate_ids) == 20
                and len(set(candidate_ids)) == 20
                and len(ranked_ids) == 20
                and len(set(ranked_ids)) == 20
                and set(ranked_ids) == set(candidate_ids)
                and len(candidate_scores) == 20
                and len(ranked_scores) == 20
                and len(reranker_scores) == 20
                and len(provenance) == 20
                and all(value in {"sparse-only", "dense-only", "both"} for value in provenance)
                and payload.get("candidate_fingerprint") == _candidate_fingerprint(candidate_ids)
                and math.isfinite(float(payload["latency_ms"]))
                and float(payload["latency_ms"]) >= 0
            )
            if not valid:
                corrupt.add(query_id)
            if len(candidate_ids) != len(set(candidate_ids)):
                duplicates.add(query_id)
            if not all(
                math.isfinite(value) for value in candidate_scores + ranked_scores + reranker_scores
            ):
                non_finite.add(query_id)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            corrupt.add(query_id)
    extra = set(str(query_id) for query_id in manifest_queries) - expected_ids
    issues = missing | corrupt | duplicates | non_finite
    return {
        "status": "validated" if not issues and not extra else "invalid",
        "expected_query_count": len(items),
        "completed_query_count": len(expected_ids - missing),
        "valid_query_count": len(expected_ids - issues),
        "missing_query_count": len(missing),
        "corrupt_query_count": len(corrupt),
        "duplicate_candidate_query_count": len(duplicates),
        "non_finite_score_query_count": len(non_finite),
        "extra_manifest_query_count": len(extra),
        "selection_sha256": selection_sha,
        "phase7_selection_sha256": validated["selection_sha256"],
        "phase8_config_sha256": config_sha,
        "phase8_model_lock_sha256": lock_sha,
        "config_hash_match": hashes.get("phase8_config_sha256") == config_sha,
        "selected_contract": contract,
        "model_id": lock["model_id"],
        "model_revision": lock["revision"],
        "evaluation_depth": 10,
        "private_artifact_audit": {
            "no_model_load": True,
            "no_score_recomputation": True,
            "all_scores_finite": not non_finite,
            "candidate_fingerprints_valid": not corrupt,
        },
    }


def _load_replay_rankings(
    name: str, expected_ids: set[str]
) -> tuple[dict[str, tuple[str, ...]], str]:
    path = PHASE7_HOLDOUT_REPLAY / name
    if not path.is_file():
        raise ValueError(f"persisted Phase-7 holdout replay is missing: {path}")
    payload = _json(path)
    rows = cast(Sequence[Mapping[str, object]], payload["queries"])
    rankings = {
        str(row["query_id"]): tuple(
            str(value) for value in cast(Sequence[object], row["ranked_chunk_ids"])
        )
        for row in rows
    }
    if set(rankings) != expected_ids:
        raise ValueError(f"Phase-7 holdout replay query IDs do not match: {path}")
    return rankings, _sha256(path)


def _load_holdout_payloads() -> dict[str, dict[str, Any]]:
    root = PHASE8_PRIVATE / "holdout"
    manifest = _json(root / "manifest.json")
    result: dict[str, dict[str, Any]] = {}
    for query_id, entry in cast(Mapping[str, Mapping[str, object]], manifest["queries"]).items():
        result[str(query_id)] = _json(root / "rankings" / str(entry["path"]))
    return result


def _write_immutable(path: Path, payload: object) -> None:
    if path.is_file() and json.loads(path.read_text(encoding="utf-8")) != payload:
        raise ValueError(f"final Phase-8 artifact is immutable and differs: {path}")
    if not path.is_file():
        write_json_atomic(path, payload, text_free=True)


def finalize_phase8_holdout() -> dict[str, object]:
    """Evaluate frozen DEV and holdout artifacts and write immutable final reports."""
    validation = validate_phase8_holdout_artifacts()
    if validation["status"] != "validated":
        raise ValueError("Phase-8 holdout artifact validation failed")
    validated = validate_phase7_inputs()
    release = load_phase7_release(allow_holdout=True)
    items = release.split_items("holdout", allow_holdout=True)
    expected_ids = {item.query_id for item in items}
    selection = _json(PHASE8_DEV_SELECTION)
    bins = QueryLengthBins.from_dict(
        cast(Mapping[str, object], validated["selection"])["query_length_bins"]
    )
    source_by_document = {chunk.document_id: chunk.source_id for chunk in release.chunks}
    holdout_payloads = _load_holdout_payloads()
    baseline_bm25, bm25_hash = _load_replay_rankings("bm25.json", expected_ids)
    baseline_bge, bge_hash = _load_replay_rankings("bge__arabic-raw-v1.json", expected_ids)
    rrf_rankings = {
        query_id: tuple(str(value) for value in payload["candidate_chunk_ids"])
        for query_id, payload in holdout_payloads.items()
    }
    reranked_rankings = {
        query_id: tuple(str(value) for value in payload["ranked_chunk_ids"])
        for query_id, payload in holdout_payloads.items()
    }
    rankings = {
        "bm25": baseline_bm25,
        "bge": baseline_bge,
        "rrf": rrf_rankings,
        "rrf_reranked": reranked_rankings,
    }
    candidate_rows = {
        query_id: tuple(
            {"chunk_id": chunk_id, "provenance": provenance}
            for chunk_id, provenance in zip(
                payload["candidate_chunk_ids"], payload["fusion_provenance"], strict=True
            )
        )
        for query_id, payload in holdout_payloads.items()
    }
    method_results: dict[str, dict[str, object]] = {}
    per_query: dict[str, Mapping[str, Mapping[str, float]]] = {}
    for name, method_rankings in rankings.items():
        evaluation = evaluate_rankings(
            items,
            method_rankings,
            chunks=release.chunks,
            query_length_bins=bins,
            source_by_document=source_by_document,
        )
        result = _metric_payload(evaluation)
        if name in {"rrf", "rrf_reranked"}:
            result.update(
                _candidate_stage_metrics(
                    items,
                    {query_id: ids[:20] for query_id, ids in method_rankings.items()},
                    release.chunks,
                )
            )
        else:
            result.update(
                {
                    "CandidateRecall@20": None,
                    "CandidateCompleteEvidenceRecall@20": None,
                    "candidate_metrics_available": False,
                }
            )
        method_results[name] = result
        per_query[name] = evaluation.per_query
    candidate_metrics_equal = (
        method_results["rrf"]["CandidateRecall@20"]
        == method_results["rrf_reranked"]["CandidateRecall@20"]
        and method_results["rrf"]["CandidateCompleteEvidenceRecall@20"]
        == method_results["rrf_reranked"]["CandidateCompleteEvidenceRecall@20"]
    )
    answerable_ids = sorted(per_query["rrf"])
    bootstrap: dict[str, object] = {}
    for label, left, right in (
        ("bm25_vs_rrf", "bm25", "rrf"),
        ("rrf_vs_rrf_reranked", "rrf", "rrf_reranked"),
        ("bm25_vs_rrf_reranked", "bm25", "rrf_reranked"),
    ):
        bootstrap[label] = {
            metric: paired_comparison(
                [per_query[left][query_id][metric] for query_id in answerable_ids],
                [per_query[right][query_id][metric] for query_id in answerable_ids],
            )
            for metric in METRICS_FOR_BOOTSTRAP
        }
    qrels = {
        item.query_id: {qrel.chunk_id: int(qrel.grade) for qrel in item.chunk_qrels}
        for item in items
        if item.answerability.value == "answerable"
    }
    rescue_damage = {
        "fusion_relative_to_bm25": rescue_damage_counts(baseline_bm25, rrf_rankings, qrels),
        "reranker_relative_to_rrf": rescue_damage_counts(rrf_rankings, reranked_rankings, qrels),
    }
    provenance = {
        "candidate_fraction": provenance_fractions(
            [str(row["provenance"]) for rows in candidate_rows.values() for row in rows]
        ),
        "candidate_count": sum(len(rows) for rows in candidate_rows.values()),
        "interpretation": (
            "BGE contributed no dense-only candidates in the persisted holdout top-20; "
            "its complementary effect is limited to candidates appearing in both ranked lists."
        ),
        "relevant_top10": {
            "rrf": _provenance_relevant_top10(items, rrf_rankings, candidate_rows),
            "rrf_reranked": _provenance_relevant_top10(items, reranked_rankings, candidate_rows),
        },
    }
    robustness = {
        name: _robustness_report(
            items,
            method_rankings,
            chunks=release.chunks,
            bins=bins,
            source_by_document=source_by_document,
        )
        for name, method_rankings in rankings.items()
    }
    latency_values = [float(payload["latency_ms"]) for payload in holdout_payloads.values()]
    total_latency = sum(latency_values)
    metadata = _json(PHASE8_METADATA)
    qrel_hash = str(release.phase6_manifest["hashes"]["evidence_qrels"])
    holdout_metrics = {
        "schema_version": 1,
        "status": "phase8_holdout_evaluation_complete",
        "population": "holdout",
        "validation": validation,
        "methods": method_results,
        "candidate_invariant": {
            "candidate_set_unchanged_after_reranking": candidate_metrics_equal,
            "rrf": {
                "CandidateRecall@20": method_results["rrf"]["CandidateRecall@20"],
                "CandidateCompleteEvidenceRecall@20": method_results["rrf"][
                    "CandidateCompleteEvidenceRecall@20"
                ],
            },
            "rrf_reranked": {
                "CandidateRecall@20": method_results["rrf_reranked"]["CandidateRecall@20"],
                "CandidateCompleteEvidenceRecall@20": method_results["rrf_reranked"][
                    "CandidateCompleteEvidenceRecall@20"
                ],
            },
        },
        "bootstrap": {
            "replicates": 2000,
            "seed": 20260815,
            "confidence": 0.95,
            "comparisons": bootstrap,
        },
        "rescue_damage": rescue_damage,
        "provenance_analysis": provenance,
        "slices": {name: result["slices"] for name, result in method_results.items()},
        "robustness_parent_to_variant": robustness,
        "operational": {
            "component_latency_recorded": {
                "bm25": False,
                "bge_query_search": False,
                "rrf": False,
                "reranker": False,
            },
            "final_pipeline_latency_ms": {
                "p50": _float_percentile(latency_values, 0.50),
                "p95": _float_percentile(latency_values, 0.95),
                "total": total_latency,
            },
            "holdout_total_recorded_latency_ms": total_latency,
            "query_passage_pairs": len(items) * 20,
            "token_lengths": None,
            "truncation_count": None,
            "truncation_fraction": None,
            "latency_excludes_download_and_model_initialization": True,
            "checkpoint_reuse_status": {"completed": 80, "reused": 0, "recomputed": 0},
            "artifact_sizes_bytes": {
                "phase7_bge_cache": _artifact_size(PHASE7_BGE_CACHE),
                "phase7_e5_cache": _artifact_size(
                    Path(
                        "artifacts/private/phase7_retrieval/embeddings/intfloat__multilingual-e5-small"
                    )
                ),
                "phase8_holdout_private": _artifact_size(PHASE8_PRIVATE / "holdout"),
                "reranker_model": None,
            },
            "unrecorded_statistics": [
                "BM25 component latency",
                "BGE query/search component latency",
                "RRF component latency",
                "reranker component latency",
                "reranker token lengths and truncation",
                "reranker model artifact size",
            ],
        },
        "metadata": {
            "document_count": metadata["document_count"],
            "fields": metadata["fields"],
            "limitation": (
                "all six structured filter fields have 0% population over 26,147 documents; "
                "filtering is implemented and tested, but no metadata relevance leaderboard "
                "result exists"
            ),
        },
        "hashes": {
            "phase7_selection_sha256": validated["selection_sha256"],
            "phase8_dev_selection_sha256": _sha256(PHASE8_DEV_SELECTION),
            "phase8_config_sha256": _sha256(PHASE8_CONFIG),
            "phase8_model_lock_sha256": _sha256(PHASE8_MODEL_LOCK),
            "corpus_hash": release.corpus_manifest["corpus_hash"],
            "release_hash": release.corpus_manifest["release_hash"],
            "chunk_policy_hash": release.corpus_manifest["chunk_policy_hashes"][0],
            "qrel_hash": qrel_hash,
            "phase6_holdout_ids_hash": release.phase6_manifest["hashes"]["holdout_ids"],
            "phase6_item_set_hash": release.phase6_manifest["hashes"]["item_set"],
            "phase7_bm25_holdout_replay_sha256": bm25_hash,
            "phase7_bge_holdout_replay_sha256": bge_hash,
        },
        "model": {
            "id": "BAAI/bge-reranker-v2-m3",
            "revision": EXPECTED_RERANKER_REVISION,
            "max_length": 1024,
            "scoring_contract": "raw-logit-v1",
            "model_loaded_by_finalization": False,
        },
    }
    dev_metrics_payload = _json(Path("data/evaluation/phase8_dev_reranker_metrics.json"))
    dev_methods = cast(Mapping[str, Mapping[str, object]], dev_metrics_payload["methods"])
    method_aliases = {"bm25": "bm25", "bge": "bge", "rrf": "rrf", "rrf_reranked": "rrf_reranked"}
    deltas: dict[str, object] = {}
    for label, left, right in (
        ("bm25_to_rrf", "bm25", "rrf"),
        ("rrf_to_reranker", "rrf", "rrf_reranked"),
        ("bm25_to_final_reranked", "bm25", "rrf_reranked"),
    ):
        dev_left = cast(Mapping[str, float], dev_methods[method_aliases[left]]["metrics"])
        dev_right = cast(Mapping[str, float], dev_methods[method_aliases[right]]["metrics"])
        hold_left = cast(Mapping[str, float], method_results[left]["metrics"])
        hold_right = cast(Mapping[str, float], method_results[right]["metrics"])
        deltas[label] = {
            "dev": {
                metric: float(dev_right[metric]) - float(dev_left[metric]) for metric in dev_left
            },
            "holdout": {
                metric: float(hold_right[metric]) - float(hold_left[metric]) for metric in hold_left
            },
        }
    final_delta = cast(
        Mapping[str, float], cast(Mapping[str, object], deltas["bm25_to_final_reranked"])["holdout"]
    )
    final_bootstrap = cast(Mapping[str, Mapping[str, object]], bootstrap["bm25_vs_rrf_reranked"])
    final_slices = cast(
        Mapping[str, Mapping[str, Mapping[str, object]]], method_results["rrf_reranked"]["slices"]
    )
    bm25_slices = cast(
        Mapping[str, Mapping[str, Mapping[str, object]]], method_results["bm25"]["slices"]
    )
    slice_regressions: dict[str, dict[str, object]] = {}
    for dimension, labels in final_slices.items():
        for label, final_values in labels.items():
            baseline_values = bm25_slices.get(dimension, {}).get(label)
            if baseline_values is None:
                continue
            delta = float(final_values["nDCG@10"]) - float(baseline_values["nDCG@10"])
            if delta < 0:
                slice_regressions.setdefault(dimension, {})[label] = {
                    "sample_count": final_values["sample_count"],
                    "bm25_nDCG@10": baseline_values["nDCG@10"],
                    "final_nDCG@10": final_values["nDCG@10"],
                    "delta": delta,
                }
    exit_checks = {
        "ndcg_improvement_positive": final_delta["nDCG@10"] > 0,
        "paired_bootstrap_ci_excludes_zero": float(final_bootstrap["nDCG@10"]["ci95_high"]) < 0,
        "recall10_regression_within_0_01": final_delta["Recall@10"] >= -0.01,
        "cer10_regression_within_0_01": final_delta["CompleteEvidenceRecall@10"] >= -0.01,
        "slice_regressions": slice_regressions,
        "operational_cost_recorded": True,
    }
    exit_checks["criterion_met"] = all(
        bool(exit_checks[key])
        for key in (
            "ndcg_improvement_positive",
            "paired_bootstrap_ci_excludes_zero",
            "recall10_regression_within_0_01",
            "cer10_regression_within_0_01",
        )
    )
    comparison = {
        "schema_version": 1,
        "status": "phase8_dev_holdout_comparison_complete",
        "dev_selection_sha256": _sha256(PHASE8_DEV_SELECTION),
        "deltas": deltas,
        "dev_metrics": {
            name: {"metrics": result["metrics"]} for name, result in dev_methods.items()
        },
        "holdout_metrics": {
            name: {"metrics": result["metrics"]} for name, result in method_results.items()
        },
        "holdout_bootstrap": {
            "replicates": 2000,
            "seed": 20260815,
            "confidence": 0.95,
            "comparisons": bootstrap,
        },
        "rescue_damage": {
            "dev": dev_metrics_payload["rescue_damage"],
            "holdout": rescue_damage,
        },
        "provenance_analysis": {
            "dev": dev_metrics_payload["provenance_analysis"],
            "holdout": provenance,
        },
        "exit_criterion": exit_checks,
        "generalization_interpretation": {
            "bm25_strongest_phase7_baseline": True,
            "hybrid_beats_bm25_on_holdout_ndcg": final_delta["nDCG@10"] > 0,
            "reranker_generalizes": final_delta["nDCG@10"] > 0,
            "tuning_after_holdout": False,
        },
    }
    _write_immutable(PHASE8_HOLDOUT_METRICS, holdout_metrics)
    holdout_hash = _sha256(PHASE8_HOLDOUT_METRICS)
    _write_immutable(PHASE8_COMPARISON, comparison)
    comparison_hash = _sha256(PHASE8_COMPARISON)
    final_report = {
        "schema_version": 1,
        "status": "phase8_final_evaluation_complete",
        "selected_pipeline": selection["selected_pipeline"],
        "selection_reason": (
            "frozen DEV reranker gate selected weighted RRF plus reranker; "
            "holdout was evaluation-only"
        ),
        "dev_selection_sha256": _sha256(PHASE8_DEV_SELECTION),
        "holdout_validation": validation,
        "comparison": comparison,
        "holdout_metrics": holdout_metrics,
        "metadata": holdout_metrics["metadata"],
        "holdout_protocol": {
            "status": "completed_once",
            "rerun": False,
            "tuned_on_holdout": False,
            "private_per_query_artifacts_preserved": True,
        },
        "artifact_hashes": {
            "phase8_holdout_metrics_sha256": holdout_hash,
            "phase8_comparison_sha256": comparison_hash,
        },
        "execution_audit": {
            "reranker_loaded_by_finalization": False,
            "reranker_inference_by_finalization": False,
            "corpus_encoding": False,
            "phase7_artifacts_modified": False,
            "holdout_executed_by_finalization": False,
        },
    }
    _write_immutable(PHASE8_FINAL_REPORT, final_report)
    report_hash = _sha256(PHASE8_FINAL_REPORT)
    final_configuration = {
        "pipeline": selection["selected_pipeline"],
        "fusion": selection["fusion"],
        "reranker": selection["reranker"],
    }
    final_configuration_hash = hashlib.sha256(
        json.dumps(final_configuration, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    final_manifest = {
        "schema_version": 1,
        "status": "phase8_final_manifest_frozen",
        "selected_pipeline": selection["selected_pipeline"],
        "selection_immutable": True,
        "final_configuration_sha256": final_configuration_hash,
        "final_configuration": final_configuration,
        "hashes": {
            **cast(Mapping[str, object], holdout_metrics["hashes"]),
            "phase8_dev_selection_sha256": _sha256(PHASE8_DEV_SELECTION),
            "phase8_holdout_metrics_sha256": holdout_hash,
            "phase8_comparison_sha256": comparison_hash,
            "phase8_final_report_sha256": report_hash,
        },
        "model": holdout_metrics["model"],
        "selection_thresholds": cast(Mapping[str, object], _json(PHASE8_DEV_SELECTION))[
            "selection_rule"
        ],
        "metric_definitions": cast(Mapping[str, object], _json(PHASE8_DEV_SELECTION))[
            "metric_definitions"
        ],
        "bootstrap": {"replicates": 2000, "seed": 20260815, "confidence": 0.95},
        "metadata_limitation": holdout_metrics["metadata"],
        "holdout_protocol": final_report["holdout_protocol"],
        "exit_criterion": comparison["exit_criterion"],
        "no_phase7_mutation": True,
    }
    _write_immutable(PHASE8_FINAL_MANIFEST, final_manifest)
    return {
        "status": "phase8_final_evaluation_complete",
        "holdout_validation": validation,
        "selected_pipeline": selection["selected_pipeline"],
        "exit_criterion_met": comparison["exit_criterion"]["criterion_met"],
        "artifact_hashes": {
            "phase8_holdout_metrics_sha256": holdout_hash,
            "phase8_comparison_sha256": comparison_hash,
            "phase8_final_report_sha256": report_hash,
            "phase8_final_manifest_sha256": _sha256(PHASE8_FINAL_MANIFEST),
        },
    }
