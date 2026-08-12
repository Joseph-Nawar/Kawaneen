"""Phase 5 corpus freeze, private chunk builds, ablation, and selection."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

from kawaneen.chunking.challenge import PRIVATE_ROOT, build_private_chunk_challenge
from kawaneen.chunking.corpus import Phase5Corpus, freeze_phase5_documents, load_phase5_units
from kawaneen.chunking.evaluation import run_chunking_ablation
from kawaneen.chunking.models import LegalChunk, SourceSpan
from kawaneen.chunking.policies import all_chunk_policies
from kawaneen.chunking.strategies import build_chunks
from kawaneen.chunking.structure import build_structure, section_units
from kawaneen.chunking.validation import summarize_chunks, validate_chunks
from kawaneen.normalization.policies import get_policy

MANIFEST_ROOT = Path("data/manifests/chunking")
METRICS_PATH = Path("data/evaluation/phase5_chunking_metrics.json")
PHASE5_CORPUS_MANIFEST = MANIFEST_ROOT / "phase5_corpus_manifest.json"
PHASE5_MANIFEST = MANIFEST_ROOT / "phase5_chunking_manifest.json"
CHUNKING_SEED = 20260812


def _phase3_canonical_hashes() -> dict[str, str]:
    inventory = json.loads(
        Path("data/manifests/canonical/inventory.json").read_text(encoding="utf-8")
    )
    hashes: dict[str, str] = {}
    for source in inventory["sources"]:
        for file in source["files"]:
            path = Path(str(file["path"]))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            hashes[path.as_posix()] = digest
    return hashes


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


def _span_dict(span: SourceSpan) -> dict[str, object]:
    return {"unit_id": span.unit_id, "start": span.start, "end": span.end}


def _chunk_dict(chunk: LegalChunk) -> dict[str, object]:
    return {
        "chunk_id": chunk.chunk_id,
        "strategy_id": chunk.strategy_id,
        "chunk_policy_hash": chunk.chunk_policy_hash,
        "source_unit_ids": list(chunk.source_unit_ids),
        "display_text": chunk.display_text,
        "search_text": chunk.search_text,
        "source_spans": [_span_dict(span) for span in chunk.source_spans],
        "parent_id": chunk.parent_id,
        "ancestor_ids": list(chunk.ancestor_ids),
        "sibling_ids": list(chunk.sibling_ids),
        "structure_path": list(chunk.structure_path),
        "citation_anchor": asdict(chunk.citation_anchor) if chunk.citation_anchor else None,
        "token_count": chunk.token_count,
        "normalization_policy_id": chunk.normalization_policy_id,
        "normalization_policy_hash": chunk.normalization_policy_hash,
        "provenance": dict(chunk.provenance),
        "context_source_spans": [_span_dict(span) for span in chunk.context_source_spans],
        "indexed_child_ids": list(chunk.indexed_child_ids),
        "fallback_reason": chunk.fallback_reason,
    }


def _write_private_chunks(root: Path, strategy_id: str, chunks: tuple[LegalChunk, ...]) -> None:
    path = root / strategy_id / "chunks.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(_chunk_dict(chunk), ensure_ascii=False, sort_keys=True) + "\n"
            for chunk in chunks
        ),
        encoding="utf-8",
    )


def _corpus_manifest(
    corpus: Phase5Corpus, canonical_hashes: Mapping[str, str]
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "phase5_corpus_frozen",
        "scope_version": "phase5-document-scope-v1",
        "document_count": len(corpus.document_ids),
        "document_count_by_source": dict(corpus.document_count_by_source),
        "unit_count": len(corpus.units),
        "source_versions": dict(corpus.source_versions),
        "document_ids_hash": corpus.document_ids_hash,
        "scope_hash": corpus.scope_hash,
        "canonical_hashes": dict(canonical_hashes),
        "ocr_included": False,
        "moj_retrieval_gold": False,
    }


def _positive_citation_gain(
    selected: str, best_fixed: str, citation_metrics: Mapping[str, Mapping[str, float]]
) -> bool:
    selected_metrics = citation_metrics[selected]
    fixed_metrics = citation_metrics[best_fixed]
    return (
        selected_metrics.get("citation_precision_at_1", 0.0)
        - fixed_metrics.get("citation_precision_at_1", 0.0)
        >= 0.05
        or selected_metrics.get("structural_anchor_accuracy_at_1", 0.0)
        - fixed_metrics.get("structural_anchor_accuracy_at_1", 0.0)
        >= 0.05
    )


def select_chunk_strategy(
    strategy_metrics: Mapping[str, Mapping[str, float]],
    citation_metrics: Mapping[str, Mapping[str, float]],
    context_metrics: Mapping[str, Mapping[str, float]],
    gates: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Apply the Phase 5 predeclared hard gates and conservative preference rule."""

    eligible = {
        strategy: bool(gates.get(strategy, {}).get("eligible", False))
        for strategy in strategy_metrics
    }
    rejected = {
        strategy: ["integrity_or_citation_gate"] for strategy, ok in eligible.items() if not ok
    }
    fixed = [
        strategy
        for strategy in strategy_metrics
        if strategy.startswith("fixed-") and eligible[strategy]
    ]
    if not fixed:
        return {
            "selected_policy_id": None,
            "best_fixed_baseline": None,
            "eligible_policies": [strategy for strategy, ok in eligible.items() if ok],
            "rejected_policies": rejected,
            "rationale": "No fixed strategy passed the hard gates.",
        }
    best_fixed = max(
        fixed, key=lambda strategy: (strategy_metrics[strategy].get("mrr_at_10", 0.0), strategy)
    )
    selected = best_fixed
    rationale = (
        f"Selected best fixed baseline {best_fixed}; no structural strategy met "
        "all conservative promotion rules."
    )
    structural = [
        strategy
        for strategy in strategy_metrics
        if strategy
        in {"legal-structure-v1", "legal-structure-neighbor-v1", "legal-parent-child-v1"}
        and eligible.get(strategy, False)
        and strategy_metrics[strategy].get("recall_at_10", 0.0)
        >= strategy_metrics[best_fixed].get("recall_at_10", 0.0) - 0.02
        and strategy_metrics[strategy].get("mrr_at_10", 0.0)
        >= strategy_metrics[best_fixed].get("mrr_at_10", 0.0) - 0.02
        and _positive_citation_gain(strategy, best_fixed, citation_metrics)
    ]
    if structural:
        selected = max(
            structural,
            key=lambda strategy: (
                citation_metrics[strategy].get("citation_precision_at_1", 0.0),
                citation_metrics[strategy].get("structural_anchor_accuracy_at_1", 0.0),
                strategy,
            ),
        )
        rationale = (
            f"Selected {selected}: retrieval stayed within tolerance and citation "
            f"quality improved over {best_fixed}."
        )
    base_structural = selected if selected.startswith("legal-") else None
    for candidate in ("legal-structure-neighbor-v1", "legal-parent-child-v1"):
        if not eligible.get(candidate, False) or base_structural is None or candidate == selected:
            continue
        context_gain = context_metrics[candidate].get(
            "context_coverage_at_5", 0.0
        ) - context_metrics[base_structural].get("context_coverage_at_5", 0.0)
        retrieval_loss = strategy_metrics[base_structural].get(
            "recall_at_10", 0.0
        ) - strategy_metrics[candidate].get("recall_at_10", 0.0)
        if context_gain >= 0.05 and retrieval_loss <= 0.02:
            selected = candidate
            rationale = (
                f"Selected {candidate}: context coverage improved without material "
                "retrieval degradation."
            )
    return {
        "selected_policy_id": selected,
        "best_fixed_baseline": best_fixed,
        "eligible_policies": [strategy for strategy, ok in eligible.items() if ok],
        "rejected_policies": rejected,
        "rationale": rationale,
    }


def chunking_plan() -> dict[str, object]:
    return {
        "schema_version": 1,
        "phase": "phase-05-legal-structure-and-chunking",
        "private_root": PRIVATE_ROOT.as_posix(),
        "normalization_policy_id": "arabic-light-v1",
        "strategies": [
            {
                "policy_id": policy.policy_id,
                "version": policy.version,
                "policy_hash": policy.policy_hash,
                "config": dict(policy.config),
            }
            for policy in all_chunk_policies()
        ],
        "scope_exclusions": [
            "embeddings",
            "dense_retrieval",
            "reranking",
            "RAG",
            "Qdrant",
            "APIs",
            "Phase 6 human evaluation",
        ],
    }


def run_phase5_chunking() -> dict[str, object]:
    canonical_before = _phase3_canonical_hashes()
    units = load_phase5_units()
    corpus = freeze_phase5_documents(units, per_source=1500)
    indexed_units = section_units(corpus.units)
    normalization_policy = get_policy("arabic-light-v1")
    private_root = PRIVATE_ROOT
    private_root.mkdir(parents=True, exist_ok=True)
    corpus_manifest = _corpus_manifest(corpus, canonical_before)
    _write_json(MANIFEST_ROOT / "phase5_corpus_manifest.json", corpus_manifest)
    challenge = build_private_chunk_challenge(corpus.units, corpus, seed=CHUNKING_SEED)
    nodes = build_structure(corpus.units, corpus)
    strategy_chunks: dict[str, tuple[LegalChunk, ...]] = {}
    strategy_stats: dict[str, dict[str, object]] = {}
    strategy_gates: dict[str, dict[str, object]] = {}
    for policy in all_chunk_policies():
        chunks = build_chunks(corpus.units, corpus, policy, normalization_policy)
        strategy_chunks[policy.policy_id] = chunks
        integrity = validate_chunks(chunks, indexed_units, nodes)
        strategy_stats[policy.policy_id] = {
            **summarize_chunks(chunks, indexed_units),
            **integrity.to_sanitized_dict(),
        }
        strategy_gates[policy.policy_id] = {
            "eligible": integrity.orphan_count == 0
            and integrity.cycle_count == 0
            and integrity.invalid_span_count == 0
            and integrity.display_text_mismatch_count == 0
            and integrity.boundary_violation_count == 0,
            **integrity.to_sanitized_dict(),
        }
        _write_private_chunks(private_root / "chunks", policy.policy_id, chunks)
    evaluation = run_chunking_ablation(strategy_chunks, challenge, seed=CHUNKING_SEED)
    decision = select_chunk_strategy(
        evaluation.strategy_metrics,
        evaluation.citation_metrics,
        evaluation.context_metrics,
        strategy_gates,
    )
    _write_json(
        private_root / "results.json",
        {
            "strategy_results": evaluation.private_results,
            "decision": decision,
        },
    )
    metrics = {
        "schema_version": 1,
        "status": "phase5_chunking_experiment_complete",
        "seed": CHUNKING_SEED,
        "corpus": {
            "document_count": len(corpus.document_ids),
            "document_count_by_source": dict(corpus.document_count_by_source),
            "unit_count": len(corpus.units),
            "document_ids_hash": corpus.document_ids_hash,
            "scope_hash": corpus.scope_hash,
        },
        "challenge": {
            "construction_version": challenge.construction_version,
            "query_count": len(challenge.items),
            "slice_counts": {
                slice_name: sum(item.slice_name == slice_name for item in challenge.items)
                for slice_name in sorted({item.slice_name for item in challenge.items})
            },
        },
        "normalization_policy_id": normalization_policy.policy_id,
        "normalization_policy_hash": normalization_policy.policy_hash,
        "strategy_stats": strategy_stats,
        "integrity_gates": strategy_gates,
        "retrieval_metrics": evaluation.strategy_metrics,
        "slice_metrics": evaluation.slice_metrics,
        "citation_metrics": evaluation.citation_metrics,
        "context_metrics": evaluation.context_metrics,
        "pairwise_wins_ties_losses": evaluation.pairwise_wins_ties_losses,
        "paired_confidence_intervals": evaluation.paired_confidence_intervals,
        "decision": decision,
        "authoritative_statutory_revalidation_required": True,
    }
    manifest = {
        **chunking_plan(),
        "status": "phase5_chunking_experiment_complete",
        "corpus_manifest": "data/manifests/chunking/phase5_corpus_manifest.json",
        "metrics": METRICS_PATH.as_posix(),
        "strategy_policy_hashes": {
            policy.policy_id: policy.policy_hash for policy in all_chunk_policies()
        },
        "decision": decision,
        "private_artifact_root": private_root.as_posix(),
    }
    _write_json(METRICS_PATH, metrics)
    _write_json(PHASE5_MANIFEST, manifest)
    canonical_after = _phase3_canonical_hashes()
    if canonical_before != canonical_after:
        raise RuntimeError("Phase 3 canonical hashes changed during Phase 5")
    return {"manifest": manifest, "metrics": metrics}


def validate_phase5_chunking() -> dict[str, object]:
    for path in (PHASE5_CORPUS_MANIFEST, PHASE5_MANIFEST, METRICS_PATH):
        if not path.is_file():
            raise ValueError(f"Phase 5 artifact missing: {path}")
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    if metrics.get("status") != "phase5_chunking_experiment_complete":
        raise ValueError("Phase 5 metrics status is invalid")
    return {
        "valid": True,
        "manifest": PHASE5_MANIFEST.as_posix(),
        "metrics": METRICS_PATH.as_posix(),
    }
