"""Content-only audit for the already-generated Phase 15 dialect variants.

The audit intentionally never reads rankings or relevance outcomes.  It is a
post-generation integrity check used to identify malformed text before the
existing paired rankings are summarized for the dialect question.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .evidence import write_json_atomic
from .reporting import write_aggregate_artifact

PRIVATE_PATH = Path("artifacts/private/phase15_evaluation/dialect/content_validity_audit.json")
TRACKED_FILENAME = "phase15_dialect_content_validity.json"


def _normalized(text: object) -> str:
    return " ".join(str(text).split())


def _base_text(
    variant: Mapping[str, Any], base_records: Mapping[str, Mapping[str, Any]] | None
) -> str:
    if base_records is not None:
        base = base_records.get(str(variant.get("base_intent_id")))
        if base is not None:
            return _normalized(base.get("query_text", base.get("text", "")))
    return _normalized(variant.get("base_text", ""))


def _is_prefix_only(base: str, variant: str) -> bool:
    if not base or variant == base or not variant.endswith(base):
        return False
    return bool(variant[: -len(base)].strip())


def _audit_dialect_content(
    variants: Sequence[Mapping[str, Any]],
    *,
    base_records: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return text-free aggregate plus private per-variant validity details."""

    seen: dict[str, str] = {}
    details: list[dict[str, Any]] = []
    valid_ids: list[str] = []
    reason_counts: Counter[str] = Counter()
    for variant in variants:
        variant_id = str(variant.get("variant_id", ""))
        text = str(variant.get("text", ""))
        normalized = _normalized(text)
        reasons: list[str] = []
        if not normalized:
            reasons.append("empty_text")
        if len([line for line in text.splitlines() if line.strip()]) > 1:
            reasons.append("multiple_paraphrases_or_concatenation")
        if normalized in seen:
            reasons.append("duplicate_variant_text")
        else:
            seen[normalized] = variant_id
        base = _base_text(variant, base_records)
        if base and normalized == base:
            reasons.append("base_identical")
        if _is_prefix_only(base, normalized):
            reasons.append("prefix_only_rewrite")
        if not variant.get("legal_intent_fingerprint") or not variant.get("qrel_fingerprint"):
            reasons.append("missing_intent_or_qrel_fingerprint")
        if reasons:
            reason_counts.update(reasons)
        else:
            valid_ids.append(variant_id)
        details.append(
            {
                "variant_id": variant_id,
                "dialect": str(variant.get("dialect", "")),
                "valid": not reasons,
                "reasons": sorted(set(reasons)),
            }
        )
    valid_ids = sorted(valid_ids)
    aggregate = {
        "schema_version": "phase15-dialect-content-validity-v1",
        "methodology_label": "PHASE15_DEV_CONTENT_ONLY_DIALECT_AUDIT",
        "outcome_independent": True,
        "total_count": len(variants),
        "valid_count": len(valid_ids),
        "invalid_count": len(variants) - len(valid_ids),
        "invalid_reasons": dict(sorted(reason_counts.items())),
        "valid_variant_ids_sha256": hashlib.sha256(
            "\n".join(valid_ids).encode("utf-8")
        ).hexdigest(),
        "valid_count_by_dialect": dict(
            sorted(
                Counter(
                    str(variant.get("dialect", ""))
                    for variant in variants
                    if str(variant.get("variant_id", "")) in set(valid_ids)
                ).items()
            )
        ),
        "effective_retrieval_pair_count": len(valid_ids),
        "identifier_contract_source": "existing pre-outcome dialect validator",
        "base_text_available_to_audit": base_records is not None
        or any("base_text" in variant for variant in variants),
    }
    return aggregate, details


def audit_dialect_content(
    variants: Sequence[Mapping[str, Any]],
    *,
    base_records: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return the text-free content-audit aggregate."""

    aggregate, _details = _audit_dialect_content(variants, base_records=base_records)
    return aggregate


def write_dialect_content_audit(
    variants: Sequence[Mapping[str, Any]],
    *,
    root: Path,
    base_records: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[Path, Path]:
    aggregate, details = _audit_dialect_content(variants, base_records=base_records)
    private_path = root / PRIVATE_PATH
    write_json_atomic(
        private_path,
        {
            **aggregate,
            "provenance": "PHASE15_DEV",
            "variants": details,
            "text_fields_are_private_only": True,
        },
    )
    aggregate_path = write_aggregate_artifact(root, TRACKED_FILENAME, aggregate)
    return private_path, aggregate_path


def validate_dialect_content_audit(private_path: Path, aggregate_path: Path) -> dict[str, int]:
    """Validate that tracked validity metadata matches private per-variant details."""

    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    private = json.loads(private_path.read_text(encoding="utf-8"))
    details = tuple(private.get("variants", ()))
    valid_ids = sorted(str(item["variant_id"]) for item in details if item.get("valid") is True)
    if aggregate.get("total_count") != 60 or len(details) != 60:
        raise ValueError("dialect content audit must cover exactly 60 variants")
    if aggregate.get("valid_count") != len(valid_ids):
        raise ValueError("dialect content audit valid count does not match private details")
    valid_hash = hashlib.sha256("\n".join(valid_ids).encode("utf-8")).hexdigest()
    if aggregate.get("valid_variant_ids_sha256") != valid_hash:
        raise ValueError("dialect content audit valid-ID hash does not match private details")
    if any(key in aggregate for key in ("text", "query", "query_text", "dialect_text")):
        raise ValueError("tracked dialect content audit must be text-free")
    return {"total_count": 60, "valid_count": len(valid_ids), "invalid_count": 60 - len(valid_ids)}
