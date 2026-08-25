"""Deterministic, pre-outcome DEV subset selection."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from pydantic import Field

from .contracts import DialectManifest, GeneratorSubsetManifest, Phase15Model


class ReviewCandidate(Phase15Model):
    case_id: str
    language: str
    pipeline_stage: str
    legal_category: str
    answerability: str
    severity: str
    holdout: bool = False
    trigger: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def _stable_order(identifier: str, seed: int) -> tuple[str, str]:
    digest = hashlib.sha256(f"{seed}:{identifier}".encode()).hexdigest()
    return digest, identifier


def _is_dev(record: Mapping[str, Any]) -> bool:
    return str(record.get("split", "dev")).lower() == "dev" and not bool(record.get("holdout"))


def select_dialect_base_intents(
    records: Sequence[Mapping[str, Any]], *, seed: int = 20260826
) -> tuple[str, ...]:
    """Select exactly 20 unique DEV intent IDs before dialect outcomes exist."""

    identifiers = {
        str(record["id"]) for record in records if _is_dev(record) and record.get("id") is not None
    }
    if len(identifiers) < 20:
        raise ValueError("at least 20 unique DEV intent IDs are required")
    return tuple(item[1] for item in sorted(_stable_order(item, seed) for item in identifiers)[:20])


def _select_group(
    records: Sequence[Mapping[str, Any]],
    *,
    predicate: Callable[[Mapping[str, Any]], bool],
    count: int,
    seed: int,
    group_name: str,
) -> tuple[str, ...]:
    candidates = {
        str(record["id"])
        for record in records
        if _is_dev(record) and record.get("id") is not None and predicate(record)
    }
    if len(candidates) < count:
        raise ValueError(
            f"not enough DEV records for {group_name}: need {count}, got {len(candidates)}"
        )
    return tuple(
        item[1] for item in sorted(_stable_order(item, seed) for item in candidates)[:count]
    )


def select_generator_subset(
    records: Sequence[Mapping[str, Any]], *, seed: int = 20260826
) -> GeneratorSubsetManifest:
    """Select the enriched, diagnostic matched-80 DEV generator population."""

    present = _select_group(
        records,
        predicate=lambda item: (
            bool(item.get("answerable")) and bool(item.get("gold_present_in_top8"))
        ),
        count=31,
        seed=seed,
        group_name="answerable gold-present",
    )
    absent = _select_group(
        records,
        predicate=lambda item: (
            bool(item.get("answerable")) and not bool(item.get("gold_present_in_top8"))
        ),
        count=30,
        seed=seed + 1,
        group_name="answerable gold-absent",
    )
    unanswerable = _select_group(
        records,
        predicate=lambda item: not bool(item.get("answerable")),
        count=19,
        seed=seed + 2,
        group_name="unanswerable",
    )
    return GeneratorSubsetManifest(
        seed=seed,
        answerable_gold_present_ids=present,
        answerable_gold_absent_ids=absent,
        unanswerable_ids=unanswerable,
    )


def select_review_cases(
    candidates: Sequence[ReviewCandidate], *, seed: int = 20260826, count: int = 120
) -> tuple[ReviewCandidate, ...]:
    """Select a stable stratified review packet from DEV diagnostics only."""

    if count != 120:
        raise ValueError("Phase 15 review packets must contain exactly 120 cases")
    if any(candidate.holdout for candidate in candidates):
        raise ValueError("HOLDOUT candidates cannot enter the Phase 15 review packet")
    unique = {candidate.case_id: candidate for candidate in candidates}
    if len(unique) < count:
        raise ValueError(f"at least {count} unique DEV review candidates are required")

    buckets: dict[tuple[str, ...], list[ReviewCandidate]] = defaultdict(list)
    for candidate in unique.values():
        key = (
            candidate.language,
            candidate.pipeline_stage,
            candidate.legal_category,
            candidate.answerability,
            candidate.severity,
        )
        buckets[key].append(candidate)
    for bucket in buckets.values():
        bucket.sort(key=lambda item: _stable_order(item.case_id, seed))

    selected: list[ReviewCandidate] = []
    while buckets and len(selected) < count:
        for key in sorted(tuple(buckets)):
            bucket = buckets[key]
            if bucket:
                selected.append(bucket.pop(0))
                if len(selected) == count:
                    break
            if not bucket:
                del buckets[key]
        if not buckets and len(selected) < count:
            break
    return tuple(selected)


def build_dialect_manifest(
    base_intent_ids: Sequence[str],
    variant_ids_by_dialect: Mapping[str, Sequence[str]],
    text_hashes: Mapping[str, str] | None = None,
) -> DialectManifest:
    """Build the aggregate-only dialect manifest after validation."""

    expected = ("egyptian", "gulf_saudi", "levantine")
    if tuple(sorted(variant_ids_by_dialect)) != expected:
        raise ValueError("dialect manifest requires Egyptian, Gulf/Saudi, and Levantine groups")
    variants = tuple(item for dialect in expected for item in variant_ids_by_dialect[dialect])
    return DialectManifest(
        base_intent_ids=tuple(base_intent_ids),
        accepted_variant_ids=variants,
        dialect_counts={dialect: len(variant_ids_by_dialect[dialect]) for dialect in expected},
        text_sha256_by_variant=dict(text_hashes or {}),
    )
