"""Leakage-resistant deterministic split planning."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict

from pydantic import BaseModel, ConfigDict

from kawaneen.evaluation.models import DatasetItem, DatasetSplit


class SplitDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dev_count: int
    holdout_count: int
    cross_split_document_count: int
    cross_split_intent_count: int
    duplicate_query_count: int
    smoke_count: int


def _group_key(item: DatasetItem) -> str:
    base = item.base_intent_id or item.intent_id
    documents = ",".join(sorted(item.source_document_ids))
    return hashlib.sha256(f"{base}|{documents}".encode()).hexdigest()


def _creation_method_value(item: DatasetItem) -> str:
    value = item.creation_method
    return str(getattr(value, "value", value))


def assign_provisional_splits(
    items: tuple[DatasetItem, ...], *, holdout_fraction: float = 0.333333
) -> tuple[DatasetItem, ...]:
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be between zero and one")
    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for item in items:
        keys = [_group_key(item), *(f"doc:{doc}" for doc in item.source_document_ids)]
        for key in keys[1:]:
            union(keys[0], key)
    groups: defaultdict[str, list[DatasetItem]] = defaultdict(list)
    for item in items:
        groups[find(_group_key(item))].append(item)
    ordered = sorted(groups.items())
    target_holdout = round(len(items) * holdout_fraction)
    holdout_groups: set[str] = set()
    reachable: dict[int, tuple[str, ...]] = {0: ()}
    for key, group in ordered:
        size = len(group)
        for total, chosen in sorted(tuple(reachable.items()), reverse=True):
            candidate_total = total + size
            if candidate_total not in reachable:
                reachable[candidate_total] = (*chosen, key)
    best_total = min(
        reachable, key=lambda total: (abs(total - target_holdout), total > target_holdout, total)
    )
    holdout_groups.update(reachable[best_total])
    dev_items = [item for item in items if _group_key(item) not in holdout_groups]
    smoke_ids = {item.query_id for item in sorted(dev_items, key=lambda value: value.query_id)[:20]}
    result: list[DatasetItem] = []
    for item in items:
        split = (
            DatasetSplit.HOLDOUT if find(_group_key(item)) in holdout_groups else DatasetSplit.DEV
        )
        result.append(item.model_copy(update={"split": split, "smoke": item.query_id in smoke_ids}))
    return tuple(result)


def split_diagnostics(items: tuple[DatasetItem, ...]) -> SplitDiagnostics:
    by_document: defaultdict[str, set[DatasetSplit]] = defaultdict(set)
    by_intent: defaultdict[str, set[DatasetSplit]] = defaultdict(set)
    for item in items:
        for doc in item.source_document_ids:
            by_document[doc].add(item.split)
        by_intent[item.base_intent_id or item.intent_id].add(item.split)
    return SplitDiagnostics(
        dev_count=sum(item.split is DatasetSplit.DEV for item in items),
        holdout_count=sum(item.split is DatasetSplit.HOLDOUT for item in items),
        cross_split_document_count=sum(len(splits) > 1 for splits in by_document.values()),
        cross_split_intent_count=sum(len(splits) > 1 for splits in by_intent.values()),
        duplicate_query_count=len(items)
        - len(
            {
                re.sub(r"[^\wء-ي]+", " ", item.query_text.casefold()).strip()
                for item in items
                if _creation_method_value(item) != "robustness_variant"
            }
        )
        - sum(_creation_method_value(item) == "robustness_variant" for item in items),
        smoke_count=sum(item.smoke for item in items),
    )


def load_split(
    items: tuple[DatasetItem, ...],
    *,
    split: DatasetSplit = DatasetSplit.DEV,
    allow_holdout: bool = False,
) -> tuple[DatasetItem, ...]:
    if split is DatasetSplit.HOLDOUT and not allow_holdout:
        raise PermissionError("holdout access requires allow_holdout=True")
    return tuple(item for item in items if item.split is split)
