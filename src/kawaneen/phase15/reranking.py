"""Pre-registered hard-query reranking analysis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .contracts import Phase15Model
from .statistics import paired_bootstrap_delta, paired_rank_biserial


class HardQueryRule(Phase15Model):
    multi_evidence: bool = True
    exact_provision: bool = True
    authority: bool = True
    deadline: bool = True
    cross_language: bool = True
    long_query: bool = True
    min_pre_rerank_relevant_rank_for_hard: int = 20


def freeze_hard_query_rule() -> HardQueryRule:
    return HardQueryRule()


def select_hard_queries(
    metadata: Sequence[Mapping[str, Any]], *, rule: HardQueryRule | None = None
) -> tuple[str, ...]:
    frozen_rule = rule or HardQueryRule()
    selected: list[str] = []
    for item in metadata:
        relevant_rank = item.get("pre_rerank_relevant_rank")
        criteria = (
            frozen_rule.multi_evidence and bool(item.get("multi_evidence")),
            frozen_rule.exact_provision and bool(item.get("exact_provision")),
            frozen_rule.authority and bool(item.get("authority")),
            frozen_rule.deadline and bool(item.get("deadline")),
            frozen_rule.cross_language and bool(item.get("cross_language")),
            frozen_rule.long_query and bool(item.get("long_query")),
            isinstance(relevant_rank, int)
            and relevant_rank >= frozen_rule.min_pre_rerank_relevant_rank_for_hard,
        )
        if any(criteria):
            selected.append(str(item["id"]))
    return tuple(dict.fromkeys(selected))


def evaluate_reranking(
    before: Mapping[str, Sequence[float]],
    after: Mapping[str, Sequence[float]],
    *,
    seed: int = 20260826,
) -> dict[str, object]:
    result: dict[str, object] = {"seed": seed, "metrics": {}}
    metrics: dict[str, object] = {}
    for metric, before_values in before.items():
        after_values = after.get(metric)
        if after_values is None:
            raise ValueError(f"missing reranked metric {metric}")
        bootstrap = paired_bootstrap_delta(after_values, before_values, seed=seed)
        metrics[metric] = {
            **bootstrap.__dict__,
            "rank_biserial": paired_rank_biserial(after_values, before_values),
        }
    result["metrics"] = metrics
    return result
