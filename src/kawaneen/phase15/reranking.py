"""Pre-registered hard-query reranking analysis."""

from __future__ import annotations

from typing import Mapping, Sequence

from .contracts import Phase15Model
from .statistics import paired_bootstrap_delta, paired_rank_biserial


class HardQueryRule(Phase15Model):
    multi_evidence: bool = True
    exact_provision: bool = True
    authority: bool = True
    deadline: bool = True
    cross_language: bool = True
    long_query: bool = True
    max_pre_rerank_relevant_rank: int = 20


def freeze_hard_query_rule() -> HardQueryRule:
    return HardQueryRule()


def select_hard_queries(
    metadata: Sequence[Mapping[str, object]], *, rule: HardQueryRule | None = None
) -> tuple[str, ...]:
    selected = []
    for item in metadata:
        relevant_rank = item.get("pre_rerank_relevant_rank")
        if bool(item.get("multi_evidence")) or bool(item.get("exact_provision")):
            selected.append(str(item["id"]))
        elif isinstance(relevant_rank, int) and relevant_rank >= (rule or HardQueryRule()).max_pre_rerank_relevant_rank:
            selected.append(str(item["id"]))
    return tuple(dict.fromkeys(selected))


def evaluate_reranking(
    before: Mapping[str, Sequence[float]], after: Mapping[str, Sequence[float]], *, seed: int = 20260826
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
