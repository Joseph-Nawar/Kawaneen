"""Frozen Phase 3 development/holdout qualification bookkeeping."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


class HoldoutAlreadyEvaluatedError(RuntimeError):
    """Raised when a holdout route would be evaluated more than once."""


def _stable_bucket(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16)


def create_frozen_split(
    selection_manifest: Mapping[str, Any],
    output_path: Path,
    *,
    development_fraction: float = 0.7,
) -> dict[str, list[str]]:
    """Create once; a different existing split is a hard failure."""

    if output_path.is_file():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if set(existing) >= {"development", "holdout"}:
            expected = {
                "development": list(existing["development"]),
                "holdout": list(existing["holdout"]),
            }
            if "selection" in selection_manifest:
                selected_ids = {str(item["id"]) for item in selection_manifest["selection"]}
                if set(expected["development"]) | set(expected["holdout"]) != selected_ids:
                    raise ValueError("frozen split cannot be changed")
            if (
                output_path.name == "phase3_split.json"
                and existing.get("status") != "frozen_before_route_results"
            ):
                raise ValueError("frozen split cannot be changed")
            return expected
        raise ValueError("frozen split cannot be changed")
    groups: dict[str, list[str]] = {}
    for item in selection_manifest.get("selection", []):
        category = str(item.get("category", "unknown"))
        groups.setdefault(category, []).append(str(item["id"]))
    development: list[str] = []
    holdout: list[str] = []
    for category in sorted(groups):
        ordered = sorted(groups[category], key=lambda value: (_stable_bucket(value), value))
        cutoff = (
            max(1, min(len(ordered) - 1, round(len(ordered) * development_fraction)))
            if len(ordered) > 1
            else 1
        )
        development.extend(sorted(ordered[:cutoff]))
        holdout.extend(sorted(ordered[cutoff:]))
    result = {"development": development, "holdout": holdout}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def select_development_configuration(
    route: str,
    candidates: Sequence[str],
    development_results: Mapping[str, Mapping[str, float]],
) -> str:
    """Select by development CER, with stable tie breaks and no holdout input."""

    if not candidates or any(candidate not in development_results for candidate in candidates):
        raise ValueError(f"development results incomplete for {route}")
    return min(
        candidates, key=lambda candidate: (float(development_results[candidate]["cer"]), candidate)
    )


def evaluate_holdout_once(
    route: str,
    configuration: str,
    pages: Sequence[str],
    ledger_path: Path,
    evaluator: Callable[[str], Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate each holdout page once and persist an immutable route ledger."""

    if ledger_path.is_file():
        raise HoldoutAlreadyEvaluatedError(f"holdout already evaluated for {route}")
    results = [dict(evaluator(page)) for page in pages]
    payload = {
        "route": route,
        "configuration": configuration,
        "pages": list(pages),
        "results": results,
        "holdout_evaluation_count": 1,
    }
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return payload
