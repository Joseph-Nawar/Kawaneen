"""Bounded private sensitivity validation for the frozen Phase 4 experiment."""

from __future__ import annotations

# ruff: noqa: RUF001
import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from kawaneen.corpus.models import CanonicalUnit
from kawaneen.corpus.serialization import write_json
from kawaneen.normalization.challenge import (
    PHENOMENA,
    PRIVATE_ROOT,
    ChallengeItem,
    PrivateChallenge,
)
from kawaneen.normalization.corpus import CandidatePolicy, freeze_candidate_policy
from kawaneen.normalization.models import NormalizationPolicy
from kawaneen.normalization.policies import all_policies, get_policy, normalize_text
from kawaneen.normalization.retrieval import (
    AblationReport,
    LexicalIndex,
    build_index,
    run_ablation,
    score_query,
)
from kawaneen.normalization.tokenization import tokenize

PRIMARY_CHALLENGE_ROOT = PRIVATE_ROOT / "challenge"
SENSITIVITY_ROOT = PRIVATE_ROOT / "sensitivity_validation"
SENSITIVITY_METRICS_PATH = Path("data/evaluation/phase4_sensitivity_metrics.json")
SENSITIVITY_MANIFEST_PATH = Path("data/manifests/normalization/phase4_sensitivity_manifest.json")
PROBE_SEED = 20260812
PROBE_COUNT = 60
SCORE_EPSILON = 1e-12
MEANINGFUL_PROBE_GAIN = 0.02
CONTROL_REGRESSION_TOLERANCE = 0.02
_TOKEN_SPAN = re.compile(r"\w+|[^\w\s]", re.UNICODE)

_EXPECTED_POLICY = {
    "alef_forms": "arabic-light-v1",
    "diacritics": "arabic-light-v1",
    "tatweel": "arabic-light-v1",
    "digit_variants": "arabic-aggressive-v1",
    "ya_maqsura": "arabic-aggressive-v1",
    "ta_marbuta": "arabic-aggressive-v1",
    "punctuation_identifiers": "arabic-aggressive-v1",
    "combined_variation": "arabic-aggressive-v1",
    "collision_risk": "arabic-aggressive-v1",
}

_ARABIC_DIGITS = {str(index): chr(0x0660 + index) for index in range(10)}
_PROBE_PUNCTUATION = {
    ",": "،",
    ";": "؛",
    "?": "؟",
    "-": "–",
}
_PROBE_COLLISION_MAP = {
    "أ": "ا",
    "إ": "ا",
    "آ": "ا",
    "ٱ": "ا",
    "ى": "ي",
    "ة": "ه",
    **{value: key for key, value in _ARABIC_DIGITS.items()},
    **{source: target for source, target in _PROBE_PUNCTUATION.items()},
}


@dataclass(frozen=True, slots=True)
class SensitivityReport:
    """Private detail plus sanitized aggregate sensitivity evidence."""

    challenge_version: str
    query_count: int
    slice_summary: dict[str, dict[str, Any]]
    policy_effects: dict[str, dict[str, Any]]
    classification_counts: dict[str, int]
    private_rows: tuple[dict[str, Any], ...]

    def to_sanitized_dict(self) -> dict[str, object]:
        return {
            "challenge_version": self.challenge_version,
            "query_count": self.query_count,
            "slice_summary": self.slice_summary,
            "policy_effects": self.policy_effects,
            "classification_counts": self.classification_counts,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def primary_artifact_hashes(output_root: Path = PRIMARY_CHALLENGE_ROOT) -> dict[str, str]:
    """Hash the existing primary challenge files without modifying them."""

    files = sorted(path for path in output_root.rglob("*") if path.is_file())
    if not files:
        raise FileNotFoundError(f"primary challenge root is empty: {output_root}")
    return {path.relative_to(output_root).as_posix(): _sha256(path) for path in files}


def load_frozen_primary_challenge(
    output_root: Path = PRIMARY_CHALLENGE_ROOT,
) -> PrivateChallenge:
    """Load the existing Phase-4 challenge and qrels read-only."""

    items_path = output_root / "challenge_items.jsonl"
    qrels_path = output_root / "qrels.json"
    item_rows: list[dict[str, Any]] = [
        json.loads(line)
        for line in items_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    qrel_rows: dict[str, list[str]] = json.loads(qrels_path.read_text(encoding="utf-8"))
    items = tuple(
        ChallengeItem(
            query_id=str(row["query_id"]),
            phenomenon=str(row["phenomenon"]),
            perturbation=str(row["perturbation"]),
            target_unit_ids=tuple(str(value) for value in row["target_unit_ids"]),
            query_text=str(row["query_text"]),
            source_display_text=str(row["source_display_text"]),
        )
        for row in item_rows
    )
    if len({item.query_id for item in items}) != len(items):
        raise ValueError("primary challenge query IDs are not unique")
    qrels = {
        str(query_id): tuple(str(unit_id) for unit_id in unit_ids)
        for query_id, unit_ids in qrel_rows.items()
    }
    if set(qrels) != {item.query_id for item in items}:
        raise ValueError("primary challenge qrels do not match query IDs")
    if any(not targets for targets in qrels.values()):
        raise ValueError("primary challenge contains an empty relevance set")
    if any(tuple(qrels[item.query_id]) != item.target_unit_ids for item in items):
        raise ValueError("primary challenge qrels differ from item targets")
    return PrivateChallenge(
        seed=20260811,
        construction_version="phase4-primary-challenge-v1",
        items=items,
        qrels=qrels,
    )


def assert_primary_frozen(
    challenge: PrivateChallenge,
    candidate_ids: Sequence[str],
    manifest_path: Path = Path("data/manifests/normalization/phase4_manifest.json"),
) -> None:
    """Verify that the loaded primary challenge matches the existing scope manifest."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidate = manifest["candidate_policy"]
    ids_hash = hashlib.sha256(",".join(candidate_ids).encode("utf-8")).hexdigest()
    if candidate["candidate_count"] != len(candidate_ids):
        raise ValueError("candidate count differs from the frozen primary manifest")
    if candidate["candidate_ids_hash"] != ids_hash:
        raise ValueError("candidate IDs differ from the frozen primary manifest")
    if len(challenge.items) != 150:
        raise ValueError("primary challenge is not the frozen 150-query challenge")
    if not all(set(item.target_unit_ids) <= set(candidate_ids) for item in challenge.items):
        raise ValueError("primary challenge target is outside the frozen candidate set")


def tokenizer_has_no_hidden_normalization() -> bool:
    """Return whether tokenization distinguishes documented Arabic orthographies."""

    pairs = (("أ", "ا"), ("١", "1"), ("ى", "ي"), ("ة", "ه"), ("،", ","))
    return all(tokenize(left) != tokenize(right) for left, right in pairs)


def _counter_overlap(left: Sequence[str], right: Sequence[str]) -> int:
    left_counts = Counter(left)
    right_counts = Counter(right)
    return sum((left_counts & right_counts).values())


def _ranked_lookup(ranking: Sequence[Any]) -> dict[str, tuple[float, int]]:
    return {hit.unit_id: (float(hit.score), rank) for rank, hit in enumerate(ranking, start=1)}


def _policy_audit(
    index: LexicalIndex,
    policy: NormalizationPolicy,
    item: ChallengeItem,
    units_by_id: dict[str, CanonicalUnit],
    raw_query_tokens: Sequence[str],
) -> dict[str, Any]:
    query = normalize_text(item.query_text, policy)
    if not isinstance(query, str):
        raise TypeError("non-audit normalization must return str")
    query_tokens = tokenize(query)
    ranking = score_query(index, query)
    lookup = _ranked_lookup(ranking)
    overlaps: dict[str, int] = {}
    targets: dict[str, dict[str, float | int | None]] = {}
    for target_id in item.target_unit_ids:
        target = normalize_text(units_by_id[target_id].text, policy)
        if not isinstance(target, str):
            raise TypeError("non-audit normalization must return str")
        target_tokens = tokenize(target)
        overlaps[target_id] = _counter_overlap(query_tokens, target_tokens)
        score, rank = lookup.get(target_id, (0.0, None))
        targets[target_id] = {"score": score, "rank": rank}
    return {
        "query_tokens": list(query_tokens),
        "target_overlap": overlaps,
        "relevant_targets": targets,
        "top10_ids": [hit.unit_id for hit in ranking[:10]],
    }


def _classify_row(
    phenomenon: str,
    expected_changed: bool,
    raw_preserves: bool,
    repaired: bool,
    score_changed: bool,
    rank_changed: bool,
) -> str:
    if phenomenon == "unchanged_control":
        return "unchanged_control"
    if not expected_changed:
        return "harness_or_perturbation_defect"
    if not raw_preserves:
        return "redundant_or_tokenizer_insensitive"
    if not repaired:
        return "harness_or_perturbation_defect"
    if rank_changed:
        return "rank_or_top10_change"
    if score_changed:
        return "score_change_no_rank_change"
    return "harness_or_perturbation_defect"


def _median(values: Iterable[float]) -> float:
    selected = tuple(values)
    return statistics.median(selected) if selected else 0.0


def select_sensitivity_policy(
    policy_metrics: dict[str, dict[str, float]],
    intervals: dict[str, dict[str, float | int]],
    slice_metrics: dict[str, dict[str, dict[str, float]]],
) -> dict[str, object]:
    """Apply the bounded sensitivity-validation decision rule."""

    def meaningful_gain(less: str, more: str) -> bool:
        for metric in ("mrr_at_10", "ndcg_at_10", "recall_at_10"):
            interval = intervals.get(f"{less}__vs__{more}__{metric}")
            if interval is None:
                continue
            estimate = -float(interval["estimate"])
            lower = -float(interval["upper"])
            if estimate >= MEANINGFUL_PROBE_GAIN and lower > 0:
                return True
        return False

    def no_guardrail_regression(less: str, more: str) -> bool:
        for slice_name in ("unchanged_control", "collision_risk"):
            left = slice_metrics.get(less, {}).get(slice_name, {})
            right = slice_metrics.get(more, {}).get(slice_name, {})
            if (
                left
                and right
                and any(
                    right.get(metric, 0.0) - left.get(metric, 0.0) < -CONTROL_REGRESSION_TOLERANCE
                    for metric in ("mrr_at_10", "ndcg_at_10", "recall_at_10")
                )
            ):
                return False
        return True

    selected = "arabic-raw-v1"
    reason = "No sensitivity-validation policy met the predefined meaningful-gain rule."
    if meaningful_gain("arabic-raw-v1", "arabic-light-v1") and no_guardrail_regression(
        "arabic-raw-v1", "arabic-light-v1"
    ):
        selected = "arabic-light-v1"
        reason = "Light showed a meaningful paired probe gain without control/collision regression."
        if meaningful_gain("arabic-light-v1", "arabic-aggressive-v1") and no_guardrail_regression(
            "arabic-light-v1", "arabic-aggressive-v1"
        ):
            selected = "arabic-aggressive-v1"
            reason = "Aggressive clearly improved over light without control/collision regression."
    return {"selected_policy_id": selected, "rationale": reason}


def _anchor_target_id(item: ChallengeItem, units_by_id: dict[str, CanonicalUnit]) -> str:
    for target_id in item.target_unit_ids:
        unit = units_by_id[target_id]
        if unit.text == item.source_display_text or item.query_id.endswith(f"-{target_id}"):
            return target_id
    return item.target_unit_ids[0]


def audit_challenge(
    units: Iterable[CanonicalUnit],
    challenge: PrivateChallenge,
    policies: Iterable[NormalizationPolicy] = all_policies(),
) -> SensitivityReport:
    """Audit token, overlap, score, rank, and top-10 sensitivity privately."""

    candidates = tuple(units)
    units_by_id = {unit.unit_id: unit for unit in candidates}
    policy_list = tuple(policies)
    indexes = {policy.policy_id: build_index(candidates, policy) for policy in policy_list}
    rows: list[dict[str, Any]] = []
    raw_policy = get_policy("arabic-raw-v1")
    for item in challenge.items:
        raw_query_tokens = tokenize(item.query_text)
        raw_target_tokens = {
            target_id: tokenize(units_by_id[target_id].text) for target_id in item.target_unit_ids
        }
        anchor_target_id = _anchor_target_id(item, units_by_id)
        max_target_length = max(len(tokens) for tokens in raw_target_tokens.values())
        raw_overlap = _counter_overlap(raw_query_tokens, raw_target_tokens[anchor_target_id])
        expected_policy_id = _EXPECTED_POLICY.get(item.phenomenon)
        policy_audits = {
            policy.policy_id: _policy_audit(
                indexes[policy.policy_id], policy, item, units_by_id, raw_query_tokens
            )
            for policy in policy_list
        }
        raw_detail = policy_audits[raw_policy.policy_id]
        raw_targets = cast(dict[str, Any], raw_detail["relevant_targets"])
        raw_top10 = tuple(raw_detail["top10_ids"])
        expected_changed = False
        expected_target_changed = False
        repaired = False
        expected_overlap = raw_overlap
        if expected_policy_id is not None:
            expected_query = policy_audits[expected_policy_id]["query_tokens"]
            expected_changed = tuple(expected_query) != tuple(raw_query_tokens)
            expected_overlaps = cast(
                dict[str, int], policy_audits[expected_policy_id]["target_overlap"]
            )
            raw_overlaps = cast(dict[str, int], raw_detail["target_overlap"])
            expected_target_changed = any(
                expected_overlaps[target_id] != raw_overlaps[target_id]
                for target_id in item.target_unit_ids
            )
            expected_overlap = expected_overlaps[anchor_target_id]
            repaired = expected_overlap > raw_overlap
        policy_effects: dict[str, object] = {}
        any_score_change = False
        any_rank_change = False
        any_top10_change = False
        for policy in policy_list:
            detail = policy_audits[policy.policy_id]
            targets = cast(dict[str, dict[str, Any]], detail["relevant_targets"])
            score_deltas: dict[str, float] = {}
            rank_deltas: dict[str, int | None] = {}
            for target_id in item.target_unit_ids:
                raw_target = raw_targets[target_id]
                current_target = targets[target_id]
                score_delta = float(current_target["score"]) - float(raw_target["score"])
                score_deltas[target_id] = score_delta
                raw_rank = raw_target["rank"]
                current_rank = current_target["rank"]
                rank_deltas[target_id] = (
                    None
                    if raw_rank is None or current_rank is None
                    else int(current_rank) - int(raw_rank)
                )
            top10 = tuple(detail["top10_ids"])
            top10_ids_changed = top10 != raw_top10
            top10_order_changed = top10_ids_changed and set(top10) == set(raw_top10)
            score_changed = any(abs(delta) > SCORE_EPSILON for delta in score_deltas.values())
            rank_changed = any(delta not in (0, None) for delta in rank_deltas.values())
            any_score_change |= score_changed
            any_rank_change |= rank_changed
            any_top10_change |= top10_ids_changed
            policy_effects[policy.policy_id] = {
                "query_tokens": detail["query_tokens"],
                "target_overlap": detail["target_overlap"],
                "relevant_targets": targets,
                "score_delta_vs_raw": score_deltas,
                "rank_delta_vs_raw": rank_deltas,
                "top10_ids_changed_vs_raw": top10_ids_changed,
                "top10_order_changed_vs_raw": top10_order_changed,
            }
        classification = _classify_row(
            item.phenomenon,
            expected_changed or expected_target_changed,
            item.phenomenon != "unchanged_control" and raw_overlap < max_target_length,
            repaired,
            any_score_change,
            any_rank_change or any_top10_change,
        )
        raw_preserves_mismatch = (
            item.phenomenon != "unchanged_control" and raw_overlap < max_target_length
        )
        rows.append(
            {
                "query_id": item.query_id,
                "phenomenon": item.phenomenon,
                "perturbation": item.perturbation,
                "lexical_token_count": len(tokenize(item.source_display_text)),
                "perturbed_token_count": len(raw_query_tokens),
                "perturbed_token_ratio": len(raw_query_tokens)
                / max(len(tokenize(item.source_display_text)), 1),
                "expected_policy_id": expected_policy_id,
                "intended_transform_changes_query": expected_changed,
                "intended_transform_changes_relevant_target": expected_target_changed,
                "raw_preserves_intended_mismatch": raw_preserves_mismatch,
                "intended_transform_repairs_overlap": repaired,
                "raw_target_overlap": raw_overlap,
                "expected_target_overlap": expected_overlap,
                "anchor_target_id": anchor_target_id,
                "classification": classification,
                "policies": policy_effects,
            }
        )
    slice_summary: dict[str, dict[str, object]] = {}
    for phenomenon in PHENOMENA:
        selected = [row for row in rows if row["phenomenon"] == phenomenon]
        score_change_pct: dict[str, float] = {}
        overlap_change_pct: dict[str, float] = {}
        top10_change_pct: dict[str, float] = {}
        query_change_pct: dict[str, float] = {}
        for policy in policy_list:
            policy_id = policy.policy_id
            score_change_pct[policy_id] = (
                100
                * sum(
                    any(
                        abs(float(delta)) > SCORE_EPSILON
                        for delta in row["policies"][policy_id]["score_delta_vs_raw"].values()
                    )
                    for row in selected
                )
                / max(len(selected), 1)
            )
            overlap_change_pct[policy_id] = (
                100
                * sum(
                    row["policies"][policy_id]["target_overlap"]
                    != row["policies"]["arabic-raw-v1"]["target_overlap"]
                    for row in selected
                )
                / max(len(selected), 1)
            )
            top10_change_pct[policy_id] = (
                100
                * sum(
                    bool(row["policies"][policy_id]["top10_ids_changed_vs_raw"]) for row in selected
                )
                / max(len(selected), 1)
            )
            query_change_pct[policy_id] = (
                100
                * sum(
                    row["policies"][policy_id]["query_tokens"]
                    != row["policies"]["arabic-raw-v1"]["query_tokens"]
                    for row in selected
                )
                / max(len(selected), 1)
            )
        slice_summary[phenomenon] = {
            "query_count": len(selected),
            "median_query_length": _median(row["lexical_token_count"] for row in selected),
            "median_perturbed_token_ratio": _median(
                row["perturbed_token_ratio"] for row in selected
            ),
            "raw_preserves_intended_mismatch_pct": 100
            * sum(bool(row["raw_preserves_intended_mismatch"]) for row in selected)
            / max(len(selected), 1),
            "intended_transform_repairs_pct": 100
            * sum(bool(row["intended_transform_repairs_overlap"]) for row in selected)
            / max(len(selected), 1),
            "query_token_change_pct_by_policy": query_change_pct,
            "target_overlap_change_pct_by_policy": overlap_change_pct,
            "nonzero_target_score_delta_pct_by_policy": score_change_pct,
            "rank_or_top10_change_pct_by_policy": top10_change_pct,
            "nonzero_target_score_delta_pct": score_change_pct.get("arabic-light-v1", 0.0),
            "rank_or_top10_change_pct": top10_change_pct.get("arabic-light-v1", 0.0),
        }
    policy_effects_summary: dict[str, dict[str, object]] = {}
    for policy in policy_list:
        changes = [
            row["policies"][policy.policy_id]
            for row in rows
            if policy.policy_id != raw_policy.policy_id
        ]
        policy_effects_summary[policy.policy_id] = {
            "queries_with_nonzero_target_score_delta": sum(
                any(
                    abs(float(delta)) > SCORE_EPSILON
                    for delta in item["score_delta_vs_raw"].values()
                )
                for item in changes
            ),
            "queries_with_target_overlap_delta": sum(
                item["target_overlap"]
                != rows[index]["policies"][raw_policy.policy_id]["target_overlap"]
                for index, item in enumerate([row["policies"][policy.policy_id] for row in rows])
            )
            if policy.policy_id != raw_policy.policy_id
            else 0,
            "queries_with_top10_change": sum(
                bool(item["top10_ids_changed_vs_raw"]) for item in changes
            ),
        }
    classification_counts = Counter(str(row["classification"]) for row in rows)
    return SensitivityReport(
        challenge_version=challenge.construction_version,
        query_count=len(rows),
        slice_summary=slice_summary,
        policy_effects=policy_effects_summary,
        classification_counts=dict(sorted(classification_counts.items())),
        private_rows=tuple(rows),
    )


def _best_window(
    text: str,
    predicate: Callable[[str], bool],
    token_frequencies: dict[str, int],
) -> str | None:
    spans = tuple(_TOKEN_SPAN.finditer(text))
    eligible_indices = [
        index
        for index, match in enumerate(spans)
        if any(predicate(char) for char in text[match.start() : match.end()])
    ]
    if not eligible_indices:
        return None
    sample_positions = sorted(
        {
            eligible_indices[index]
            for index in (
                0,
                len(eligible_indices) // 4,
                len(eligible_indices) // 2,
                (3 * len(eligible_indices)) // 4,
                len(eligible_indices) - 1,
            )
        }
    )
    choices: list[tuple[tuple[float, int, int], str]] = []
    for token_index in sample_positions:
        for width in range(2, 9):
            if width > len(spans):
                continue
            start = max(0, min(token_index - width // 2, len(spans) - width))
            end = start + width
            window = text[spans[start].start() : spans[end - 1].end()]
            if not any(
                any(predicate(char) for char in text[match.start() : match.end()])
                for match in spans[start:end]
            ):
                continue
            tokens = tokenize(window)
            word_tokens = tuple(token for token in tokens if token.isalnum())
            rarity = sum(1.0 / max(token_frequencies.get(token, 1), 1) for token in word_tokens)
            choices.append(((rarity, len(word_tokens), -start), window))
    return max(choices, key=lambda item: item[0])[1] if choices else None


def _replace_first(
    text: str, mapping: dict[str, str], predicate: Callable[[str], bool]
) -> str | None:
    for index, char in enumerate(text):
        if predicate(char) and char in mapping:
            return text[:index] + mapping[char] + text[index + 1 :]
    return None


def _probe_perturbation(phenomenon: str, text: str) -> str | None:
    def arabic_letters(char: str) -> bool:
        return "\u0600" <= char <= "\u06ff" and char.isalpha()

    if phenomenon == "unchanged_control":
        return text
    if phenomenon == "alef_forms":
        return _replace_first(
            text,
            {"ا": "أ", "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا"},
            lambda char: char in "اأإآٱ",
        )
    if phenomenon == "diacritics":
        return next(
            (
                text[: index + 1] + "َ" + text[index + 1 :]
                for index, char in enumerate(text)
                if arabic_letters(char)
            ),
            None,
        )
    if phenomenon == "tatweel":
        return next(
            (
                text[: index + 1] + "ـ" + text[index + 1 :]
                for index, char in enumerate(text)
                if arabic_letters(char)
            ),
            None,
        )
    if phenomenon == "digit_variants":
        return _replace_first(text, _ARABIC_DIGITS, lambda char: char in _ARABIC_DIGITS)
    if phenomenon == "ya_maqsura":
        return _replace_first(text, {"ي": "ى", "ى": "ي"}, lambda char: char in "يى")
    if phenomenon == "ta_marbuta":
        return _replace_first(text, {"ة": "ه", "ه": "ة"}, lambda char: char in "ةه")
    if phenomenon == "punctuation_identifiers":
        return _replace_first(text, _PROBE_PUNCTUATION, lambda char: char in _PROBE_PUNCTUATION)
    if phenomenon == "combined_variation":
        first = _probe_perturbation("alef_forms", text)
        if first is None:
            return None
        second = _probe_perturbation("tatweel", first)
        if second is None:
            return None
        return _probe_perturbation("digit_variants", second)
    if phenomenon == "collision_risk":
        return _replace_first(text, {"ة": "ه", "ه": "ة"}, lambda char: char in "ةه")
    raise ValueError(f"unknown sensitivity phenomenon: {phenomenon}")


def _probe_key(text: str) -> str:
    return " ".join("".join(_PROBE_COLLISION_MAP.get(char, char) for char in text).split())


def build_private_sensitivity_probe(
    units: Iterable[CanonicalUnit],
    candidate_policy: CandidatePolicy,
    *,
    seed: int = PROBE_SEED,
    output_root: Path = SENSITIVITY_ROOT / "probe",
) -> PrivateChallenge:
    """Build 10 balanced, policy-independent slices of six short private queries."""

    candidates = tuple(sorted(units, key=lambda unit: unit.unit_id))
    if {unit.unit_id for unit in candidates} != set(candidate_policy.candidate_ids):
        raise ValueError("probe candidates differ from the frozen candidate policy")

    def arabic_letters(char: str) -> bool:
        return "\u0600" <= char <= "\u06ff" and char.isalpha()

    predicates: dict[str, Callable[[str], bool]] = {
        "alef_forms": lambda char: char in "اأإآٱ",
        "digit_variants": lambda char: char in "0123456789",
        "ya_maqsura": lambda char: char in "يى",
        "ta_marbuta": lambda char: char in "ةه",
        "punctuation_identifiers": lambda char: char in ",;?-",
        "combined_variation": lambda char: char in "اأإآٱ0123456789",
        "collision_risk": lambda char: char in "ةه",
    }
    token_frequencies = Counter(
        token for unit in candidates for token in tokenize(unit.text) if token.isalnum()
    )
    records_by_slice: dict[str, list[tuple[CanonicalUnit, str, str]]] = defaultdict(list)
    for phenomenon in PHENOMENA:
        predicate = predicates.get(phenomenon, arabic_letters)
        for unit in candidates:
            window = _best_window(unit.text, predicate, token_frequencies)
            if window is None:
                continue
            query = _probe_perturbation(phenomenon, window)
            if query is not None:
                records_by_slice[phenomenon].append((unit, window, query))
    for phenomenon in PHENOMENA:
        records_by_slice[phenomenon] = list(dict.fromkeys(records_by_slice[phenomenon]))
    full_text_groups: defaultdict[str, set[str]] = defaultdict(set)
    for unit in candidates:
        full_text_groups[_probe_key(unit.text)].add(unit.unit_id)
    items: list[ChallengeItem] = []
    for phenomenon in PHENOMENA:
        selected: list[tuple[CanonicalUnit, str, str]] = []
        for record in records_by_slice[phenomenon]:
            unit, window, query = record
            targets = full_text_groups[_probe_key(unit.text)]
            if phenomenon == "collision_risk" and len(targets) < 2:
                continue
            if not any(existing[0].unit_id == unit.unit_id for existing in selected):
                selected.append(record)
            if len(selected) == 6:
                break
        if len(selected) != 6:
            raise ValueError(f"insufficient short-query probe records for {phenomenon}")
        for unit, window, query in selected:
            query_id = f"phase4-probe-{seed:08d}-{len(items):03d}-{phenomenon}-{unit.unit_id}"
            targets = tuple(sorted(full_text_groups[_probe_key(unit.text)]))
            items.append(
                ChallengeItem(
                    query_id=query_id,
                    phenomenon=phenomenon,
                    perturbation=f"sensitivity_probe_{phenomenon}",
                    target_unit_ids=targets,
                    query_text=query,
                    source_display_text=window,
                )
            )
    challenge = PrivateChallenge(
        seed=seed,
        construction_version="phase4-sensitivity-probe-v1",
        items=tuple(items),
        qrels={item.query_id: item.target_unit_ids for item in items},
    )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "challenge_items.jsonl").write_text(
        "".join(
            json.dumps(asdict(item), ensure_ascii=False, sort_keys=True) + "\n" for item in items
        ),
        encoding="utf-8",
    )
    write_json(output_root / "qrels.json", challenge.qrels)
    return challenge


def _report_metrics(report: AblationReport) -> dict[str, object]:
    return {
        "policy_metrics": report.policy_metrics,
        "slice_metrics": report.slice_metrics,
        "pairwise_wins_ties_losses": report.pairwise_wins_ties_losses,
        "paired_confidence_intervals": report.paired_confidence_intervals,
    }


def run_sensitivity_validation() -> dict[str, object]:
    """Run the bounded primary audit and 60-item probe without rewriting primary data."""

    from kawaneen.normalization.corpus import load_candidate_units, select_representative_subset

    primary_hashes_before = primary_artifact_hashes()
    units = select_representative_subset(load_candidate_units(Path("data/interim/canonical")))
    candidate_policy = freeze_candidate_policy(units)
    primary = load_frozen_primary_challenge()
    assert_primary_frozen(primary, tuple(unit.unit_id for unit in units))
    primary_report = audit_challenge(units, primary)
    probe = build_private_sensitivity_probe(units, candidate_policy)
    probe_report = audit_challenge(units, probe)
    primary_ablation = run_ablation(units, primary, all_policies())
    probe_ablation = run_ablation(units, probe, all_policies())
    primary_hashes_after = primary_artifact_hashes()
    if primary_hashes_before != primary_hashes_after:
        raise RuntimeError("primary challenge artifacts changed during sensitivity validation")
    private_root = SENSITIVITY_ROOT
    private_root.mkdir(parents=True, exist_ok=True)
    write_json(
        private_root / "primary_sensitivity_audit.json",
        {"rows": list(primary_report.private_rows)},
    )
    write_json(
        private_root / "probe_sensitivity_audit.json",
        {"rows": list(probe_report.private_rows)},
    )
    validation_decision = select_sensitivity_policy(
        probe_ablation.policy_metrics,
        probe_ablation.paired_confidence_intervals,
        probe_ablation.slice_metrics,
    )
    metrics = {
        "schema_version": 1,
        "status": "phase4_sensitivity_validation_complete",
        "primary_challenge_version": primary.construction_version,
        "primary_artifact_hashes": primary_hashes_before,
        "candidate_ids_hash": candidate_policy.candidate_ids_hash,
        "candidate_count": len(units),
        "tokenizer_hidden_normalization": not tokenizer_has_no_hidden_normalization(),
        "primary_sensitivity": primary_report.to_sanitized_dict(),
        "primary_ablation": _report_metrics(primary_ablation),
        "probe": {
            "query_count": len(probe.items),
            "slice_counts": dict(sorted(Counter(item.phenomenon for item in probe.items).items())),
            "sensitivity": probe_report.to_sanitized_dict(),
            "ablation": _report_metrics(probe_ablation),
        },
        "decision": {
            "primary_selected_policy_id": "arabic-raw-v1",
            "validation_selected_policy_id": validation_decision["selected_policy_id"],
            "rationale": validation_decision["rationale"],
            "phase7_revalidation_required": True,
        },
    }
    manifest = {
        "schema_version": 1,
        "status": "phase4_sensitivity_validation_complete",
        "primary_challenge_version": primary.construction_version,
        "primary_artifact_hashes": primary_hashes_before,
        "candidate_policy": candidate_policy.to_sanitized_dict(),
        "probe": {
            "construction_version": probe.construction_version,
            "seed": probe.seed,
            "query_count": len(probe.items),
            "slice_counts": dict(sorted(Counter(item.phenomenon for item in probe.items).items())),
        },
        "private_artifact_root": SENSITIVITY_ROOT.as_posix(),
        "decision": metrics["decision"],
    }
    write_json(SENSITIVITY_MANIFEST_PATH, manifest)
    write_json(SENSITIVITY_METRICS_PATH, metrics)
    return {"manifest": manifest, "metrics": metrics}
