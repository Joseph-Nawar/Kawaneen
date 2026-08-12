"""Sanitized corpus-level diagnostics for normalization policies."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from kawaneen.corpus.models import CanonicalUnit
from kawaneen.normalization.models import NormalizationPolicy, NormalizationResult
from kawaneen.normalization.policies import normalize_text
from kawaneen.normalization.tokenization import tokenize


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _change_ratio(original: str, normalized: str) -> float:
    mismatches = sum(left != right for left, right in zip(original, normalized, strict=False))
    mismatches += abs(len(original) - len(normalized))
    return mismatches / max(len(original), len(normalized), 1)


@dataclass(frozen=True, slots=True)
class CorpusDiagnostics:
    policy_id: str
    policy_hash: str
    unit_count: int
    character_change_rate: float
    token_change_rate: float
    vocabulary_compression_rate: float
    distinct_form_collision_rate: float
    unit_collision_rate: float
    collision_group_count: int
    transformation_frequencies: dict[str, int]
    top_risky_collisions: tuple[dict[str, object], ...]

    def to_sanitized_dict(self) -> dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "policy_hash": self.policy_hash,
            "unit_count": self.unit_count,
            "character_change_rate": self.character_change_rate,
            "token_change_rate": self.token_change_rate,
            "vocabulary_compression_rate": self.vocabulary_compression_rate,
            "distinct_form_collision_rate": self.distinct_form_collision_rate,
            "unit_collision_rate": self.unit_collision_rate,
            "collision_group_count": self.collision_group_count,
            "transformation_frequencies": dict(sorted(self.transformation_frequencies.items())),
            "collision_groups": list(self.top_risky_collisions),
        }


def diagnose_policy(
    units: Iterable[CanonicalUnit], policy: NormalizationPolicy
) -> CorpusDiagnostics:
    selected = tuple(units)
    raw_vocabulary: set[str] = set()
    normalized_vocabulary: set[str] = set()
    collision_forms: defaultdict[str, set[str]] = defaultdict(set)
    collision_units: defaultdict[str, list[CanonicalUnit]] = defaultdict(list)
    frequencies: Counter[str] = Counter()
    char_change = 0.0
    token_change = 0.0
    raw_token_count = 0
    normalized_token_count = 0
    for unit in selected:
        result = normalize_text(unit.text, policy, audit=True)
        if not isinstance(result, NormalizationResult):
            raise TypeError("audit normalization must return NormalizationResult")
        normalized = result.search_text
        char_change += _change_ratio(unit.text, normalized)
        raw_tokens = tokenize(unit.text)
        normalized_tokens = tokenize(normalized)
        token_change += _change_ratio("\x1f".join(raw_tokens), "\x1f".join(normalized_tokens))
        raw_token_count += len(raw_tokens)
        normalized_token_count += len(normalized_tokens)
        raw_vocabulary.update(raw_tokens)
        normalized_vocabulary.update(normalized_tokens)
        normalized_key = _sha256(normalized)
        source_key = _sha256(unit.text)
        collision_forms[normalized_key].add(source_key)
        collision_units[normalized_key].append(unit)
        frequencies.update(result.transform_counts)

    collision_groups = [key for key, forms in collision_forms.items() if len(forms) > 1]
    collision_unit_total = sum(len(collision_units[key]) for key in collision_groups)
    risky: list[dict[str, object]] = []
    for key in sorted(collision_groups, key=lambda value: (-len(collision_units[value]), value))[
        :10
    ]:
        ids = sorted(unit.unit_id for unit in collision_units[key])
        risky.append(
            {
                "normalized_text_sha256": key,
                "unit_count": len(ids),
                "distinct_source_form_count": len(collision_forms[key]),
                "unit_ids_sha256": _sha256(",".join(ids)),
            }
        )
    return CorpusDiagnostics(
        policy_id=policy.policy_id,
        policy_hash=policy.policy_hash,
        unit_count=len(selected),
        character_change_rate=char_change / max(len(selected), 1),
        token_change_rate=token_change / max(len(selected), 1),
        vocabulary_compression_rate=1 - (len(normalized_vocabulary) / max(len(raw_vocabulary), 1)),
        distinct_form_collision_rate=len(collision_groups) / max(len(collision_forms), 1),
        unit_collision_rate=collision_unit_total / max(len(selected), 1),
        collision_group_count=len(collision_groups),
        transformation_frequencies=dict(frequencies),
        top_risky_collisions=tuple(risky),
    )
