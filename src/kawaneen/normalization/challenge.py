"""Deterministic, policy-independent private normalization challenge generation."""

from __future__ import annotations

# ruff: noqa: RUF001
import json
import random
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from kawaneen.corpus.models import CanonicalUnit
from kawaneen.normalization.corpus import CandidatePolicy

PRIVATE_ROOT = Path("artifacts/private/phase4_normalization")
PHENOMENA = (
    "unchanged_control",
    "alef_forms",
    "diacritics",
    "tatweel",
    "digit_variants",
    "ya_maqsura",
    "ta_marbuta",
    "punctuation_identifiers",
    "combined_variation",
    "collision_risk",
)
_DIGITS_TO_ARABIC = {str(index): chr(0x0660 + index) for index in range(10)}
_COLLISION_MAP = {
    "أ": "ا",
    "إ": "ا",
    "آ": "ا",
    "ٱ": "ا",
    "ى": "ي",
    "ة": "ه",
    **_DIGITS_TO_ARABIC,
    "،": ",",
    "؛": ";",
    "؟": "?",
    "٫": ".",
    "٬": ",",
    "–": "-",
    "—": "-",
    "−": "-",
}


@dataclass(frozen=True, slots=True)
class ChallengeItem:
    query_id: str
    phenomenon: str
    perturbation: str
    target_unit_ids: tuple[str, ...]
    query_text: str
    source_display_text: str


@dataclass(frozen=True, slots=True)
class PrivateChallenge:
    seed: int
    construction_version: str
    items: tuple[ChallengeItem, ...]
    qrels: dict[str, tuple[str, ...]]


def _first_index(text: str, predicate: Callable[[str], bool]) -> int | None:
    for index, char in enumerate(text):
        if predicate(char):
            return index
    return None


def _arabic_letter(char: str) -> bool:
    return "\u0600" <= char <= "\u06ff" and char.isalpha()


def _replace_first(text: str, source: str, target: str) -> str | None:
    index = text.find(source)
    if index < 0:
        return None
    return text[:index] + target + text[index + len(source) :]


def _alef_variant(text: str) -> str | None:
    candidate = _replace_first(text, "ا", "أ")
    if candidate is not None:
        return candidate
    for source in ("أ", "إ", "آ", "ٱ"):
        candidate = _replace_first(text, source, "ا")
        if candidate is not None:
            return candidate
    return None


def _add_diacritic(text: str) -> str | None:
    index = _first_index(text, _arabic_letter)
    if index is None:
        return None
    return text[: index + 1] + "َ" + text[index + 1 :]


def _add_tatweel(text: str) -> str | None:
    index = _first_index(text, _arabic_letter)
    if index is None:
        return None
    return text[: index + 1] + "ـ" + text[index + 1 :]


def _digit_variant(text: str) -> str | None:
    index = _first_index(text, lambda char: char in _DIGITS_TO_ARABIC or "٠" <= char <= "٩")
    if index is None:
        return None
    char = text[index]
    replacement = _DIGITS_TO_ARABIC.get(char, str(ord(char) - ord("٠")))
    return text[:index] + replacement + text[index + 1 :]


def _ya_variant(text: str) -> str | None:
    return _replace_first(text, "ي", "ى") or _replace_first(text, "ى", "ي")


def _ta_variant(text: str) -> str | None:
    return _replace_first(text, "ة", "ه") or _replace_first(text, "ه", "ة")


def _punctuation_variant(text: str) -> str | None:
    for source, target in ((",", "،"), (";", "؛"), ("?", "؟"), ("-", "–")):
        candidate = _replace_first(text, source, target)
        if candidate is not None:
            return candidate
    return None


def _combined_variant(text: str) -> str | None:
    candidate = _alef_variant(text)
    if candidate is None:
        return None
    candidate = _add_diacritic(candidate)
    if candidate is None:
        return None
    candidate = _add_tatweel(candidate)
    return _digit_variant(candidate) if candidate is not None else None


def _collision_key(text: str) -> str:
    collapsed = "".join(_COLLISION_MAP.get(char, char) for char in text)
    return " ".join(collapsed.split())


_PERTURBATIONS: dict[str, tuple[str, Callable[[str], str | None]]] = {
    "unchanged_control": ("identity_control", lambda text: text),
    "alef_forms": ("alef_visual_variant", _alef_variant),
    "diacritics": ("insert_fatha", _add_diacritic),
    "tatweel": ("insert_tatweel", _add_tatweel),
    "digit_variants": ("arabic_digit_variant", _digit_variant),
    "ya_maqsura": ("ya_maqsura_variant", _ya_variant),
    "ta_marbuta": ("ta_marbuta_variant", _ta_variant),
    "punctuation_identifiers": ("arabic_punctuation_variant", _punctuation_variant),
    "combined_variation": ("combined_controlled_variants", _combined_variant),
    "collision_risk": ("ta_marbuta_collision_variant", _ta_variant),
}


def _query_id(seed: int, index: int, phenomenon: str, unit_id: str) -> str:
    return f"phase4-q{seed:08d}-{index:03d}-{phenomenon}-{unit_id}"


def _write_private(challenge: PrivateChallenge, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    items_path = output_root / "challenge_items.jsonl"
    items_path.write_text(
        "".join(
            json.dumps(asdict(item), ensure_ascii=False, sort_keys=True) + "\n"
            for item in challenge.items
        ),
        encoding="utf-8",
    )
    (output_root / "qrels.json").write_text(
        json.dumps(challenge.qrels, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_private_challenge(
    units: Iterable[CanonicalUnit],
    *,
    seed: int = 20260811,
    target_count: int = 150,
    output_root: Path = PRIVATE_ROOT,
) -> PrivateChallenge:
    """Build the same private query/qrel set for every normalization policy."""

    if target_count < len(PHENOMENA):
        raise ValueError(f"target_count must be at least {len(PHENOMENA)}")
    candidates = tuple(sorted(units, key=lambda unit: (unit.unit_id, unit.provenance.source_row)))
    by_collision_key: dict[str, list[CanonicalUnit]] = {}
    for unit in candidates:
        by_collision_key.setdefault(_collision_key(unit.text), []).append(unit)
    rng = random.Random(seed)
    quota, remainder = divmod(target_count, len(PHENOMENA))
    items: list[ChallengeItem] = []
    used: set[tuple[str, str]] = set()
    for phenomenon_index, phenomenon in enumerate(PHENOMENA):
        desired = quota + (1 if phenomenon_index < remainder else 0)
        perturbation_name, perturb = _PERTURBATIONS[phenomenon]
        eligible = [
            unit
            for unit in candidates
            if perturb(unit.text) is not None
            and (
                phenomenon != "collision_risk"
                or len(by_collision_key[_collision_key(unit.text)]) > 1
            )
        ]
        rng.shuffle(eligible)
        selected = 0
        for unit in eligible:
            key = (unit.unit_id, phenomenon)
            if key in used:
                continue
            query_text = perturb(unit.text)
            if query_text is None:
                continue
            targets = tuple(
                sorted(item.unit_id for item in by_collision_key[_collision_key(unit.text)])
            )
            index = len(items)
            items.append(
                ChallengeItem(
                    query_id=_query_id(seed, index, phenomenon, unit.unit_id),
                    phenomenon=phenomenon,
                    perturbation=perturbation_name,
                    target_unit_ids=targets,
                    query_text=query_text,
                    source_display_text=unit.text,
                )
            )
            used.add(key)
            selected += 1
            if selected == desired:
                break
        if selected < desired:
            raise ValueError(f"insufficient eligible challenge units for {phenomenon}")
    qrels = {item.query_id: item.target_unit_ids for item in items}
    challenge = PrivateChallenge(
        seed=seed,
        construction_version="phase4-challenge-v1",
        items=tuple(items),
        qrels=qrels,
    )
    _write_private(challenge, output_root)
    return challenge


def validate_challenge(challenge: PrivateChallenge, candidate_policy: CandidatePolicy) -> None:
    candidate_ids: set[str] = set()
    if len({item.query_id for item in challenge.items}) != len(challenge.items):
        raise ValueError("challenge query IDs are not unique")
    for item in challenge.items:
        candidate_ids.update(item.target_unit_ids)
        if item.phenomenon not in PHENOMENA:
            raise ValueError("challenge contains unknown phenomenon")
        if not item.target_unit_ids:
            raise ValueError("challenge item has no target")
    if not candidate_ids.issubset(candidate_policy.candidate_ids):
        raise ValueError("challenge target is not in the frozen candidate set")
    if set(challenge.qrels) != {item.query_id for item in challenge.items}:
        raise ValueError("qrels do not match challenge items")
