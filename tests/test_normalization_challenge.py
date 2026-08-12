from __future__ import annotations

from pathlib import Path

# ruff: noqa: RUF001
from kawaneen.corpus.models import CanonicalUnit, SourceProvenance, UnitType
from kawaneen.normalization.challenge import build_private_challenge, validate_challenge
from kawaneen.normalization.corpus import freeze_candidate_policy


def _units() -> tuple[CanonicalUnit, ...]:
    texts = (
        "هذه مادة قانونية",
        "هذا قانون",
        "هذا حكم",
        "حكم قانون",
        "المادة ١٢",
        "في القضية",
        "المادة",
        "المادة 12, 2024",
        "ألف في المادة 12",
        "مادة ة",
        "مادة ه",
    )
    return tuple(
        CanonicalUnit(
            unit_id=f"unit-{index:03d}",
            document_id=f"doc-{index:03d}",
            unit_type=UnitType.CASE_TEXT,
            text=text,
            provenance=SourceProvenance(
                source_id="alarb",
                source_version="v1",
                source_path="fixture",
                source_row=index,
                source_field="text",
            ),
        )
        for index, text in enumerate(texts, start=1)
    )


def test_challenge_has_policy_independent_strata_and_stable_qrels(tmp_path: Path) -> None:
    units = _units()
    challenge = build_private_challenge(units, seed=7, target_count=10, output_root=tmp_path)
    again = build_private_challenge(units, seed=7, target_count=10, output_root=tmp_path / "again")
    assert len(challenge.items) == 10
    assert {item.phenomenon for item in challenge.items} == {
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
    }
    assert challenge.qrels == again.qrels
    assert [item.query_id for item in challenge.items] == [item.query_id for item in again.items]
    validate_challenge(challenge, freeze_candidate_policy(units))
    assert (tmp_path / "challenge_items.jsonl").is_file()
    assert (tmp_path / "qrels.json").is_file()


def test_ambiguous_parallel_units_receive_multi_relevant_qrels(tmp_path: Path) -> None:
    base = _units()[8].model_copy(update={"text": "ألف في قضية 12, 2024"})
    parallel = _units()[8].model_copy(update={"text": "ألف في قضيه 12, 2024"})
    units = tuple(
        item.model_copy(update={"unit_id": f"parallel-a-{index}"})
        for index, item in enumerate((base,) * 10)
    ) + tuple(
        item.model_copy(update={"unit_id": f"parallel-b-{index}"})
        for index, item in enumerate((parallel,) * 10)
    )
    challenge = build_private_challenge(units, seed=11, target_count=10, output_root=tmp_path)
    assert any(len(relevant) > 1 for relevant in challenge.qrels.values())
