from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from kawaneen.corpus.models import CanonicalUnit, SourceProvenance, UnitType
from kawaneen.normalization.challenge import ChallengeItem, PrivateChallenge
from kawaneen.normalization.corpus import freeze_candidate_policy
from kawaneen.normalization.policies import all_policies
from kawaneen.normalization.sensitivity import (
    audit_challenge,
    build_private_sensitivity_probe,
    load_frozen_primary_challenge,
    primary_artifact_hashes,
    select_sensitivity_policy,
    tokenizer_has_no_hidden_normalization,
)
from kawaneen.normalization.tokenization import tokenize


def _units() -> tuple[CanonicalUnit, ...]:
    texts = (
        "المادة 12/2024 في القضية",
        "الماده 12/2024 في القضيه",
        "ألف في حكم قانوني",
        "الف في حكم قانوني",
        "حـكم في المادة 12",
        "حكم في الماده ١٢",
        "قرار 123/2024 مرجع",
        "قرار ١٢٣/٢٠٢٤ مرجع",
        "قضية 12-2024 أمام المحكمة",
        "قضيه ١٢-٢٠٢٤ أمام المحكمه",
        "المادة 13/2025 في القضية",
        "المادة 14/2026 في القضية",
        "المادة 15/2027 في القضية",
        "المادة 16/2028 في القضية",
        "مرجع 17-2029 أمام المحكمة",
        "مرجع 18-2030 أمام المحكمة",
        "مرجع 19-2031 أمام المحكمة",
        "مرجع 20-2032 أمام المحكمة",
        "مادة 21/2033 في القضية",
        "مادة 21/2033 في القضية",
        "مادة 22/2034 في القضية",
        "مادة 22/2034 في القضية",
        "مادة 23/2035 في القضية",
        "مادة 23/2035 في القضية",
        "مادة 24/2036 في القضية",
        "مادة 24/2036 في القضية",
        "مادة 25/2037 في القضية",
        "مادة 25/2037 في القضية",
        "مادة 26/2038 في القضية",
        "مادة 26/2038 في القضية",
        "هذا نص قانوني ثابت",
        "ذلك نص قانوني آخر",
    )
    return tuple(
        CanonicalUnit(
            unit_id=f"unit-{index:03d}",
            document_id=f"doc-{index:03d}",
            unit_type=UnitType.CASE_TEXT,
            text=text,
            provenance=SourceProvenance(
                source_id="synthetic",
                source_version="v1",
                source_path="fixture",
                source_row=index,
                source_field="text",
            ),
        )
        for index, text in enumerate(texts, start=1)
    )


def test_primary_loader_reads_private_items_and_qrels_without_rewriting(tmp_path: Path) -> None:
    units = _units()
    challenge = build_private_sensitivity_probe(
        units, freeze_candidate_policy(units), output_root=tmp_path
    )
    primary_root = tmp_path / "primary"
    primary_root.mkdir()
    (primary_root / "challenge_items.jsonl").write_text(
        "".join(json.dumps(asdict(item), ensure_ascii=False) + "\n" for item in challenge.items),
        encoding="utf-8",
    )
    (primary_root / "qrels.json").write_text(json.dumps(challenge.qrels), encoding="utf-8")
    before = primary_artifact_hashes(primary_root)
    loaded = load_frozen_primary_challenge(primary_root)
    assert len(loaded.items) == 60
    assert loaded.qrels == challenge.qrels
    assert primary_artifact_hashes(primary_root) == before


def test_sensitivity_probe_is_deterministic_balanced_and_candidate_bound(tmp_path: Path) -> None:
    units = _units()
    candidate_policy = freeze_candidate_policy(units)
    first = build_private_sensitivity_probe(
        units, candidate_policy, seed=7, output_root=tmp_path / "one"
    )
    second = build_private_sensitivity_probe(
        units, candidate_policy, seed=7, output_root=tmp_path / "two"
    )
    assert first.items == second.items
    assert first.qrels == second.qrels
    assert len(first.items) == 60
    assert {item.phenomenon for item in first.items} == {
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
    assert all(2 <= len(tokenize(item.query_text)) <= 8 for item in first.items)
    assert all(set(item.target_unit_ids) <= candidate_policy.candidate_ids for item in first.items)
    assert sum(len(targets) > 1 for targets in first.qrels.values()) >= 1


def test_tokenizer_is_orthography_sensitive_and_audit_records_repair(tmp_path: Path) -> None:
    assert tokenizer_has_no_hidden_normalization()
    units = _units()
    item = ChallengeItem(
        query_id="sensitivity-q1",
        phenomenon="alef_forms",
        perturbation="synthetic_alef",
        target_unit_ids=("unit-003",),
        query_text="الف في حكم قانوني",
        source_display_text="ألف في حكم قانوني",
    )
    challenge = PrivateChallenge(
        seed=1,
        construction_version="synthetic-sensitivity-v1",
        items=(item,),
        qrels={item.query_id: item.target_unit_ids},
    )
    report = audit_challenge(units, challenge, all_policies())
    row = report.private_rows[0]
    assert row["raw_preserves_intended_mismatch"] is True
    assert row["intended_transform_repairs_overlap"] is True
    assert row["anchor_target_id"] == "unit-003"
    assert row["intended_transform_changes_query"] is False
    assert row["intended_transform_changes_relevant_target"] is True
    assert (
        row["policies"]["arabic-light-v1"]["target_overlap"]["unit-003"]
        > row["policies"]["arabic-raw-v1"]["target_overlap"]["unit-003"]
    )
    assert report.to_sanitized_dict()["query_count"] == 1


def test_probe_construction_does_not_invoke_candidate_normalizers(
    tmp_path: Path, monkeypatch: object
) -> None:
    import kawaneen.normalization.sensitivity as sensitivity

    def fail(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("probe construction invoked a candidate normalizer")

    monkeypatch.setattr(sensitivity, "normalize_text", fail)
    units = _units()
    probe = build_private_sensitivity_probe(
        units, freeze_candidate_policy(units), seed=9, output_root=tmp_path
    )
    assert len(probe.items) == 60


def test_sensitivity_selection_promotes_light_for_meaningful_probe_gain() -> None:
    decision = select_sensitivity_policy(
        {
            "arabic-raw-v1": {"mrr_at_10": 0.50, "ndcg_at_10": 0.50},
            "arabic-light-v1": {"mrr_at_10": 0.60, "ndcg_at_10": 0.58},
            "arabic-aggressive-v1": {"mrr_at_10": 0.59, "ndcg_at_10": 0.57},
        },
        {
            "arabic-raw-v1__vs__arabic-light-v1__mrr_at_10": {
                "estimate": -0.10,
                "lower": -0.15,
                "upper": -0.05,
            },
            "arabic-light-v1__vs__arabic-aggressive-v1__mrr_at_10": {
                "estimate": 0.01,
                "lower": -0.02,
                "upper": 0.04,
            },
        },
        {"arabic-raw-v1": {}, "arabic-light-v1": {}, "arabic-aggressive-v1": {}},
    )
    assert decision["selected_policy_id"] == "arabic-light-v1"
