from __future__ import annotations

import json
from pathlib import Path

from kawaneen.chunking.challenge import (
    build_private_chunk_challenge,
    validate_challenge_independence,
)
from kawaneen.chunking.corpus import freeze_phase5_documents
from kawaneen.corpus.models import CanonicalUnit, SourceProvenance, UnitType


def _units() -> tuple[CanonicalUnit, ...]:
    return tuple(
        CanonicalUnit(
            unit_id=f"unit-{index}",
            document_id=f"doc-{index // 4}",
            unit_type=(UnitType.FACTS if index % 4 == 0 else UnitType.VERDICT),
            text=(" ".join(f"قانوني{index}-{word}" for word in range(80)) + "\n\nفقرة ثانية"),
            provenance=SourceProvenance(
                source_id="alarb" if index < 8 else "arabiccr",
                source_version="v1",
                source_path="fixture",
                source_row=index + 1,
                source_field="facts",
            ),
            ordinal=index + 1,
        )
        for index in range(16)
    )


def test_private_chunk_challenge_is_balanced_and_policy_independent(tmp_path: Path) -> None:
    units = _units()
    corpus = freeze_phase5_documents(units, per_source=2)
    challenge = build_private_chunk_challenge(corpus.units, corpus, seed=7, output_root=tmp_path)
    assert len(challenge.items) == 180
    assert {item.slice_name for item in challenge.items} == {
        "local_passage",
        "long_legal_section",
        "multi_paragraph_evidence",
        "structural_boundary_proximity",
        "fixed_window_boundary_stress",
        "parent_context_evidence",
    }
    assert all(item.gold_spans for item in challenge.items)
    assert validate_challenge_independence(challenge)
    assert json.loads((tmp_path / "qrels.json").read_text())
    assert challenge == build_private_chunk_challenge(
        corpus.units, corpus, seed=7, output_root=tmp_path / "repeat"
    )
