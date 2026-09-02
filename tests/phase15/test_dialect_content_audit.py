from __future__ import annotations

import json
from pathlib import Path

import pytest

from kawaneen.phase15.content_audit import (
    audit_dialect_content,
    validate_dialect_content_audit,
    write_dialect_content_audit,
)


def _variant(variant_id: str, dialect: str, text: str) -> dict[str, object]:
    return {
        "variant_id": variant_id,
        "base_intent_id": f"base-{variant_id}",
        "dialect": dialect,
        "text": text,
        "article_identifiers": [],
        "date_identifiers": [],
        "number_identifiers": [],
        "legal_intent_fingerprint": "intent",
        "qrel_fingerprint": "qrel",
    }


def test_content_audit_rejects_multiline_concatenation_without_scores() -> None:
    variants = [_variant("gulf-1", "gulf_saudi", "السؤال الأول؟\nوش السؤال الثاني؟")]
    result = audit_dialect_content(variants)
    assert result["valid_count"] == 0
    assert result["invalid_reasons"] == {"multiple_paraphrases_or_concatenation": 1}


def test_content_audit_tracks_duplicate_text_and_text_free_aggregate(tmp_path: Path) -> None:
    variants = [
        _variant("egyptian-1", "egyptian", "إيه الحكم؟"),
        _variant("gulf-1", "gulf_saudi", "إيه الحكم؟"),
    ]
    result = audit_dialect_content(variants)
    assert result["valid_count"] == 1
    assert result["invalid_count"] == 1
    _, aggregate_path = write_dialect_content_audit(variants, root=tmp_path)
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    assert "إيه الحكم" not in json.dumps(aggregate, ensure_ascii=False)
    assert aggregate["total_count"] == 2


def test_content_audit_validator_requires_sixty_private_records(tmp_path: Path) -> None:
    variants = [_variant(f"egyptian-{i}", "egyptian", f"إيه الحكم {i}؟") for i in range(60)]
    private_path, aggregate_path = write_dialect_content_audit(variants, root=tmp_path)
    assert validate_dialect_content_audit(private_path, aggregate_path)["valid_count"] == 60
    payload = json.loads(private_path.read_text(encoding="utf-8"))
    payload["variants"] = payload["variants"][:-1]
    private_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly 60"):
        validate_dialect_content_audit(private_path, aggregate_path)
