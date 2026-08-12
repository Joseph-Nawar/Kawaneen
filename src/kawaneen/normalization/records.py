"""Private derived normalization records with immutable-corpus checks."""

from __future__ import annotations

import hashlib

from pydantic import BaseModel, ConfigDict, Field

from kawaneen.corpus.models import CanonicalUnit, SourceProvenance, UnitType
from kawaneen.normalization.models import NormalizationPolicy, NormalizationResult
from kawaneen.normalization.policies import get_policy, normalize_text


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class NormalizedRecord(BaseModel):
    """A private derived view row; canonical text is never overwritten."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    unit_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    unit_type: UnitType
    display_text: str
    search_text: str
    policy_id: str = Field(min_length=1)
    policy_hash: str = Field(min_length=64, max_length=64)
    source_text_sha256: str = Field(min_length=64, max_length=64)
    search_text_sha256: str = Field(min_length=64, max_length=64)
    provenance: SourceProvenance
    ordinal: int | None = Field(default=None, ge=1)
    transform_counts: dict[str, int] = Field(default_factory=dict)

    @classmethod
    def from_canonical(cls, unit: CanonicalUnit, policy: NormalizationPolicy) -> NormalizedRecord:
        result = normalize_text(unit.text, policy, audit=True)
        if not isinstance(result, NormalizationResult):
            raise TypeError("audit normalization must return NormalizationResult")
        return cls(
            unit_id=unit.unit_id,
            document_id=unit.document_id,
            unit_type=unit.unit_type,
            display_text=unit.text,
            search_text=result.search_text,
            policy_id=policy.policy_id,
            policy_hash=policy.policy_hash,
            source_text_sha256=_sha256(unit.text),
            search_text_sha256=_sha256(result.search_text),
            provenance=unit.provenance,
            ordinal=unit.ordinal,
            transform_counts=dict(result.transform_counts),
        )


def validate_record_contract(record: NormalizedRecord, source_text: str) -> None:
    """Raise if a derived record no longer matches its canonical source."""

    if record.display_text != source_text:
        raise ValueError("display_text does not preserve canonical source text")
    if record.source_text_sha256 != _sha256(source_text):
        raise ValueError("source_text_sha256 does not match display_text")
    if record.search_text_sha256 != _sha256(record.search_text):
        raise ValueError("search_text_sha256 does not match search_text")
    policy = get_policy(record.policy_id)
    if record.policy_hash != policy.policy_hash:
        raise ValueError("policy_hash does not match policy_id")
    if normalize_text(source_text, policy) != record.search_text:
        raise ValueError("search_text does not match policy normalization")
