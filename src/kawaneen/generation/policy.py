"""Deterministic jurisdiction and legal-safety gates before generation."""

from __future__ import annotations

import hashlib
import json
import re
from enum import StrEnum
from pathlib import Path
from typing import cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kawaneen.generation.contracts import AbstentionReason
from kawaneen.grounding.contracts import ContextPack


class SourceStatus(StrEnum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    UNKNOWN = "unknown"


JURISDICTION_SCOPE_PATH = Path("data/manifests/generation/phase10_jurisdiction_scope.json")


class JurisdictionScope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    active_jurisdiction: str | None = None
    authoritative_jurisdiction: str | None = None
    allowed_jurisdictions: tuple[str, ...] = ()
    mode: str = "unverified"
    required: bool = False

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_authority(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        payload = dict(cast(dict[str, object], value))
        active = payload.get("active_jurisdiction") or payload.get("authoritative_jurisdiction")
        if active is not None:
            payload.setdefault("active_jurisdiction", active)
            payload.setdefault("authoritative_jurisdiction", active)
            payload.setdefault("mode", "single")
        return payload

    @model_validator(mode="after")
    def validate_scope(self) -> JurisdictionScope:
        if (
            self.active_jurisdiction
            and self.authoritative_jurisdiction
            and self.active_jurisdiction != self.authoritative_jurisdiction
        ):
            raise ValueError("active and authoritative jurisdictions must match")
        if (
            self.active_jurisdiction
            and self.allowed_jurisdictions
            and self.active_jurisdiction not in self.allowed_jurisdictions
        ):
            raise ValueError("active jurisdiction must be allowed")
        if (
            self.authoritative_jurisdiction
            and self.allowed_jurisdictions
            and self.authoritative_jurisdiction not in self.allowed_jurisdictions
        ):
            raise ValueError("authoritative jurisdiction must be allowed")
        return self


def default_deployment_scope(
    path: Path = JURISDICTION_SCOPE_PATH,
) -> JurisdictionScope:
    """Load and verify the governed deployment jurisdiction contract."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("governed jurisdiction scope is unavailable") from error
    if not isinstance(payload, dict):
        raise ValueError("governed jurisdiction scope is not an object")
    value = cast(dict[str, object], payload)
    if value.get("status") != "active":
        raise ValueError("governed jurisdiction scope is not active")
    source = value.get("authoritative_source")
    if not isinstance(source, dict):
        raise ValueError("governed jurisdiction scope has no authoritative source")
    source_value = cast(dict[str, object], source)
    registry_path = source_value.get("registry_path")
    registry_sha = source_value.get("registry_sha256")
    if not isinstance(registry_path, str) or not isinstance(registry_sha, str):
        raise ValueError("governed jurisdiction scope has no registry lock")
    try:
        actual_sha = hashlib.sha256(Path(registry_path).read_bytes()).hexdigest()
    except OSError as error:
        raise ValueError("governed jurisdiction source registry is unavailable") from error
    if actual_sha != registry_sha:
        raise ValueError("governed jurisdiction source registry hash mismatch")
    return JurisdictionScope.model_validate(
        {
            "active_jurisdiction": value.get("active_jurisdiction"),
            "allowed_jurisdictions": value.get("allowed_jurisdictions", ()),
            "mode": value.get("mode", "unverified"),
            "required": True,
        }
    )


class PolicyContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    context_pack: ContextPack
    scope: JurisdictionScope = Field(default_factory=default_deployment_scope)
    source_status: SourceStatus = SourceStatus.UNKNOWN
    source_status_available: bool = False
    conflicting_evidence: bool = False
    context_jurisdictions: tuple[str, ...] = ()
    retrieval_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    minimum_retrieval_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class PolicyOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    allowed: bool
    reason: AbstentionReason | None = None
    jurisdiction_text: str | None = None
    jurisdiction_verified: bool = False
    query_for_generation: str | None = None
    detail: str | None = None


_ENGLISH_ADVICE = re.compile(
    r"\b(?:should\s+i|should\s+we|do\s+i\s+need\s+to|can\s+i\s+sue|"
    r"should\s+i\s+(?:sign|pay|sue)|may\s+i\s+(?:sign|pay))\b",
    re.IGNORECASE,
)
_ARABIC_ADVICE = re.compile(r"هل\s+(?:يجب|ينبغي)|هل\s+أ(?:وقع|دفع|قاضي|رفع\s+دعوى)")
_CURRENTNESS = re.compile(
    r"\b(?:current|currently|latest|in\s+force|valid\s+today|up[- ]to[- ]date)\b"
    r"|(?:ساري|نافذ|حالي|الأحدث|آخر\s+نسخة|نافذ المفعول)",
    re.IGNORECASE,
)
_JURISDICTIONS = {
    "saudi arabia": "SA",
    "saudi": "SA",
    "السعودية": "SA",
    "المملكة العربية السعودية": "SA",
    "egypt": "EG",
    "egyptian": "EG",
    "مصر": "EG",
    "مصري": "EG",
    "uae": "AE",
    "united arab emirates": "AE",
    "الإمارات": "AE",
}


def evaluate_pre_generation_policy(query: str, context: PolicyContext) -> PolicyOutcome:
    if not query.strip():
        return _refused(AbstentionReason.REQUESTED_INFO_NOT_FOUND, "query is blank")
    if _ENGLISH_ADVICE.search(query) or _ARABIC_ADVICE.search(query):
        return _refused(
            AbstentionReason.PERSONALIZED_LEGAL_ADVICE,
            "personalized legal advice requests are refused before generation",
        )

    requested = _requested_jurisdictions(query)
    context_jurisdictions = _canonicalize_jurisdictions(context.context_jurisdictions)
    if len(context_jurisdictions) > 1:
        return _refused(
            AbstentionReason.JURISDICTION_AMBIGUOUS,
            "context contains conflicting jurisdictions",
        )
    jurisdiction_text, jurisdiction_verified, jurisdiction_reason = _check_jurisdiction(
        requested,
        context_jurisdictions,
        context.scope,
    )
    if jurisdiction_reason is not None:
        return _refused(jurisdiction_reason, "requested jurisdiction is outside the server scope")
    if context.conflicting_evidence:
        return _refused(AbstentionReason.CONFLICTING_EVIDENCE, "evidence status is conflicting")
    if context.source_status is SourceStatus.SUPERSEDED:
        return _refused(AbstentionReason.SUPERSEDED_SOURCE, "source is superseded")
    if _CURRENTNESS.search(query) and not context.source_status_available:
        return _refused(
            AbstentionReason.CURRENTNESS_UNVERIFIED,
            "source status is unavailable for a currentness question",
        )
    if not context.context_pack.evidence:
        return _refused(AbstentionReason.NO_CONTEXT, "ContextPack contains no evidence")
    if (
        context.retrieval_confidence is not None
        and context.retrieval_confidence < context.minimum_retrieval_confidence
    ):
        return _refused(
            AbstentionReason.LOW_RETRIEVAL_CONFIDENCE,
            "retrieval confidence is below policy",
        )
    return PolicyOutcome(
        allowed=True,
        jurisdiction_text=jurisdiction_text,
        jurisdiction_verified=jurisdiction_verified,
        query_for_generation=query,
    )


def _requested_jurisdictions(query: str) -> frozenset[str]:
    lowered = query.casefold()
    return frozenset(
        jurisdiction
        for marker, jurisdiction in _JURISDICTIONS.items()
        if marker.casefold() in lowered
    )


def _canonicalize_jurisdictions(values: tuple[str, ...]) -> frozenset[str]:
    aliases = {value.casefold(): value for value in _JURISDICTIONS.values()}
    return frozenset(aliases.get(value.casefold(), value) for value in values)


def _check_jurisdiction(
    requested: frozenset[str],
    context_jurisdictions: frozenset[str],
    scope: JurisdictionScope,
) -> tuple[str | None, bool, AbstentionReason | None]:
    if len(requested) > 1:
        return None, False, AbstentionReason.JURISDICTION_AMBIGUOUS
    active = scope.active_jurisdiction or scope.authoritative_jurisdiction
    allowed = frozenset(scope.allowed_jurisdictions)
    if scope.required and active is None:
        return None, False, AbstentionReason.JURISDICTION_AMBIGUOUS
    if context_jurisdictions and active and context_jurisdictions != frozenset({active}):
        return active, True, AbstentionReason.JURISDICTION_MISMATCH
    if context_jurisdictions and not active:
        return None, False, AbstentionReason.JURISDICTION_MISMATCH
    if requested:
        requested_value = next(iter(requested))
        if active and requested_value != active:
            return active, True, AbstentionReason.JURISDICTION_MISMATCH
        if allowed and requested_value not in allowed:
            return active, bool(active), AbstentionReason.JURISDICTION_MISMATCH
        if not active:
            return None, False, AbstentionReason.JURISDICTION_MISMATCH
        return active or requested_value, bool(active), None
    if active:
        return active, True, None
    return None, False, None


def _refused(reason: AbstentionReason, detail: str) -> PolicyOutcome:
    return PolicyOutcome(allowed=False, reason=reason, detail=detail)
