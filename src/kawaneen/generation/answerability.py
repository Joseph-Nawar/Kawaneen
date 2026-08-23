"""Deterministic Stage-D answerability and source-eligibility policy."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from pydantic import BaseModel, ConfigDict, Field

from kawaneen.generation.contracts import AbstentionReason
from kawaneen.generation.policy import (
    PolicyContext,
    PolicyOutcome,
    evaluate_pre_generation_policy,
)
from kawaneen.grounding.contracts import ContextPack

ANSWERABILITY_POLICY_VERSION = "phase10-stage-d-answerability-policy-v1"
SOURCE_REGISTRY_PATH = Path("data/manifests/source_registry.csv")
CANONICAL_UNITS_PATH = Path(
    "artifacts/private/phase6_evaluation/ai-reviewed-v1/corpus/canonical_units.json"
)


class SourceEligibility(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str = Field(min_length=1)
    source_type: str = "unknown"
    source_role: str = "unknown"
    authority_level: str = "unknown"
    decision: str = "unknown"
    scope_terms: tuple[str, ...] = ()

    @property
    def is_primary_legal_source(self) -> bool:
        role = self.source_role.casefold()
        authority = self.authority_level.casefold()
        decision = self.decision.casefold()
        source_type = self.source_type.casefold()
        return (
            "primary" in role
            and authority == "official"
            and decision in {"approved", "confirmed"}
            and any(term in source_type for term in ("statute", "regulation", "official"))
        )

    @property
    def normalized_scope_terms(self) -> frozenset[str]:
        return frozenset(_tokens(" ".join(self.scope_terms)))


def load_source_eligibility_registry(
    path: Path = SOURCE_REGISTRY_PATH,
) -> dict[str, SourceEligibility]:
    """Load governed source roles/types; unknown rows never become eligible."""

    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = csv.DictReader(handle)
            result: dict[str, SourceEligibility] = {}
            for row in rows:
                source_id = (row.get("source_id") or "").strip()
                if not source_id:
                    continue
                scope_terms = tuple(
                    value
                    for value in (
                        row.get("task"),
                        row.get("description"),
                        row.get("source_type"),
                        row.get("source_role"),
                    )
                    if isinstance(value, str) and value.strip()
                )
                result[source_id] = SourceEligibility(
                    source_id=source_id,
                    source_type=(row.get("source_type") or "unknown").strip(),
                    source_role=(row.get("source_role") or "unknown").strip(),
                    authority_level=(row.get("authority_level") or "unknown").strip(),
                    decision=(row.get("decision") or "unknown").strip(),
                    scope_terms=scope_terms,
                )
            return result
    except (OSError, csv.Error) as error:
        raise ValueError("governed source eligibility registry is unavailable") from error


def answerability_policy_hash(
    source_registry_path: Path = SOURCE_REGISTRY_PATH,
    structural_metadata_path: Path = CANONICAL_UNITS_PATH,
) -> str:
    try:
        registry_hash = hashlib.sha256(source_registry_path.read_bytes()).hexdigest()
        structural_hash = hashlib.sha256(structural_metadata_path.read_bytes()).hexdigest()
    except OSError as error:
        raise ValueError("governed answerability metadata is unavailable") from error
    payload = {
        "version": ANSWERABILITY_POLICY_VERSION,
        "source_registry_sha256": registry_hash,
        "structural_metadata_sha256": structural_hash,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load_structural_roles(
    path: Path = CANONICAL_UNITS_PATH,
) -> dict[str, str]:
    """Load only canonical unit structural roles, never evaluation labels."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("canonical structural metadata is unavailable") from error
    if not isinstance(value, dict):
        raise ValueError("canonical structural metadata is invalid")
    value_mapping = cast(dict[str, object], value)
    if not isinstance(value_mapping.get("units"), list):
        raise ValueError("canonical structural metadata is invalid")
    result: dict[str, str] = {}
    for raw in cast(list[object], value_mapping["units"]):
        if not isinstance(raw, dict):
            continue
        raw_mapping = cast(dict[str, object], raw)
        unit_id = raw_mapping.get("unit_id")
        unit_type = raw_mapping.get("unit_type")
        if isinstance(unit_id, str) and isinstance(unit_type, str):
            result[unit_id] = unit_type
    return result


def evaluate_stage_d_policy(
    query: str,
    context: PolicyContext,
    *,
    source_registry: Mapping[str, SourceEligibility] | None = None,
    structural_roles: Mapping[str, str] | None = None,
    case_specific_evidence_available: bool = False,
) -> PolicyOutcome:
    """Apply the qrel-free Stage-D eligibility gate after context assembly."""

    base = evaluate_pre_generation_policy(query, context)
    source_entries = _source_entries(context.context_pack, source_registry)

    if _is_future_law_request(query):
        return _refused(
            AbstentionReason.FUTURE_LAW_UNKNOWABLE,
            "future legal enactments or amendments cannot be established by present evidence",
        )

    official_request = _is_authoritative_text_request(query)
    if official_request and not any(entry.is_primary_legal_source for entry in source_entries):
        return _refused(
            AbstentionReason.AUTHORITATIVE_SOURCE_UNAVAILABLE,
            "the context has no governed eligible primary legal-text source",
        )

    if _is_temporal_boundary_request(query) and not context.source_status_available:
        return _refused(
            AbstentionReason.CURRENTNESS_UNVERIFIED,
            "the context cannot establish legal status across the requested time boundary",
        )

    if _is_unspecified_case_fact_request(query) and not (
        case_specific_evidence_available or _identifies_case(query)
    ):
        return _refused(
            AbstentionReason.CASE_FACTS_NOT_ESTABLISHED,
            "precedent material cannot establish facts of an unspecified matter",
        )

    if _is_dispositive_request(query):
        if _MISSING_REQUIRED_SECTION.search(query):
            return _refused(
                AbstentionReason.REQUIRED_CASE_SECTION_MISSING,
                "the query identifies the required operative section as unavailable",
            )
        roles = structural_roles or {}
        available_roles = {
            roles.get(evidence.unit_id, "unknown").casefold()
            for evidence in context.context_pack.evidence
        }
        if not available_roles & {"ruling", "verdict", "disposition", "operative", "holding"}:
            return _refused(
                AbstentionReason.REQUIRED_CASE_SECTION_MISSING,
                "the context does not contain an operative holding or disposition section",
            )

    if _is_explicit_forum_request(query) and not _scope_matches(query, source_entries):
        return _refused(
            AbstentionReason.FORUM_OR_SOURCE_SCOPE_MISMATCH,
            "authoritative source scope for the requested forum or proceeding is unavailable",
        )

    if not base.allowed:
        return base
    return base


_LEGAL_TERM = r"(?:law|rule|amendment|regulation|statute|text|نظام|قانون|لائحة|تعديل|مادة|قاعدة)"
_FUTURE = re.compile(
    rf"(?:future\s+{_LEGAL_TERM}|{_LEGAL_TERM}\s+.*(?:will\s+be|is\s+going\s+to\s+be)\s+(?:issued|enacted|applied)|"
    rf"what\s+{_LEGAL_TERM}.*\bwill\s+be\s+(?:issued|enacted|applied)|"
    rf"if\s+(?:a\s+)?(?:future\s+)?(?:{_LEGAL_TERM})\s+(?:is|were)\s+enacted|"
    rf"(?:سيصدر|ستصدر|سيطبق|مستقبلاً|في\s+المستقبل|لاحق)\D{{0,30}}{_LEGAL_TERM}|"
    rf"{_LEGAL_TERM}\D{{0,30}}(?:سيصدر|ستصدر|سيطبق|مستقبلاً|في\s+المستقبل|لاحق))",
    re.IGNORECASE,
)
_TEMPORAL = re.compile(
    r"(?:changed|change|updated|amended|superseded|remained\s+current|current\s+since|"
    r"after\s+the\s+judgment|since\s+the\s+judgment|"
    r"تغير|تعدلت|تم\s+تعديل|استبدل|نسخ|بعد\s+تاريخ|منذ\s+تاريخ|بعد\s+الحكم|"
    r"ساري|نافذ|حالي|الأحدث|آخر\s+نسخة)",
    re.IGNORECASE,
)
_AUTHORITATIVE_TEXT = re.compile(
    r"(?:official|authoritative|statutory\s+text|current\s+regulation|updated\s+provision|"
    r"النص\s+الرسمي|النص\s+النظامي|نص\s+اللائحة|المادة\s+المحدثة|النظام\s+الحالي)",
    re.IGNORECASE,
)
_CASE_FACTS = re.compile(
    r"(?:\b(?:was|were|did|has)\b.{0,60}\b(?:established|made|given|paid|served|occurred|"
    r"act|good\s+faith)\b|هل\s+(?:ثبت|كان|تم|وقع|دفع|سلم|أعطي|أعطى|قدم|بلغ|أخطر))",
    re.IGNORECASE,
)
_DISPOSITIVE = re.compile(
    r"(?:dispositive|operative|final\s+(?:outcome|ruling|order)|relief\s+(?:granted|denied)|"
    r"what\s+did\s+the\s+court\s+order|منطوق|النتيجة\s+التي\s+تلزم|ماذا\s+قضت|"
    r"الحكم\s+النهائي|القرار\s+النهائي)",
    re.IGNORECASE,
)
_MISSING_REQUIRED_SECTION = re.compile(
    r"(?:missing|absent|unavailable|not\s+(?:provided|available|included)|"
    r"غاب|مفقود|غير\s+(?:متاح|موجود|مرفق)|لا\s+يوجد)",
    re.IGNORECASE,
)
_FORUM = re.compile(
    r"(?:labor\s+court|employment\s+court|commercial\s+court|administrative\s+court|"
    r"forum|proceeding|محكمة\s+عمالية|محكمة\s+تجارية|محكمة\s+إدارية|اختصاص\s+المحكمة)",
    re.IGNORECASE,
)
_IDENTIFIED_CASE = re.compile(
    r"(?:judgment\s+[A-Za-z0-9/-]+|case\s+(?:no\.?\s*)?[A-Za-z0-9/-]+|"
    r"decision\s+[A-Za-z0-9/-]+|في\s+(?:الحكم|القرار|القضية)\s*\d*)",
    re.IGNORECASE,
)


def _is_future_law_request(query: str) -> bool:
    return bool(_FUTURE.search(query))


def _is_temporal_boundary_request(query: str) -> bool:
    return bool(_TEMPORAL.search(query))


def _is_authoritative_text_request(query: str) -> bool:
    return bool(_AUTHORITATIVE_TEXT.search(query))


def _is_unspecified_case_fact_request(query: str) -> bool:
    return bool(_CASE_FACTS.search(query))


def _identifies_case(query: str) -> bool:
    return bool(_IDENTIFIED_CASE.search(query))


def _is_dispositive_request(query: str) -> bool:
    return bool(_DISPOSITIVE.search(query))


def _is_explicit_forum_request(query: str) -> bool:
    return bool(_FORUM.search(query))


def _source_entries(
    pack: ContextPack,
    source_registry: Mapping[str, SourceEligibility] | None,
) -> tuple[SourceEligibility, ...]:
    registry = source_registry or load_source_eligibility_registry()
    source_ids = {
        evidence.source.source_id
        for evidence in pack.evidence
        if evidence.source.source_id
    }
    return tuple(
        registry.get(source_id, SourceEligibility(source_id=source_id))
        for source_id in source_ids
    )


def _scope_matches(query: str, entries: tuple[SourceEligibility, ...]) -> bool:
    requested = frozenset(_tokens(query))
    return any(requested & entry.normalized_scope_terms for entry in entries)


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+|[\u0600-\u06ff]+", value.casefold()))


def _refused(reason: AbstentionReason, detail: str) -> PolicyOutcome:
    return PolicyOutcome(allowed=False, reason=reason, detail=detail)
