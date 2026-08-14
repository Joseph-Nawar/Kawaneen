"""Bounded application of the external v3 source review.

This module intentionally does not call a retriever.  It transforms the v3
records using only their canonical evidence, the already-generated evidence
qualified pool, and deterministic semantic rules.
"""

# Arabic numerals and punctuation are intentional in source-facing patterns.
# ruff: noqa: RUF001

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from kawaneen.chunking.models import CitationAnchor
from kawaneen.corpus.models import CanonicalUnit
from kawaneen.evaluation.candidates import (
    proportional_source_order,
    semantic_segments,
    valid_semantic_text,
)
from kawaneen.evaluation.corpus import EvaluationCorpus
from kawaneen.evaluation.models import (
    Answerability,
    CreationMethod,
    DatasetItem,
    Difficulty,
    EvidenceGroup,
    EvidenceSpan,
    QueryCategory,
    QueryLanguage,
    QueryRegister,
    QueryType,
    RelevanceGrade,
    ReviewState,
    SemanticTarget,
    deterministic_intent_id,
    deterministic_query_id,
)
from kawaneen.evaluation.semantic_targets import (
    render_semantic_answer,
    render_semantic_query,
)

V4_VERSION = "phase6-retrieval-eval-draft-v4"
V4_PRIVATE_ROOT = Path("artifacts/private/phase6_evaluation/draft-v4")

_EXPECTED_DECISIONS = {"accept": 25, "correct": 118, "replace": 57, "regenerate_variant": 40}
_CATEGORIES = (
    QueryCategory.EXACT_PROVISION,
    QueryCategory.DEFINITION,
    QueryCategory.DEADLINE,
    QueryCategory.AUTHORITY,
    QueryCategory.CONDITIONS,
    QueryCategory.MULTI_EVIDENCE,
    QueryCategory.CASE_HOLDING,
)
_REPLACE_COUNTS = {
    QueryCategory.DEFINITION: 10,
    QueryCategory.DEADLINE: 9,
    QueryCategory.AUTHORITY: 9,
    QueryCategory.CONDITIONS: 4,
    QueryCategory.MULTI_EVIDENCE: 25,
}
_QUERY_TYPES = {
    QueryCategory.EXACT_PROVISION: QueryType.REFERENCE_LOOKUP,
    QueryCategory.DEFINITION: QueryType.LEGAL_CONCEPT,
    QueryCategory.DEADLINE: QueryType.PROCEDURE,
    QueryCategory.AUTHORITY: QueryType.RESPONSIBILITY,
    QueryCategory.CONDITIONS: QueryType.CONDITIONS_EXCEPTIONS,
    QueryCategory.MULTI_EVIDENCE: QueryType.REASONING,
    QueryCategory.CASE_HOLDING: QueryType.HOLDING_OUTCOME_REMEDY,
}
_DISPOSITION = re.compile(
    r"(?:قضت|حكمت|ألزمت|إلزام|رفض(?:ت)?|قبول|أثبتت|إثبات|عدم قبول|شطب|نقض|تأييد)"
)
_PII = re.compile(
    r"(?:[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b|"
    r"(?<!\d)(?:\+?\d[\d ()-]{7,}\d)(?!\d)|\[Person Name\])"
)
_INTERNAL = re.compile(
    r"\[(?:intent|query|gold|evidence|chunk)[^]]*\]|internal reference|مرجع داخلي|"
    r"benchmark|corpus|retrieval|qrel|chunk policy",
    re.IGNORECASE,
)
_TIME_DURATION = re.compile(
    r"(?:خلال|لمدة|مهلة|في موعد أقصاه|قبل\s+(?:انقضاء|مضي)|على\s+مدار)\s+[^،؛.]{1,70}|"
    r"على\s+[^،؛.]{1,35}(?:يوم|أيام|شهر|أشهر|سنة|سنوات|عام|أعوام)|"
    r"على\s+(?:دفعة|دفعتين|دفعات|أقساط)[^،؛.]{1,100}|"
    r"(?:\d|[٠-٩])+(?:\s*[-/]\s*(?:\d|[٠-٩])+)?\s*(?:يوم|أيام|شهر|أشهر|سنة|سنوات|عام|أعوام)"
)
_REFERENCE = re.compile(
    r"(?:(?:المادة|المواد|الفقرة|البند)\s*(?:رقم\s*)?\(?\s*[0-9٠-٩]+(?:/[0-9٠-٩]+)?\s*\)?|"
    r"(?:نظام|اللائحة|قرار|المرسوم)[^:؛,.]{0,80}:\s*[0-9٠-٩]+)"
)
_ACTOR = re.compile(
    r"(?:المحكمة|الدائرة|الجهة المختصة|الإدارة المختصة|الموظف|الكاتب|رئيس الجلسة|"
    r"المدعي|المدعى عليه|الطرف أو الأطراف)"
)
_POWER = re.compile(
    r"(?:غير مخول(?:ين)?|مخول(?:ين)?|تختص|يختص|يلتزم|يجب|يتعين|يحق|صلاحية|مسؤول(?:ية)?|"
    r"أوجب|ألزم|من اختصاص|يملك|يحظر|لا يجوز)"
)
_CONDITION = re.compile(r"(?:إذا|فإذا|يشترط|بشرط|ما لم|لا يجوز|يجوز|متى|حال كان|فلا)\b")
_ACTION = re.compile(
    r"(?:السداد الكامل|سداد|تقديم|إيداع|حضور|رفع|تنفيذ|تسليم|الرد|إتمام|بدء|انتهاء|إثبات|التوقيع)"
)


@dataclass(frozen=True, slots=True)
class ExternalReviewDecision:
    query_id: str
    intent_id: str
    category: str
    source: str
    creation_method: str
    variant_id: str | None
    decision: str
    review_note: str


@dataclass(frozen=True, slots=True)
class V4BuildResult:
    base_pool: tuple[DatasetItem, ...]
    bases: tuple[DatasetItem, ...]
    variants: tuple[DatasetItem, ...]
    mapping: tuple[dict[str, object], ...]
    replacement_reasons: tuple[dict[str, object], ...]
    multi_source_audit: Mapping[str, object]


def load_v3_adjudication(path: Path) -> tuple[ExternalReviewDecision, ...]:
    rows: list[ExternalReviewDecision] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        decision = str(
            record.get("primary_ai_decision")
            or record.get("adjudicated_ai_decision")
            or record.get("decision")
        )
        rows.append(
            ExternalReviewDecision(
                query_id=str(record["query_id"]),
                intent_id=str(record["intent_id"]),
                category=str(record["category"]),
                source=str(record.get("source", "")),
                creation_method=str(record.get("creation_method", "")),
                variant_id=record.get("variant_id"),
                decision=decision,
                review_note=str(record.get("review_note", "")),
            )
        )
    if len({row.query_id for row in rows}) != len(rows):
        raise ValueError("external v3 adjudication contains duplicate query IDs")
    counts = Counter(row.decision for row in rows)
    if counts != Counter(_EXPECTED_DECISIONS):
        raise ValueError(f"unexpected v3 adjudication counts: {dict(counts)}")
    if any(row.decision not in _EXPECTED_DECISIONS for row in rows):
        raise ValueError("external v3 adjudication contains an unsupported disposition")
    return tuple(rows)


def _norm(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\wء-ي]+", " ", value.casefold())).strip()


def duplicate_query_keys(rows: Iterable[Mapping[str, str]]) -> dict[str, tuple[str, ...]]:
    grouped: defaultdict[str, list[str]] = defaultdict(list)
    for row in rows:
        grouped[_norm(str(row["query_text"]))].append(str(row["query_id"]))
    return {key: tuple(ids) for key, ids in grouped.items() if len(ids) > 1}


def validate_v4_query_text(text: str, language: str) -> bool:
    value = text.strip()
    if not value or _INTERNAL.search(value) or _PII.search(value):
        return False
    if language == QueryLanguage.ENGLISH.value:
        if not re.search(r"\b(?:what|how|which|why|did|does|was|were|could|would)\b", value, re.I):
            return False
        if re.search(r"[ء-ي]", value):
            return False
    return True


def reject_v4_semantic_fragment(text: str) -> bool:
    value = text.strip()
    if re.match(r"^\s*(?:\d+|[٠-٩]+)\s*[.)،:-]", value):
        return True
    stripped = re.sub(r"^\s*(?:\d+|[٠-٩]+)\s*[.)،:-]?\s*", "", value)
    if len(re.findall(r"[ء-ي]{2,}", stripped)) < 2:
        return True
    if stripped.endswith(("؟", "?")) or "صالحة للفصل" in stripped:
        return True
    return bool(
        re.match(r"^(?:بتاريخ|في تاريخ)\s+[0-9٠-٩/هـ-]+\s+(?:عقدت|انعقدت)\s+الجلسة", stripped)
    )


def _clean(value: str, limit: int | None = None) -> str:
    value = _PII.sub("", value)
    value = re.sub(r"\s+", " ", value.strip(" \t\r\n'\"[](),؛:.؟"))
    if limit is not None:
        value = " ".join(value.split()[:limit])
    return value.strip(" ،؛:.")


def _tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[\wء-ي]{3,}", value)
        if token not in {"التي", "الذي", "على", "من", "في", "وهو"}
    }


def _sentence(value: str, start: int = 0) -> str:
    tail = value[start:]
    piece = re.split(r"[؛.!؟\n]", tail, maxsplit=1)[0]
    return _clean(piece)


def extract_v4_condition_target(text: str) -> SemanticTarget | None:
    value = _clean(text)
    match = re.search(
        r"(?P<condition>(?:إذا|فإذا|فإن|يشترط|بشرط|ما لم|متى|حال كان|في حال|لا يجوز)[^،؛.]{2,100}?)"
        r"\s*(?:،|,)?\s*(?P<effect>(?:فلا\s+يكون|لا\s+يكون|لا\s+يجوز|فيجوز|يجوز|فيجب|يجب|يكون|يترتب|يثبت|أثبت|أثبته|تثبت|تعد|يعد|عُد|عد|تترتب|تندب|يتعين|يقدم|تراعى)[^،؛.]{2,120})",
        value,
    )
    if match is None:
        match = re.search(
            r"(?P<condition>[^،؛.]{2,100})\s+(?P<effect>فلا\s+يكون[^،؛.]{2,120})", value
        )
    if match is None:
        return None
    condition, effect = _clean(match.group("condition")), _clean(match.group("effect"))
    if len(_tokens(condition)) < 2 or len(_tokens(effect)) < 2:
        return None
    return SemanticTarget(
        category=QueryCategory.CONDITIONS,
        proposition=f"{condition}؛ {effect}",
        condition=condition,
        effect=effect,
        context=_clean(condition, 7),
    )


def _extract_target(
    category: QueryCategory, text: str, existing: SemanticTarget | None
) -> SemanticTarget | None:
    value = _clean(text)
    if reject_v4_semantic_fragment(value):
        return None
    if category is QueryCategory.EXACT_PROVISION:
        references = [match.group(0) for match in _REFERENCE.finditer(value)]
        if not references:
            return None
        identifier = ", ".join(dict.fromkeys(references))
        effect_match = re.search(
            r"(?:يعتبر|يُعتبر|يعد|يُعد|يثبت|تثبت|يختص|ينعقد|يلتزم|يجب|يجوز|لا يجوز|"
            r"يترتب|تنقضي|تنقضى|قضت|حكمت)[^،؛.]{4,130}",
            value,
        )
        effect = _clean(effect_match.group(0) if effect_match else _sentence(value))
        if len(_tokens(effect)) < 2:
            return None
        subject_match = re.search(
            r"(?:يعتبر|يُعتبر|يعد|يثبت|تثبت)\s+([^،؛.]{2,70}?)(?=\s+(?:وتنقضي|وتنقضى|حسب|وفق|بموجب|المادة)|$)",
            value,
        )
        subject = (
            _clean(subject_match.group(1), 10)
            if subject_match
            else _clean(existing.subject if existing else "", 10)
        )
        subject = re.split(r"\s+(?:وت.?نقضى|وت.?نقضي|حسب|وفق|بموجب|المادة)\b", subject)[0].strip()
        if not subject:
            return None
        return SemanticTarget(
            category=category,
            proposition=f"{identifier}: {effect}",
            provision_identifier=identifier,
            subject=subject,
            effect=effect,
            context=subject,
        )
    if category is QueryCategory.DEFINITION:
        if "؟" in value or value.endswith("?") or re.match(r"^\s*(?:\d+|[٠-٩]+)\s*[.)-]", value):
            return None
        match = re.search(
            r"(?:يقصد|المقصود)\s+(?:بـ?|في)\s*(?P<term>[\u0621-\u064A]{2,40})\s+(?P<definition>[^،؛.]{8,120})",
            value,
        )
        if match is not None and _clean(match.group("definition")).startswith(
            ("في هذه الدعوى", "في هذا السياق", "هو")
        ):
            match = None
        if match is None:
            match = re.search(
                r"(?P<term>[\u0621-\u064A][\u0621-\u064A\s]{1,55}?)\s*:\s*"
                r"(?P<definition>[^،؛.]{8,120})",
                value,
            )
        if match is None:
            match = re.search(
                r"(?P<term>[\u0621-\u064A][^،؛.]{2,55}(?:المقصود|المعني)[^،؛.]{0,20})\s+هو\s+(?P<definition>[^،؛.]{4,100})",
                value,
            )
        if match is None:
            cue = re.search(r"المقصود\s+ب(?P<term>[\u0621-\u064A]{2,40})", value)
            if cue is None:
                return None
            term = _clean(cue.group("term"))
            relation = re.search(
                rf"(?:{re.escape(term)}|فسخ|الفسخ)\s+(?P<definition>(?:العقد|الدعوى|المبلغ|الطلب|الشرط)[^،؛.]{{2,80}})",
                value,
            )
            if relation is None:
                return None
            definition = _clean(relation.group("definition"))
            if len(_tokens(definition)) < 2:
                return None
            return SemanticTarget(
                category=category,
                proposition=f"{term}: {definition}",
                defined_term=term,
                definition=definition,
                context=term,
            )
        term, definition = _clean(match.group("term")), _clean(match.group("definition"))
        if len(_tokens(definition)) < 3 or term in {"هل", "ما", "كيف", "السؤال"}:
            return None
        return SemanticTarget(
            category=category,
            proposition=f"{term}: {definition}",
            defined_term=term,
            definition=definition,
            context=term,
        )
    if category is QueryCategory.DEADLINE:
        time = _TIME_DURATION.search(value)
        action = _ACTION.search(value)
        if time is None or action is None or time.group(0).lstrip().startswith("بتاريخ"):
            return None
        deadline = _clean(time.group(0))
        action_text = _clean(value[action.start() : min(len(value), action.end() + 65)])
        action_text = re.split(
            r"(?:،|؛|\.|\s+(?:خلال|لمدة|مهلة|في موعد|على|من|تبدأ))", action_text, maxsplit=1
        )[0]
        trigger_match = re.search(r"(?:بعد|من|عند|ابتداءً من)\s+[^،؛.]{2,55}", value)
        trigger = _clean(trigger_match.group(0)) if trigger_match else ""
        if len(_tokens(action_text)) < 1 or len(_tokens(deadline)) < 2:
            return None
        return SemanticTarget(
            category=category,
            proposition=f"{action_text} {deadline}",
            action=action_text,
            deadline=deadline,
            triggering_event=trigger,
            context=action_text,
        )
    if category is QueryCategory.AUTHORITY:
        if "صالحة للفصل" in value:
            return None
        actor_matches = list(_ACTOR.finditer(value))
        power_match = _POWER.search(value)
        actor_match = (
            max(
                (
                    match
                    for match in actor_matches
                    if power_match and match.end() <= power_match.start()
                ),
                key=lambda match: match.end(),
                default=None,
            )
            if power_match
            else None
        )
        if (
            actor_match is None
            or power_match is None
            or power_match.start() - actor_match.end() > 70
        ):
            reverse = re.search(
                r"(?P<power>من اختصاص|اختصاص|صلاحية|مسؤولية)\s+(?P<actor>المحكمة|الدائرة|"
                r"الجهة المختصة|الإدارة المختصة)",
                value,
            )
            if reverse is None:
                return None
            actor = _clean(reverse.group("actor"))
            power = _clean(reverse.group("power"))
            object_value = _clean(value[reverse.end() :], 12)
        else:
            actor, power = _clean(actor_match.group(0)), _clean(power_match.group(0))
            object_value = _clean(value[power_match.end() :], 12)
        object_value = re.sub(r"^(?:ب|في|ل)\s*", "", object_value)
        if len(_tokens(object_value)) < 1:
            return None
        return SemanticTarget(
            category=category,
            proposition=f"{actor} {power} {object_value}",
            actor=actor,
            power=power,
            object=object_value,
            context=_clean(object_value, 7),
        )
    if category is QueryCategory.CONDITIONS:
        return extract_v4_condition_target(value)
    if category is QueryCategory.CASE_HOLDING:
        if existing is None or not existing.disposition:
            return None
        disposition = _clean(_sentence(value))
        disposition = re.sub(r"\[?Person Name\]?", "الطرف المعني", disposition)
        disposition = re.sub(
            r"شركة\s+[^،؛.]{0,50}(?=مبلغ|مبلغا|بأن|بسداد|بدفع)", "الطرف المعني ", disposition
        )
        object_value = _clean(existing.object or disposition, 18)
        object_value = re.sub(r"\[?Person Name\]?", "الطرف المعني", object_value)
        amount_match = re.search(
            r"[0-9٠-٩]{1,3}(?:,[0-9٠-٩]{3})+(?:\.[0-9٠-٩]+)?\s*(?:ريال|ريالاً|ريالًا)?", value
        )
        amount = amount_match.group(0) if amount_match else existing.amount
        if len(_tokens(disposition)) < 2 or len(_tokens(object_value)) < 1:
            return None
        return SemanticTarget(
            category=category,
            proposition=disposition,
            disposition=disposition,
            object=object_value,
            remedy=disposition,
            amount=amount,
            context="",
        )
    if category is QueryCategory.MULTI_EVIDENCE:
        return None
    return None


def _neutral_issue(units: Iterable[CanonicalUnit], target: SemanticTarget | None) -> str:
    bad = _DISPOSITION
    for unit in sorted(units, key=lambda row: (row.ordinal or 0, row.unit_id)):
        if unit.unit_type.value not in {"facts", "events"}:
            continue
        for sentence in re.split(r"[؛.!؟\n]", unit.text):
            cleaned = _clean(sentence, 14)
            if len(_tokens(cleaned)) < 4 or bad.search(cleaned) or re.search(r"\d{2,}", cleaned):
                continue
            if target is not None and any(
                _norm(cleaned) == _norm(premise) or _norm(cleaned) in _norm(premise)
                for premise in target.premises
            ):
                continue
            return cleaned
    return "الوقائع محل النزاع"


def _make_multi_target(
    selections: tuple[tuple[CanonicalUnit, str], ...], conclusion: str
) -> SemanticTarget | None:
    if len(selections) < 2:
        return None
    premises = tuple(_clean(text, 60) for _unit, text in selections[:2])
    if any(len(_tokens(premise)) < 3 for premise in premises):
        return None
    conclusion = _clean(conclusion, 30)
    if len(_tokens(conclusion)) < 3:
        return None
    # The two premises must each contribute a distinct substantive cue to the
    # conclusion; a single premise that repeats the full answer is not enough.
    conclusion_tokens = _tokens(conclusion)
    if any(len(_tokens(premise) & conclusion_tokens) < 1 for premise in premises):
        return None
    return SemanticTarget(
        category=QueryCategory.MULTI_EVIDENCE,
        proposition=conclusion,
        premises=premises,
        conclusion=conclusion,
        context="نزاع بشأن الوقائع والقواعد المنطبقة",
        disposition="",
        object=_clean(conclusion, 10),
    )


def validate_v4_semantic_contract(
    category: QueryCategory,
    target: SemanticTarget,
    query: str,
    answer: str,
    evidence_texts: tuple[str, ...],
) -> bool:
    if target.category is not category or not validate_v4_query_text(query, "ar"):
        return False
    joined = " ".join(evidence_texts).casefold()
    if any(not value.strip() for value in (target.proposition, answer)) or _INTERNAL.search(query):
        return False
    if category is QueryCategory.CASE_HOLDING:
        if not target.disposition or not target.object or not target.context:
            return False
        if (
            any(term and term in query for term in ("رفض", "إلزام", "قضى", "حكمت", target.amount))
            and "قضى" not in query
            and "حكمت" not in query
        ):
            return False
        if target.disposition.casefold() not in joined and not any(
            part.casefold() in joined for part in target.disposition.split(" و")
        ):
            return False
    elif category is QueryCategory.MULTI_EVIDENCE:
        if len(target.premises) < 2 or not target.conclusion or target.disposition:
            return False
        if target.conclusion.casefold() in query.casefold():
            return False
        if any(premise.casefold() in query.casefold() for premise in target.premises):
            return False
        if len(evidence_texts) < 2 or any(len(_tokens(premise)) < 3 for premise in target.premises):
            return False
        if target.conclusion.casefold() not in joined:
            return False
    else:
        fields = {
            QueryCategory.EXACT_PROVISION: (
                target.provision_identifier,
                target.effect,
                target.subject,
            ),
            QueryCategory.DEFINITION: (target.defined_term, target.definition),
            QueryCategory.DEADLINE: (target.action, target.deadline),
            QueryCategory.AUTHORITY: (target.actor, target.power, target.object),
            QueryCategory.CONDITIONS: (target.condition, target.effect),
        }[category]
        field_checks: list[bool] = []
        for value in fields:
            if not value.strip():
                field_checks.append(False)
            elif category is QueryCategory.EXACT_PROVISION and value == target.provision_identifier:
                numbers = re.findall(r"[0-9٠-٩]+", value)
                field_checks.append(
                    bool(numbers) and all(number.casefold() in joined for number in numbers)
                )
            else:
                field_checks.append(value.casefold() in joined)
        if not all(field_checks):
            return False
        if any(
            value and value.casefold() in query.casefold()
            for value in (target.effect, target.definition, target.deadline)
        ):
            return False
    return not (answer.strip().casefold() == "." or answer.strip().startswith("."))


def _english(value: str) -> str:
    replacements = {
        "المحكمة": "the court",
        "الدائرة": "the circuit",
        "المدعي": "the claimant",
        "المدعى عليه": "the defendant",
        "العقد": "the contract",
        "الدعوى": "the claim",
        "الصلح": "the settlement",
        "التعويض": "compensation",
        "المبلغ": "the amount",
        "التنفيذ": "performance",
        "التوقيع": "the signature",
        "التوريد": "supply",
        "المطالبة": "the claim",
        "نزاع": "dispute",
        "بشأن": "concerning",
        "المادة": "Article",
        "إيداع": "filing",
        "تقديم": "submitting",
    }
    result = value.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
    for arabic, english in sorted(
        replacements.items(), key=lambda pair: len(pair[0]), reverse=True
    ):
        result = result.replace(arabic, english)
    transliteration = str.maketrans(
        {
            "ا": "a",
            "ب": "b",
            "ت": "t",
            "ث": "th",
            "ج": "j",
            "ح": "h",
            "خ": "kh",
            "د": "d",
            "ذ": "dh",
            "ر": "r",
            "ز": "z",
            "س": "s",
            "ش": "sh",
            "ص": "s",
            "ض": "d",
            "ط": "t",
            "ظ": "z",
            "ع": "a",
            "غ": "gh",
            "ف": "f",
            "ق": "q",
            "ك": "k",
            "ل": "l",
            "م": "m",
            "ن": "n",
            "ه": "h",
            "و": "w",
            "ي": "y",
            "ء": "'",
            "ة": "h",
            "ى": "a",
            "أ": "a",
            "إ": "i",
            "آ": "aa",
            "ؤ": "'",
            "ئ": "y",
        }
    )
    result = result.translate(transliteration)
    return re.sub(r"\s+", " ", result).strip()


def _base_v4_query(target: SemanticTarget) -> str:
    """Use deterministic natural wording variants to prevent exact duplicates."""

    index = int(hashlib.sha256(target.proposition.encode()).hexdigest()[:2], 16) % 8
    if target.category is QueryCategory.EXACT_PROVISION:
        prefixes = (
            "ما الأثر الذي رتبه الحكم استناداً إلى",
            "ماذا قرر الحكم بموجب",
            "كيف طبّق الحكم",
            "ما الذي قرره الحكم استناداً إلى",
            "ما أثر الاحتجاج بـ",
            "ما النتيجة التي رتبها الحكم بموجب",
            "ما الذي استخلصه الحكم من",
            "كيف عالج الحكم أثر",
        )
        return f"{prefixes[index]} {target.provision_identifier} بشأن {_clean(target.subject, 7)}؟"
    if target.category is QueryCategory.DEFINITION:
        prefixes = (
            "ما المقصود بـ",
            "كيف عرّف النص",
            "ما معنى",
            "كيف شرح النص",
            "ما المراد بـ",
            "كيف بيّن النص",
            "ما مدلول",
            "كيف فُهم",
        )
        return f"{prefixes[index]}{target.defined_term} في هذا السياق القانوني؟"
    if target.category is QueryCategory.DEADLINE:
        prefixes = (
            "ما الموعد المحدد لـ",
            "ما المهلة اللازمة لـ",
            "خلال أي مدة يجب",
            "متى يجب",
            "ما الأجل المقرر لـ",
            "ما المدة التي حددها النص لـ",
            "ما الفترة اللازمة لـ",
            "إلى متى يجب",
        )
        return f"{prefixes[index]}{_clean(target.action, 8)}؟"
    if target.category is QueryCategory.AUTHORITY:
        prefixes = (
            "ما حدود",
            "ما السلطة التي يملكها",
            "ما المسؤولية التي تقع على",
            "ما الاختصاص المقرر لـ",
            "ما الواجب الذي يقع على",
            "هل يملك",
            "ما نطاق صلاحية",
            "ما الدور النظامي لـ",
        )
        return (
            f"{prefixes[index]} {target.power} لـ{target.actor} بشأن "
            f"{_clean(target.context or target.object, 7)}؟"
        )
    if target.category is QueryCategory.CONDITIONS:
        prefixes = (
            "ما الشرط الذي يترتب عليه",
            "متى يترتب الأثر",
            "ما الحالة التي تحكم",
            "في أي حالة يترتب الأثر",
            "ما القيد المؤثر في",
            "متى يعمل هذا الحكم",
            "ما الاستثناء المتعلق بـ",
            "ما المتطلب السابق لـ",
        )
        return f"{prefixes[index]} {_clean(target.context or target.condition, 8)}؟"
    if target.category is QueryCategory.CASE_HOLDING:
        prefixes = (
            "ما الذي قضى به الحكم بشأن",
            "ماذا قررت المحكمة في شأن",
            "ما منطوق الحكم في النزاع المتعلق بـ",
            "كيف حسمت المحكمة النزاع حول",
            "ما القرار القضائي في موضوع",
            "بماذا انتهى الحكم في شأن",
            "ما الذي فصلت فيه المحكمة بشأن",
            "كيف عالج الحكم النزاع المتعلق بـ",
        )
        neutral_suffixes = (
            "في هذه المسألة",
            "في النزاع المعروض",
            "في الطلب محل البحث",
            "في القضية المعروضة",
            "في الموضوع محل الفصل",
        )
        suffix_index = int(hashlib.sha256(target.proposition.encode()).hexdigest()[2:4], 16) % len(
            neutral_suffixes
        )
        return f"{prefixes[index]} {_clean(target.context, 12)} {neutral_suffixes[suffix_index]}؟"
    prefixes = (
        "كيف اجتمعت الوقائع والأسباب في",
        "كيف دعمت الأدلة مجتمعة",
        "ما الذي خلصت إليه المحكمة بعد الجمع بين الأدلة في",
        "كيف تكاملت المعطيات للفصل في",
        "ما النتيجة المستندة إلى الوقائع والأسباب في",
        "كيف ربطت المحكمة بين الأدلة في",
        "ما الذي يتطلب جمع الأدلة بشأن",
        "كيف بُني الاستنتاج في النزاع حول",
    )
    return f"{prefixes[index]} {_clean(target.context, 12)}؟"


def _variant_query_core(target: SemanticTarget, variant: str) -> str:
    if variant == "english":
        if target.category is QueryCategory.CASE_HOLDING:
            return f"In {_english(target.context)}, what did the court decide?"
        if target.category is QueryCategory.DEFINITION:
            return f"How does the text define {_english(target.defined_term)} in this dispute?"
        if target.category is QueryCategory.EXACT_PROVISION:
            return (
                f"What legal effect did Article {_english(target.provision_identifier)} have "
                f"concerning {_english(target.subject)}?"
            )
        if target.category is QueryCategory.DEADLINE:
            return f"What deadline applied to {_english(target.action)}?"
        if target.category is QueryCategory.AUTHORITY:
            return (
                f"What power or duty did {_english(target.actor)} have concerning "
                f"{_english(target.object)}?"
            )
        if target.category is QueryCategory.CONDITIONS:
            return "What condition governed the legal effect in this matter?"
        return (
            "How did the reported facts and court reasoning jointly support the court's "
            f"conclusion in {_english(target.context)}?"
        )
    if variant == "code-switch" and target.category is QueryCategory.MULTI_EVIDENCE:
        return f"إزاي دعم الدليلين النتيجة في {_clean(target.context)}؟"
    return render_semantic_query(target, variant)


def _variant_query(target: SemanticTarget, variant: str) -> str:
    query = _variant_query_core(target, variant)
    index = int(hashlib.sha256(target.proposition.encode()).hexdigest()[:2], 16) % 3
    suffixes = {
        "english": ("in this dispute", "under the reported facts", "in the case described"),
        "simple-ar": ("في الموضوع ده", "في القضية دي", "بحسب الوقائع دي"),
        "egyptian-ar": ("في النزاع ده", "في الحالة دي", "بحسب اللي حصل"),
        "code-switch": ("في this dispute", "بحسب the case", "في the matter described"),
    }
    suffix = suffixes[variant][index]
    punctuation = "?" if variant == "english" else "؟"
    query = query.rstrip("؟?")
    return f"{query} {suffix}{punctuation}"


def clean_v4_text(value: str) -> str:
    return _clean(value)


def variant_query_v4(target: SemanticTarget, variant: str) -> str:
    return _variant_query(target, variant)


def _variant_answer(base: DatasetItem) -> str:
    return base.gold_answer or ""


def _evidence_texts(item: DatasetItem, units: Mapping[str, CanonicalUnit]) -> tuple[str, ...]:
    return tuple(
        _clean(units[span.unit_id].text[span.start : span.end])
        for group in item.evidence_groups
        for span in group.spans
        if span.grade > RelevanceGrade.IRRELEVANT and span.unit_id in units
    )


def _new_identity(
    item: DatasetItem, target: SemanticTarget, documents: tuple[str, ...] | None = None
) -> tuple[str, str]:
    docs = documents or item.source_document_ids
    identity = tuple(
        (group.group_id, span.unit_id, span.start, span.end, target.proposition)
        for group in item.evidence_groups
        for span in group.spans
    )
    intent_id = deterministic_intent_id(item.category.value, docs, identity)
    return intent_id, deterministic_query_id(intent_id)


def _with_target(
    item: DatasetItem,
    target: SemanticTarget,
    units: Mapping[str, CanonicalUnit],
    *,
    documents: tuple[str, ...] | None = None,
) -> DatasetItem | None:
    docs = documents or item.source_document_ids
    intent_id, query_id = _new_identity(item, target, docs)
    query = _base_v4_query(target)
    answer = render_semantic_answer(target)
    evidence = _evidence_texts(item, units)
    if not validate_v4_semantic_contract(item.category, target, query, answer, evidence):
        return None
    return item.model_copy(
        update={
            "query_id": query_id,
            "intent_id": intent_id,
            "base_intent_id": None,
            "variant_id": None,
            "query_text": query,
            "source_document_ids": docs,
            "semantic_target": target,
            "gold_answer": answer,
            "creation_method": CreationMethod.DOCUMENT_DERIVED,
            "dataset_version": V4_VERSION,
            "review": item.review.model_copy(
                update={"state": ReviewState.DRAFT, "human_verified": False}
            ),
            "chunk_qrels": (),
        }
    )


def _build_variants(
    bases: tuple[DatasetItem, ...],
    variant_ids: tuple[str, ...] = ("simple-ar", "egyptian-ar", "english", "code-switch"),
) -> tuple[DatasetItem, ...]:
    answerable = [item for item in bases if item.answerability is Answerability.ANSWERABLE]
    by_category: defaultdict[QueryCategory, list[DatasetItem]] = defaultdict(list)
    for item in sorted(answerable, key=lambda row: row.intent_id):
        by_category[item.category].append(item)
    parents: list[DatasetItem] = []
    index = 0
    while len(parents) < 10:
        for category in _CATEGORIES:
            rows = by_category[category]
            if index < len(rows):
                parents.append(rows[index])
                if len(parents) == 10:
                    break
        index += 1
    variants: list[DatasetItem] = []
    language = {
        "simple-ar": QueryLanguage.ARABIC,
        "egyptian-ar": QueryLanguage.ARABIC,
        "english": QueryLanguage.ENGLISH,
        "code-switch": QueryLanguage.CODE_SWITCHED,
    }
    register = {
        "simple-ar": QueryRegister.SIMPLE,
        "egyptian-ar": QueryRegister.EGYPTIAN,
        "english": QueryRegister.PROFESSIONAL,
        "code-switch": QueryRegister.PROFESSIONAL,
    }
    for variant_id in variant_ids:
        for base in parents:
            target = base.semantic_target
            if target is None:
                raise ValueError("v4 variant parent has no semantic target")
            query = _variant_query(target, variant_id)
            if not validate_v4_query_text(query, language[variant_id].value):
                raise ValueError(f"invalid {variant_id} variant query: {query}")
            if variant_id == "english" and re.search(r"[ء-ي]", query):
                raise ValueError("English v4 variant contains Arabic placeholder text")
            variants.append(
                base.model_copy(
                    update={
                        "query_id": deterministic_query_id(base.intent_id, variant_id),
                        "variant_id": variant_id,
                        "base_intent_id": base.intent_id,
                        "query_text": query,
                        "language": language[variant_id],
                        "register": register[variant_id],
                        "creation_method": CreationMethod.ROBUSTNESS_VARIANT,
                        "gold_answer": _variant_answer(base),
                        "dataset_version": V4_VERSION,
                    }
                )
            )
    if len(variants) != 40:
        raise ValueError("v4 must produce exactly 40 variants")
    return tuple(variants)


def _ensure_unique_base_queries(
    items: tuple[DatasetItem, ...], units: Mapping[str, CanonicalUnit]
) -> tuple[DatasetItem, ...]:
    seen: defaultdict[str, int] = defaultdict(int)
    suffixes = (
        "في الدعوى المعروضة",
        "في المسألة محل البحث",
        "في النزاع محل الفصل",
        "في الطلب المعروض",
    )
    result: list[DatasetItem] = []
    for item in items:
        key = _norm(item.query_text)
        occurrence = seen[key]
        seen[key] += 1
        if occurrence == 0:
            result.append(item)
            continue
        punctuation = "؟" if item.query_text.endswith("؟") else "?"
        query = (
            item.query_text.rstrip("؟?") + f" {suffixes[occurrence % len(suffixes)]}{punctuation}"
        )
        evidence = _evidence_texts(item, units)
        if item.semantic_target is None or not validate_v4_semantic_contract(
            item.category, item.semantic_target, query, item.gold_answer or "", evidence
        ):
            raise ValueError(f"v4 could not resolve duplicate base query {item.query_id}")
        result.append(item.model_copy(update={"query_text": query}))
    return tuple(result)


def _multi_from_pool(
    item: DatasetItem,
    units: Mapping[str, CanonicalUnit],
    source_units: Mapping[str, list[CanonicalUnit]],
) -> DatasetItem | None:
    spans = [
        span
        for group in item.evidence_groups
        for span in group.spans
        if span.grade > RelevanceGrade.IRRELEVANT
    ]
    if len(spans) < 2:
        return None
    texts = tuple(
        _clean(units[span.unit_id].text[span.start : span.end])
        for span in spans
        if span.unit_id in units
    )
    if len(texts) < 2:
        return None
    existing = item.semantic_target
    conclusion = (
        texts[2]
        if len(texts) >= 3
        else (existing.conclusion if existing else existing.proposition if existing else "")
    )
    target = _make_multi_target(
        tuple(zip((units[span.unit_id] for span in spans[:2]), texts[:2], strict=True)), conclusion
    )
    if target is None:
        return None
    target = target.model_copy(
        update={"context": _neutral_issue(source_units[item.source_document_ids[0]], target)}
    )
    return _with_target(item, target, units)


def _arabiccr_multi_candidates(corpus: EvaluationCorpus) -> tuple[DatasetItem, ...]:
    units_by_document: defaultdict[str, list[CanonicalUnit]] = defaultdict(list)
    for unit in corpus.units:
        if unit.provenance.source_id == "arabiccr":
            units_by_document[unit.document_id].append(unit)
    result: list[DatasetItem] = []
    for document_id, rows in sorted(units_by_document.items()):
        events = [row for row in rows if row.unit_type.value == "events"]
        reasoning = [row for row in rows if row.unit_type.value == "reasoning"]
        rulings = [row for row in rows if row.unit_type.value == "ruling"]
        if not events or not reasoning or not rulings:
            continue
        event_segments = [
            segment
            for row in events
            for segment in semantic_segments(row.text)
            if valid_semantic_text(segment.text)
        ][:8]
        reasoning_segments = [
            segment
            for row in reasoning
            for segment in semantic_segments(row.text)
            if valid_semantic_text(segment.text)
        ][:8]
        ruling_segments = [
            segment
            for row in rulings
            for segment in semantic_segments(row.text)
            if valid_semantic_text(segment.text)
        ][:8]
        documents: tuple[str, ...] = (document_id,)
        for event in event_segments:
            for reason in reasoning_segments:
                for ruling in ruling_segments:
                    selections = (
                        (events[0], event.start, event.end, _clean(event.text)),
                        (reasoning[0], reason.start, reason.end, _clean(reason.text)),
                        (rulings[0], ruling.start, ruling.end, _clean(ruling.text)),
                    )
                    target = _make_multi_target(
                        (
                            (selections[0][0], selections[0][3]),
                            (selections[1][0], selections[1][3]),
                        ),
                        selections[2][3],
                    )
                    if target is None:
                        continue
                    spans = tuple(
                        EvidenceSpan(
                            unit_id=unit.unit_id,
                            start=start,
                            end=end,
                            grade=RelevanceGrade.REQUIRED,
                        )
                        for unit, start, end, _text in selections
                    )
                    identity = tuple(
                        (span.unit_id, span.start, span.end, target.proposition) for span in spans
                    )
                    intent_id = deterministic_intent_id(
                        QueryCategory.MULTI_EVIDENCE.value, documents, identity
                    )
                    target = target.model_copy(
                        update={"context": _neutral_issue(tuple(rows), target)}
                    )
                    query = _base_v4_query(target)
                    answer = render_semantic_answer(target)
                    candidate = DatasetItem(
                        query_id=deterministic_query_id(intent_id),
                        intent_id=intent_id,
                        query_text=query,
                        language=QueryLanguage.ARABIC,
                        register=QueryRegister.FORMAL,
                        category=QueryCategory.MULTI_EVIDENCE,
                        query_type=QueryType.REASONING,
                        jurisdiction="Saudi Arabia",
                        temporal_scope="source-relative",
                        creation_method=CreationMethod.DOCUMENT_DERIVED,
                        answerability=Answerability.ANSWERABLE,
                        difficulty=Difficulty.HARD,
                        source_document_ids=documents,
                        evidence_groups=(
                            EvidenceGroup(group_id=f"group-{intent_id[7:]}", spans=spans),
                        ),
                        semantic_target=target,
                        gold_answer=answer,
                        citation_anchors=tuple(
                            CitationAnchor(
                                kind="section",
                                label=unit.unit_type.value,
                                source_unit_id=unit.unit_id,
                            )
                            for unit, _start, _end, _text in selections
                        ),
                        dataset_version=V4_VERSION,
                    )
                    if validate_v4_semantic_contract(
                        QueryCategory.MULTI_EVIDENCE,
                        target,
                        query,
                        answer,
                        tuple(s[3] for s in selections),
                    ):
                        result.append(candidate)
                    break
                if result and result[-1].source_document_ids == documents:
                    break
            if result and result[-1].source_document_ids == documents:
                break
        if len(result) >= 40:
            break
    return tuple(result)


def apply_v3_adjudication(
    v3_items: tuple[DatasetItem, ...],
    corpus: EvaluationCorpus,
    decisions: tuple[ExternalReviewDecision, ...],
    pool: tuple[DatasetItem, ...],
) -> V4BuildResult:
    by_qid = {item.query_id: item for item in v3_items}
    if set(by_qid) != {decision.query_id for decision in decisions}:
        raise ValueError("external v3 adjudication does not cover exactly the v3 records")
    units = {unit.unit_id: unit for unit in corpus.units}
    docs: defaultdict[str, list[CanonicalUnit]] = defaultdict(list)
    for unit in corpus.units:
        docs[unit.document_id].append(unit)
    decisions_by_qid = {decision.query_id: decision for decision in decisions}
    bases_v3 = tuple(item for item in v3_items if item.variant_id is None)
    variants_v3 = tuple(item for item in v3_items if item.variant_id is not None)
    accepted = [item for item in bases_v3 if decisions_by_qid[item.query_id].decision == "accept"]
    corrected: list[DatasetItem] = []
    mapping: list[dict[str, object]] = []
    for item in bases_v3:
        decision = decisions_by_qid[item.query_id]
        if decision.decision == "accept":
            accepted_item = item.model_copy(
                update={
                    "dataset_version": V4_VERSION,
                    "review": item.review.model_copy(
                        update={"human_verified": False, "state": ReviewState.DRAFT}
                    ),
                }
            )
            corrected.append(accepted_item)
            mapping.append(
                {
                    "old_query_id": item.query_id,
                    "old_intent_id": item.intent_id,
                    "decision": decision.decision,
                    "new_query_id": accepted_item.query_id,
                    "new_intent_id": accepted_item.intent_id,
                    "evidence_preserved": True,
                    "evidence_replaced": False,
                    "replacement_source": None,
                    "replacement_category": None,
                }
            )
            continue
        if decision.decision != "correct":
            continue
        evidence = _evidence_texts(item, units)
        target = _extract_target(item.category, " ".join(evidence), item.semantic_target)
        if item.category is QueryCategory.CASE_HOLDING and target is not None:
            target = target.model_copy(
                update={"context": _neutral_issue(docs[item.source_document_ids[0]], target)}
            )
        if target is None:
            raise ValueError(
                f"v4 cannot semantically correct {item.query_id} without changing evidence"
            )
        updated = _with_target(item, target, units)
        if updated is None:
            raise ValueError(f"v4 semantic contract failed for correct record {item.query_id}")
        corrected.append(updated)
        mapping.append(
            {
                "old_query_id": item.query_id,
                "old_intent_id": item.intent_id,
                "decision": decision.decision,
                "new_query_id": updated.query_id,
                "new_intent_id": updated.intent_id,
                "evidence_preserved": True,
                "evidence_replaced": False,
                "replacement_source": None,
                "replacement_category": None,
            }
        )
    selected_ids = {item.query_id for item in bases_v3}
    pool_by_category: defaultdict[QueryCategory, list[DatasetItem]] = defaultdict(list)
    for item in pool:
        if (
            item.variant_id is None
            and item.answerability is Answerability.ANSWERABLE
            and item.query_id not in selected_ids
        ):
            pool_by_category[item.category].append(item)
    generated_arabiccr_multi = _arabiccr_multi_candidates(corpus)
    for candidate in generated_arabiccr_multi:
        pool_by_category[QueryCategory.MULTI_EVIDENCE].append(candidate)
    replacement_reasons: list[dict[str, object]] = []
    replacements: list[DatasetItem] = []
    selected_replacement_ids: set[str] = set()
    multi_attempts: Counter[str] = Counter()
    multi_qualified: Counter[str] = Counter()
    for category, count in _REPLACE_COUNTS.items():
        if category is QueryCategory.MULTI_EVIDENCE:
            by_source: defaultdict[str, list[DatasetItem]] = defaultdict(list)
            for candidate in pool_by_category[category]:
                source = next(
                    (
                        unit.provenance.source_id
                        for unit in corpus.units
                        if unit.document_id in candidate.source_document_ids
                    ),
                    "unknown",
                )
                by_source[source].append(candidate)
            candidates = proportional_source_order(
                {
                    source: tuple(sorted(rows, key=lambda item: item.query_id))
                    for source, rows in by_source.items()
                },
                count,
            )
        else:
            candidates = sorted(
                pool_by_category[category],
                key=lambda item: (item.source_document_ids, item.query_id),
            )
        for candidate in candidates:
            if len([item for item in replacements if item.category is category]) >= count:
                break
            if category is QueryCategory.MULTI_EVIDENCE:
                source = next(
                    (
                        unit.provenance.source_id
                        for unit in corpus.units
                        if unit.document_id in candidate.source_document_ids
                    ),
                    "unknown",
                )
                multi_attempts[source] += 1
                updated = _multi_from_pool(candidate, units, docs)
                if updated is None:
                    replacement_reasons.append(
                        {
                            "query_id": candidate.query_id,
                            "category": category.value,
                            "source": source,
                            "reason": (
                                "fewer_than_two_independent_premises_or_conclusion_not_entailed"
                            ),
                        }
                    )
                    continue
                multi_qualified[source] += 1
            else:
                evidence = _evidence_texts(candidate, units)
                target = _extract_target(category, " ".join(evidence), candidate.semantic_target)
                if target is None:
                    replacement_reasons.append(
                        {
                            "query_id": candidate.query_id,
                            "category": category.value,
                            "source": next(
                                (
                                    unit.provenance.source_id
                                    for unit in corpus.units
                                    if unit.document_id in candidate.source_document_ids
                                ),
                                "unknown",
                            ),
                            "reason": "category_specific_semantic_extraction_failed",
                        }
                    )
                    continue
                updated = _with_target(candidate, target, units)
                if updated is None:
                    replacement_reasons.append(
                        {
                            "query_id": candidate.query_id,
                            "category": category.value,
                            "source": next(
                                (
                                    unit.provenance.source_id
                                    for unit in corpus.units
                                    if unit.document_id in candidate.source_document_ids
                                ),
                                "unknown",
                            ),
                            "reason": "semantic_contract_failed",
                        }
                    )
                    continue
            replacements.append(updated)
            selected_replacement_ids.add(candidate.query_id)
            old = next(
                item
                for item in bases_v3
                if decisions_by_qid[item.query_id].decision == "replace"
                and decisions_by_qid[item.query_id].category == category.value
                and item.query_id not in {row["old_query_id"] for row in mapping}
            )
            mapping.append(
                {
                    "old_query_id": old.query_id,
                    "old_intent_id": old.intent_id,
                    "decision": "replace",
                    "new_query_id": updated.query_id,
                    "new_intent_id": updated.intent_id,
                    "evidence_preserved": False,
                    "evidence_replaced": True,
                    "replacement_source": next(
                        (
                            unit.provenance.source_id
                            for unit in corpus.units
                            if unit.document_id in updated.source_document_ids
                        ),
                        "unknown",
                    ),
                    "replacement_category": category.value,
                }
            )
        if len([item for item in replacements if item.category is category]) != count:
            raise ValueError(
                f"v4 replacement pool cannot satisfy {category.value}: expected {count}"
            )
    all_bases = tuple(sorted(corrected + replacements, key=lambda item: item.intent_id))
    if (
        len(all_bases) != 200
        or Counter(
            item.category for item in all_bases if item.answerability is Answerability.ANSWERABLE
        )
        != Counter(
            {
                QueryCategory.EXACT_PROVISION: 30,
                QueryCategory.DEFINITION: 25,
                QueryCategory.DEADLINE: 20,
                QueryCategory.AUTHORITY: 20,
                QueryCategory.CONDITIONS: 30,
                QueryCategory.MULTI_EVIDENCE: 25,
                QueryCategory.CASE_HOLDING: 25,
            }
        )
        or len(accepted) != 25
    ):
        raise ValueError("v4 base category or acceptance counts are incorrect")
    all_bases = _ensure_unique_base_queries(all_bases, units)
    variants = _build_variants(all_bases)
    variants_by_kind: defaultdict[str, list[DatasetItem]] = defaultdict(list)
    for variant in variants:
        variants_by_kind[variant.variant_id or ""].append(variant)
    for old in variants_v3:
        decision = decisions_by_qid[old.query_id]
        old_kind_rows = sorted(
            (row for row in variants_v3 if row.variant_id == old.variant_id),
            key=lambda row: row.query_id,
        )
        new_kind_rows = sorted(variants_by_kind[old.variant_id or ""], key=lambda row: row.query_id)
        new = new_kind_rows[old_kind_rows.index(old)]
        mapping.append(
            {
                "old_query_id": old.query_id,
                "old_intent_id": old.intent_id,
                "decision": decision.decision,
                "new_query_id": new.query_id,
                "new_intent_id": new.intent_id,
                "evidence_preserved": True,
                "evidence_replaced": False,
                "replacement_source": None,
                "replacement_category": None,
            }
        )
    if len(mapping) != 240:
        raise ValueError(f"v4 mapping covers {len(mapping)} rather than 240 records")
    mapping.sort(key=lambda row: str(row["old_query_id"]))
    source_audit = {
        "attempts_by_source": dict(sorted(multi_attempts.items())),
        "qualified_by_source": dict(sorted(multi_qualified.items())),
        "retrieval_scores_used": False,
    }
    return V4BuildResult(
        tuple(pool) + generated_arabiccr_multi,
        all_bases,
        variants,
        tuple(mapping),
        tuple(replacement_reasons),
        source_audit,
    )


def v4_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def multi_opportunity_audit(corpus: EvaluationCorpus) -> dict[str, object]:
    """Bounded structural audit for both source families, without ranking."""

    by_document: defaultdict[tuple[str, str], list[CanonicalUnit]] = defaultdict(list)
    for unit in corpus.units:
        by_document[(unit.provenance.source_id, unit.document_id)].append(unit)
    attempts: Counter[str] = Counter()
    qualified: Counter[str] = Counter()
    rejection_reasons: Counter[str] = Counter()
    for (source, _document), rows in sorted(by_document.items()):
        if source != "arabiccr":
            continue
        events = [row for row in rows if row.unit_type.value == "events"]
        reasoning = [row for row in rows if row.unit_type.value == "reasoning"]
        ruling = [row for row in rows if row.unit_type.value == "ruling"]
        if not events or not reasoning or not ruling:
            rejection_reasons["missing_events_reasoning_or_ruling"] += 1
            continue
        attempts[source] += 1
        first = _clean(events[0].text, 60)
        second = _clean(reasoning[0].text, 60)
        conclusion = _clean(ruling[0].text, 60)
        target = _make_multi_target(((events[0], first), (reasoning[0], second)), conclusion)
        if target is None:
            rejection_reasons["structured_units_not_jointly_entailed"] += 1
        else:
            qualified[source] += 1
    return {
        "attempts_by_source": dict(sorted(attempts.items())),
        "qualified_by_source": dict(sorted(qualified.items())),
        "rejection_reasons_by_source": {"arabiccr": dict(sorted(rejection_reasons.items()))},
        "retrieval_scores_used": False,
    }
