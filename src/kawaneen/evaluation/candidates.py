"""Evidence-qualified, retrieval-independent Phase 6 draft generation."""

from __future__ import annotations

import ast
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TypeVar

from kawaneen.chunking.models import CitationAnchor
from kawaneen.corpus.models import CanonicalUnit
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
    UnanswerableReason,
    deterministic_intent_id,
    deterministic_query_id,
)
from kawaneen.evaluation.serialization import write_items_jsonl

PRIVATE_ROOT = Path("artifacts/private/phase6_evaluation")

# The selected 200-intent allocation remains unchanged. The larger pool is
# deliberately predeclared and is generated from validated semantic spans.
CATEGORY_TARGETS = {
    QueryCategory.EXACT_PROVISION: 45,
    QueryCategory.DEFINITION: 40,
    QueryCategory.DEADLINE: 40,
    QueryCategory.AUTHORITY: 40,
    QueryCategory.CONDITIONS: 50,
    QueryCategory.MULTI_EVIDENCE: 50,
    QueryCategory.CASE_HOLDING: 45,
    QueryCategory.UNANSWERABLE: 50,
}
BASE_TARGETS = {
    QueryCategory.EXACT_PROVISION: 30,
    QueryCategory.DEFINITION: 25,
    QueryCategory.DEADLINE: 20,
    QueryCategory.AUTHORITY: 20,
    QueryCategory.CONDITIONS: 30,
    QueryCategory.MULTI_EVIDENCE: 25,
    QueryCategory.CASE_HOLDING: 25,
    QueryCategory.UNANSWERABLE: 25,
}

_SourceItem = TypeVar("_SourceItem")
_REFERENCE_TYPES = frozenset({"applicable_laws", "court_reasoning", "reasoning"})
_CASE_TYPES = frozenset({"verdict", "ruling"})
_MULTI_TYPES = frozenset({"facts", "court_reasoning", "events", "reasoning", "verdict", "ruling"})
_PATTERNS = {
    QueryCategory.EXACT_PROVISION: re.compile(
        r"(?:المادة|المواد|الفقرة|البند|نظام|اللائحة|قرار|المرسوم)\s*(?:رقم\s*)?(?:\(?\s*[0-9\u0660-\u0669]+)"
    ),
    QueryCategory.DEFINITION: re.compile(
        r"(?:يقصد\s+(?:ب|بـ)|المقصود\s+(?:ب|بـ)|يعرّف|يُعرّف|تعرف|تعريف|هو\s+كل|هي\s+كل)"
    ),
    QueryCategory.DEADLINE: re.compile(
        r"(?:خلال|مدة|مهلة|يوم(?:ين|اً|ان)?|أيام|شهر|سنة|بتاريخ|قبل\s+الجلسة|موعد|ينتهي)"
    ),
    QueryCategory.AUTHORITY: re.compile(
        r"(?:تختص|الاختصاص|اختصاص|يلتزم|يجب|مسؤول|صلاحية|يحق|أوجب|ألزم|من\s+اختصاص)"
    ),
    QueryCategory.CONDITIONS: re.compile(
        r"(?:إذا|فإذا|يشترط|بشرط|ما\s+لم|لا\s+يجوز|يجوز|استثناء|متى|حال\s+كان)"
    ),
}
_ACTION = re.compile(r"(?:يودع|يقدم|يحضر|يلتزم|يجب|يبدأ|ينتهي|إجراء|جلسة|يطلب)")
_ACTOR = re.compile(r"(?:المحكمة|الدائرة|الجهة المختصة|الإدارة المختصة|المدعي|المدعى عليه)")
_TIME = re.compile(
    r"(?:خلال\s+[\w\u0660-\u0669]+(?:\s+[\w\u0660-\u0669]+)?|مدة\s+[^،؛.]{1,30}|مهلة\s+[^،؛.]{1,30}|"
    r"[0-9\u0660-\u0669]+\s*(?:يوم|أيام|شهر|سنة)|بتاريخ\s+[0-9\u0660-\u0669/]+)"
)
_DISPOSITION = re.compile(
    r"(?:قضت|حكمت|ألزمت|إلزام|رفض|قبول|إثبات\s+الصلح|عدم\s+قبول|شطب|نقض|تأييد|انتهت\s+الدائرة)"
)
_LIST_PREFIX = re.compile(r"^\s*(?:\d+|[\u0660-\u0669]+)\s*[.)،:-]?\s*")
_WHITESPACE = re.compile(r"\s+")
_PII = re.compile(
    r"(?:[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b|"
    r"(?<!\d)(?:\+?\d[\d ()-]{7,}\d)(?!\d))"
)
_TOPICS = {
    QueryCategory.EXACT_PROVISION: (
        "الاختصاص التجاري",
        "الإجراءات القضائية",
        "الصلح القضائي",
        "الإثبات",
        "العقود التجارية",
        "الاختصاص المكاني",
        "المواعيد القضائية",
        "تنفيذ الالتزام",
        "المطالبة المالية",
        "الدفوع",
        "حجية المستند",
        "التبليغ",
        "قيد الدعوى",
        "المحضر القضائي",
        "الوكالة",
        "المسؤولية العقدية",
        "الطلب العارض",
        "الاختصاص النوعي",
        "المصالحة",
        "المقابل المالي",
        "التنفيذ",
        "البيّنة",
        "الإقرار",
        "الجزاء الإجرائي",
        "نطاق الدعوى",
        "الالتزام التجاري",
        "المعاملة التجارية",
        "المطالبة بالتعويض",
        "الطعن",
        "المرافعة",
    ),
    QueryCategory.DEFINITION: (
        "المفهوم محل النزاع",
        "العلاقة التجارية",
        "الالتزام التعاقدي",
        "المستند القضائي",
        "الصلح",
        "الإقرار",
        "البيّنة",
        "الضرر",
        "التعويض",
        "التاجر",
        "الدعوى التجارية",
        "الاختصاص",
        "التسليم",
        "الوفاء",
        "الوكالة",
        "المطالبة",
        "الشرط التعاقدي",
        "المدعى عليه",
        "المبلغ المستحق",
        "الحق محل الحماية",
        "الإجراء النظامي",
        "المسؤولية",
        "التعاقد",
        "الطلب القضائي",
        "الحكم",
        "المصلحة",
        "الدفع",
        "الالتزام",
        "المعاملة",
        "المخالفة",
    ),
    QueryCategory.DEADLINE: (
        "إيداع مذكرة الدفاع",
        "تقديم الجواب",
        "حضور الجلسة",
        "تنفيذ الالتزام",
        "المطالبة بالدين",
        "إتمام التبليغ",
        "رفع الدعوى",
        "استكمال المستندات",
        "سداد المبلغ",
        "بدء العقد",
        "انتهاء المدة",
        "طلب المهلة",
        "تحديد الجلسة",
        "تقديم البينة",
        "إيداع الطلب",
        "الوفاء بالعقد",
        "المراجعة القضائية",
        "إصدار المحضر",
        "تنفيذ الصلح",
        "إتمام الإجراء",
    ),
    QueryCategory.AUTHORITY: (
        "نظر الدعوى",
        "تحديد الاختصاص",
        "إدارة الجلسة",
        "إيداع المذكرة",
        "تنفيذ الحكم",
        "تقديم الطلب",
        "إثبات الصلح",
        "سداد الالتزام",
        "الرد على الدعوى",
        "إصدار القرار",
        "تقدير البينة",
        "تحديد المسؤولية",
        "إجراء التبليغ",
        "حماية الحق",
        "فحص المستند",
        "تطبيق القاعدة",
        "إلزام الطرف",
        "قبول الطلب",
        "رفض الدفع",
        "متابعة التنفيذ",
    ),
    QueryCategory.CONDITIONS: (
        "قبول الدعوى",
        "تطبيق الالتزام",
        "نفاذ الصلح",
        "استحقاق المبلغ",
        "قبول البينة",
        "إلزام الطرف",
        "نظر الطلب",
        "صحة التعاقد",
        "استعمال الدفع",
        "تحديد المسؤولية",
        "إجراء التنفيذ",
        "سماع الدعوى",
        "قبول المستند",
        "تقرير التعويض",
        "نفاذ الحكم",
        "الاحتجاج بالإقرار",
        "إعمال الاستثناء",
        "تحديد الحق",
        "ترتيب الجزاء",
        "إثبات الواقعة",
        "تطبيق النص",
        "قبول الطعن",
        "إتمام الإجراء",
        "تقدير الضرر",
        "إثبات الوفاء",
        "استمرار الالتزام",
        "تحديد الأجل",
        "إلزام المدين",
        "حماية المركز القانوني",
        "إصدار المنطوق",
    ),
    QueryCategory.MULTI_EVIDENCE: (
        "المطالبة والنتيجة",
        "الوقائع والمنطوق",
        "الدفع والحكم",
        "العقد والالتزام",
        "المستند والجزاء",
        "الإقرار والنتيجة",
        "الطلب والرفض",
        "الصلح والتنفيذ",
        "البينة والاستحقاق",
        "التبليغ والحكم",
        "الوقائع والتعويض",
        "الدعوى والاختصاص",
        "الإنكار والإثبات",
        "المبلغ والإلزام",
        "المحضر والصلح",
        "التعاقد والوفاء",
        "الدفوع والمنطوق",
        "المطالبة والرفض",
        "الضرر والتعويض",
        "أصل التسليم وتمام التنفيذ",
        "الاختصاص والموضوع",
        "الإجراء والنتيجة",
        "الحضور والقرار",
        "الالتزام والجزاء",
        "الدليل والحكم",
    ),
    QueryCategory.CASE_HOLDING: (
        "المطالبة الأصلية",
        "الطلب المالي",
        "الدعوى التجارية",
        "الصلح",
        "التعويض",
        "الوفاء بالعقد",
        "الدفع المقابل",
        "المبلغ المطالب به",
        "تنفيذ الالتزام",
        "الطلب العارض",
        "المسؤولية",
        "الاستحقاق",
        "رفض الدعوى",
        "قبول الطلب",
        "منطوق الحكم",
        "الجزاء",
        "المصاريف",
        "المطالبة بالتسليم",
        "الإقرار",
        "الإنكار",
        "التعاقد",
        "المستند",
        "الدفوع",
        "الطعن",
        "الاختصاص",
        "الصلح التنفيذي",
        "المبلغ المستحق",
        "الحق المدعى به",
        "النتيجة القضائية",
        "الحكم النهائي",
    ),
}


@dataclass(frozen=True, slots=True)
class EvidenceDiscovery:
    start: int
    end: int
    semantic_text: str
    score: int


@dataclass(frozen=True, slots=True)
class _Segment:
    start: int
    end: int
    text: str
    raw_text: str
    raw_content_start: int
    raw_content_end: int


@dataclass(frozen=True, slots=True)
class DraftBuildResult:
    base_candidates: tuple[DatasetItem, ...]
    selected_base_candidates: tuple[DatasetItem, ...]
    variants: tuple[DatasetItem, ...]

    @property
    def all_items(self) -> tuple[DatasetItem, ...]:
        return self.selected_base_candidates + self.variants


def _redact(value: str) -> str:
    value = _PII.sub("[REDACTED]", value)
    return re.sub(r"\s+", " ", value).strip()


@lru_cache(maxsize=100_000)
def _segments(value: str) -> tuple[_Segment, ...]:
    """Return decoded semantic segments with offsets into untouched canonical text."""

    stripped = value.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        result: list[_Segment] = []
        for match in re.finditer(r"(['\"])(?:\\.|(?!\1).)*\1", stripped):
            literal = match.group(0)
            try:
                decoded = ast.literal_eval(literal)
            except (SyntaxError, ValueError):
                continue
            if isinstance(decoded, str):
                offset = len(value) - len(stripped)
                raw_start = offset + match.start()
                raw_end = offset + match.end()
                result.append(
                    _Segment(
                        raw_start,
                        raw_end,
                        decoded,
                        literal,
                        raw_start + 1,
                        raw_end - 1,
                    )
                )
        if result:
            return tuple(result)
    result = []
    starts = [0]
    for match in re.finditer(r"(?<=[.!?؟؛:])\s+|\n+", value):
        starts.append(match.end())
    for start, end in zip(starts, [*starts[1:], len(value)], strict=True):
        while start < end and value[start].isspace():
            start += 1
        while end > start and value[end - 1].isspace():
            end -= 1
        if end > start:
            result.append(_Segment(start, end, value[start:end], value[start:end], start, end))
    return tuple(result)


def _clean_semantic_text(text: str) -> str:
    cleaned = _LIST_PREFIX.sub("", text)
    cleaned = cleaned.strip(" \t\r\n'\"[](),؛:.؟")
    return cleaned


def _candidate_score(text: str, category: QueryCategory) -> int:
    cues = len(_PATTERNS.get(category, _DISPOSITION).findall(text))
    return cues * 100 + min(len(text), 900) + min(len(text) // 8, 80)


def _valid_semantic_text(text: str) -> bool:
    cleaned = _clean_semantic_text(text)
    return len(cleaned) >= 24 and sum(character.isalpha() for character in cleaned) >= 8


def _map_decoded_span(segment: _Segment, value: str, start: int, end: int) -> tuple[int, int]:
    target = value[start:end]
    local = segment.raw_text.find(target)
    if local >= 0:
        return segment.start + local, segment.start + local + len(target)
    # Escaped list literals cannot always be mapped character-for-character;
    # retain the literal content bounds as an exact source span.
    return segment.raw_content_start, segment.raw_content_end


def _category_valid(category: QueryCategory, text: str, unit_type: str) -> bool:
    cleaned = _clean_semantic_text(text)
    if category is QueryCategory.EXACT_PROVISION:
        return unit_type in _REFERENCE_TYPES and bool(_PATTERNS[category].search(cleaned))
    if category is QueryCategory.DEFINITION:
        return bool(_PATTERNS[category].search(cleaned))
    if category is QueryCategory.DEADLINE:
        return bool(_PATTERNS[category].search(cleaned) and _ACTION.search(cleaned))
    if category is QueryCategory.AUTHORITY:
        return bool(_PATTERNS[category].search(cleaned) and _ACTOR.search(cleaned))
    if category is QueryCategory.CONDITIONS:
        return bool(_PATTERNS[category].search(cleaned) and _ACTION.search(cleaned))
    if category is QueryCategory.CASE_HOLDING:
        return unit_type in _CASE_TYPES and bool(_DISPOSITION.search(cleaned))
    return False


def discover_evidence(
    category: QueryCategory, unit: CanonicalUnit
) -> tuple[EvidenceDiscovery, ...]:
    """Discover validated category-specific spans in untouched canonical offsets."""

    if category in {QueryCategory.MULTI_EVIDENCE, QueryCategory.UNANSWERABLE}:
        return ()
    return _discover_all(unit).get(category, ())


def _discover_all(
    unit: CanonicalUnit,
) -> dict[QueryCategory, tuple[EvidenceDiscovery, ...]]:
    """Classify all semantic opportunities while decoding each unit only once."""

    result: dict[QueryCategory, list[EvidenceDiscovery]] = defaultdict(list)
    categories = (
        QueryCategory.EXACT_PROVISION,
        QueryCategory.DEFINITION,
        QueryCategory.DEADLINE,
        QueryCategory.AUTHORITY,
        QueryCategory.CONDITIONS,
        QueryCategory.CASE_HOLDING,
    )
    for segment in _segments(unit.text):
        cleaned = _clean_semantic_text(segment.text)
        if not _valid_semantic_text(cleaned):
            continue
        local_start = max(segment.text.find(cleaned), 0)
        start, end = _map_decoded_span(
            segment, segment.text, local_start, local_start + len(cleaned)
        )
        if end <= start:
            continue
        for category in categories:
            if _category_valid(category, cleaned, unit.unit_type.value):
                result[category].append(
                    EvidenceDiscovery(
                        start=start,
                        end=end,
                        semantic_text=cleaned,
                        score=_candidate_score(cleaned, category),
                    )
                )
    return {
        category: tuple(sorted(values, key=lambda item: (-item.score, item.start, item.end)))
        for category, values in result.items()
    }


def category_match(category: QueryCategory, text: str, unit_type: str) -> bool:
    if category is QueryCategory.MULTI_EVIDENCE:
        return unit_type in _MULTI_TYPES and _valid_semantic_text(_clean_semantic_text(text))
    if category is QueryCategory.UNANSWERABLE:
        return False
    return any(
        _valid_semantic_text(segment.text) and _category_valid(category, segment.text, unit_type)
        for segment in _segments(text)
    )


def clean_semantic_text(text: str) -> str:
    """Expose the canonical-text cleanup used by semantic candidate builders."""

    return _clean_semantic_text(text)


def semantic_segments(value: str) -> tuple[_Segment, ...]:
    """Expose lossless segment offsets for downstream semantic extraction."""

    return _segments(value)


def valid_semantic_text(text: str) -> bool:
    return _valid_semantic_text(text)


def proportional_source_order(
    groups: Mapping[str, Sequence[_SourceItem]], target: int
) -> list[_SourceItem]:
    return _proportional_source_order(groups, target)


def _allocate_source_quotas(opportunities: Mapping[str, int], target: int) -> dict[str, int]:
    if target < 0:
        raise ValueError("target must be non-negative")
    available = {source: max(0, count) for source, count in opportunities.items() if count > 0}
    if not available or target == 0:
        return {source: 0 for source in sorted(opportunities)}
    target = min(target, sum(available.values()))
    total = sum(available.values())
    quotas = {source: (target * count) // total for source, count in available.items()}
    remainders = {source: (target * count) % total for source, count in available.items()}
    for source in sorted(available, key=lambda value: (-remainders[value], value)):
        if sum(quotas.values()) >= target:
            break
        if quotas[source] < available[source]:
            quotas[source] += 1
    return {source: quotas.get(source, 0) for source in sorted(opportunities)}


def _proportional_source_order(
    groups: Mapping[str, Sequence[_SourceItem]], target: int
) -> list[_SourceItem]:
    quotas = _allocate_source_quotas(
        {source: len(items) for source, items in groups.items()}, target
    )
    selected = {source: 0 for source in groups}
    result: list[_SourceItem] = []
    while len(result) < min(target, sum(quotas.values())):
        source = min(
            (source for source in groups if selected[source] < quotas.get(source, 0)),
            key=lambda value: (selected[value] / quotas[value], value),
        )
        result.append(groups[source][selected[source]])
        selected[source] += 1
    return result


def _reference_answer(text: str) -> str:
    match = re.search(
        r"(?:المادة|الفقرة|البند|نظام|اللائحة|قرار|المرسوم)\s*(?:رقم\s*)?"
        r"(?:\(?\s*[0-9\u0660-\u0669]+(?:/[0-9\u0660-\u0669]+)?\s*\)?)",
        text,
    )
    return (
        f"المرجع النظامي المشار إليه هو {match.group(0)}."
        if match
        else "المرجع النظامي محدد في الأساس النظامي للحكم."
    )


def _time_answer(text: str) -> str:
    match = _TIME.search(text)
    return (
        f"المدة أو الموعد المحدد هو {match.group(0)}."
        if match
        else "يحدد النص مدة أو موعداً إجرائياً صريحاً."
    )


def _authority_answer(text: str) -> str:
    match = _ACTOR.search(text)
    return (
        f"الجهة أو الطرف المعني هو {match.group(0)}."
        if match
        else "حدد النص الجهة أو الطرف المسؤول عن الإجراء."
    )


def _holding_answer(text: str) -> str:
    if re.search(r"إثبات\s+الصلح|أثبتت?\s+(?:الدائرة|المحكمة)\s+الصلح", text):
        return "أثبتت الدائرة الصلح بين الطرفين."
    if re.search(r"رفض|عدم\s+قبول", text):
        return "انتهى الحكم إلى رفض الدعوى أو الطلب محل النظر."
    if re.search(r"إلزام|ألزمت", text):
        return "انتهى الحكم إلى إلزام الطرف المحكوم عليه بالأداء."
    if re.search(r"قبول|تأييد", text):
        return "انتهى الحكم إلى قبول الطلب أو تأييد النتيجة محل النظر."
    return "أثبت منطوق الحكم النتيجة القضائية المبينة في ختامه."


def _answer_for(category: QueryCategory, texts: tuple[str, ...]) -> str:
    text = " ".join(texts)
    if category is QueryCategory.EXACT_PROVISION:
        return _reference_answer(text)
    if category is QueryCategory.DEADLINE:
        return _time_answer(text)
    if category is QueryCategory.AUTHORITY:
        return _authority_answer(text)
    if category is QueryCategory.CASE_HOLDING:
        return _holding_answer(text)
    if category is QueryCategory.DEFINITION:
        return "يحدد النص المفهوم القانوني الذي تُبنى عليه القاعدة محل التطبيق."
    if category is QueryCategory.CONDITIONS:
        return "يتوقف تطبيق القاعدة على تحقق الشرط أو الاستثناء الذي يورده النص."
    return f"انتهت الدائرة إلى النتيجة المبينة بعد موازنة الوقائع والأسباب: {_holding_answer(text)}"


def _query_for(
    category: QueryCategory, ordinal: int, *, variant: str | None = None
) -> tuple[str, QueryType, QueryRegister, QueryLanguage]:
    if category is QueryCategory.UNANSWERABLE:
        if variant is None:
            return (
                _UNANSWERABLE_SPECS[ordinal % len(_UNANSWERABLE_SPECS)][1],
                QueryType.ABSTENTION,
                QueryRegister.FORMAL,
                QueryLanguage.ARABIC,
            )
        topic = _UNANSWERABLE_VARIANT_TOPICS[ordinal % len(_UNANSWERABLE_VARIANT_TOPICS)]
        unanswerable_variants = {
            "simple-ar": f"هل يمكن الإجابة عن سؤال {topic} من الأحكام المتاحة؟",
            "egyptian-ar": f"ينفع نعرف حكم {topic} من المستندات الموجودة؟",
            "english": f"Can the available judgments answer the question about {topic}?",
            "code-switch": f"هل نقدر نحدد الـ answer عن {topic} من الأحكام المتاحة؟",
        }
        registers = {
            "simple-ar": (QueryRegister.SIMPLE, QueryLanguage.ARABIC),
            "egyptian-ar": (QueryRegister.EGYPTIAN, QueryLanguage.ARABIC),
            "english": (QueryRegister.PROFESSIONAL, QueryLanguage.ENGLISH),
            "code-switch": (QueryRegister.PROFESSIONAL, QueryLanguage.CODE_SWITCHED),
        }
        register, language = registers[variant]
        return unanswerable_variants[variant], QueryType.ABSTENTION, register, language
    structures = {
        QueryCategory.EXACT_PROVISION: (
            "ما المرجع النظامي الذي اعتمد عليه الحكم في مسألة {topic}؟",
            "أي نص نظامي يحكم مسألة {topic} في هذا الحكم؟",
            "ما موضع الأساس النظامي المتعلق بـ{topic}؟",
        ),
        QueryCategory.DEFINITION: (
            "كيف يحدد الحكم المفهوم القانوني المتعلق بـ{topic}؟",
            "ما المعنى القانوني الذي يقرره الحكم بشأن {topic}؟",
            "كيف يفسر النص مفهوم {topic}؟",
        ),
        QueryCategory.DEADLINE: (
            "ما المدة أو الموعد الذي يحكم {topic}؟",
            "متى ينبغي إتمام {topic} وفق الحكم؟",
            "ما الأجل الإجرائي المتصل بـ{topic}؟",
        ),
        QueryCategory.AUTHORITY: (
            "من الجهة أو الطرف المسؤول عن {topic}؟",
            "على من يقع الاختصاص في شأن {topic}؟",
            "من يملك صلاحية {topic} بحسب الحكم؟",
        ),
        QueryCategory.CONDITIONS: (
            "ما الشرط أو الاستثناء اللازم لـ{topic}؟",
            "متى يمكن ترتيب أثر {topic}؟",
            "ما المتطلب السابق لتطبيق القاعدة على {topic}؟",
        ),
        QueryCategory.MULTI_EVIDENCE: (
            "كيف قادت عناصر {topic} إلى النتيجة؟",
            "ما الصلة بين {topic} في أسباب الحكم ومنطوقه؟",
            "كيف فسرت المحكمة {topic} قبل إصدار القرار؟",
        ),
        QueryCategory.CASE_HOLDING: (
            "ما الذي قضت به الدائرة بشأن {topic}؟",
            "ما نتيجة الحكم في موضوع {topic}؟",
            "ما الجزاء أو التدبير الذي انتهى إليه الحكم في {topic}؟",
        ),
    }
    query_types = {
        QueryCategory.EXACT_PROVISION: QueryType.REFERENCE_LOOKUP,
        QueryCategory.DEFINITION: QueryType.LEGAL_CONCEPT,
        QueryCategory.DEADLINE: QueryType.PROCEDURE,
        QueryCategory.AUTHORITY: QueryType.RESPONSIBILITY,
        QueryCategory.CONDITIONS: QueryType.CONDITIONS_EXCEPTIONS,
        QueryCategory.MULTI_EVIDENCE: QueryType.REASONING,
        QueryCategory.CASE_HOLDING: QueryType.HOLDING_OUTCOME_REMEDY,
    }
    topic = _TOPICS[category][ordinal % len(_TOPICS[category])]
    question = structures[category][ordinal % 3].format(topic=topic)
    query_type = query_types[category]
    if variant is None:
        return question, query_type, QueryRegister.FORMAL, QueryLanguage.ARABIC
    specs = {
        "simple-ar": (
            f"إيه الحكم أو القاعدة بخصوص {topic}؟",
            QueryRegister.SIMPLE,
            QueryLanguage.ARABIC,
        ),
        "egyptian-ar": (
            f"إيه اللي المحكمة قررته عن {topic}؟",
            QueryRegister.EGYPTIAN,
            QueryLanguage.ARABIC,
        ),
        "english": (
            f"What does the judgment establish about {topic}?",
            QueryRegister.PROFESSIONAL,
            QueryLanguage.ENGLISH,
        ),
        "code-switch": (
            f"إيه الـ legal rule بخصوص {topic}؟",
            QueryRegister.PROFESSIONAL,
            QueryLanguage.CODE_SWITCHED,
        ),
    }
    text, register, language = specs[variant]
    return text, query_type, register, language


def _anchor(unit_id: str, unit_type: str) -> CitationAnchor:
    return CitationAnchor(kind="section", label=unit_type, source_unit_id=unit_id)


def _item_from_evidence(
    corpus: EvaluationCorpus,
    category: QueryCategory,
    selections: tuple[tuple[CanonicalUnit, EvidenceDiscovery, RelevanceGrade], ...],
    *,
    ordinal: int,
    answer: str | None = None,
) -> DatasetItem:
    document_ids = tuple(sorted({unit.document_id for unit, _span, _grade in selections}))
    identity = tuple((span.start, span.end, unit.unit_id) for unit, span, _grade in selections)
    intent_id = deterministic_intent_id(category.value, document_ids, identity)
    evidence = tuple(
        EvidenceSpan(unit_id=unit.unit_id, start=span.start, end=span.end, grade=grade)
        for unit, span, grade in selections
    )
    groups = (EvidenceGroup(group_id=f"group-{intent_id[7:]}", spans=evidence),)
    query_text, query_type, register, language = _query_for(category, ordinal)
    return DatasetItem(
        query_id=deterministic_query_id(intent_id),
        intent_id=intent_id,
        query_text=query_text,
        language=language,
        register=register,
        category=category,
        query_type=query_type,
        jurisdiction="Saudi Arabia",
        temporal_scope="source-relative",
        creation_method=CreationMethod.DOCUMENT_DERIVED,
        answerability=Answerability.ANSWERABLE,
        difficulty=Difficulty.HARD
        if category is QueryCategory.MULTI_EVIDENCE
        else Difficulty.MEDIUM,
        source_document_ids=document_ids,
        evidence_groups=groups,
        gold_answer=_redact(
            answer
            or _answer_for(
                category, tuple(span.semantic_text for _unit, span, _grade in selections)
            )
        ),
        citation_anchors=tuple(
            _anchor(unit.unit_id, unit.unit_type.value) for unit, _span, _grade in selections
        ),
        dataset_version="phase6-retrieval-eval-draft-v2",
    )


_UNANSWERABLE_SPECS = (
    (UnanswerableReason.OUTSIDE_CORPUS_SCOPE, "ما حكم محكمة الأسرة في نزاع حضانة بين الوالدين؟"),
    (
        UnanswerableReason.OUTSIDE_CORPUS_SCOPE,
        "ما القاعدة الخاصة بترخيص منشأة طبية في مدينة غير سعودية؟",
    ),
    (UnanswerableReason.OUTSIDE_CORPUS_SCOPE, "ما أثر حكم جنائي أجنبي على هذه المسألة؟"),
    (UnanswerableReason.OUTSIDE_CORPUS_SCOPE, "كيف تعالج محكمة عمالية دعوى فصل تعسفي؟"),
    (UnanswerableReason.OUTSIDE_CORPUS_SCOPE, "ما الحكم في نزاع إيجار سكني غير تجاري؟"),
    (
        UnanswerableReason.AUTHORITATIVE_CURRENT_STATUTE_UNAVAILABLE,
        "ما النص الرسمي النافذ حالياً الذي يحدد ضريبة نشاط جديد؟",
    ),
    (
        UnanswerableReason.AUTHORITATIVE_CURRENT_STATUTE_UNAVAILABLE,
        "ما العقوبة الحالية في اللائحة الرسمية لمخالفة تقنية مستحدثة؟",
    ),
    (
        UnanswerableReason.AUTHORITATIVE_CURRENT_STATUTE_UNAVAILABLE,
        "ما المادة النافذة اليوم التي تنظم ترخيص منصة رقمية؟",
    ),
    (
        UnanswerableReason.AUTHORITATIVE_CURRENT_STATUTE_UNAVAILABLE,
        "ما الحد النظامي الحالي للرسوم على خدمة لم يرد نصها في المواد المتاحة؟",
    ),
    (
        UnanswerableReason.AUTHORITATIVE_CURRENT_STATUTE_UNAVAILABLE,
        "ما النص الرسمي المحدث بشأن حماية نموذج ذكاء اصطناعي؟",
    ),
    (
        UnanswerableReason.TEMPORAL_AMBIGUITY,
        "ما القاعدة السارية في عام 2035 بشأن العقد محل السؤال؟",
    ),
    (
        UnanswerableReason.TEMPORAL_AMBIGUITY,
        "هل كان الحكم النافذ قبل تعديل عام 1447هـ يقرر النتيجة نفسها؟",  # noqa: RUF001
    ),
    (UnanswerableReason.TEMPORAL_AMBIGUITY, "ما أثر تعديل مستقبلي على مدة الإجراء المذكور؟"),
    (UnanswerableReason.TEMPORAL_AMBIGUITY, "هل تغيرت القاعدة بعد تاريخ الحكم إلى صيغة أخرى؟"),
    (UnanswerableReason.TEMPORAL_AMBIGUITY, "ما النص الذي سيطبق إذا صدر تعديل لاحق على النظام؟"),
    (
        UnanswerableReason.INSUFFICIENT_SOURCE_EVIDENCE,
        "هل يثبت الاستحقاق دون معرفة العقد أو المبلغ أو مستند السداد؟",
    ),
    (
        UnanswerableReason.INSUFFICIENT_SOURCE_EVIDENCE,
        "هل تكفي الدعوى المجردة لإثبات المسؤولية دون وقائع قابلة للتحقق؟",
    ),
    (
        UnanswerableReason.INSUFFICIENT_SOURCE_EVIDENCE,
        "ما مقدار التعويض إذا لم يذكر السؤال الضرر أو سببه؟",
    ),
    (
        UnanswerableReason.INSUFFICIENT_SOURCE_EVIDENCE,
        "هل ثبت التسليم عندما لا يبين السؤال أي محضر أو إقرار؟",
    ),
    (
        UnanswerableReason.INSUFFICIENT_SOURCE_EVIDENCE,
        "ما النتيجة التي تلزم الطرفين إذا غاب منطوق الحكم عن المواد المتاحة؟",
    ),
    (
        UnanswerableReason.OUTSIDE_CORPUS_SCOPE,
        "ما رقم القضية المدنية التي لا تظهر ضمن السجلات التجارية المتاحة؟",
    ),
    (
        UnanswerableReason.OUTSIDE_CORPUS_SCOPE,
        "من هو مالك الشركة في سجل الشركات خارج نصوص الأحكام؟",
    ),
    (
        UnanswerableReason.OUTSIDE_CORPUS_SCOPE,
        "ما تاريخ تسجيل العلامة التجارية إذا لم يرد في الحكم؟",
    ),
    (
        UnanswerableReason.INSUFFICIENT_SOURCE_EVIDENCE,
        "هل كان الطرف حسن النية عندما لا تعرض المواد دليلاً على قصده؟",
    ),
    (UnanswerableReason.TEMPORAL_AMBIGUITY, "أي قانون سيصدر مستقبلاً ليحكم واقعة لم يحدد تاريخها؟"),
)

_UNANSWERABLE_VARIANT_TOPICS = (
    "حضانة بين والدين",
    "ترخيص منشأة طبية خارج السعودية",
    "أثر حكم جنائي أجنبي",
    "فصل عامل تعسفياً",
    "إيجار سكني غير تجاري",
    "ضريبة نشاط جديد",
    "مخالفة تقنية مستحدثة",
    "ترخيص منصة رقمية",
    "رسوم خدمة غير منصوص عليها",
    "حماية نموذج ذكاء اصطناعي",
    "قاعدة سارية في عام 2035",
    "أثر تعديل عام 1447هـ",  # noqa: RUF001
    "تعديل مستقبلي على مدة إجراء",
    "تغير القاعدة بعد تاريخ الحكم",
    "تعديل لاحق على النظام",
    "استحقاق بلا عقد أو مستند سداد",
    "إثبات مسؤولية بلا وقائع قابلة للتحقق",
    "تعويض بلا بيان للضرر أو سببه",
    "إثبات التسليم بلا محضر أو إقرار",
    "نتيجة بلا منطوق حكم",
    "رقم قضية مدنية خارج السجلات التجارية",
    "مالك شركة خارج نص الحكم",
    "تاريخ تسجيل علامة تجارية غير مذكور",
    "حسن نية بلا دليل على القصد",
    "قانون مستقبلي لواقعة مجهولة التاريخ",
)


def _unanswerable(corpus: EvaluationCorpus, document_id: str, ordinal: int) -> DatasetItem:
    reason, query = _UNANSWERABLE_SPECS[ordinal % len(_UNANSWERABLE_SPECS)]
    intent_id = deterministic_intent_id("unanswerable", (document_id,), (reason.value, ordinal))
    return DatasetItem(
        query_id=deterministic_query_id(intent_id),
        intent_id=intent_id,
        query_text=query,
        language=QueryLanguage.ARABIC,
        register=QueryRegister.FORMAL,
        category=QueryCategory.UNANSWERABLE,
        query_type=QueryType.ABSTENTION,
        jurisdiction="Saudi Arabia",
        temporal_scope="source-relative",
        creation_method=CreationMethod.DOCUMENT_DERIVED,
        answerability=Answerability.UNANSWERABLE,
        unanswerable_reason=reason,
        difficulty=Difficulty.HARD,
        source_document_ids=(document_id,),
        dataset_version="phase6-retrieval-eval-draft-v2",
    )


def _multi_opportunities(
    corpus: EvaluationCorpus,
) -> defaultdict[
    str, list[tuple[CanonicalUnit, EvidenceDiscovery, CanonicalUnit, EvidenceDiscovery]]
]:
    by_document: defaultdict[tuple[str, str], list[CanonicalUnit]] = defaultdict(list)
    for unit in corpus.units:
        by_document[(unit.provenance.source_id, unit.document_id)].append(unit)
    result: defaultdict[
        str, list[tuple[CanonicalUnit, EvidenceDiscovery, CanonicalUnit, EvidenceDiscovery]]
    ] = defaultdict(list)
    for (source, _document_id), units in sorted(by_document.items()):
        first: tuple[CanonicalUnit, EvidenceDiscovery] | None = None
        second: tuple[CanonicalUnit, EvidenceDiscovery] | None = None
        for unit in sorted(units, key=lambda row: (row.ordinal or 0, row.unit_id)):
            if unit.unit_type.value not in _MULTI_TYPES:
                continue
            discoveries = _discover_all(unit)
            spans = discoveries.get(QueryCategory.CASE_HOLDING, ())
            if not spans:
                spans = discoveries.get(QueryCategory.DEADLINE, ())
            if not spans:
                segments = [
                    segment
                    for segment in _segments(unit.text)
                    if _valid_semantic_text(segment.text)
                ]
                spans = tuple(
                    EvidenceDiscovery(
                        segment.start, segment.end, _clean_semantic_text(segment.text), 1
                    )
                    for segment in segments[:1]
                )
            if not spans:
                continue
            if first is None:
                first = (unit, spans[0])
            elif unit.unit_type.value != first[0].unit_type.value:
                second = (unit, spans[0])
                break
        if first and second:
            result[source].append((first[0], first[1], second[0], second[1]))
    return result


def _variants(base_items: tuple[DatasetItem, ...]) -> tuple[DatasetItem, ...]:
    by_category: defaultdict[QueryCategory, list[DatasetItem]] = defaultdict(list)
    for item in base_items:
        by_category[item.category].append(item)
    selected_parents: list[DatasetItem] = []
    for round_index in range(10):
        for category in BASE_TARGETS:
            rows = sorted(by_category[category], key=lambda item: item.query_id)
            if round_index < len(rows) and len(selected_parents) < 10:
                selected_parents.append(rows[round_index])
    result: list[DatasetItem] = []
    for variant_id in ("simple-ar", "egyptian-ar", "english", "code-switch"):
        for ordinal, base in enumerate(selected_parents):
            if base.category is QueryCategory.UNANSWERABLE:
                ordinal = next(
                    (
                        index
                        for index, (_reason, query) in enumerate(_UNANSWERABLE_SPECS)
                        if query == base.query_text
                    ),
                    ordinal,
                )
            else:
                ordinal = next(
                    (
                        index
                        for index, row in enumerate(
                            sorted(by_category[base.category], key=lambda item: item.query_id)
                        )
                        if row.intent_id == base.intent_id
                    ),
                    ordinal,
                )
            query_text, query_type, register, language = _query_for(
                base.category, ordinal, variant=variant_id
            )
            data = base.model_dump()
            data.update(
                {
                    "query_id": deterministic_query_id(base.intent_id, variant_id),
                    "variant_id": variant_id,
                    "base_intent_id": base.intent_id,
                    "query_text": query_text,
                    "query_type": query_type,
                    "language": language,
                    "register": register,
                    "creation_method": CreationMethod.ROBUSTNESS_VARIANT,
                    "dataset_version": "phase6-retrieval-eval-draft-v2",
                }
            )
            result.append(DatasetItem.model_validate(data))
    return tuple(result)


def _candidate_pool(corpus: EvaluationCorpus) -> tuple[DatasetItem, ...]:
    by_category: dict[QueryCategory, list[DatasetItem]] = defaultdict(list)
    opportunities: defaultdict[
        QueryCategory, defaultdict[str, list[tuple[CanonicalUnit, EvidenceDiscovery]]]
    ] = defaultdict(lambda: defaultdict(list))
    for unit in corpus.units:
        for category, discoveries in _discover_all(unit).items():
            if discoveries:
                opportunities[category][unit.provenance.source_id].append((unit, discoveries[0]))
    for category, target in CATEGORY_TARGETS.items():
        if category is QueryCategory.UNANSWERABLE:
            docs_by_source: defaultdict[str, list[str]] = defaultdict(list)
            for unit in sorted(
                corpus.units, key=lambda row: (row.provenance.source_id, row.document_id)
            ):
                if unit.document_id not in docs_by_source[unit.provenance.source_id]:
                    docs_by_source[unit.provenance.source_id].append(unit.document_id)
            docs = _proportional_source_order(docs_by_source, target)
            by_category[category].extend(
                _unanswerable(corpus, document_id, ordinal)
                for ordinal, document_id in enumerate(docs)
            )
            continue
        if category is QueryCategory.MULTI_EVIDENCE:
            multi_opps = _multi_opportunities(corpus)
            selected = _proportional_source_order(multi_opps, target)
            for ordinal, (first, first_span, second, second_span) in enumerate(selected):
                by_category[category].append(
                    _item_from_evidence(
                        corpus,
                        category,
                        (
                            (first, first_span, RelevanceGrade.REQUIRED),
                            (second, second_span, RelevanceGrade.REQUIRED),
                        ),
                        ordinal=ordinal,
                    )
                )
            continue
        selected = _proportional_source_order(opportunities[category], target)
        for ordinal, (unit, span) in enumerate(selected):
            by_category[category].append(
                _item_from_evidence(
                    corpus, category, ((unit, span, RelevanceGrade.REQUIRED),), ordinal=ordinal
                )
            )
        if len(by_category[category]) < target:
            raise ValueError(
                f"could not build {target} evidence-qualified candidates for {category.value}"
            )
    return tuple(item for category in CATEGORY_TARGETS for item in by_category[category])


def build_draft_candidates(
    corpus: EvaluationCorpus, *, output_root: Path = PRIVATE_ROOT
) -> DraftBuildResult:
    base_candidates = _candidate_pool(corpus)
    selected = tuple(
        item
        for category, target in BASE_TARGETS.items()
        for item in tuple(
            candidate for candidate in base_candidates if candidate.category is category
        )[:target]
    )
    variants = _variants(selected)
    if output_root:
        write_items_jsonl(output_root / "draft" / "base_candidates.jsonl", base_candidates)
        write_items_jsonl(
            output_root / "draft" / "selected_and_variants.jsonl", selected + variants
        )
    return DraftBuildResult(base_candidates, selected, variants)
