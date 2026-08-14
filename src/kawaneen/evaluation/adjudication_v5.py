"""Bounded application of the final external review to draft-v4."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from kawaneen.corpus.models import CanonicalUnit
from kawaneen.evaluation.adjudication_v4 import (
    _CATEGORIES,  # pyright: ignore[reportPrivateUsage]
    _base_v4_query,  # pyright: ignore[reportPrivateUsage]
    _clean,  # pyright: ignore[reportPrivateUsage]
    _evidence_texts,  # pyright: ignore[reportPrivateUsage]
    _extract_target,  # pyright: ignore[reportPrivateUsage]
    _tokens,  # pyright: ignore[reportPrivateUsage]
    clean_v4_text,
    validate_v4_semantic_contract,
)
from kawaneen.evaluation.candidates import proportional_source_order
from kawaneen.evaluation.corpus import EvaluationCorpus
from kawaneen.evaluation.models import (
    Answerability,
    CreationMethod,
    DatasetItem,
    EvidenceGroup,
    EvidenceSpan,
    QueryCategory,
    QueryLanguage,
    QueryRegister,
    RelevanceGrade,
    ReviewState,
    SemanticTarget,
    deterministic_intent_id,
    deterministic_query_id,
)
from kawaneen.evaluation.semantic_targets import render_semantic_answer

V5_VERSION = "phase6-retrieval-eval-draft-v5"
V5_PRIVATE_ROOT = Path("artifacts/private/phase6_evaluation/draft-v5")
_EXPECTED_DECISIONS = {"accept": 25, "correct": 138, "replace": 37, "regenerate_variant": 40}
_REPLACE_COUNTS = {
    QueryCategory.DEFINITION: 4,
    QueryCategory.AUTHORITY: 8,
    QueryCategory.MULTI_EVIDENCE: 25,
}
_TRANSLATIONS = {
    "المحكمة": "the court",
    "الدائرة": "the circuit",
    "المدعي": "the claimant",
    "المدعية": "the claimant",
    "المدعى عليه": "the defendant",
    "المدعى عليها": "the defendant",
    "الدعوى": "the claim",
    "المذكرة": "the memorandum",
    "إيداع": "filing",
    "تقديم": "submitting",
    "السداد": "payment",
    "التوقيع": "signing",
    "السبب": "the cause",
    "التأسيس": "establishment",
    "الشرط": "the condition",
    "الاختصاص": "jurisdiction",
    "الصلاحية": "authority",
    "المسؤولية": "responsibility",
    "العقد": "the contract",
    "المبلغ": "the amount",
    "التعويض": "compensation",
    "العلامة": "the trademark",
    "الجدية": "seriousness",
    "رفع الجلسة لموعد آخر": "adjourning the hearing to another date",
    "تقديم جوابه الكترونيا": "submitting the response electronically",
}
_TRANSLITERATION = re.compile(
    r"\b(?:aljdyh|rfa|aljlsh|almdk|altqdm|almswolyh|alathr|alshrt|alhkm)\b",
    re.IGNORECASE,
)
_ENGLISH_NAME = re.compile(r"\[Person Name\]|\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b")


@dataclass(frozen=True, slots=True)
class ExternalReviewDecisionV5:
    query_id: str
    intent_id: str
    base_intent_id: str | None
    category: str
    source: str
    variant_id: str | None
    decision: str
    evidence_action: str
    required_correction: str


@dataclass(frozen=True, slots=True)
class V5BuildResult:
    base_pool: tuple[DatasetItem, ...]
    bases: tuple[DatasetItem, ...]
    variants: tuple[DatasetItem, ...]
    mapping: tuple[dict[str, object], ...]
    replacement_reasons: tuple[dict[str, object], ...]
    multi_audit: Mapping[str, object]


def load_v4_final_adjudication(path: Path) -> tuple[ExternalReviewDecisionV5, ...]:
    rows: list[ExternalReviewDecisionV5] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        rows.append(
            ExternalReviewDecisionV5(
                query_id=str(record["query_id"]),
                intent_id=str(record["intent_id"]),
                base_intent_id=record.get("base_intent_id"),
                category=str(record["category"]),
                source=str(record.get("source", "")),
                variant_id=record.get("variant_id"),
                decision=str(
                    record.get("adjudicated_ai_decision") or record.get("primary_ai_decision")
                ),
                evidence_action=str(record.get("evidence_action", "preserve")),
                required_correction=str(record.get("required_correction", "")),
            )
        )
    if len(rows) != 240 or len({row.query_id for row in rows}) != 240:
        raise ValueError("v4 final adjudication must cover 240 unique records")
    if Counter(row.decision for row in rows) != Counter(_EXPECTED_DECISIONS):
        raise ValueError("unexpected v4 final adjudication counts")
    if sum(row.decision == "replace" and row.category == "multi_evidence" for row in rows) != 25:
        raise ValueError("all 25 multi-evidence bases must be replaced")
    if sum(row.evidence_action == "extend_same_unit" for row in rows) != 3:
        raise ValueError("expected exactly three same-unit extensions")
    return tuple(rows)


def neutralize_v5_case_issue(value: str) -> str:
    result = re.sub(r"\[Person Name\]", "", value)
    result = re.sub(
        r"(?:شركة|مصنع|مؤسسة)\s+[ء-ي]+(?:\s+[ء-ي]+){0,7}",
        lambda match: "الشركة" if match.group(0).startswith("شركة") else "المصنع",
        result,
    )
    result = re.sub(r"إلزام", "المطالبة", result)
    result = re.sub(r"رفض", "الموقف من", result)
    result = re.sub(r"(?:هوية وطنية|سجل تجاري)\s+رقم\s*[^،؛.]{0,35}", "", result)
    result = re.sub(r"\s+", " ", result).strip(" ،؛:.؟")
    return result or "النزاع التعاقدي محل الدعوى"


def _case_issue_from_document(item: DatasetItem, units: Mapping[str, CanonicalUnit]) -> str:
    preferred = {"facts", "events"}
    candidates = [
        unit.text
        for unit in units.values()
        if unit.document_id in item.source_document_ids and unit.unit_type.value in preferred
    ]
    markers = re.compile(r"(?:دعوى|مطالبة|عقد|اتفاق|بيع|إيجار|نزاع|طلب)")
    for text in candidates:
        sentences = re.split(r"[\n.!؟؛]+", text)
        ranked = sorted(
            (sentence.strip() for sentence in sentences if len(_tokens(sentence)) >= 8),
            key=lambda sentence: (not bool(markers.search(sentence)), -len(_tokens(sentence))),
        )
        for sentence in ranked:
            if re.search(r"(?:حكمت|قضت|ألزمت|رفضت|قررت الدائرة)", sentence):
                continue
            issue = neutralize_v5_case_issue(sentence)
            issue = re.sub(r"\(?[0-9٠-٩][0-9٠-٩/().، -]{1,}\)?", "المبلغ", issue)  # noqa: RUF001
            issue = re.sub(r"\s+", " ", issue).strip()
            if issue and issue != "النزاع التعاقدي محل الدعوى":
                return _clean(issue, 14)
    return "النزاع التعاقدي محل الدعوى"


def validate_v5_variant_query(text: str, variant: str) -> bool:
    if _TRANSLITERATION.search(text) or "Article Article" in text:
        return False
    if variant == "english":
        return bool(
            re.search(r"\b(?:what|how|which|why|did|does|was|were)\b", text, re.I)
            and not re.search(r"[ء-ي]", text)
            and not _ENGLISH_NAME.search(text)
        )
    if variant == "code-switch":
        return bool(
            re.search(r"[ء-ي]", text)
            and re.search(
                r"(?:^|[^A-Za-z])(?:legal|deadline|power|condition|holding|facts|court)(?:$|[^A-Za-z])",
                text,
                re.I,
            )
        )
    return bool(re.search(r"[ء-ي]", text))


def validate_v5_deadline_relation(text: str, action: str) -> bool:
    return bool(
        action.strip()
        and re.search(
            r"(?:خلال|لمدة|مهلة|فوراً|دفعة|دفعت|أقساط|على\s+\S+\s+(?:أشهر|شهر|أيام|يوم|سنوات|سنة))",
            text,
        )
    ) and not bool(re.search(r"(?:انعقدت|عقدت)\s+الجلسة\s+بتاريخ", text))


def validate_v5_authority_relation(text: str) -> bool:
    return "صالحة للفصل" not in text and bool(
        re.search(
            r"(?:تختص|الاختصاص|صلاحية|مسؤولية|يلتزم|يحق|مخول|من اختصاص|يجب أن تشتمل|يجب أن تتضمن)",
            text,
        )
    )


def validate_v5_condition_polarity(evidence: str, effect: str) -> bool:
    if "لا" in effect or "فلا" in effect:
        return True
    markers = re.compile(r"(?:إذا|متى|في حالة|في حال|ما دام|ما لم)")
    return not any(
        re.search(r"(?<!و)(?:فلا|لا )", evidence[match.end() : match.end() + 100])
        for match in markers.finditer(evidence)
    )


def validate_v5_ruling_span(text: str) -> bool:
    # A verb plus an actor alone is an incomplete disposition.  Require a
    # concrete object/result after the ruling verb so spans ending at the
    # boilerplate phrase "ألزمت المدعى عليه" fail closed.
    complete_verb = bool(
        re.search(
            r"(?:حكمت|قضت|ألزمت|رفضت|أثبتت|عدم قبول).{3,}"
            r"(?:بـ|ب|رفض|قبول|إلزام|إثبات|عدم|المبلغ|الدعوى|الطلب)",
            text,
        )
    )
    complete_disposition = bool(
        re.search(
            r"(?:ب?إلزام|ب?رفض|ب?قبول|ب?إثبات).{1,}?"
            r"(?:\d|ريال|الدعوى|الطلب|مطالبة|ما زاد)",
            text,
        )
    )
    return (complete_verb and len(_tokens(text)) >= 6) or (
        complete_disposition and len(_tokens(text)) >= 4
    )


def validate_v5_multi_necessity(
    *,
    conclusion: str,
    premises: tuple[str, ...],
    group_texts: tuple[str, ...],
    query: str,
    answer: str,
    grade2_group_count: int,
) -> bool:
    if grade2_group_count < 2 or len(premises) < 2 or len(group_texts) < 2:
        return False
    conclusion_norm = _clean(conclusion).casefold()
    if not conclusion_norm or conclusion_norm in query.casefold():
        return False
    if any(conclusion_norm in _clean(text).casefold() for text in group_texts):
        return False
    return (
        all(_clean(premise).casefold() in answer.casefold() for premise in premises[:2])
        and conclusion_norm in answer.casefold()
    )


def _english_fragment(value: str, fallback: str) -> str:
    result = value.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
    for arabic, english in sorted(_TRANSLATIONS.items(), key=lambda pair: -len(pair[0])):
        result = result.replace(arabic, english)
    result = re.sub(r"[^A-Za-z0-9/,.?()' -]", " ", result)
    result = re.sub(r"\s+", " ", result).strip()
    return result if result and not re.search(r"[ء-ي]", result) else fallback


def _english_identifier(value: str) -> str:
    identifier = _english_fragment(value, "the cited provision")
    if re.fullmatch(r"\d+(?:/\d+)?", identifier):
        return f"Article {identifier}"
    return identifier


def _english_case_issue(value: str) -> str:
    normalized = _clean(value).casefold()
    if "مخالصة" in normalized or "تسوية" in normalized:
        return "the settlement agreement"
    if "إيجار" in normalized:
        return "the lease dispute"
    if "بيع" in normalized:
        return "the sale dispute"
    if "عقد" in normalized or "اتفاق" in normalized:
        return "the contractual dispute"
    if "مطالبة" in normalized or "دعوى" in normalized:
        return "the claim"
    return "the underlying dispute"


def _english_context(value: str) -> str:
    normalized = _clean(value).casefold()
    mappings = (
        ("السجل التجاري", "the commercial registration"),
        ("سجل تجاري", "the commercial registration"),
        ("اختصاص", "the jurisdictional issue"),
        ("المصالحة", "the settlement requirement"),
        ("سبب", "the stated legal cause"),
        ("العقد", "the contract"),
        ("الدعوى", "the claim"),
        ("القرار", "the decision"),
    )
    for arabic, english in mappings:
        if arabic in normalized:
            return english
    return "the legal issue"


def _v5_query(target: SemanticTarget) -> str:
    if target.category is QueryCategory.DEFINITION:
        return f"ما المقصود بـ {target.defined_term} في هذا السياق القانوني؟"
    if target.category is QueryCategory.DEADLINE and "اقترحت" in target.proposition:
        return "ما جدول السداد الذي اقترحته المدعى عليها؟"
    if target.category is QueryCategory.CASE_HOLDING:
        return f"في النزاع المتعلق بـ{neutralize_v5_case_issue(target.context)}، ماذا قضت المحكمة؟"
    if target.category is QueryCategory.MULTI_EVIDENCE:
        prefixes = (
            "ما العنصران اللذان يلزم جمعهما للفصل في",
            "كيف تكاملت الواقعة والقاعدة في",
            "ما المقدمات اللازمة معاً لفهم",
        )
        index = int(hashlib.sha256(target.proposition.encode()).hexdigest()[:2], 16) % len(prefixes)
        topic = neutralize_v5_case_issue(target.context)
        if topic == "المحكمة":
            topic = (
                "الاختصاص النوعي محل النزاع"
                if any("اختصاص" in premise for premise in target.premises)
                else "المطالبة المالية محل النزاع"
            )
        return f"{prefixes[index]} {topic}؟"
    return _base_v4_query(target)


def _v5_answer(target: SemanticTarget) -> str:
    if target.category is QueryCategory.DEADLINE and "اقترحت" in target.proposition:
        return "اقترحت المدعى عليها سداد المبلغ على ستة أشهر تبدأ من أغسطس 2023."
    if target.category is QueryCategory.MULTI_EVIDENCE:
        return (
            f"تتمثل المقدمة الأولى في {target.premises[0]}، والثانية في "
            f"{target.premises[1]}؛ وباجتماعهما {target.conclusion}."
        )
    return render_semantic_answer(target)


def _evidence_identity(item: DatasetItem, target: SemanticTarget) -> tuple[str, str]:
    identity = tuple(
        (item.query_id, group.group_id, span.unit_id, span.start, span.end, target.proposition)
        for group in item.evidence_groups
        for span in group.spans
    )
    intent_id = deterministic_intent_id(item.category.value, item.source_document_ids, identity)
    return intent_id, deterministic_query_id(intent_id)


def _with_v5_target(item: DatasetItem, target: SemanticTarget) -> DatasetItem:
    intent_id, query_id = _evidence_identity(item, target)
    return item.model_copy(
        update={
            "query_id": query_id,
            "intent_id": intent_id,
            "variant_id": None,
            "base_intent_id": None,
            "query_text": _v5_query(target),
            "semantic_target": target,
            "gold_answer": _v5_answer(target),
            "creation_method": CreationMethod.DOCUMENT_DERIVED,
            "dataset_version": V5_VERSION,
            "review": item.review.model_copy(
                update={"state": ReviewState.DRAFT, "human_verified": False}
            ),
            "chunk_qrels": (),
        }
    )


def _extend_same_unit(
    item: DatasetItem, decision: ExternalReviewDecisionV5, units: Mapping[str, CanonicalUnit]
) -> DatasetItem:
    if len(item.evidence_groups) != 1 or len(item.evidence_groups[0].spans) != 1:
        raise ValueError(f"same-unit extension requires one span: {item.query_id}")
    old = item.evidence_groups[0].spans[0]
    text = units[old.unit_id].text
    if "Extend backward" in decision.required_correction:
        marker = "المقصود بالسبب في الدعوى هو السبب الذي أدى إلى قيام النزاع"
        start, end = text.rfind(marker, 0, old.start + 1), old.end
    elif "Article 133" in decision.required_correction:
        marker = "في جميع الأحوال; لا توجه اليمين إلى الشخصية الاعتبارية"
        start, marker_start = old.start, text.find(marker, old.end)
        end = marker_start + len(marker) if marker_start >= 0 else -1
    else:
        start_marker, end_marker = "حكمت الدائرة", "برفض هذه الدعوى"
        start = text.find(start_marker, max(0, old.start - 800), old.end + 800)
        end_start = text.find(end_marker, max(start, old.start), old.end + 1200)
        end = end_start + len(end_marker) if end_start >= 0 else -1
    if start < 0 or end <= start:
        raise ValueError(f"same-unit extension marker not found: {item.query_id}")
    span = EvidenceSpan(unit_id=old.unit_id, start=start, end=end, grade=old.grade)
    group = item.evidence_groups[0].model_copy(update={"spans": (span,)})
    return item.model_copy(update={"evidence_groups": (group,)})


def _target_for_item(item: DatasetItem, units: Mapping[str, CanonicalUnit]) -> SemanticTarget:
    evidence = _evidence_texts(item, units)
    joined = " ".join(evidence)
    if item.query_id == "query-dfeabc78e9a8056306b80668":
        return SemanticTarget(
            category=QueryCategory.EXACT_PROVISION,
            proposition="المادة (133): في جميع الأحوال لا توجه اليمين إلى الشخصية الاعتبارية",
            provision_identifier="المادة (133)",
            subject="الشخصية الاعتبارية",
            effect="في جميع الأحوال; لا توجه اليمين إلى الشخصية الاعتبارية",
            context="الشخصية الاعتبارية",
        )
    if item.query_id == "query-299d8f0136394740bb312b2a":
        return SemanticTarget(
            category=QueryCategory.DEFINITION,
            proposition=(
                "السبب في الدعوى: السبب الذي أدى إلى قيام النزاع؛ لا يقصد به الأدلة والمبررات"
            ),
            defined_term="السبب في الدعوى",
            definition="السبب الذي أدى إلى قيام النزاع؛ لا يقصد به الأدلة والمبررات",
            context="السبب في الدعوى",
        )
    target = _extract_target(item.category, joined, item.semantic_target) or item.semantic_target
    if target is None:
        raise ValueError(f"v5 could not extract a target: {item.query_id}")
    if (
        item.category is QueryCategory.AUTHORITY
        and "الاختصاص القضائي" in joined
        and "يجب بحثها" in joined
    ):
        target = target.model_copy(
            update={
                "actor": "المحكمة",
                "power": "يجب بحثها تلقائياً",
                "object": "الاختصاص القضائي",
                "proposition": "تبحث المحكمة الاختصاص القضائي تلقائياً",
            }
        )
    if item.category is QueryCategory.AUTHORITY and "نظام المرافعات الشرعية:202" in joined:
        target = target.model_copy(
            update={
                "actor": "الالتماس",
                "power": "يجب أن تشتمل الصـحيفة",
                "object": "بيان الحكم الملـتمس إعادة النظر فيه ورقمه وتاريخه",
                "proposition": "يجب أن تشتمل صحيفة الالتماس على بيان الحكم ورقمه وتاريخه",
            }
        )
    if item.category is QueryCategory.AUTHORITY and "نظام المحاكم التجارية:20" in joined:
        target = target.model_copy(
            update={
                "actor": "صحيفة الدعوى",
                "power": "يجب أن تتضمن صحيفة الدعوى",
                "object": "بيانات الأطراف وممثليهم وصفاتهم وعناوينهم",
                "proposition": (
                    "يجب أن تتضمن صحيفة الدعوى بيانات الأطراف وممثليهم وصفاتهم وعناوينهم"
                ),
            }
        )
    if (
        item.category is QueryCategory.CONDITIONS
        and "لا يعتد به" in joined
        and "ما لم يتحقق الشرط" in joined
    ):
        target = target.model_copy(
            update={
                "condition": "متى ما علق شيء على تحقق شرط معين",
                "effect": "لا يعتد به ولا تعد حجة ما لم يتحقق الشرط المعلق عليه الشيء",
                "proposition": "لا يعتد بالمعلق بالشرط ولا يعد حجة ما لم يتحقق الشرط",
            }
        )
    if (
        item.category is QueryCategory.DEADLINE
        and "سداد المبلغ على ستة أشهر" in joined
        and "عرض" in joined
    ):
        target = target.model_copy(
            update={
                "action": "سداد المبلغ",
                "deadline": "على ستة أشهر",
                "triggering_event": "تبدأ من أغسطس 2023",
                "proposition": "اقترحت المدعى عليها سداد المبلغ على ستة أشهر تبدأ من أغسطس 2023",
                "context": "جدول السداد المقترح",
            }
        )
    if (
        item.category is QueryCategory.CONDITIONS
        and "فلا يجوز للفرع المطالبة" in joined
        and "التعاقد كان مع الأصل" in joined
    ):
        target = target.model_copy(
            update={
                "condition": "ما دام التعاقد كان مع الأصل",
                "effect": "فلا يجوز للفرع المطالبة به",
                "proposition": "ما دام التعاقد كان مع الأصل فلا يجوز للفرع المطالبة به",
            }
        )
    if (
        item.category is QueryCategory.CONDITIONS
        and "في حالة الإخفاق في التوصل إلى الحل الودي" in joined
        and "يجب أن يحال ذلك النزاع" in joined
    ):
        target = target.model_copy(
            update={
                "condition": "في حالة الإخفاق في التوصل إلى الحل الودي",
                "effect": "يجب أن يحال ذلك النزاع إلى مكتب",
                "proposition": "عند إخفاق الحل الودي يجب إحالة النزاع إلى مكتب",
            }
        )
    if item.category is QueryCategory.CASE_HOLDING:
        issue = _case_issue_from_document(item, units)
        target = target.model_copy(update={"context": issue})
        if not validate_v5_ruling_span(joined):
            raise ValueError(f"incomplete ruling evidence: {item.query_id}")
    if item.category is QueryCategory.DEADLINE and not validate_v5_deadline_relation(
        joined, target.action
    ):
        raise ValueError(f"deadline relation failed: {item.query_id}")
    if item.category is QueryCategory.AUTHORITY and not validate_v5_authority_relation(joined):
        raise ValueError(f"authority relation failed: {item.query_id}")
    if item.category is QueryCategory.CONDITIONS and not validate_v5_condition_polarity(
        joined, target.effect
    ):
        raise ValueError(f"condition polarity failed: {item.query_id}")
    return target


def _multi_candidate(item: DatasetItem, units: Mapping[str, CanonicalUnit]) -> DatasetItem | None:
    spans = [
        span
        for group in item.evidence_groups
        for span in group.spans
        if span.grade == RelevanceGrade.REQUIRED
    ]
    if len(spans) < 2 or item.semantic_target is None:
        return None
    selected = spans[:2]
    group_texts = tuple(
        _clean(units[span.unit_id].text[span.start : span.end]) for span in selected
    )
    target = item.semantic_target
    conclusion = target.conclusion or target.proposition
    if any(_clean(conclusion).casefold() in text.casefold() for text in group_texts):
        return None
    target = target.model_copy(update={"context": neutralize_v5_case_issue(target.context)})
    groups = tuple(
        EvidenceGroup(
            group_id=f"group-{hashlib.sha256(f'{item.query_id}:{index}'.encode()).hexdigest()[:24]}",
            spans=(span,),
        )
        for index, span in enumerate(selected, start=1)
    )
    candidate = _with_v5_target(item.model_copy(update={"evidence_groups": groups}), target)
    if not validate_v5_multi_necessity(
        conclusion=target.conclusion,
        premises=target.premises,
        group_texts=group_texts,
        query=candidate.query_text,
        answer=candidate.gold_answer or "",
        grade2_group_count=2,
    ):
        return None
    return candidate


def _variant_query(target: SemanticTarget, variant: str) -> str:
    if variant == "english":
        if target.category is QueryCategory.EXACT_PROVISION:
            identifier = _english_identifier(
                re.sub(r"المادة\s*", "Article ", target.provision_identifier)
            )
            return f"What legal effect did {identifier} establish concerning the legal issue?"
        if target.category is QueryCategory.DEFINITION:
            defined = _english_fragment(target.defined_term, "the defined legal term")
            return f"How does the text define {defined} in this legal context?"
        if target.category is QueryCategory.DEADLINE:
            action = _english_fragment(target.action, "the required filing")
            return f"What deadline applied to {action}?"
        if target.category is QueryCategory.AUTHORITY:
            actor = _english_fragment(target.actor, "the identified actor")
            obj = _english_fragment(target.object, "the matter at issue")
            return f"What legal power or duty did {actor} have regarding {obj}?"
        if target.category is QueryCategory.CONDITIONS:
            condition = _english_context(target.condition)
            return f"What legal consequence followed when {condition} applied?"
        if target.category is QueryCategory.CASE_HOLDING:
            issue = _english_case_issue(target.context)
            return f"In the dispute concerning {issue}, what did the court decide?"
        context = _english_context(target.context)
        return (
            f"How did the facts and legal reasoning jointly support the conclusion about {context}?"
        )
    if variant == "code-switch":
        if target.category is QueryCategory.EXACT_PROVISION:
            identifier = _clean(target.provision_identifier or "النص محل السؤال", 5)
            return f"ما الـlegal effect لـ {identifier}؟"  # noqa: RUF001
        if target.category is QueryCategory.DEFINITION:
            return f"إيه الـlegal definition لـ {_clean(target.defined_term or target.subject, 5)}؟"  # noqa: RUF001
        if target.category is QueryCategory.DEADLINE:
            return f"إيه الـdeadline لـ {_clean(target.action, 6)}؟"  # noqa: RUF001
        if target.category is QueryCategory.AUTHORITY:
            actor = _english_fragment(target.actor, "the identified actor")
            obj = _english_context(target.object)
            return (
                f"إيه الـpower أو الـduty القانوني لـ {actor} "  # noqa: RUF001
                f"بخصوص {obj}؟"
            )
        if target.category is QueryCategory.CONDITIONS:
            condition = _english_context(target.condition)
            return f"إيه الـlegal consequence لما {condition} applies؟"  # noqa: RUF001
        if target.category is QueryCategory.MULTI_EVIDENCE:
            context = _english_context(target.context or target.subject)
            return f"إزاي the facts والقاعدة القانونية اتجمعوا في {context} عشان يوضحوا النتيجة؟"
        if target.category is QueryCategory.CASE_HOLDING:
            issue = _english_case_issue(target.context)
            return f"في النزاع بخصوص {issue}، what did the court decide؟"
        return f"إيه الـlegal issue بخصوص {_clean(target.context or target.subject, 6)}؟"  # noqa: RUF001
    if target.category is QueryCategory.CASE_HOLDING:
        return f"المحكمة حكمت بإيه في {neutralize_v5_case_issue(target.context)}؟"
    if variant == "simple-ar":
        if target.category is QueryCategory.EXACT_PROVISION:
            return f"ما أثر {_clean(target.provision_identifier or target.subject, 5)}؟"
        if target.category is QueryCategory.DEFINITION:
            return f"ما تعريف {_clean(target.defined_term or target.subject, 5)}؟"
        if target.category is QueryCategory.DEADLINE:
            return f"متى يجب {_clean(target.action, 6)}؟"
        if target.category is QueryCategory.AUTHORITY:
            return f"ما واجب {_clean(target.actor, 4)} بخصوص {_clean(target.object, 6)}؟"
        if target.category is QueryCategory.CONDITIONS:
            return f"ماذا يحدث إذا {_clean(target.condition, 7)}؟"
        if target.category is QueryCategory.MULTI_EVIDENCE:
            context = _clean(target.context or target.subject or "المسألة محل النزاع", 6)
            return f"كيف اجتمعت الواقعة والقاعدة في {context}؟"
        context = _clean(target.context or target.subject or target.action, 7)
        return f"ما المسألة القانونية في {context}؟"
    if target.category is QueryCategory.EXACT_PROVISION:
        return f"إيه أثر {_clean(target.provision_identifier or target.subject, 5)}؟"
    if target.category is QueryCategory.DEFINITION:
        return f"يعني إيه {_clean(target.defined_term or target.subject, 5)} قانونياً؟"
    if target.category is QueryCategory.DEADLINE:
        return f"المطلوب {_clean(target.action, 6)} خلال قد إيه؟"
    if target.category is QueryCategory.AUTHORITY:
        power = _clean(target.power or "الواجب القانوني", 4)
        obj = _clean(target.object, 6)
        return f"مين عليه {power} بخصوص {obj}؟"
    if target.category is QueryCategory.CONDITIONS:
        return f"إيه اللي يحصل لو {_clean(target.condition, 7)}؟"
    if target.category is QueryCategory.MULTI_EVIDENCE:
        context = _clean(target.context or target.subject or "المسألة محل النزاع", 6)
        return f"إزاي الواقعة والقاعدة اتجمعوا في {context}؟"
    context = _clean(target.context or target.subject or target.action, 7)
    return f"إيه اللي النص أو الحكم قرره بخصوص {context}؟"


def variant_query_v5(target: SemanticTarget, variant: str | None) -> str:
    """Return the deterministic v5 robustness query for diagnostics/tests."""

    return _variant_query(target, variant) if variant else _v5_query(target)


def _query_matches_generated(actual: str, expected: str) -> bool:
    if actual == expected:
        return True
    suffixes = (
        " في القضية محل السؤال؟",
        " في النزاع المعروض؟",
        " in this matter?",
        " in the case at issue?",
        " في this matter؟",
    )
    return any(actual == expected.rstrip("؟") + suffix for suffix in suffixes)


def _deduplicate_queries(items: tuple[DatasetItem, ...]) -> tuple[DatasetItem, ...]:
    seen: set[str] = set()
    suffixes = {
        QueryLanguage.ARABIC: (" في القضية محل السؤال؟", " في النزاع المعروض؟"),
        QueryLanguage.ENGLISH: (" in this matter?", " in the case at issue?"),
        QueryLanguage.CODE_SWITCHED: (" في this matter؟", " في the case at issue؟"),
    }
    result: list[DatasetItem] = []
    for item in sorted(
        items, key=lambda row: (row.query_text, row.intent_id, row.variant_id or "")
    ):
        key = _clean(item.query_text).casefold()
        if key in seen:
            options = suffixes[item.language]
            index = 0
            while True:
                suffix = options[index % len(options)]
                candidate = item.query_text.rstrip("؟?") + suffix
                candidate_key = _clean(candidate).casefold()
                if candidate_key not in seen:
                    item = item.model_copy(update={"query_text": candidate})
                    key = candidate_key
                    break
                index += 1
        seen.add(key)
        result.append(item)
    return tuple(sorted(result, key=lambda row: row.intent_id))


def validate_v5_item(item: DatasetItem, evidence_texts: tuple[str, ...]) -> bool:
    """Validate the v5 semantic contract without consulting retrieval output."""

    if item.answerability is Answerability.UNANSWERABLE:
        return not item.gold_answer and not item.evidence_groups and not item.chunk_qrels
    target = item.semantic_target
    if target is None:
        return False
    if item.creation_method is CreationMethod.ROBUSTNESS_VARIANT:
        if not _query_matches_generated(
            item.query_text, _variant_query(target, item.variant_id or "")
        ):
            return False
        if item.gold_answer != _v5_answer(target):
            return False
        if item.category is QueryCategory.MULTI_EVIDENCE:
            return validate_v5_multi_necessity(
                conclusion=target.conclusion,
                premises=target.premises,
                group_texts=tuple(clean_v4_text(value) for value in evidence_texts),
                query=item.query_text,
                answer=item.gold_answer or "",
                grade2_group_count=sum(
                    1
                    for group in item.evidence_groups
                    if any(span.grade == RelevanceGrade.REQUIRED for span in group.spans)
                ),
            )
        joined = " ".join(clean_v4_text(value).casefold() for value in evidence_texts)
        fields = {
            QueryCategory.EXACT_PROVISION: (target.provision_identifier, target.effect),
            QueryCategory.DEFINITION: (target.defined_term, target.definition),
            QueryCategory.DEADLINE: (target.action, target.deadline),
            QueryCategory.AUTHORITY: (target.actor, target.power, target.object),
            QueryCategory.CONDITIONS: (target.condition, target.effect),
            QueryCategory.CASE_HOLDING: (target.disposition, target.object),
        }.get(item.category, ())
        return all(value.strip() and clean_v4_text(value).casefold() in joined for value in fields)
    if item.category is QueryCategory.MULTI_EVIDENCE:
        group_texts = tuple(clean_v4_text(value) for value in evidence_texts)
        return (
            validate_v5_multi_necessity(
                conclusion=target.conclusion,
                premises=target.premises,
                group_texts=group_texts,
                query=item.query_text,
                answer=item.gold_answer or "",
                grade2_group_count=sum(
                    1
                    for group in item.evidence_groups
                    if any(span.grade == RelevanceGrade.REQUIRED for span in group.spans)
                ),
            )
            and item.gold_answer == _v5_answer(target)
            and item.query_text == _v5_query(target)
        )
    if item.category is QueryCategory.DEFINITION:
        joined = " ".join(clean_v4_text(value).casefold() for value in evidence_texts)
        definition_tokens = set(_tokens(target.definition))
        covered = sum(token.casefold() in joined for token in definition_tokens)
        return (
            target.defined_term.casefold() in joined
            and covered >= max(1, int(len(definition_tokens) * 0.7))
            and item.gold_answer == _v5_answer(target)
            and _query_matches_generated(item.query_text, _v5_query(target))
        )
    if item.category is QueryCategory.CASE_HOLDING:
        joined = " ".join(clean_v4_text(value).casefold() for value in evidence_texts)
        disposition_tokens = set(_tokens(target.disposition))
        covered = sum(token.casefold() in joined for token in disposition_tokens)
        return (
            covered >= max(3, int(len(disposition_tokens) * 0.7))
            and validate_v5_ruling_span(" ".join(evidence_texts))
            and not any(term in item.query_text for term in ("رفض", "إلزام"))
            and item.gold_answer == _v5_answer(target)
            and _query_matches_generated(item.query_text, _v5_query(target))
        )
    valid = validate_v4_semantic_contract(
        item.category,
        target,
        item.query_text,
        item.gold_answer or "",
        tuple(clean_v4_text(value) for value in evidence_texts),
    )
    return valid and _query_matches_generated(item.query_text, _v5_query(target))


def _build_variants(
    bases: tuple[DatasetItem, ...], preferred_parent_intents: tuple[str, ...] = ()
) -> tuple[DatasetItem, ...]:
    answerable = [item for item in bases if item.answerability is Answerability.ANSWERABLE]
    by_category: defaultdict[QueryCategory, list[DatasetItem]] = defaultdict(list)
    for item in sorted(answerable, key=lambda row: row.intent_id):
        by_category[item.category].append(item)
    by_intent = {item.intent_id: item for item in answerable}
    parents: list[DatasetItem] = [
        by_intent[intent] for intent in preferred_parent_intents if intent in by_intent
    ]
    if len(parents) != 10:
        parents = []
        index = 0
        while len(parents) < 10:
            for category in _CATEGORIES:
                if index < len(by_category[category]):
                    parents.append(by_category[category][index])
                    if len(parents) == 10:
                        break
            index += 1
    specs = (
        ("simple-ar", QueryLanguage.ARABIC, QueryRegister.SIMPLE),
        ("egyptian-ar", QueryLanguage.ARABIC, QueryRegister.EGYPTIAN),
        ("english", QueryLanguage.ENGLISH, QueryRegister.PROFESSIONAL),
        ("code-switch", QueryLanguage.CODE_SWITCHED, QueryRegister.PROFESSIONAL),
    )
    variants: list[DatasetItem] = []
    for variant_id, language, register in specs:
        for base in parents:
            if base.semantic_target is None:
                raise ValueError("v5 variant parent lacks semantic target")
            query = _variant_query(base.semantic_target, variant_id)
            if not validate_v5_variant_query(query, variant_id):
                raise ValueError(f"invalid v5 variant: {query}")
            variants.append(
                base.model_copy(
                    update={
                        "query_id": deterministic_query_id(base.intent_id, variant_id),
                        "variant_id": variant_id,
                        "base_intent_id": base.intent_id,
                        "query_text": query,
                        "language": language,
                        "register": register,
                        "creation_method": CreationMethod.ROBUSTNESS_VARIANT,
                        "dataset_version": V5_VERSION,
                        "review": base.review.model_copy(
                            update={"state": ReviewState.DRAFT, "human_verified": False}
                        ),
                    }
                )
            )
    return tuple(variants)


def _source(item: DatasetItem, units: Mapping[str, CanonicalUnit]) -> str:
    for document_id in item.source_document_ids:
        for unit in units.values():
            if unit.document_id == document_id:
                return str(unit.provenance.source_id)
    return "unknown"


def _mapping(
    old: DatasetItem,
    new: DatasetItem,
    decision: ExternalReviewDecisionV5,
    action: str,
    source: str | None,
) -> dict[str, object]:
    return {
        "old_query_id": old.query_id,
        "old_intent_id": old.intent_id,
        "old_category": old.category.value,
        "decision": decision.decision,
        "new_query_id": new.query_id,
        "new_intent_id": new.intent_id,
        "new_category": new.category.value,
        "evidence_action": action,
        "evidence_preserved": action in {"preserve", "inherit_final_base"},
        "evidence_extended": action == "extend_same_unit",
        "evidence_replaced": action == "replace",
        "replacement_source": source,
        "replacement_document_ids": list(new.source_document_ids) if action == "replace" else [],
        "query_changed": old.query_text != new.query_text,
        "answer_changed": old.gold_answer != new.gold_answer,
        "semantic_target_changed": old.semantic_target != new.semantic_target,
        "qrels_changed": old.chunk_qrels != new.chunk_qrels,
    }


def apply_v4_adjudication(
    v4_items: tuple[DatasetItem, ...],
    corpus: EvaluationCorpus,
    decisions: tuple[ExternalReviewDecisionV5, ...],
    pool: tuple[DatasetItem, ...],
) -> V5BuildResult:
    by_qid = {item.query_id: item for item in v4_items}
    if set(by_qid) != {row.query_id for row in decisions}:
        raise ValueError("v5 adjudication does not cover exactly the v4 records")
    decision_by_qid = {row.query_id: row for row in decisions}
    units = {unit.unit_id: unit for unit in corpus.units}
    bases_v4 = tuple(item for item in v4_items if item.variant_id is None)
    variants_v4 = tuple(item for item in v4_items if item.variant_id is not None)
    accepted = [item for item in bases_v4 if decision_by_qid[item.query_id].decision == "accept"]
    corrected: list[DatasetItem] = []
    mapping: list[dict[str, object]] = []
    old_to_new_intent: dict[str, str] = {}
    for item in bases_v4:
        decision = decision_by_qid[item.query_id]
        if decision.decision == "accept":
            updated = item.model_copy(
                update={
                    "dataset_version": V5_VERSION,
                    "review": item.review.model_copy(
                        update={"state": ReviewState.DRAFT, "human_verified": False}
                    ),
                }
            )
            corrected.append(updated)
            old_to_new_intent[item.intent_id] = updated.intent_id
            mapping.append(_mapping(item, updated, decision, "preserve", None))
        elif decision.decision == "correct":
            working = (
                _extend_same_unit(item, decision, units)
                if decision.evidence_action == "extend_same_unit"
                else item
            )
            updated = _with_v5_target(working, _target_for_item(working, units))
            corrected.append(updated)
            old_to_new_intent[item.intent_id] = updated.intent_id
            mapping.append(_mapping(item, updated, decision, decision.evidence_action, None))
    selected_ids = {item.query_id for item in bases_v4}
    candidates: defaultdict[QueryCategory, list[DatasetItem]] = defaultdict(list)
    for candidate in pool:
        if (
            candidate.variant_id is None
            and candidate.query_id not in selected_ids
            and candidate.answerability is Answerability.ANSWERABLE
        ):
            candidates[candidate.category].append(candidate)
    replace_rows: defaultdict[str, list[ExternalReviewDecisionV5]] = defaultdict(list)
    for decision in decisions:
        if decision.decision == "replace":
            replace_rows[decision.category].append(decision)
    replacements: list[DatasetItem] = []
    rejection_reasons: list[dict[str, object]] = []
    attempts: Counter[str] = Counter()
    qualified: Counter[str] = Counter()
    for category, count in _REPLACE_COUNTS.items():
        by_source: defaultdict[str, list[DatasetItem]] = defaultdict(list)
        for candidate in candidates[category]:
            by_source[_source(candidate, units)].append(candidate)
        ordered = proportional_source_order(
            {
                key: tuple(sorted(value, key=lambda row: row.query_id))
                for key, value in by_source.items()
            },
            count,
        )
        ordered_ids = {candidate.query_id for candidate in ordered}
        ordered.extend(
            candidate
            for candidate in sorted(candidates[category], key=lambda row: row.query_id)
            if candidate.query_id not in ordered_ids
        )
        chosen: list[DatasetItem] = []
        for candidate in ordered:
            if len(chosen) == count:
                break
            if category is QueryCategory.MULTI_EVIDENCE:
                source = _source(candidate, units)
                attempts[source] += 1
                updated = _multi_candidate(candidate, units)
                if updated is None:
                    rejection_reasons.append(
                        {
                            "query_id": candidate.query_id,
                            "category": category.value,
                            "source": source,
                            "reason": "less_than_two_nonredundant_grade2_groups",
                        }
                    )
                    continue
                qualified[source] += 1
            else:
                updated = _with_v5_target(candidate, _target_for_item(candidate, units))
            chosen.append(updated)
        if len(chosen) != count:
            raise ValueError(f"v5 replacement pool cannot satisfy {category.value}")
        old_rows = sorted(replace_rows[category.value], key=lambda row: row.query_id)
        for old_decision, updated in zip(
            old_rows, sorted(chosen, key=lambda row: row.query_id), strict=True
        ):
            old_item = by_qid[old_decision.query_id]
            replacements.append(updated)
            old_to_new_intent[old_item.intent_id] = updated.intent_id
            mapping.append(
                _mapping(old_item, updated, old_decision, "replace", _source(updated, units))
            )
    bases = tuple(sorted(corrected + replacements, key=lambda item: item.intent_id))
    expected = Counter(
        {
            QueryCategory.EXACT_PROVISION: 30,
            QueryCategory.DEFINITION: 25,
            QueryCategory.DEADLINE: 20,
            QueryCategory.AUTHORITY: 20,
            QueryCategory.CONDITIONS: 30,
            QueryCategory.MULTI_EVIDENCE: 25,
            QueryCategory.CASE_HOLDING: 25,
            QueryCategory.UNANSWERABLE: 25,
        }
    )
    if (
        len(bases) != 200
        or Counter(item.category for item in bases) != expected
        or len(accepted) != 25
    ):
        raise ValueError("v5 base quotas are incorrect")
    old_variant_parents = tuple(
        dict.fromkeys(
            old_to_new_intent[item.base_intent_id or item.intent_id]
            for item in sorted(variants_v4, key=lambda row: row.query_id)
            if (item.base_intent_id or item.intent_id) in old_to_new_intent
        )
    )
    variants = _build_variants(bases, old_variant_parents)
    final_items = _deduplicate_queries(bases + variants)
    bases = tuple(item for item in final_items if item.variant_id is None)
    variants = tuple(item for item in final_items if item.variant_id is not None)
    base_by_intent = {item.intent_id: item for item in bases}
    variant_by_key = {(item.base_intent_id, item.variant_id): item for item in variants}
    mapping = []
    for old in v4_items:
        decision = decision_by_qid[old.query_id]
        parent = old_to_new_intent.get(old.base_intent_id or old.intent_id)
        if old.variant_id is not None:
            new = variant_by_key[(parent, old.variant_id)]
            action = "inherit_final_base"
            source = None
        else:
            new = base_by_intent[parent or ""]
            action = "replace" if decision.decision == "replace" else decision.evidence_action
            source = _source(new, units) if action == "replace" else None
        mapping.append(_mapping(old, new, decision, action, source))
    if len(mapping) != 240:
        raise ValueError("v5 mapping must cover 240 records")
    mapping.sort(key=lambda row: str(row["old_query_id"]))
    return V5BuildResult(
        tuple(pool),
        bases,
        variants,
        tuple(mapping),
        tuple(rejection_reasons),
        {
            "attempts_by_source": dict(sorted(attempts.items())),
            "qualified_by_source": dict(sorted(qualified.items())),
            "retrieval_scores_used": False,
        },
    )


def v5_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
