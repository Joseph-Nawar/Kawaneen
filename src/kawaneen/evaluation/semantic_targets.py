"""Evidence-derived semantic propositions for Phase 6 draft-v3."""

from __future__ import annotations

import re
from collections.abc import Sequence

from kawaneen.corpus.models import CanonicalUnit
from kawaneen.evaluation.candidates import EvidenceDiscovery
from kawaneen.evaluation.models import QueryCategory, SemanticTarget

_REFERENCE = re.compile(
    r"(?:المادة|المواد|الفقرة|البند)\s*(?:رقم\s*)?"
    r"(?:\(?\s*[0-9\u0660-\u0669]+(?:/[0-9\u0660-\u0669]+)?\s*\)?)"
)
_LAW_REFERENCE = re.compile(r"(?:نظام|اللائحة|قرار|المرسوم)[^:؛,.]{0,80}:\s*([0-9\u0660-\u0669]+)")
_TIME = re.compile(
    r"(?:خلال|لمدة|مدة|مهلة)\s+[^،؛.]{1,40}|"
    r"على\s+[^،؛.]{1,25}(?:يوم|أيام|شهر|أشهر|سنة|سنوات)|"
    r"(?:بتاريخ|في موعد أقصاه|قبل الجلسة)\s+[0-9\u0660-\u0669/هـ-]+"
)
_ACTION = re.compile(
    r"(?:السداد الكامل|سداد المبلغ|سداد|تقديم|إيداع|حضور|رفع|تنفيذ|تسليم|"
    r"الرد|إتمام|بدء|انتهاء|إثبات|التوقيع)"
)
_ACTOR = re.compile(
    r"(?:المحكمة|الدائرة|الجهة المختصة|الإدارة المختصة|الموظف|الكاتب|رئيس الجلسة|"
    r"المدعي|المدعى عليه|الطرف أو الأطراف)"
)
_POWER = re.compile(
    r"(?:غير مخول(?:ين)?|مخول(?:ين)?|تختص|يختص|يلتزم|يجب|يتعين|يحق|صلاحية|مسؤول|"
    r"أوجب|ألزم|من اختصاص|يملك)"
)
_DISPOSITION = re.compile(
    r"(?:قضت|حكمت|ألزمت|إلزام|رفضت|رفض|قبول|أثبتت|إثبات|عدم قبول|شطب|نقض|تأييد)"
)
_AMOUNT = re.compile(
    r"(?:[0-9\u0660-\u0669]{1,3}(?:,[0-9\u0660-\u0669]{3})+(?:\.[0-9]+)?|"
    r"[0-9\u0660-\u0669]+(?:\.[0-9]+)?)\s*(?:ريال|ريالاً|ريالًا)?"
)
_PII = re.compile(r"(?:\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b|(?<!\d)(?:\+?\d[\d ()-]{7,}\d)(?!\d))")
_EFFECT_CUE = re.compile(
    r"(?:اختصاص|يختص|يلتزم|يجب|يتعين|يحق|صلاحية|مسؤول|أوجب|ألزم|يجوز|لا يجوز|"
    r"يثبت|تثبت|يعد|يُعد|ينقضي|تنقضي|ينعقد|ينطبق|يسقط|يترتب|قضت|حكمت|"
    r"رفض|قبول|إلزام|عدم قبول|شطب|نقض|تأييد|صحة|بطلان)"
)
_GENERIC = (
    "يحدد النص المفهوم القانوني",
    "يتوقف تطبيق القاعدة",
    "الجهة أو الطرف المعني هو",
    "المرجع النظامي محدد",
    "النتيجة القضائية المبينة",
)
_STOPWORDS = frozenset(
    "من في على عن إلى أو و ثم أن إن هذا هذه ذلك التي الذي وقد كان تكون هو هي تم حيث"
)


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip(" \t\r\n'\"[](),؛:.؟")).strip()


def _tokens(value: str) -> set[str]:
    return {token for token in re.findall(r"[\wء-ي]{3,}", value) if token not in _STOPWORDS}


def _phrase_after(value: str, match: re.Match[str], limit: int = 80) -> str:
    tail = value[match.end() : match.end() + limit]
    tail = re.split(r"[،؛.]", tail, maxsplit=1)[0]
    return _clean(tail)


def _compact(value: str, limit: int = 10) -> str:
    """Keep a short semantic cue for a query without copying source text."""

    value = _PII.sub("", value)
    tokens = value.split()
    return " ".join(tokens[:limit]).strip(" ،؛:.")


def _shared_context(values: Sequence[str], limit: int = 5) -> str:
    if not values:
        return ""
    token_sets: list[set[str]] = [_tokens(value) for value in values]
    shared: set[str] = token_sets[0].copy() if token_sets else set()
    for token_set in token_sets[1:]:
        shared.intersection_update(token_set)
    ordered = [token for token in _clean(values[0]).split() if token in shared]
    return _compact(" ".join(dict.fromkeys(ordered)), limit)


def _action_phrase(value: str, match: re.Match[str]) -> str:
    tail = value[match.end() : match.end() + 45]
    tail = re.split(r"(?:،|؛|\.|\s+(?:خلال|لمدة|بتاريخ|في موعد))", tail, maxsplit=1)[0]
    return _clean(f"{match.group(0)} {_compact(tail, 5)}")


def _extract_exact(text: str) -> SemanticTarget | None:
    match = _REFERENCE.search(text)
    identifier = ""
    if match:
        number = re.search(r"[0-9\u0660-\u0669]+(?:/[0-9\u0660-\u0669]+)?", match.group(0))
        identifier = number.group(0) if number else ""
    if not identifier:
        law = _LAW_REFERENCE.search(text)
        if law:
            identifier = law.group(1)
    if not identifier:
        return None
    before = _clean(text[: match.start()] if match else text)
    effect = before
    if not effect or len(_tokens(effect)) < 2:
        effect = _clean(text[match.end() :] if match else text)
    subject_match = re.search(
        r"(?:يُعتبر|يعتبر|يعد|يُعد|يثبت|تثبت|تنقضي|ينقضي|يكون)\s+(.{2,60}?)(?=\s+(?:سند|بذلك|حسب|وفق|من|وتنقضي|وتنقضى)|$)",
        text,
    )
    subject = _clean(subject_match.group(1)) if subject_match else ""
    if not subject:
        noun = re.search(r"(الصلح|العقد|الدعوى|المحضر|الحكم|الطلب|المستند)", text)
        subject = noun.group(1) if noun else ""
    if not subject or len(_tokens(effect)) < 2 or not _EFFECT_CUE.search(effect):
        return None
    return SemanticTarget(
        category=QueryCategory.EXACT_PROVISION,
        proposition=f"{identifier}: {effect}",
        provision_identifier=identifier,
        subject=subject,
        effect=effect,
        context=_compact(effect, 6),
    )


def _extract_definition(text: str) -> SemanticTarget | None:
    pair = re.search(
        r"(?P<term>النظام|اللائحة|المجلس|المحكمة|الوزارة|الوزير|الوكالة|الصلح|"
        r"الإقرار|البيّنة|البينة|التاجر|الدعوى|الطرف أو الأطراف)\s*:\s*"
        r"(?P<definition>[^،؛.]{3,100})",
        text,
    )
    if pair is None:
        pair = re.search(
            r"(?:يقصد|المقصود)\s+ب(?P<term>ال?[\u0621-\u064A]{2,35})\s+"
            r"(?P<definition>[^،؛.]{8,100})",
            text,
        )
    if pair is None:
        return None
    term = _clean(pair.group("term"))
    definition = _clean(pair.group("definition"))
    if len(_tokens(term)) < 1 or len(_tokens(definition)) < 2:
        return None
    return SemanticTarget(
        category=QueryCategory.DEFINITION,
        proposition=f"{term}: {definition}",
        defined_term=term,
        definition=definition,
        context=_compact(definition, 5),
    )


def _extract_deadline(text: str) -> SemanticTarget | None:
    time_matches = list(_TIME.finditer(text))
    action_matches = list(_ACTION.finditer(text))
    if not time_matches or not action_matches:
        return None
    time_match, action_match = min(
        ((time, action) for time in time_matches for action in action_matches),
        key=lambda pair: (abs(pair[0].start() - pair[1].start()), -pair[0].start()),
    )
    if abs(time_match.start() - action_match.end()) > 100:
        return None
    action = _action_phrase(text, action_match)
    deadline = _clean(time_match.group(0))
    trigger_match = re.search(r"(?:بعد|من|عند|في جلسة)\s+[^،؛.]{2,45}", text)
    trigger = _clean(trigger_match.group(0)) if trigger_match else ""
    return SemanticTarget(
        category=QueryCategory.DEADLINE,
        proposition=f"{action} {deadline}",
        action=action,
        deadline=deadline,
        triggering_event=trigger,
        context=_compact(trigger or action, 6),
    )


def _extract_authority(text: str) -> SemanticTarget | None:
    actor_matches = list(_ACTOR.finditer(text))
    if not actor_matches:
        return None
    actor_match = actor_matches[0]
    power_match = _POWER.search(text, actor_match.end())
    if power_match is not None:
        preceding = [match for match in actor_matches if match.end() <= power_match.start()]
        if preceding:
            actor_match = preceding[-1]
    if power_match is None or power_match.start() - actor_match.end() > 45:
        reverse = re.search(
            r"(?P<power>من اختصاص|صلاحية|مسؤولية)\s+(?P<actor>المحكمة|الدائرة|"
            r"الجهة المختصة|الإدارة المختصة)",
            text,
        )
        if reverse is None:
            return None
        actor = _clean(reverse.group("actor"))
        power = _clean(reverse.group("power"))
        object_value = _clean(text[reverse.end() :])
    else:
        actor = _clean(actor_match.group(0))
        power = _clean(power_match.group(0))
        object_value = _phrase_after(text, power_match)
        if object_value.startswith(("ب", "في", "ل")):
            object_value = object_value[1:].strip()
    if not object_value or len(_tokens(object_value)) < 1:
        return None
    return SemanticTarget(
        category=QueryCategory.AUTHORITY,
        proposition=f"{actor} {power} {object_value}",
        actor=actor,
        power=power,
        object=object_value,
        context=_compact(object_value, 6),
    )


def _extract_condition(text: str) -> SemanticTarget | None:
    match = re.search(
        r"(?P<condition>(?:إذا|فإذا|يشترط|بشرط|ما لم|لا يجوز|يجوز|متى|حال كان)[^،؛.]{2,100}?)"
        r"\s+(?P<effect>(?:أثبت|أثبته|يثبت|يعد|يكون|صار|تصبح|تترتب|يترتب|تسلم|تنقضي|"
        r"لا يعتد)[^،؛.]{2,120})",
        text,
    )
    if match is None:
        return None
    condition = _clean(match.group("condition"))
    effect = _clean(match.group("effect"))
    if len(_tokens(condition)) < 2 or len(_tokens(effect)) < 2:
        return None
    return SemanticTarget(
        category=QueryCategory.CONDITIONS,
        proposition=f"{condition}؛ {effect}",
        condition=condition,
        effect=effect,
        context=_compact(condition, 7),
    )


def _extract_holding(text: str) -> SemanticTarget | None:
    disposition_matches = list(_DISPOSITION.finditer(text))
    if not disposition_matches:
        return None
    first_disposition = disposition_matches[0]
    amount_match = _AMOUNT.search(text)
    amount = amount_match.group(0) if amount_match else ""
    object_match = re.search(
        r"(?:بسداد|بتسليم|بإلزام|إلزام|رفضت|رفض|إثبات|أثبتت|قبول)\s+([^،؛.]+)",
        text,
    )
    object_value = _clean(object_match.group(1)) if object_match else ""
    if not object_value:
        object_value = _clean(text)
    disposition_end = len(text)
    punctuation = re.search(r"[،؛.]", text[first_disposition.end() :])
    if punctuation:
        disposition_end = first_disposition.end() + punctuation.start()
    disposition = _clean(text[first_disposition.start() : disposition_end])
    if len(_tokens(disposition)) < 2:
        disposition = _clean(first_disposition.group(0))
    context = _compact(object_value, 8)
    if context in {"هذه الدعوى", "الدعوى", "الطلب", "المبلغ"}:
        context = _compact(text, 8)
    if len(_tokens(object_value)) < 1:
        return None
    return SemanticTarget(
        category=QueryCategory.CASE_HOLDING,
        proposition=_clean(text),
        disposition=disposition,
        object=object_value,
        remedy=_clean(text),
        amount=amount,
        context=context,
    )


def _extract_multi(evidence_texts: Sequence[str], conclusion: str | None) -> SemanticTarget | None:
    if len(evidence_texts) < 2 or not conclusion:
        return None
    premises = tuple(_clean(value) for value in evidence_texts[:2])
    conclusion = _clean(conclusion)
    conclusion_tokens = _tokens(conclusion)
    if any(len(_tokens(premise)) < 2 for premise in premises):
        return None
    if any(not (_tokens(premise) & conclusion_tokens) for premise in premises):
        return None
    holding = _extract_holding(conclusion)
    context = _shared_context((premises[0], premises[1], conclusion), 5)
    return SemanticTarget(
        category=QueryCategory.MULTI_EVIDENCE,
        proposition=conclusion,
        premises=premises,
        conclusion=conclusion,
        disposition=holding.disposition if holding else "",
        object=holding.object if holding else _compact(conclusion, 8),
        remedy=holding.remedy if holding else "",
        amount=holding.amount if holding else "",
        context=context or _compact(conclusion, 5),
    )


def extract_semantic_target(
    category: QueryCategory,
    unit: CanonicalUnit,
    discovery: EvidenceDiscovery,
    *,
    evidence_texts: Sequence[str] = (),
    conclusion: str | None = None,
) -> SemanticTarget | None:
    if category is QueryCategory.CASE_HOLDING and unit.unit_type.value not in {"verdict", "ruling"}:
        return None
    text = _clean(discovery.semantic_text or unit.text[discovery.start : discovery.end])
    if category is QueryCategory.EXACT_PROVISION:
        return _extract_exact(text)
    if category is QueryCategory.DEFINITION:
        return _extract_definition(text)
    if category is QueryCategory.DEADLINE:
        return _extract_deadline(text)
    if category is QueryCategory.AUTHORITY:
        return _extract_authority(text)
    if category is QueryCategory.CONDITIONS:
        return _extract_condition(text)
    if category is QueryCategory.CASE_HOLDING:
        return _extract_holding(text)
    if category is QueryCategory.MULTI_EVIDENCE:
        return _extract_multi(evidence_texts, conclusion)
    return None


def validate_semantic_target(
    category: QueryCategory,
    target: SemanticTarget,
    evidence_texts: Sequence[str],
) -> bool:
    joined = " ".join(evidence_texts)
    if target.category is not category:
        return False
    if any(value.casefold() in _GENERIC for value in (target.proposition, target.effect)):
        return False
    fields = {
        QueryCategory.EXACT_PROVISION: (target.provision_identifier, target.effect),
        QueryCategory.DEFINITION: (target.defined_term, target.definition),
        QueryCategory.DEADLINE: (target.action, target.deadline),
        QueryCategory.AUTHORITY: (target.actor, target.power, target.object),
        QueryCategory.CONDITIONS: (target.condition, target.effect),
        QueryCategory.CASE_HOLDING: (target.disposition, target.object),
    }
    if category is QueryCategory.MULTI_EVIDENCE:
        if len(target.premises) < 2 or not target.conclusion:
            return False
        conclusion_tokens = _tokens(target.conclusion)
        return (
            all(
                len(_tokens(premise)) >= 2 and _tokens(premise) & conclusion_tokens
                for premise in target.premises
            )
            and target.conclusion.casefold() in joined.casefold()
        )
    return all(
        value
        and (
            value.casefold() in joined.casefold()
            or (
                category is QueryCategory.CASE_HOLDING
                and any(part.casefold() in joined.casefold() for part in value.split(" و"))
            )
        )
        for value in fields[category]
    )


def _english(value: str) -> str:
    replacements = (
        ("المحكمة", "the court"),
        ("الدائرة", "the circuit"),
        ("الموظف", "the employee"),
        ("المدعى عليه", "the defendant"),
        ("المدعي", "the claimant"),
        ("الصلح", "the settlement"),
        ("الدعوى", "the claim"),
        ("التوقيع", "the signature"),
        ("السداد", "payment"),
        ("المبلغ", "the amount"),
        ("المادة", "Article"),
        ("سنداً تنفيذياً", "an enforceable instrument"),
        ("سندًا تنفيذياً", "an enforceable instrument"),
        ("رفضت ما زاد", "rejected the excess"),
        ("رفض", "rejection"),
        ("إلزام", "an order to pay"),
    )
    result = value.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
    for arabic, english in replacements:
        result = result.replace(arabic, english)
    result = re.sub(r"[^A-Za-z0-9/,.?()' -]", " ", result)
    return re.sub(r"\s+", " ", result).strip()


def _generated(value: str) -> str:
    """Redact unnecessary case identifiers from generated representations."""

    return re.sub(r"\s+", " ", _PII.sub("[redacted]", value)).strip()


def render_semantic_query(target: SemanticTarget, variant: str | None = None) -> str:
    if variant == "english":
        if target.category is QueryCategory.EXACT_PROVISION:
            return _generated(
                "What effect did the judgment attach to provision "
                f"{_english(target.provision_identifier)}"
                f" regarding {_english(_compact(target.context or target.subject, 6))}?"
            )
        if target.category is QueryCategory.DEFINITION:
            return (
                f"How does the text define {_english(target.defined_term)} in this legal context?"
            )
        if target.category is QueryCategory.DEADLINE:
            return _generated(
                f"What deadline applied to {_english(target.action)}"
                f" in the context of {_english(_compact(target.context, 6))}?"
            )
        if target.category is QueryCategory.AUTHORITY:
            return (
                f"What power did {_english(target.actor)} have regarding "
                f"{_english(_compact(target.context or target.object, 6))}?"
            )
        if target.category is QueryCategory.CONDITIONS:
            return f"What condition produced {_english(_compact(target.context or target.effect))}?"
        if target.category is QueryCategory.CASE_HOLDING:
            return _generated(
                "What did the judgment decide about "
                f"{_english(_compact(target.context or target.object))}?"
            )
        return _generated(
            f"How did the stated premises support the conclusion about "
            f"{_english(_compact(target.context or target.object))}?"
        )
    if variant == "simple-ar":
        if target.category is QueryCategory.EXACT_PROVISION:
            return _generated(
                f"إيه الأثر اللي رتبه الحكم بموجب {target.provision_identifier} "
                f"عن {_compact(target.context or target.subject, 6)}؟"
            )
        if target.category is QueryCategory.DEFINITION:
            return f"يعني إيه {target.defined_term} في السياق ده؟"
        if target.category is QueryCategory.DEADLINE:
            return _generated(f"إمتى يتم {target.action} في سياق {_compact(target.context, 6)}؟")
        if target.category is QueryCategory.AUTHORITY:
            return _generated(
                f"مين له {target.power} بخصوص {_compact(target.context or target.object, 6)}؟"
            )
        if target.category is QueryCategory.CONDITIONS:
            return f"إيه الشرط اللي يترتب عليه {_compact(target.effect)}؟"
        if target.category is QueryCategory.CASE_HOLDING:
            return _generated(f"الحكم قضى بإيه بخصوص {_compact(target.context or target.object)}؟")
        return _generated(
            f"إزاي أدت الوقائع والأسباب إلى {target.disposition or 'النتيجة'} "
            f"بشأن {_compact(target.context)}؟"
        )
    if variant == "egyptian-ar":
        if target.category is QueryCategory.EXACT_PROVISION:
            return _generated(
                f"المحكمة رتبت إيه بموجب {target.provision_identifier} "
                f"بخصوص {_compact(target.context or target.subject, 6)}؟"
            )
        if target.category is QueryCategory.DEFINITION:
            return f"النص عرّف {target.defined_term} إزاي؟"
        if target.category is QueryCategory.DEADLINE:
            return _generated(
                f"الموعد كان إمتى لـ{target.action} في سياق {_compact(target.context, 6)}؟"
            )
        if target.category is QueryCategory.AUTHORITY:
            return _generated(
                f"مين المسؤول عن {_compact(target.context or target.object, 6)} وإيه صلاحيته؟"
            )
        if target.category is QueryCategory.CONDITIONS:
            return f"إيه الشرط اللي بيترتب عليه {_compact(target.effect)}؟"
        if target.category is QueryCategory.CASE_HOLDING:
            return _generated(f"المحكمة حكمت بإيه في {_compact(target.context or target.object)}؟")
        return _generated(f"إزاي الوقائع والأسباب أدت لـ{target.disposition or 'النتيجة'}؟")
    if variant == "code-switch":
        if target.category is QueryCategory.EXACT_PROVISION:
            return _generated(
                f"ما هو الـ legal effect الذي رتبه الحكم بموجب {target.provision_identifier} "
                f"بخصوص {_compact(target.context or target.subject, 6)}؟"
            )
        if target.category is QueryCategory.DEFINITION:
            return f"ما الـlegal meaning المقصود بـ{target.defined_term} في النص؟"  # noqa: RUF001
        if target.category is QueryCategory.DEADLINE:
            return _generated(
                f"ما هو الـdeadline لـ{target.action} في سياق {_compact(target.context, 6)}؟"  # noqa: RUF001
            )
        if target.category is QueryCategory.AUTHORITY:
            return _generated(
                f"مين عنده الـpower بخصوص {_compact(target.context or target.object, 6)}؟"  # noqa: RUF001
            )
        if target.category is QueryCategory.CONDITIONS:
            return f"ما هو الـcondition الذي يترتب عليه {_compact(target.effect)}؟"  # noqa: RUF001
        if target.category is QueryCategory.CASE_HOLDING:
            return _generated(
                f"ما هو الـholding في موضوع {_compact(target.context or target.object)}؟"  # noqa: RUF001
            )
        return _generated(
            f"إزاي الـpremises دعمت {target.disposition or 'النتيجة'} "  # noqa: RUF001
            f"في {_compact(target.context)}؟"
        )
    if target.category is QueryCategory.EXACT_PROVISION:
        return _generated(
            f"ما الأثر الذي رتبه الحكم استناداً إلى {target.provision_identifier}"
            f" بشأن {_compact(target.context or target.subject, 6)}؟"
        )
    if target.category is QueryCategory.DEFINITION:
        return f"ما المقصود بـ{target.defined_term} في هذا السياق القانوني؟"
    if target.category is QueryCategory.DEADLINE:
        return _generated(
            f"ما الموعد المحدد لـ{target.action} في سياق {_compact(target.context, 6)}؟"
        )
    if target.category is QueryCategory.AUTHORITY:
        return _generated(
            f"ما حدود {target.power} التي يقررها النص لـ{target.actor} بشأن "
            f"{_compact(target.context or target.object, 6)}؟"
        )
    if target.category is QueryCategory.CONDITIONS:
        return f"ما الشرط الذي يترتب عليه {_compact(target.context or target.effect)}؟"
    if target.category is QueryCategory.CASE_HOLDING:
        return _generated(f"ما الذي قضى به الحكم بشأن {_compact(target.context or target.object)}؟")
    return _generated(
        f"كيف أدت الوقائع والأسباب المبينة إلى {target.disposition or 'النتيجة'} "
        f"بشأن {_compact(target.context or target.object)}؟"
    )


def render_semantic_answer(target: SemanticTarget) -> str:
    if target.category is QueryCategory.EXACT_PROVISION:
        return _generated(
            f"استند الحكم إلى {target.provision_identifier} لتقرير أن {target.effect}."
        )
    if target.category is QueryCategory.DEFINITION:
        return _generated(f"عرّف النص {target.defined_term} بأنه {target.definition}.")
    if target.category is QueryCategory.DEADLINE:
        trigger = f" {target.triggering_event}" if target.triggering_event else ""
        return _generated(f"كان يتعين {target.action} في {target.deadline}{trigger}.")
    if target.category is QueryCategory.AUTHORITY:
        return _generated(f"قرر النص أن {target.actor} {target.power} بشأن {target.object}.")
    if target.category is QueryCategory.CONDITIONS:
        return _generated(f"عند تحقق الشرط: {target.condition}، يترتب الأثر: {target.effect}.")
    if target.category is QueryCategory.CASE_HOLDING:
        answer = f"قضى الحكم بـ{target.disposition} بشأن {target.object}"
        if target.amount and target.amount not in answer:
            answer += f"، وبمقدار {target.amount}"
        if "رفض" in target.remedy and "رفض" not in answer:
            answer += "، مع رفض الجزء الزائد"
        return _generated(answer + ".")
    return _generated(f"أدت المقدمتان إلى النتيجة الآتية: {target.conclusion}.")
