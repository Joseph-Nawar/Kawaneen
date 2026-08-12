"""Policy-independent private source-span challenge construction."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from kawaneen.chunking.corpus import Phase5Corpus
from kawaneen.chunking.models import SourceSpan
from kawaneen.chunking.structure import section_units
from kawaneen.corpus.models import CanonicalUnit

PRIVATE_ROOT = Path("artifacts/private/phase5_chunking")
SLICES = (
    "local_passage",
    "long_legal_section",
    "multi_paragraph_evidence",
    "structural_boundary_proximity",
    "fixed_window_boundary_stress",
    "parent_context_evidence",
)
_TOKEN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


@dataclass(frozen=True, slots=True)
class ChunkChallengeItem:
    query_id: str
    slice_name: str
    query_text: str
    document_id: str
    gold_spans: tuple[SourceSpan, ...]
    construction_version: str


@dataclass(frozen=True, slots=True)
class PrivateChunkChallenge:
    seed: int
    construction_version: str
    items: tuple[ChunkChallengeItem, ...]
    qrels: dict[str, tuple[SourceSpan, ...]]


def _token_spans(text: str) -> tuple[tuple[int, int], ...]:
    return tuple((match.start(), match.end()) for match in _TOKEN.finditer(text))


def _window(unit: CanonicalUnit, index: int, width: int) -> SourceSpan:
    tokens = _token_spans(unit.text)
    if not tokens:
        raise ValueError("challenge source unit has no lexical tokens")
    start_token = (index * 17) % max(len(tokens) - width + 1, 1)
    return SourceSpan(
        unit.unit_id, tokens[start_token][0], tokens[min(start_token + width, len(tokens)) - 1][1]
    )


def _challenge_unit(
    units: tuple[CanonicalUnit, ...], item_index: int, slice_index: int
) -> CanonicalUnit:
    return units[(item_index * 11 + slice_index * 7) % len(units)]


def build_private_chunk_challenge(
    units: Iterable[CanonicalUnit],
    corpus: Phase5Corpus,
    *,
    seed: int = 20260812,
    output_root: Path = PRIVATE_ROOT / "challenge",
) -> PrivateChunkChallenge:
    """Build six balanced source-span slices without invoking chunk strategies."""

    selected = tuple(
        sorted(
            section_units(units),
            key=lambda unit: (unit.document_id, unit.ordinal or 0, unit.unit_id),
        )
    )
    if not selected or not {unit.document_id for unit in selected}.issubset(corpus.document_ids):
        raise ValueError("challenge units must belong to the frozen corpus")
    by_text: defaultdict[str, list[CanonicalUnit]] = defaultdict(list)
    for unit in selected:
        by_text[unit.text].append(unit)
    items: list[ChunkChallengeItem] = []
    for slice_index, slice_name in enumerate(SLICES):
        for item_index in range(30):
            unit = _challenge_unit(selected, item_index, slice_index)
            width = 12 if slice_name in {"long_legal_section", "parent_context_evidence"} else 8
            span = _window(unit, item_index + slice_index, width)
            spans = [span]
            if slice_name == "structural_boundary_proximity":
                same_document = [
                    candidate for candidate in selected if candidate.document_id == unit.document_id
                ]
                next_unit = next(
                    (
                        candidate
                        for candidate in same_document
                        if (candidate.ordinal or 0) > (unit.ordinal or 0)
                    ),
                    None,
                )
                if next_unit is not None:
                    spans.append(_window(next_unit, item_index, 4))
            if slice_name == "multi_paragraph_evidence" and "\n" in unit.text:
                second_start = unit.text.find("\n") + 1
                if second_start < len(unit.text):
                    spans.append(SourceSpan(unit.unit_id, second_start, len(unit.text)))
            # Exact duplicate source forms are represented as multi-relevant spans.
            for duplicate in by_text[unit.text]:
                if duplicate.unit_id != unit.unit_id:
                    spans.append(SourceSpan(duplicate.unit_id, span.start, span.end))
            unique_spans = tuple(dict.fromkeys(spans))
            query_parts = [unit.text[span.start : span.end] for span in unique_spans[:2]]
            query_text = " ".join(query_parts)
            digest = hashlib.sha256(query_text.encode()).hexdigest()[:10]
            query_id = f"phase5-{seed:08d}-{slice_index:02d}-{item_index:02d}-{digest}"
            items.append(
                ChunkChallengeItem(
                    query_id=query_id,
                    slice_name=slice_name,
                    query_text=query_text,
                    document_id=unit.document_id,
                    gold_spans=unique_spans,
                    construction_version="phase5-chunk-challenge-v1",
                )
            )
    challenge = PrivateChunkChallenge(
        seed=seed,
        construction_version="phase5-chunk-challenge-v1",
        items=tuple(items),
        qrels={item.query_id: item.gold_spans for item in items},
    )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "challenge_items.jsonl").write_text(
        "".join(
            json.dumps(asdict(item), ensure_ascii=False, sort_keys=True) + "\n" for item in items
        ),
        encoding="utf-8",
    )
    (output_root / "qrels.json").write_text(
        json.dumps(
            {
                query_id: [asdict(span) for span in spans]
                for query_id, spans in challenge.qrels.items()
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return challenge


def validate_challenge_independence(challenge: PrivateChunkChallenge) -> bool:
    return (
        challenge.construction_version == "phase5-chunk-challenge-v1"
        and len(challenge.items) == 180
        and len({item.query_id for item in challenge.items}) == 180
        and not any(
            hasattr(item, "strategy_id") or hasattr(item, "policy_id") for item in challenge.items
        )
    )
