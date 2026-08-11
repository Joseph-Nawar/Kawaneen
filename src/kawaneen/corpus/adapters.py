"""Phase 2 source adapters for exact-text canonical views."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, cast

import pyarrow.parquet as _parquet

from kawaneen.acquisition.specs import load_specifications
from kawaneen.acquisition.storage import source_root
from kawaneen.corpus.ids import canonical_id
from kawaneen.corpus.models import RawAccounting, SourceFragment, SourceProvenance, UnitType
from kawaneen.corpus.serialization import (
    canonical_root,
    documents_schema,
    fragments_schema,
    units_schema,
    write_parquet,
)

pq: Any = cast(Any, _parquet)


def _provenance(
    source_id: str, version: str, path: str, row: int, field: str, split: str = ""
) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "source_version": version,
        "source_path": path,
        "source_row": row,
        "source_field": field,
        "split": split,
    }


def _document_row(
    document_id: str,
    kind: str,
    title: str,
    provenance: dict[str, Any],
    **extra: Any,
) -> dict[str, Any]:
    return {
        "document_id": document_id,
        "kind": kind,
        "title": title,
        **provenance,
        "raw_article_label": extra.get("raw_article_label", ""),
        "derived_article_ordinal": extra.get("derived_article_ordinal"),
        "reconstruction_status": extra.get("reconstruction_status", ""),
        "source_metadata_json": json.dumps(extra.get("source_metadata", {}), sort_keys=True),
    }


def _unit_row(
    document_id: str,
    unit_id: str,
    unit_type: str,
    text: str,
    provenance: dict[str, Any],
    ordinal: int,
) -> dict[str, Any]:
    return {
        "unit_id": unit_id,
        "document_id": document_id,
        "unit_type": unit_type,
        "text": text,
        **provenance,
        "ordinal": ordinal,
    }


def _build_alarb(
    raw_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], RawAccounting]:
    spec = load_specifications()["alarb"]
    documents: list[dict[str, Any]] = []
    units: list[dict[str, Any]] = []
    total = 0
    for expected in spec.files:
        if expected.format != "parquet":
            continue
        path = source_root(raw_root, spec.source_id, spec.version) / expected.path
        table = pq.read_table(path)
        for index, row in enumerate(table.to_pylist(), start=1):
            total += 1
            split = expected.split
            document_id = canonical_id("alarb", spec.version, expected.path, index, "row", "case")
            base = _provenance("alarb", spec.version, expected.path, index, "row", split)
            documents.append(
                _document_row(document_id, "case", f"ALARB {split} {index}", base, split=split)
            )
            for ordinal, (field, unit_type) in enumerate(
                (
                    ("case_facts", UnitType.FACTS),
                    ("court_reasoning", UnitType.COURT_REASONING),
                    ("applicable_laws", UnitType.APPLICABLE_LAWS),
                    ("verdict", UnitType.VERDICT),
                ),
                start=1,
            ):
                text = str(row.get(field, ""))
                provenance = _provenance("alarb", spec.version, expected.path, index, field, split)
                units.append(
                    _unit_row(
                        document_id,
                        canonical_id("alarb", spec.version, expected.path, index, field),
                        unit_type.value,
                        text,
                        provenance,
                        ordinal,
                    )
                )
    return (
        documents,
        units,
        [],
        RawAccounting(
            source_id="alarb",
            expected_records=spec.expected_records,
            accounted_records=total,
            canonical_documents=total,
            canonical_units=len(units),
            excluded_records=0,
            error_records=0,
        ),
    )


def _build_arabiccr(
    raw_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], RawAccounting]:
    spec = load_specifications()["arabiccr"]
    path = source_root(raw_root, spec.source_id, spec.version) / spec.files[0].path
    documents: list[dict[str, Any]] = []
    units: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader, start=1):
            document_id = canonical_id(
                "arabiccr", spec.version, spec.files[0].path, index, "row", "case"
            )
            base = _provenance("arabiccr", spec.version, spec.files[0].path, index, "row")
            metadata = {
                field: row.get(field, "")
                for field in (
                    "judgment_number",
                    "case_number",
                    "court_name",
                    "case_type",
                    "judgment_date",
                    "year",
                    "city",
                    "details_url",
                )
            }
            documents.append(
                _document_row(
                    document_id, "case", row.get("case_number", ""), base, source_metadata=metadata
                )
            )
            for ordinal, field in enumerate(
                ("case_text", "EVENTS", "REASONING", "RULING"), start=1
            ):
                units.append(
                    _unit_row(
                        document_id,
                        canonical_id("arabiccr", spec.version, spec.files[0].path, index, field),
                        field.lower(),
                        row.get(field, ""),
                        _provenance("arabiccr", spec.version, spec.files[0].path, index, field),
                        ordinal,
                    )
                )
    count = len(documents)
    return (
        documents,
        units,
        [],
        RawAccounting(
            source_id="arabiccr",
            expected_records=spec.expected_records,
            accounted_records=count,
            canonical_documents=count,
            canonical_units=len(units),
            excluded_records=0,
            error_records=0,
        ),
    )


def _build_moj(
    raw_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], RawAccounting]:
    spec = load_specifications()["saudi-moj-derived"]
    expected = next(item for item in spec.files if item.format == "parquet")
    path = source_root(raw_root, spec.source_id, spec.version) / expected.path
    fragments: list[dict[str, Any]] = []
    rows = pq.read_table(path).to_pylist()
    for index, row in enumerate(rows, start=1):
        fragments.append(
            {
                "fragment_id": canonical_id(
                    spec.source_id, spec.version, expected.path, index, "text", "fragment"
                ),
                **_provenance(spec.source_id, spec.version, expected.path, index, "text"),
                "raw_label": str(row.get("article_number", "")),
                "law_name": str(row.get("law_name", "")),
                "law_type": str(row.get("law_type", "")),
                "derived_article_ordinal": None,
                "explicit_part": None,
                "article_label_structural_key": None,
                "article_parse_confidence": "unresolved",
                "article_status_marker": None,
                "part_index": None,
                "text": str(row.get("text", "")),
            }
        )
    # Statutory documents and units are created from fragments without changing their text.
    from kawaneen.corpus.statutory import classify_fragment_group, parse_article_label

    by_law: dict[str, list[dict[str, Any]]] = {}
    for item in fragments:
        by_law.setdefault(item["law_name"], []).append(item)
        label = parse_article_label(item["raw_label"])
        item["derived_article_ordinal"] = label.ordinal
        item["explicit_part"] = label.part
        item["article_label_structural_key"] = label.article_label_structural_key
        item["article_parse_confidence"] = label.article_parse_confidence.value
        item["article_status_marker"] = label.article_status_marker
        item["part_index"] = label.part_index
    documents: list[dict[str, Any]] = []
    units: list[dict[str, Any]] = []
    for law_name, law_fragments in sorted(by_law.items()):
        first = law_fragments[0]
        document_id = canonical_id(
            spec.source_id, spec.version, expected.path, first["source_row"], "law_name", "statute"
        )
        groups: dict[str, list[SourceFragment]] = {}
        for item in law_fragments:
            grouping_key = item["article_label_structural_key"]
            if grouping_key is None:
                grouping_key = f"unresolved:{item['source_row']}"
            groups.setdefault(grouping_key, []).append(
                SourceFragment(
                    fragment_id=item["fragment_id"],
                    provenance=SourceProvenance(
                        **{
                            key: item[key]
                            for key in (
                                "source_id",
                                "source_version",
                                "source_path",
                                "source_row",
                                "source_field",
                                "split",
                            )
                        }
                    ),
                    raw_label=item["raw_label"],
                    law_name=law_name,
                    law_type=item["law_type"],
                    derived_article_ordinal=item["derived_article_ordinal"],
                    explicit_part=item["explicit_part"],
                    article_label_structural_key=item["article_label_structural_key"],
                    article_parse_confidence=item["article_parse_confidence"],
                    article_status_marker=item["article_status_marker"],
                    part_index=item["part_index"],
                    unit_type=UnitType.ARTICLE_FRAGMENT,
                    text=item["text"],
                )
            )
        classifications = [
            classify_fragment_group(law_name, values[0].raw_label, values)
            for values in sorted(groups.values(), key=lambda items: items[0].raw_label)
        ]
        status = (
            "unique"
            if all(group.status.value == "unique" for group in classifications)
            else "unresolved"
        )
        base = {
            key: first[key]
            for key in (
                "source_id",
                "source_version",
                "source_path",
                "source_row",
                "source_field",
                "split",
            )
        }
        documents.append(
            _document_row(
                document_id,
                "statute",
                law_name,
                base,
                raw_article_label="",
                reconstruction_status=status,
            )
        )
        for ordinal, item in enumerate(
            sorted(law_fragments, key=lambda value: value["source_row"]), start=1
        ):
            units.append(
                _unit_row(
                    document_id,
                    canonical_id(
                        spec.source_id, spec.version, expected.path, item["source_row"], "text"
                    ),
                    UnitType.ARTICLE.value,
                    item["text"],
                    {
                        key: item[key]
                        for key in (
                            "source_id",
                            "source_version",
                            "source_path",
                            "source_row",
                            "source_field",
                            "split",
                        )
                    },
                    ordinal,
                )
            )
    return (
        documents,
        units,
        fragments,
        RawAccounting(
            source_id=spec.source_id,
            expected_records=spec.expected_records,
            accounted_records=len(rows),
            canonical_documents=len(documents),
            canonical_units=len(units),
            excluded_records=0,
            error_records=0,
        ),
    )


def build_source(
    source_id: str,
    raw_root: Path = Path("data/raw"),
    output_root: Path = Path("data/interim/canonical"),
) -> RawAccounting:
    """Build one source without modifying its raw namespace."""

    spec = load_specifications()[source_id]
    if source_id == "alarb":
        documents, units, fragments, accounting = _build_alarb(raw_root)
    elif source_id == "arabiccr":
        documents, units, fragments, accounting = _build_arabiccr(raw_root)
    elif source_id == "saudi-moj-derived":
        documents, units, fragments, accounting = _build_moj(raw_root)
    else:
        raise ValueError(f"no canonical adapter for {source_id}")
    root = canonical_root(output_root, source_id, spec.version)
    if documents:
        write_parquet(
            sorted(documents, key=lambda row: row["document_id"]),
            root / "documents.parquet",
            documents_schema(),
        )
    if units:
        write_parquet(
            sorted(units, key=lambda row: row["unit_id"]), root / "units.parquet", units_schema()
        )
    if fragments:
        write_parquet(
            sorted(fragments, key=lambda row: row["fragment_id"]),
            root / "fragments.parquet",
            fragments_schema(),
        )
    return accounting
