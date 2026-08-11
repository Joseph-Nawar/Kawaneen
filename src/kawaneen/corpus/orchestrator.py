"""Offline corpus planning, construction, validation, and sanitized inventory."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import pyarrow.parquet as _parquet

from kawaneen.acquisition.specs import load_specifications
from kawaneen.acquisition.storage import source_root
from kawaneen.corpus.adapters import build_source
from kawaneen.corpus.ids import canonical_id
from kawaneen.corpus.models import SourceFragment, SourceProvenance, UnitType
from kawaneen.corpus.serialization import (
    canonical_root,
    reconstruction_schema,
    write_json,
    write_parquet,
)
from kawaneen.corpus.statutory import (
    build_statutory_review_samples,
    classify_all,
    duplicate_diagnostics,
    parse_article_label,
    reconstruction_counts,
)

pq: Any = cast(Any, _parquet)

RAW_ROOT = Path("data/raw")
CANONICAL_ROOT = Path("data/interim/canonical")
MANIFEST_ROOT = Path("data/manifests/canonical")


def build_statutory_review_handoff(
    destination: Path, external_adjudication: Path
) -> dict[str, object]:
    """Regenerate a private reconciliation handoff from exact parsed ordinals.

    The external adjudication is the frozen schedule of requested targets. Raw text
    is only read at this ignored private-export boundary and is never written into
    a version-controlled manifest.
    """

    records = [
        cast(dict[str, Any], json.loads(line))
        for line in external_adjudication.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    targets_by_law: dict[str, dict[str, int]] = {}
    for record in records:
        law = cast(dict[str, Any], record["law"])["dataset_law_name"]
        results = cast(dict[str, Any], record["sample_review"])["results"]
        targets_by_law[str(law)] = {
            str(item["sample_role"]): int(item["intended_article_ordinal"])
            for item in cast(list[dict[str, Any]], results)
        }
    fragments, _laws = _statutory_fragments()
    fragments_by_law: dict[str, list[SourceFragment]] = defaultdict(list)
    for fragment in fragments:
        fragments_by_law[fragment.law_name].append(fragment)
    exported: list[dict[str, object]] = []
    for law_name in sorted(targets_by_law):
        fragments_for_law = fragments_by_law[law_name]
        fragments_by_row = {
            fragment.provenance.source_row: fragment for fragment in fragments_for_law
        }
        for sample in build_statutory_review_samples(
            law_name, fragments_for_law, targets_by_law[law_name]
        ):
            members = cast(list[dict[str, object]], sample["members"])
            sample["members"] = [
                {
                    **member,
                    "fragment_id": fragments_by_row[cast(int, member["source_row"])].fragment_id,
                    "text": fragments_by_row[cast(int, member["source_row"])].text,
                }
                for member in members
            ]
            exported.append(sample)
    if not all(cast(bool, item["target_present"]) for item in exported):
        raise ValueError("statutory review exporter emitted a target that is not present")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(exported, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "law_count": len(targets_by_law),
        "sample_count": len(exported),
        "target_present_count": sum(cast(bool, item["target_present"]) for item in exported),
    }


def plan() -> list[dict[str, Any]]:
    return [
        {
            "source_id": source_id,
            "version": spec.version,
            "canonical_output": canonical_root(CANONICAL_ROOT, source_id, spec.version).as_posix(),
            "decision": "private_local_canonicalization",
            "phase4_eligibility": "manual_review_required",
        }
        for source_id, spec in sorted(load_specifications().items())
    ]


def _statutory_fragments() -> tuple[list[SourceFragment], list[str]]:
    spec = load_specifications()["saudi-moj-derived"]
    expected = next(item for item in spec.files if item.format == "parquet")
    path = source_root(RAW_ROOT, spec.source_id, spec.version) / expected.path
    rows = pq.read_table(path).to_pylist()
    fragments: list[SourceFragment] = []
    laws: list[str] = []
    for row_number, row in enumerate(rows, start=1):
        label = str(row.get("article_number", ""))
        parsed = parse_article_label(label)
        fragments.append(
            SourceFragment(
                fragment_id=canonical_id(
                    spec.source_id, spec.version, expected.path, row_number, "text", "fragment"
                ),
                provenance=SourceProvenance(
                    source_id=spec.source_id,
                    source_version=spec.version,
                    source_path=expected.path,
                    source_row=row_number,
                    source_field="text",
                ),
                raw_label=label,
                law_name=str(row.get("law_name", "")),
                law_type=str(row.get("law_type", "")),
                derived_article_ordinal=parsed.ordinal,
                explicit_part=parsed.part,
                article_label_structural_key=parsed.article_label_structural_key,
                article_parse_confidence=parsed.article_parse_confidence,
                article_status_marker=parsed.article_status_marker,
                part_index=parsed.part_index,
                unit_type=UnitType.ARTICLE_FRAGMENT,
                text=str(row.get("text", "")),
            )
        )
        laws.append(str(row.get("law_name", "")))
    return fragments, laws


def statutory_status() -> dict[str, Any]:
    fragments, laws = _statutory_fragments()
    groups = classify_all(fragments, laws)
    counts = reconstruction_counts(groups)
    parsed = [fragment for fragment in fragments if fragment.derived_article_ordinal is not None]
    by_law: dict[str, list[SourceFragment]] = defaultdict(list)
    for fragment in fragments:
        by_law[fragment.law_name].append(fragment)
    law_diagnostics: list[dict[str, Any]] = []
    for law_name, items in sorted(by_law.items()):
        ordinals = [item.derived_article_ordinal for item in items if item.derived_article_ordinal]
        ordinal_keys: dict[int, set[str]] = defaultdict(set)
        for item in items:
            if item.derived_article_ordinal is not None and item.article_label_structural_key:
                ordinal_keys[item.derived_article_ordinal].add(item.article_label_structural_key)
        law_diagnostics.append(
            {
                "law_name": law_name,
                "raw_fragment_count": len(items),
                "high_confidence_parsed_labels": sum(
                    item.article_parse_confidence.value == "high" for item in items
                ),
                "unresolved_labels": sum(item.derived_article_ordinal is None for item in items),
                "unique_structural_article_ordinals": len(set(ordinals)),
                "part_marked_records": sum(item.part_index is not None for item in items),
                "genuine_duplicate_ordinal_groups": sum(
                    len(keys) > 1 for keys in ordinal_keys.values()
                ),
            }
        )
    return {
        "schema_version": 2,
        "source_id": "saudi-moj-derived",
        "fragment_count": len(fragments),
        "duplicate_key_situations": sum(max(0, len(group.fragment_ids) - 1) for group in groups),
        "reconstruction_counts": counts,
        "groups": len(groups),
        "high_confidence_parsed_labels": len(parsed),
        "unresolved_labels": len(fragments) - len(parsed),
        "unique_structural_article_ordinals": len(
            {
                fragment.derived_article_ordinal
                for fragment in fragments
                if fragment.derived_article_ordinal is not None
            }
        ),
        "part_marked_records": sum(fragment.part_index is not None for fragment in fragments),
        "law_diagnostics": law_diagnostics,
        "superseded_baseline": {
            "groups": 1192,
            "duplicate_groups": 446,
            "duplicate_key_situations": 2439,
            "reason": "partial substring and part-number parsing bug",
        },
        "phase4_status": "private_parsing_seed_only",
    }


def _write_duplicate_diagnostics() -> dict[str, object]:
    fragments, laws = _statutory_fragments()
    groups = classify_all(fragments, laws)
    diagnostics = duplicate_diagnostics(fragments, groups)
    diagnostics["superseded_baseline"] = {
        "groups": 1192,
        "duplicate_groups": 446,
        "duplicate_key_situations": 2439,
        "reason": "partial substring and part-number parsing bug",
    }
    write_json(MANIFEST_ROOT / "duplicate_diagnostics.json", diagnostics)
    write_json(
        MANIFEST_ROOT / "duplicate_review_sample.json",
        {
            "schema_version": 1,
            "sample_size": len(diagnostics["review_sample"]),
            "selection": "first 25 groups after stable law/key ordering",
            "review_sample": diagnostics["review_sample"],
        },
    )
    return diagnostics


def _write_inventory(source_id: str, version: str, accounting: dict[str, Any]) -> dict[str, Any]:
    root = canonical_root(CANONICAL_ROOT, source_id, version)
    files: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.parquet")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files.append({"path": path.as_posix(), "size": path.stat().st_size, "sha256": digest})
    payload = {
        "schema_version": 1,
        "source_id": source_id,
        "version": version,
        "accounting": accounting,
        "files": files,
    }
    write_json(MANIFEST_ROOT / f"{source_id}.json", payload)
    return payload


def _reconstruction_record(group: Any) -> dict[str, Any]:
    parsed = parse_article_label(group.raw_article_label)
    return {
        "law_name": group.law_name,
        "raw_article_label": group.raw_article_label,
        "status": group.status.value,
        "article_label_structural_key": parsed.article_label_structural_key,
        "article_ordinal": parsed.article_ordinal,
        "article_parse_confidence": parsed.article_parse_confidence.value,
        "part_index": parsed.part_index,
        "article_status_marker": parsed.article_status_marker,
        "fragment_ids_json": json.dumps(group.fragment_ids, ensure_ascii=False),
        "operations_json": json.dumps(group.operations, ensure_ascii=False),
    }


def _write_reconstruction(version: str) -> None:
    fragments, laws = _statutory_fragments()
    groups = classify_all(fragments, laws)
    records = [_reconstruction_record(group) for group in groups]
    write_parquet(
        records,
        canonical_root(CANONICAL_ROOT, "saudi-moj-derived", version) / "reconstruction.parquet",
        reconstruction_schema(),
    )


def build(sources: list[str] | None = None) -> list[dict[str, Any]]:
    selected = sources or sorted(load_specifications())
    results: list[dict[str, Any]] = []
    selected_inventories: dict[str, dict[str, Any]] = {}
    for source_id in selected:
        accounting = build_source(source_id)
        if source_id == "saudi-moj-derived":
            _write_reconstruction(load_specifications()[source_id].version)
        selected_inventories[source_id] = _write_inventory(
            source_id, load_specifications()[source_id].version, accounting.model_dump()
        )
        results.append(accounting.model_dump())
    inventories: list[dict[str, Any]] = []
    accounting_by_source: dict[str, dict[str, Any]] = {}
    for source_id in sorted(load_specifications()):
        if source_id in selected_inventories:
            payload = selected_inventories[source_id]
        else:
            path = MANIFEST_ROOT / f"{source_id}.json"
            if not path.is_file():
                raise ValueError(f"missing existing canonical inventory: {source_id}")
            payload = json.loads(path.read_text(encoding="utf-8"))
        inventories.append(payload)
        accounting_by_source[source_id] = payload["accounting"]
    if "saudi-moj-derived" in selected:
        write_json(MANIFEST_ROOT / "reconstruction.json", statutory_status())
        _write_duplicate_diagnostics()
    write_json(MANIFEST_ROOT / "inventory.json", {"schema_version": 1, "sources": inventories})
    write_json(
        MANIFEST_ROOT / "quality.json",
        {
            "schema_version": 1,
            "raw_accounting": [
                accounting_by_source[source] for source in sorted(accounting_by_source)
            ],
            "statutory": (
                statutory_status()
                if "saudi-moj-derived" in selected
                else json.loads((MANIFEST_ROOT / "reconstruction.json").read_text(encoding="utf-8"))
            ),
        },
    )
    write_json(
        MANIFEST_ROOT / "snapshot.json",
        {
            "schema_version": 1,
            "sources": [accounting_by_source[source] for source in sorted(accounting_by_source)],
        },
    )
    return results


def validate() -> dict[str, Any]:
    missing: list[str] = []
    for source_id, spec in sorted(load_specifications().items()):
        root = canonical_root(CANONICAL_ROOT, source_id, spec.version)
        if not root.is_dir() or not list(root.glob("*.parquet")):
            missing.append(source_id)
    if missing:
        raise ValueError(f"missing canonical outputs: {', '.join(missing)}")
    return {"valid": True, "sources": sorted(load_specifications())}


def inventory() -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for source_id, _spec in sorted(load_specifications().items()):
        path = MANIFEST_ROOT / f"{source_id}.json"
        if path.is_file():
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
    return payloads


def gaps() -> list[dict[str, str]]:
    targets = [
        "Companies and business formation",
        "Digital commerce",
        "Commercial Courts",
        "Civil Transactions",
        "Bankruptcy",
        "Arbitration",
        "Enforcement",
        "Evidence",
        "Sharia Pleadings",
        "Documentation",
        "Legal Practice",
        "Judicial Costs",
        "Registered Mortgage",
        "Real Estate Finance",
        "Secured Transactions",
        "Consumer Protection",
        "Competition",
        "Agency and Distribution",
        "Intellectual Property commercial rules",
        "Electronic Signatures",
    ]
    reconciliation_path = Path("data/manifests/reconciliation/core-commercial-civil-v1.csv")
    reconciliation: dict[str, dict[str, str]] = {}
    if reconciliation_path.is_file():
        with reconciliation_path.open(encoding="utf-8", newline="") as handle:
            reconciliation = {row["dataset_law_name"]: row for row in csv.DictReader(handle)}
    target_laws = {
        "Commercial Courts": "نظام المحاكم التجارية",
        "Civil Transactions": "نظام المعاملات المدنية",
        "Bankruptcy": "نظام الإفلاس",
        "Arbitration": "نظام التحكيم",
        "Enforcement": "نظام التنفيذ",
        "Evidence": "نظام الإثبات",
        "Sharia Pleadings": "نظام المرافعات الشرعية",
        "Documentation": "نظام التوثيق",
        "Legal Practice": "نظام المحاماة",
        "Judicial Costs": "نظام التكاليف القضائية",
        "Registered Mortgage": "نظام الرهن العقاري المسجل",
        "Real Estate Finance": "نظام التمويل العقاري",
    }
    rows = [
        {
            "instrument": target,
            "desired_scope": "commercial_civil_15_25",
            "seed_status": (
                "present_untrusted"
                if reconciliation.get(target_laws.get(target, ""), {}).get("reconciliation_status")
                else "missing_targeted_acquisition"
            ),
            "targeted_acquisition_need": (
                "authoritative_article_count_and_manual_samples"
                if target in target_laws
                else "official_machine_readable_source_or_manual_authority_record"
            ),
        }
        for target in targets
    ]
    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
    path = MANIFEST_ROOT / "statutory_gap_report.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows
