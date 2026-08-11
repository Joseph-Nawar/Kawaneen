import json

from kawaneen.cli import main


def test_corpus_plan_is_sanitized(capsys) -> None:
    assert main(["corpus", "plan"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert {row["source_id"] for row in payload} == {"alarb", "arabiccr", "saudi-moj-derived"}
    assert all(row["canonical_output"].startswith("data/interim/canonical/") for row in payload)


def test_corpus_inventory_and_status_commands(capsys) -> None:
    assert main(["corpus", "inventory"]) == 0
    inventory = json.loads(capsys.readouterr().out)
    assert len(inventory) == 3
    assert main(["corpus", "statutory-status"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["fragment_count"] == 3185
    assert main(["corpus", "gaps"]) == 0
    gaps = json.loads(capsys.readouterr().out)
    assert len(gaps) == 20


def test_parser_benchmark_does_not_turn_unqualified_results_into_a_pass(capsys) -> None:
    assert main(["parsing", "benchmark"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "geometry_benchmark_validated_routes_measured"
    assert payload["qualification"]["ocr"] == "not_qualified"


def test_parser_preflight_is_sanitized_for_empty_directory(tmp_path, capsys) -> None:
    assert main(["parsing", "preflight", "--path", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out) == {"schema_version": 1, "sources": []}


def test_duplicate_diagnostics_command_is_sanitized(capsys) -> None:
    assert main(["corpus", "duplicate-diagnostics"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["duplicate_group_count"] == 450
    assert payload["superseded_baseline"]["duplicate_groups"] == 446
    assert "review_sample" in payload
