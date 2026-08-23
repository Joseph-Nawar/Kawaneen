from __future__ import annotations

import json
from pathlib import Path

import kawaneen.cli as cli
from kawaneen.generation.ollama import OllamaModelIdentity


def test_generation_parser_exposes_status_and_resumable_dev_run() -> None:
    status = cli.build_parser().parse_args(
        ["generation", "status", "--generator", "qwen-ollama"]
    )
    run = cli.build_parser().parse_args(
        ["generation", "run-dev", "--generator", "qwen-ollama", "--resume"]
    )
    readiness = cli.build_parser().parse_args(
        ["generation", "readiness", "--generator", "qwen-ollama"]
    )
    diagnostic = cli.build_parser().parse_args(
        ["generation", "diagnose-stage-b-timeouts", "--resume"]
    )
    diagnostic_status = cli.build_parser().parse_args(
        ["generation", "timeout-diagnostic-status"]
    )
    diagnostic_evaluate = cli.build_parser().parse_args(
        ["generation", "evaluate-timeout-diagnostic"]
    )
    v2_status = cli.build_parser().parse_args(
        ["generation", "timeout-diagnostic-v2-status"]
    )
    v2_run = cli.build_parser().parse_args(
        ["generation", "diagnose-stage-b-timeouts-v2", "--resume"]
    )
    v2_evaluate = cli.build_parser().parse_args(
        ["generation", "evaluate-timeout-diagnostic-v2"]
    )
    stage_c_status = cli.build_parser().parse_args(
        ["generation", "status", "--generator", "qwen-ollama-stage-c"]
    )
    stage_c_readiness = cli.build_parser().parse_args(
        ["generation", "readiness", "--generator", "qwen-ollama-stage-c"]
    )
    stage_c_run = cli.build_parser().parse_args(
        ["generation", "run-dev", "--generator", "qwen-ollama-stage-c", "--resume"]
    )

    assert status.generation_command == "status"
    assert status.generator == "qwen-ollama"
    assert run.generation_command == "run-dev"
    assert run.resume is True
    assert readiness.generation_command == "readiness"
    assert diagnostic.generation_command == "diagnose-stage-b-timeouts"
    assert diagnostic.resume is True
    assert diagnostic_status.generation_command == "timeout-diagnostic-status"
    assert diagnostic_evaluate.generation_command == "evaluate-timeout-diagnostic"
    assert v2_status.generation_command == "timeout-diagnostic-v2-status"
    assert v2_run.generation_command == "diagnose-stage-b-timeouts-v2"
    assert v2_run.resume is True
    assert v2_evaluate.generation_command == "evaluate-timeout-diagnostic-v2"
    assert stage_c_status.generator == "qwen-ollama-stage-c"
    assert stage_c_readiness.generator == "qwen-ollama-stage-c"
    assert stage_c_run.generator == "qwen-ollama-stage-c"
    assert stage_c_run.resume is True


def test_lock_ollama_queries_local_tags_and_persists_private_lock(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    identity = OllamaModelIdentity(
        model="qwen3:4b-instruct-2507-q4_K_M",
        digest="sha256:" + "a" * 64,
    )
    calls: list[tuple[str, str]] = []

    def inspect(endpoint: str, expected_model: str, _transport: object) -> OllamaModelIdentity:
        calls.append((endpoint, expected_model))
        return identity

    def persist(path: Path, value: OllamaModelIdentity) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value.model_dump(mode="json")), encoding="utf-8")

    monkeypatch.setattr(cli, "inspect_ollama_model", inspect)
    monkeypatch.setattr(cli, "write_local_model_lock", persist)
    lock_path = tmp_path / "model-lock.json"

    result = cli.main(
        [
            "generation",
            "lock-ollama",
            "--model",
            identity.model,
            "--endpoint",
            "http://localhost:11434",
            "--lock-path",
            str(lock_path),
        ]
    )

    assert result == 0
    assert calls == [("http://localhost:11434", identity.model)]
    assert json.loads(capsys.readouterr().out)["digest"] == identity.digest
