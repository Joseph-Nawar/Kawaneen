from __future__ import annotations

import pytest

import kawaneen.cli as cli


@pytest.mark.parametrize(
    "command,function_name",
    (
        ("embedding", "phase15_embedding"),
        ("reranking", "phase15_reranking"),
        ("counterfactuals", "phase15_counterfactuals"),
        ("generation-run", "phase15_generation_run"),
        ("latency", "phase15_latency"),
        ("abstention", "phase15_abstention"),
        ("dialect-prepare", "phase15_dialect_prepare"),
        ("dialect-evaluate", "phase15_dialect_evaluate"),
    ),
)
def test_phase15_cli_dispatches_experiment_commands(
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    function_name: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, function_name, lambda *args: {"status": "TEST"})
    assert cli.main(["phase15", command]) == 0
    assert '"status": "TEST"' in capsys.readouterr().out
