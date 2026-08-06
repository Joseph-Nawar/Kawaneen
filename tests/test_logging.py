from __future__ import annotations

import json
import logging

import structlog

from kawaneen.core.logging import configure_logging


def test_console_logging_is_readable(capsys) -> None:
    configure_logging(log_level="INFO", log_format="console")
    structlog.get_logger("test").info("foundation_ready", component="logging")
    output = capsys.readouterr().err
    assert "foundation_ready" in output
    assert "component=logging" in output


def test_json_logging_contains_timestamp_level_and_event(capsys) -> None:
    configure_logging(log_level="INFO", log_format="json")
    structlog.get_logger("test").info("foundation_ready")
    record = json.loads(capsys.readouterr().err)
    assert record["event"] == "foundation_ready"
    assert record["level"] == "info"
    assert record["timestamp"]


def test_repeated_configuration_does_not_duplicate_output(capsys) -> None:
    configure_logging(log_level="INFO", log_format="console")
    configure_logging(log_level="INFO", log_format="console")
    structlog.get_logger("test").info("once")
    output = capsys.readouterr().err
    assert output.count("once") == 1
    assert len(logging.getLogger().handlers) == 1
