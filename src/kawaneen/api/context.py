"""Request-scoped correlation IDs with structlog context isolation."""

from __future__ import annotations

import re
import uuid

import structlog

_REQUEST_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


def normalize_request_id(value: str | None) -> str:
    if value is not None and _REQUEST_ID.fullmatch(value):
        return value
    return str(uuid.uuid4())


def bind_request_id(request_id: str) -> None:
    structlog.contextvars.bind_contextvars(request_id=request_id)


def clear_request_context() -> None:
    structlog.contextvars.clear_contextvars()
