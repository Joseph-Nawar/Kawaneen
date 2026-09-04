"""Safe domain errors translated into stable public API envelopes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ErrorCode = Literal[
    "VALIDATION_ERROR",
    "REQUEST_TOO_LARGE",
    "DOCUMENT_NOT_FOUND",
    "SERVICE_UNAVAILABLE",
    "MODEL_UNAVAILABLE",
    "REQUEST_TIMEOUT",
    "RATE_LIMITED",
    "INTERNAL_ERROR",
]


@dataclass(frozen=True, slots=True)
class ApiException(Exception):
    code: ErrorCode
    message: str
    status_code: int

    def __str__(self) -> str:
        return self.message


def document_not_found(document_id: str) -> ApiException:
    del document_id
    return ApiException("DOCUMENT_NOT_FOUND", "document not found", 404)


def service_unavailable(message: str = "required service is not ready") -> ApiException:
    return ApiException("SERVICE_UNAVAILABLE", message, 503)


def model_unavailable(message: str = "required model is not ready") -> ApiException:
    return ApiException("MODEL_UNAVAILABLE", message, 503)


def request_too_large() -> ApiException:
    return ApiException("REQUEST_TOO_LARGE", "request body is too large", 413)


def request_timeout() -> ApiException:
    return ApiException("REQUEST_TIMEOUT", "request timed out", 504)


def rate_limited(message: str = "public demo request limit exceeded") -> ApiException:
    return ApiException("RATE_LIMITED", message, 429)
