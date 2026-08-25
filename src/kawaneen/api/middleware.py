# pyright: reportArgumentType=false
"""ASGI request-ID, body-limit, and safe unexpected-error middleware."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

from kawaneen.api.context import bind_request_id, clear_request_context, normalize_request_id
from kawaneen.api.contracts import ErrorDetail, ErrorResponse
from kawaneen.api.errors import ApiException, request_too_large


class ServingMiddleware:
    def __init__(
        self,
        app: Callable[..., Awaitable[None]],
        *,
        max_body_bytes: int = 128 * 1024,
    ) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(
        self,
        scope: MutableMapping[str, Any],
        receive: Callable[..., Awaitable[Any]],
        send: Callable[..., Awaitable[None]],
    ) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        request_id = normalize_request_id(_header(scope, b"x-request-id"))
        state = scope.setdefault("state", {})
        state["request_id"] = request_id
        bind_request_id(request_id)
        response_started = False

        async def send_with_request_id(message: Any) -> None:
            nonlocal response_started
            if message.get("type") == "http.response.start":
                response_started = True
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", request_id.encode("ascii")))
                message["headers"] = headers
            await send(message)

        content_length = _header(scope, b"content-length")
        try:
            if scope.get("method") == "POST" and content_length is not None:
                try:
                    too_large = int(content_length) > self.max_body_bytes
                except ValueError:
                    too_large = True
                if too_large:
                    await _send_error(send_with_request_id, request_too_large(), request_id)
                    return

            async def limited_receive() -> Any:
                message = await receive()
                if message.get("type") == "http.request":
                    body = message.get("body", b"")
                    state["received_bytes"] = int(state.get("received_bytes", 0)) + len(body)
                    if state["received_bytes"] > self.max_body_bytes:
                        raise request_too_large()
                return message

            await self.app(scope, limited_receive, send_with_request_id)
        except ApiException as error:
            if not response_started:
                await _send_error(send_with_request_id, error, request_id)
        except Exception:
            if not response_started:
                await _send_raw_error(
                    send_with_request_id,
                    "INTERNAL_ERROR",
                    "internal server error",
                    500,
                    request_id,
                )
        finally:
            clear_request_context()


def _header(scope: MutableMapping[str, Any], wanted: bytes) -> str | None:
    for key, value in scope.get("headers", []):
        if key.lower() == wanted:
            return value.decode("latin-1")
    return None


async def _send_error(
    send: Callable[..., Awaitable[None]], error: ApiException, request_id: str
) -> None:
    await _send_raw_error(send, error.code, error.message, error.status_code, request_id)


async def _send_raw_error(
    send: Callable[..., Awaitable[None]],
    code: str,
    message: str,
    status_code: int,
    request_id: str,
) -> None:
    payload = ErrorResponse(
        error=ErrorDetail(code=code, message=message), request_id=request_id
    ).model_dump_json()
    body = payload.encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status_code,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})
