# pyright: reportUnusedFunction=false, reportArgumentType=false
"""FastAPI application factory and production lifecycle."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic_settings import BaseSettings, SettingsConfigDict

from kawaneen.api.contracts import ErrorDetail, ErrorResponse
from kawaneen.api.errors import ApiException
from kawaneen.api.middleware import ServingMiddleware
from kawaneen.api.routers import build_router
from kawaneen.api.runtime import ServiceContainer, build_default_container
from kawaneen.core.config import Settings


class ApiSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="KAWANEEN_API_", extra="ignore")

    search_timeout_seconds: float = 10.0
    answer_timeout_seconds: float = 65.0
    extract_timeout_seconds: float = 35.0


def create_app(
    container_factory: Callable[[], ServiceContainer] | None = None,
    *,
    settings: Settings | None = None,
    api_settings: ApiSettings | None = None,
) -> FastAPI:
    effective_settings = settings or Settings()
    timeouts = api_settings or ApiSettings()
    factory = container_factory or (lambda: build_default_container(effective_settings))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        container = factory()
        app.state.container = container
        await asyncio.to_thread(container.initialize)
        try:
            yield
        finally:
            await asyncio.to_thread(container.close)

    app = FastAPI(
        title="Kawaneen API",
        version="v1",
        description="Serving-safe Saudi legal and regulatory intelligence boundary.",
        lifespan=lifespan,
    )
    app.add_middleware(ServingMiddleware)
    app.include_router(
        build_router(
            search_timeout=timeouts.search_timeout_seconds,
            answer_timeout=timeouts.answer_timeout_seconds,
            extract_timeout=timeouts.extract_timeout_seconds,
        )
    )

    @app.exception_handler(ApiException)
    async def api_error_handler(request: Request, error: ApiException) -> JSONResponse:
        return JSONResponse(
            status_code=error.status_code,
            content=ErrorResponse(
                error=ErrorDetail(code=error.code, message=error.message),
                request_id=_request_id(request),
            ).model_dump(mode="json"),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, error: RequestValidationError
    ) -> JSONResponse:
        del error
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                error=ErrorDetail(code="VALIDATION_ERROR", message="request validation failed"),
                request_id=_request_id(request),
            ).model_dump(mode="json"),
        )

    @app.exception_handler(Exception)
    async def internal_error_handler(request: Request, error: Exception) -> JSONResponse:
        del error
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=ErrorDetail(code="INTERNAL_ERROR", message="internal server error"),
                request_id=_request_id(request),
            ).model_dump(mode="json"),
        )

    return app


def _request_id(request: Request) -> str:
    return str(getattr(request.state, "request_id", "missing-request-id"))
