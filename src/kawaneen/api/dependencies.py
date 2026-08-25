"""FastAPI dependency accessors for the application-owned service container."""

from __future__ import annotations

from typing import cast

from fastapi import Request

from kawaneen.api.errors import service_unavailable
from kawaneen.api.runtime import ServiceContainer


def get_container(request: Request) -> ServiceContainer:
    container = getattr(request.app.state, "container", None)
    if container is None:
        raise service_unavailable("serving runtime is not initialized")
    return cast(ServiceContainer, container)
