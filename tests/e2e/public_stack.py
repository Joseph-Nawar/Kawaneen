from __future__ import annotations

from fastapi import FastAPI

from kawaneen.api.app import create_app
from tests.phase14_support import build_phase14_service_container

app: FastAPI = create_app(build_phase14_service_container)
