"""Optional real Phase 12/UI boundary smoke; never runs in public CI."""

from __future__ import annotations

import os

import pytest

from kawaneen.ui.client import HttpUiClient, UiApiError


@pytest.mark.private_artifact
def test_private_phase12_ui_smoke_uses_only_normal_serving_paths() -> None:
    base_url = os.environ.get("KAWANEEN_PRIVATE_PHASE12_API_URL")
    if not base_url:
        pytest.skip("set KAWANEEN_PRIVATE_PHASE12_API_URL for the local Phase 12 smoke")
    client = HttpUiClient(base_url)
    try:
        try:
            health = client.health()
            models = client.models()
            assert health.status in {"ready", "degraded"}
            assert models.capabilities
            if health.status == "ready":
                response = client.search("ما هي مدة الاعتراض؟", limit=1)
                assert response.retrieval.returned_count <= 1
        except UiApiError as error:
            pytest.skip(f"local Phase 12 API unavailable: {error.code}")
    finally:
        client.close()
