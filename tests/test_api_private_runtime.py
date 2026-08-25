from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.mark.private_artifact
def test_real_local_serving_composition_smoke() -> None:
    """Exercise only bounded, non-evaluation serving requests on local assets."""

    from kawaneen.api.app import create_app

    with TestClient(create_app()) as client:
        health = client.get("/v1/health")
        models = client.get("/v1/models")
        page = client.get("/v1/documents?offset=0&limit=1")
        document_id = page.json()["items"][0]["document_id"]
        detail = client.get(f"/v1/documents/{document_id}")
        deterministic = client.post(
            "/v1/extract",
            json={"text": "يلتزم الطرف بالسداد خلال ثلاثين يوماً.", "mode": "deterministic"},
        )
        search = client.post("/v1/search", json={"query": "ما هي مدة الاعتراض؟", "limit": 1})
        answer = client.post("/v1/answer", json={"query": "ما هي مدة الاعتراض؟"})
        hybrid = client.post(
            "/v1/extract",
            json={"text": "يلتزم الطرف بالسداد خلال ثلاثين يوماً.", "mode": "hybrid"},
        )

    assert health.status_code == 200
    assert health.json()["status"] == "ready"
    assert models.status_code == 200
    assert detail.status_code == 200
    assert deterministic.status_code == 200
    assert search.status_code == 200
    assert search.json()["retrieval"]["strategy"] == "hybrid_reranked"
    assert answer.status_code == 200
    assert hybrid.status_code == 200
    assert hybrid.json()["capability_status"] == "experimental_limited"
