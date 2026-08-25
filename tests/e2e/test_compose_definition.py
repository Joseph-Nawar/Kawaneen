from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e

ROOT = Path(__file__).parents[2]


def test_public_compose_is_native_multi_arch_test_only_and_has_health_ordering() -> None:
    compose = (ROOT / "docker-compose.e2e.yml").read_text(encoding="utf-8")

    assert "platform: linux/amd64" not in compose
    assert "condition: service_healthy" in compose
    assert "tests.e2e.public_stack:app" in compose
    assert "tests.e2e.run_public_e2e" in compose
    assert "production" in compose.lower()
