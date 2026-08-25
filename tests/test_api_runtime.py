from __future__ import annotations


def test_service_container_initializes_once_and_cleans_up_once() -> None:
    from kawaneen.api.runtime import ServiceContainer

    counts = {"init": 0, "close": 0, "loads": 0}

    def initialize() -> None:
        counts["init"] += 1

    def close() -> None:
        counts["close"] += 1

    container = ServiceContainer(
        initializer=initialize,
        closer=close,
        capabilities=lambda: (counts.__setitem__("loads", counts["loads"] + 1), ())[1],
    )
    container.initialize()
    container.initialize()
    assert counts["init"] == 1
    assert container.health().status == "degraded"

    container.close()
    container.close()
    assert counts["close"] == 1
    assert counts["loads"] == 0
    assert container.models() == ()


def test_ready_container_reports_required_capabilities_without_loading_models() -> None:
    from kawaneen.api.runtime import ComponentReadiness, ServiceContainer

    container = ServiceContainer(
        components=(ComponentReadiness("corpus", True, True),),
        capabilities=lambda: (),
    )
    container.initialize()

    assert container.health().status == "ready"
    assert container.models() == ()
