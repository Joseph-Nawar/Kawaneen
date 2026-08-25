from kawaneen.ui.config import HealthProbe, UiMode, UiSettings, resolve_mode


def test_auto_mode_requires_explicit_demo_activation_when_api_is_unavailable() -> None:
    settings = UiSettings.from_env({"KAWANEEN_UI_MODE": "auto"})

    resolution = resolve_mode(settings, HealthProbe.unavailable("connection refused"))

    assert resolution.active_mode is None
    assert resolution.requires_demo_activation is True
    assert resolution.status_label == "Degraded"


def test_demo_mode_is_explicit_and_persistent() -> None:
    settings = UiSettings.from_env({"KAWANEEN_UI_MODE": "demo"})

    resolution = resolve_mode(settings, None)

    assert settings.mode is UiMode.DEMO
    assert resolution.active_mode is UiMode.DEMO
    assert resolution.status_label == "Demo data"
    assert resolution.requires_demo_activation is False


def test_live_mode_does_not_downgrade_to_demo() -> None:
    settings = UiSettings.from_env(
        {"KAWANEEN_UI_MODE": "live", "KAWANEEN_API_URL": "http://localhost:9010"}
    )

    resolution = resolve_mode(settings, HealthProbe.unavailable("offline"))

    assert resolution.active_mode is None
    assert resolution.status_label == "Degraded"
    assert resolution.requires_demo_activation is False
    assert resolution.detail == "offline"


def test_invalid_mode_falls_back_to_auto() -> None:
    settings = UiSettings.from_env({"KAWANEEN_UI_MODE": "unknown"})

    assert settings.mode is UiMode.AUTO
    assert settings.api_url == "http://127.0.0.1:8000"
