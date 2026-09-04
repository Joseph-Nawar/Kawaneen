from __future__ import annotations


def test_public_demo_ui_flag_preserves_live_mode() -> None:
    from kawaneen.ui.config import UiMode, UiSettings

    settings = UiSettings.from_env({"KAWANEEN_UI_MODE": "live", "KAWANEEN_UI_PUBLIC_DEMO": "true"})
    assert settings.mode is UiMode.LIVE
    assert settings.public_demo is True
