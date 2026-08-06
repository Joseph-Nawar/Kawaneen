from __future__ import annotations

import os


def clear_kawaneen_environment(monkeypatch) -> None:
    for name in tuple(os.environ):
        if name.startswith("KAWANEEN_"):
            monkeypatch.delenv(name, raising=False)
