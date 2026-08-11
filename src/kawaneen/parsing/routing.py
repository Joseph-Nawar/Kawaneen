"""Configured PDF health routing without parser or model imports."""

from __future__ import annotations

import tomllib
from pathlib import Path

from kawaneen.parsing.models import PageHealth, ParseRoute


def load_routing_config(path: Path) -> dict[str, int]:
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    thresholds = payload.get("thresholds", {})
    return {str(key): int(value) for key, value in thresholds.items()}


def route_page(health: PageHealth, config: dict[str, int]) -> ParseRoute:
    if health.text_chars < config["image_only_max_text_chars"] and health.image_count > 0:
        return ParseRoute.FULL_PAGE_OCR
    if (
        health.suspicious_text
        or health.text_chars < config["mixed_max_text_chars"]
        or health.image_count > 0
    ):
        return ParseRoute.DAMAGED_MIXED
    return ParseRoute.EMBEDDED_TEXT
