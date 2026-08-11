from pathlib import Path

from kawaneen.parsing.models import PageHealth, ParseRoute
from kawaneen.parsing.routing import load_routing_config, route_page


def test_routes_healthy_mixed_and_image_only_pages() -> None:
    config = load_routing_config(Path("configs/parsing/default.toml"))
    assert (
        route_page(PageHealth(page_number=1, text_chars=500, image_count=0), config)
        is ParseRoute.EMBEDDED_TEXT
    )
    assert (
        route_page(PageHealth(page_number=2, text_chars=20, image_count=1), config)
        is ParseRoute.DAMAGED_MIXED
    )
    assert (
        route_page(PageHealth(page_number=3, text_chars=0, image_count=1), config)
        is ParseRoute.FULL_PAGE_OCR
    )
