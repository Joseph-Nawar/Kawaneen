from __future__ import annotations

import pytest


def test_demo_limits_reject_oversized_query_and_document() -> None:
    from kawaneen.demo.limits import DemoRequestLimiter

    limiter = DemoRequestLimiter()
    with pytest.raises(ValueError, match="500"):
        limiter.validate_query("س" * 501)
    with pytest.raises(ValueError, match="8,000"):
        limiter.validate_extraction("س" * 8_001)


def test_demo_limiter_caps_results_and_rejects_second_concurrent_slot() -> None:
    from kawaneen.demo.limits import DemoRequestLimiter

    limiter = DemoRequestLimiter()
    assert limiter.result_limit(8) == 5
    slot = limiter.slot()
    with slot, pytest.raises(RuntimeError, match="concurrent"), limiter.slot():
        pass


def test_demo_limiter_fixed_window_rejects_after_budget() -> None:
    from kawaneen.demo.limits import DemoRequestLimiter

    limiter = DemoRequestLimiter(rate_limit=2, window_seconds=60)
    for _ in range(2):
        with limiter.slot():
            pass
    with pytest.raises(RuntimeError, match="rate"), limiter.slot():
        pass
