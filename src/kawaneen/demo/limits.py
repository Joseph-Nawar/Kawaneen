from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Generator
from contextlib import contextmanager


class DemoRequestLimiter:
    def __init__(
        self, *, max_concurrent: int = 1, rate_limit: int = 30, window_seconds: float = 60.0
    ) -> None:
        self._semaphore = threading.BoundedSemaphore(max_concurrent)
        self._rate_limit = rate_limit
        self._window_seconds = window_seconds
        self._timestamps: deque[float] = deque()
        self._lock = threading.Lock()

    def validate_query(self, query: str) -> None:
        if len(query) > 500:
            raise ValueError("public demo query limit is 500 characters")

    def validate_extraction(self, text: str) -> None:
        if len(text) > 8_000:
            raise ValueError("public demo extraction limit is 8,000 characters")

    def result_limit(self, limit: int) -> int:
        return min(limit, 5)

    def acquire(self) -> None:
        now = time.monotonic()
        with self._lock:
            while self._timestamps and now - self._timestamps[0] >= self._window_seconds:
                self._timestamps.popleft()
            if len(self._timestamps) >= self._rate_limit:
                raise RuntimeError("public demo rate limit exceeded")
            if not self._semaphore.acquire(blocking=False):
                raise RuntimeError("public demo concurrent request limit exceeded")
            self._timestamps.append(now)

    def release(self) -> None:
        self._semaphore.release()

    @contextmanager
    def slot(self) -> Generator[None, None, None]:
        self.acquire()
        try:
            yield
        finally:
            self.release()


__all__ = ["DemoRequestLimiter"]
