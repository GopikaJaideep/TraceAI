"""Sliding-window rate limiter. In-memory, so per process: run a shared store (e.g. Redis) or an
edge limiter in front of multiple instances."""
from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import HTTPException


class RateLimiter:
    def __init__(self, limit: int, window_seconds: float):
        self.limit, self.window = limit, window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str) -> None:
        now = time.monotonic()
        hits = self._hits[key]
        while hits and now - hits[0] > self.window:
            hits.popleft()
        if len(hits) >= self.limit:
            retry = int(self.window - (now - hits[0])) + 1
            raise HTTPException(429, "Too many requests", headers={"Retry-After": str(retry)})
        hits.append(now)

    def reset(self) -> None:
        self._hits.clear()
