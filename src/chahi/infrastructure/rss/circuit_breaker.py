"""Domain circuit breaker với OPEN/HALF_OPEN/CLOSED."""

from __future__ import annotations

import threading
import time
from enum import Enum


class BreakerState(Enum):
    """Các trạng thái của breaker."""

    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class DomainCircuitBreaker:
    """Circuit breaker state machine theo domain.

    - ``CLOSED``: hoạt động bình thường.
    - ``OPEN``: tạm ngắt deep scrape cho domain.
    - ``HALF_OPEN``: mở thử 1 request probe để kiểm tra hồi phục.
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        open_ttl: float = 900.0,
        max_open_ttl: float = 3600.0,
    ) -> None:
        self._threshold = failure_threshold
        self._ttl = open_ttl
        self._max_ttl = max_open_ttl

        self._lock = threading.Lock()
        self._failures: dict[str, int] = {}
        self._states: dict[str, BreakerState] = {}
        self._open_until: dict[str, float] = {}
        self._cooldown_level: dict[str, int] = {}

    def get_state(self, domain: str) -> BreakerState:
        """Lấy state hiện tại; OPEN hết TTL sẽ tự chuyển HALF_OPEN."""
        with self._lock:
            state = self._states.get(domain, BreakerState.CLOSED)
            if state == BreakerState.OPEN and time.time() > self._open_until.get(
                domain, 0.0
            ):
                self._states[domain] = BreakerState.HALF_OPEN
                return BreakerState.HALF_OPEN
            return state

    def record_success(self, domain: str) -> None:
        """Reset breaker về CLOSED sau request thành công."""
        with self._lock:
            self._failures[domain] = 0
            self._states[domain] = BreakerState.CLOSED
            self._open_until.pop(domain, None)
            self._cooldown_level[domain] = 0

    def record_failure(self, domain: str) -> BreakerState:
        """Ghi nhận failure và cập nhật state.

        Returns:
            State mới của breaker sau khi cập nhật.
        """
        with self._lock:
            state = self._states.get(domain, BreakerState.CLOSED)
            if state == BreakerState.HALF_OPEN:
                self._trip_open_locked(domain, escalate=True)
                return BreakerState.OPEN

            failures = self._failures.get(domain, 0) + 1
            self._failures[domain] = failures
            if failures >= self._threshold:
                self._trip_open_locked(domain, escalate=False)
                return BreakerState.OPEN
            return BreakerState.CLOSED

    def get_failure_count(self, domain: str) -> int:
        """Số lần failure liên tiếp hiện tại của domain."""
        with self._lock:
            return self._failures.get(domain, 0)

    def get_open_remaining(self, domain: str) -> float:
        """Số giây còn lại trước khi OPEN chuyển HALF_OPEN."""
        with self._lock:
            if self._states.get(domain, BreakerState.CLOSED) != BreakerState.OPEN:
                return 0.0
            return max(0.0, self._open_until.get(domain, 0.0) - time.time())

    def _trip_open_locked(self, domain: str, *, escalate: bool) -> None:
        """Đưa domain sang OPEN, với cooldown exponential cho half-open failure."""
        level = self._cooldown_level.get(domain, 0)
        if escalate:
            level += 1
        else:
            level = max(level, 1)
        self._cooldown_level[domain] = level

        cooldown = min(self._ttl * (2 ** max(0, level - 1)), self._max_ttl)
        self._states[domain] = BreakerState.OPEN
        self._open_until[domain] = time.time() + cooldown
