"""Resilience primitives for third-party dependencies (plan/008 §3).

Two reusable pieces, applied to both external dependencies — Gemini (``llm/``) and
BigQuery (``bigquery/``):

- :func:`resilient_call` retries a callable on *transient* failures with exponential
  backoff + jitter (via ``tenacity``), while failing fast on permanent errors so we
  don't waste time/cost on an auth or bad-request failure.
- :class:`CircuitBreaker` trips after repeated failures so we stop hammering a
  struggling dependency, degrade immediately for a cool-down, then half-open to
  probe recovery.

Transient vs. permanent is classified by :func:`is_transient` (rate limits, 5xx,
timeouts -> retry; auth / 4xx -> fail fast).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TypeVar

import tenacity

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Markers (substrings of the error text/type) that indicate a *transient* failure.
_TRANSIENT_MARKERS = (
    "429",
    "resource_exhausted",
    "rate limit",
    "rate_limit",
    "quota",
    "503",
    "service unavailable",
    "unavailable",
    "500",
    "internal error",
    "internalservererror",
    "timeout",
    "timed out",
    "deadline",
    "connection",
    "temporarily",
    "try again",
)

# Markers that indicate a *permanent* failure — never retried (overrides the above).
_PERMANENT_MARKERS = (
    "permissiondenied",
    "permission denied",
    "unauthenticated",
    "invalid_argument",
    "invalidargument",
    "badrequest",
    "bad request",
    "notfound",
    "not found",
    "400",
    "401",
    "403",
    "404",
)


def is_transient(exc: BaseException) -> bool:
    """True if ``exc`` looks transient (worth retrying); False for permanent errors."""
    text = f"{type(exc).__name__}: {exc}".lower()
    if any(marker in text for marker in _PERMANENT_MARKERS):
        return False
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    return any(marker in text for marker in _TRANSIENT_MARKERS)


class CircuitOpenError(RuntimeError):
    """Raised when a circuit breaker is open (the dependency is considered down)."""


class CircuitBreaker:
    """A minimal circuit breaker: closed -> open (on repeated failures) -> half-open.

    After ``threshold`` consecutive transient failures it opens for ``cooldown_s``;
    calls during that window fail fast with :class:`CircuitOpenError`. After the
    cool-down it half-opens to allow one probe — success closes it, failure re-opens.
    """

    def __init__(
        self,
        name: str,
        threshold: int = 5,
        cooldown_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.name = name
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self._clock = clock
        self._failures = 0
        self._opened_at: float | None = None

    def before(self) -> None:
        """Raise :class:`CircuitOpenError` if the breaker is open and still cooling down."""
        if self._opened_at is None:
            return
        if self._clock() - self._opened_at < self.cooldown_s:
            raise CircuitOpenError(f"{self.name} circuit is open; cooling down")
        logger.info("Circuit %s half-open (probing recovery)", self.name)

    def on_success(self) -> None:
        if self._failures or self._opened_at is not None:
            logger.info("Circuit %s closed after a successful call", self.name)
        self._failures = 0
        self._opened_at = None

    def on_failure(self) -> None:
        self._failures += 1
        if self._opened_at is not None:
            self._opened_at = self._clock()  # probe failed -> extend the cool-down
        elif self._failures >= self.threshold:
            self._opened_at = self._clock()
            logger.warning("Circuit %s OPENED after %d failures", self.name, self._failures)


def resilient_call(
    func: Callable[[], T],
    *,
    breaker: CircuitBreaker | None = None,
    max_attempts: int = 4,
    base_delay: float = 1.0,
    retry_on: Callable[[BaseException], bool] = is_transient,
) -> T:
    """Run ``func`` with retry-on-transient + optional circuit breaker.

    Permanent errors (``retry_on`` returns False) and :class:`CircuitOpenError` are
    re-raised immediately. ``base_delay=0`` disables waiting (used in tests).
    """

    def _guarded() -> T:
        if breaker is not None:
            breaker.before()  # may raise CircuitOpenError (fail fast)
        try:
            result = func()
        except CircuitOpenError:
            raise
        except BaseException as exc:  # noqa: BLE001 — classify, record, then re-raise
            if breaker is not None and retry_on(exc):
                breaker.on_failure()
            raise
        if breaker is not None:
            breaker.on_success()
        return result

    retryer = tenacity.Retrying(
        stop=tenacity.stop_after_attempt(max(1, max_attempts)),
        wait=tenacity.wait_exponential_jitter(
            initial=base_delay, max=base_delay * 8, jitter=base_delay
        ),
        retry=tenacity.retry_if_exception(
            lambda exc: not isinstance(exc, CircuitOpenError) and retry_on(exc)
        ),
        reraise=True,
    )
    return retryer(_guarded)
