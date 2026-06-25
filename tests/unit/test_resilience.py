"""Unit tests for the resilience primitives (plan/008 §3, plan/011 §1.1).

Pure logic, no network, no sleeping (base_delay=0). Covers transient classification,
retry-then-succeed / give-up / fail-fast-on-permanent, and the circuit breaker
(opens after threshold, half-opens after cooldown, closes on success).
"""

import pytest

from assistant.resilience import (
    CircuitBreaker,
    CircuitOpenError,
    is_transient,
    resilient_call,
)

# --- is_transient ------------------------------------------------------------


def test_transient_errors_are_retryable():
    assert is_transient(TimeoutError("slow"))
    assert is_transient(ConnectionError("reset"))
    assert is_transient(Exception("429 RESOURCE_EXHAUSTED: rate limit"))
    assert is_transient(Exception("503 Service Unavailable"))


def test_permanent_errors_are_not_retryable():
    assert not is_transient(Exception("403 PermissionDenied"))
    assert not is_transient(Exception("400 invalid_argument: bad SQL"))
    assert not is_transient(Exception("401 Unauthenticated"))
    assert not is_transient(ValueError("some unknown error"))  # unknown -> conservative


# --- resilient_call ----------------------------------------------------------


def test_retries_transient_then_succeeds():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise TimeoutError("transient")
        return "ok"

    assert resilient_call(flaky, max_attempts=5, base_delay=0) == "ok"
    assert calls["n"] == 3


def test_gives_up_after_max_attempts():
    calls = {"n": 0}

    def always_fail():
        calls["n"] += 1
        raise TimeoutError("still down")

    with pytest.raises(TimeoutError):
        resilient_call(always_fail, max_attempts=3, base_delay=0)
    assert calls["n"] == 3


def test_permanent_error_is_not_retried():
    calls = {"n": 0}

    def permanent():
        calls["n"] += 1
        raise Exception("403 PermissionDenied")

    with pytest.raises(Exception, match="PermissionDenied"):
        resilient_call(permanent, max_attempts=5, base_delay=0)
    assert calls["n"] == 1  # tried once, not retried


# --- CircuitBreaker ----------------------------------------------------------


def test_breaker_opens_after_threshold_failures():
    cb = CircuitBreaker("t", threshold=3, cooldown_s=10, clock=lambda: 0.0)
    for _ in range(3):
        cb.before()  # closed: allowed
        cb.on_failure()
    with pytest.raises(CircuitOpenError):
        cb.before()  # now open


def test_breaker_half_opens_after_cooldown_then_closes_on_success():
    clock = {"t": 0.0}
    cb = CircuitBreaker("t", threshold=2, cooldown_s=10, clock=lambda: clock["t"])
    cb.on_failure()
    cb.on_failure()  # open at t=0
    with pytest.raises(CircuitOpenError):
        cb.before()

    clock["t"] = 11  # past cooldown
    cb.before()  # half-open: probe allowed (no raise)
    cb.on_success()  # closes

    clock["t"] = 12
    cb.before()  # closed again, no raise


def test_breaker_trips_during_retries_and_fails_fast():
    cb = CircuitBreaker("t", threshold=2, cooldown_s=100, clock=lambda: 0.0)
    calls = {"n": 0}

    def always_fail():
        calls["n"] += 1
        raise TimeoutError("down")

    with pytest.raises(CircuitOpenError):
        resilient_call(always_fail, breaker=cb, max_attempts=10, base_delay=0)
    # Two real failures opened the breaker; the next attempt was blocked (no 3rd call).
    assert calls["n"] == 2
