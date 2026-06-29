from __future__ import annotations

import pytest

from harness.circuit_breaker import CircuitBreaker, CircuitOpenError
from harness.state import AgentState


def make_state() -> AgentState:
    return {
        "messages": [],
        "plan": [],
        "current_step": 0,
        "iterations": 0,
        "file_edits": {},
        "verification": {"passed": False, "failures": [], "attempts": 0},
        "budget": {"tokens_used": 0, "tokens_max": 10000, "cost_usd": 0.0},
        "harness_events": [],
        "task": "test task",
        "output": "",
    }


def test_circuit_opens_after_configured_failures() -> None:
    breaker = CircuitBreaker(max_failures=2)

    breaker.record_failure()
    assert breaker.allow_request()

    breaker.record_failure()
    assert not breaker.allow_request()


def test_success_resets_failure_count() -> None:
    breaker = CircuitBreaker(max_failures=2)

    breaker.record_failure()
    breaker.record_success()

    assert breaker.failure_count == 0
    assert breaker.allow_request()


def test_call_rejects_when_open() -> None:
    breaker = CircuitBreaker(max_failures=1)
    breaker.record_failure()

    with pytest.raises(CircuitOpenError):
        breaker.call(lambda: "unreachable")


def test_check_trips_when_iterations_exceed_limit() -> None:
    state = make_state()
    state["iterations"] = 16
    breaker = CircuitBreaker(max_iterations=15)

    is_open, reason = breaker.check(state)

    assert is_open is True
    assert reason == "iterations exceeded maximum"
    assert state["harness_events"][-1]["condition"] == "max_iterations"


def test_check_trips_when_file_edit_count_exceeds_limit() -> None:
    state = make_state()
    state["file_edits"] = {"src/app.py": 6}
    breaker = CircuitBreaker(max_edits_per_file=5)

    is_open, reason = breaker.check(state)

    assert is_open is True
    assert reason == "src/app.py exceeded edit limit"
    assert state["harness_events"][-1]["condition"] == "max_edits_per_file"


def test_check_trips_when_same_error_repeats_too_often() -> None:
    state = make_state()
    state["verification"]["failures"] = ["boom", "boom", "boom", "boom"]
    breaker = CircuitBreaker(max_same_error=3)

    is_open, reason = breaker.check(state)

    assert is_open is True
    assert reason == "same verification error repeated too often"
    assert state["harness_events"][-1]["condition"] == "max_same_error"


def test_is_tripped_returns_false_when_no_condition_matches() -> None:
    state = make_state()
    breaker = CircuitBreaker()

    assert breaker.is_tripped(state) is False
    assert state["harness_events"] == []
