from __future__ import annotations

import pytest

from harness.circuit_breaker import CircuitBreaker, CircuitOpenError, CircuitState
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


def test_circuit_transitions_to_half_open_after_cooldown() -> None:
    state = make_state()
    state["verification"]["failures"] = ["boom", "boom", "boom", "boom"]
    state["iterations"] = 4
    breaker = CircuitBreaker(max_same_error=3, cooldown_iterations=5)

    is_open, _ = breaker.check(state)
    assert is_open is True
    assert breaker.state == CircuitState.OPEN

    state["iterations"] = 9
    is_open, reason = breaker.check(state)

    assert is_open is False
    assert reason == ""
    assert breaker.state == CircuitState.HALF_OPEN


def test_half_open_trial_success_closes_circuit() -> None:
    state = make_state()
    breaker = CircuitBreaker(cooldown_iterations=1)
    breaker.state = CircuitState.OPEN
    breaker.opened_at_iteration = 1
    state["iterations"] = 2
    breaker.check(state)

    breaker.record_trial_success(state)

    assert breaker.state == CircuitState.CLOSED
    assert breaker.failure_count == 0
    assert breaker.opened_at_iteration is None


def test_half_open_trial_failure_reopens_circuit_with_extended_cooldown() -> None:
    state = make_state()
    breaker = CircuitBreaker(cooldown_iterations=5)
    breaker.state = CircuitState.OPEN
    breaker.opened_at_iteration = 1
    state["iterations"] = 6
    breaker.check(state)

    state["iterations"] = 7
    breaker.record_trial_failure(state)

    assert breaker.state == CircuitState.OPEN
    assert breaker.opened_at_iteration == 7
    state["iterations"] = 11
    is_open, _ = breaker.check(state)
    assert is_open is True


def test_state_transitions_are_logged_to_harness_events() -> None:
    state = make_state()
    state["verification"]["failures"] = ["boom", "boom", "boom", "boom"]
    state["iterations"] = 4
    breaker = CircuitBreaker(max_same_error=3, cooldown_iterations=1)

    breaker.check(state)
    state["iterations"] = 5
    breaker.check(state)
    breaker.record_trial_success(state)

    transitions = [
        event
        for event in state["harness_events"]
        if event.get("type") == "circuit_breaker_state_change"
    ]
    assert [event["old_state"] for event in transitions] == ["closed", "open", "half_open"]
    assert [event["new_state"] for event in transitions] == ["open", "half_open", "closed"]
