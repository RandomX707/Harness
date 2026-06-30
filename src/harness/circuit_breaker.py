"""Circuit breaker for repeated harness failures."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, TypeVar

import structlog

from harness.state import AgentState


T = TypeVar("T")


class CircuitOpenError(RuntimeError):
    """Raised when the circuit is open and work should stop."""


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Stop an agent loop when bounded harness conditions are exceeded."""

    def __init__(
        self,
        max_iterations: int = 15,
        max_edits_per_file: int = 5,
        max_same_error: int = 3,
        cooldown_iterations: int = 5,
        max_failures: int | None = None,
    ) -> None:
        self.max_iterations = max_iterations
        self.max_edits_per_file = max_edits_per_file
        self.max_same_error = max_same_error
        self.cooldown_iterations = cooldown_iterations
        self.max_failures = max_failures if max_failures is not None else max_same_error
        self.failure_count = 0
        self.opened = False
        self.state = CircuitState.CLOSED
        self.opened_at_iteration: int | None = None

    def check(self, state: AgentState) -> tuple[bool, str]:
        iterations = int(state.get("iterations", 0))
        if self.state == CircuitState.OPEN:
            opened_at = self.opened_at_iteration if self.opened_at_iteration is not None else iterations
            if iterations - opened_at >= self.cooldown_iterations:
                self._transition(state, CircuitState.HALF_OPEN)
                return False, ""
            return True, "circuit breaker open"

        if self.state == CircuitState.HALF_OPEN:
            return False, ""

        if iterations > self.max_iterations:
            self._open(state)
            self._log_event(
                state=state,
                condition="max_iterations",
                value=iterations,
                threshold=self.max_iterations,
            )
            return True, "iterations exceeded maximum"

        for filename, edit_count in state.get("file_edits", {}).items():
            if int(edit_count) > self.max_edits_per_file:
                self._open(state)
                self._log_event(
                    state=state,
                    condition="max_edits_per_file",
                    value={"file": filename, "edits": edit_count},
                    threshold=self.max_edits_per_file,
                )
                return True, f"{filename} exceeded edit limit"

        failures = state.get("verification", {}).get("failures", [])
        repeated_errors = Counter(str(error) for error in failures)
        for error, count in repeated_errors.items():
            if count > self.max_same_error:
                self._open(state)
                self._log_event(
                    state=state,
                    condition="max_same_error",
                    value={"error": error, "count": count},
                    threshold=self.max_same_error,
                )
                return True, "same verification error repeated too often"

        return False, ""

    def is_tripped(self, state: AgentState) -> bool:
        return self.check(state)[0]

    def allow_request(self) -> bool:
        return not self.opened

    def record_success(self) -> None:
        self.failure_count = 0
        self.opened = False
        self.state = CircuitState.CLOSED
        self.opened_at_iteration = None

    def record_failure(self) -> None:
        self.failure_count += 1
        if self.failure_count >= self.max_failures:
            self.opened = True
            self.state = CircuitState.OPEN

    def call(self, operation: Callable[[], T]) -> T:
        if not self.allow_request():
            raise CircuitOpenError("Circuit is open after repeated failures.")

        try:
            result = operation()
        except Exception:
            self.record_failure()
            raise

        self.record_success()
        return result

    def record_trial_success(self, state: AgentState) -> None:
        if self.state == CircuitState.HALF_OPEN:
            self._transition(state, CircuitState.CLOSED)
            self.failure_count = 0
            self.opened = False
            self.opened_at_iteration = None

    def record_trial_failure(self, state: AgentState) -> None:
        if self.state == CircuitState.HALF_OPEN:
            self._transition(state, CircuitState.OPEN)
            self.opened = True
            self.opened_at_iteration = int(state.get("iterations", 0))

    def _log_event(
        self,
        state: AgentState,
        condition: str,
        value: object,
        threshold: int,
    ) -> None:
        events = state.setdefault("harness_events", [])
        events.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "type": "circuit_breaker",
                "condition": condition,
                "value": value,
                "threshold": threshold,
            }
        )

    def _open(self, state: AgentState) -> None:
        self.opened = True
        self.opened_at_iteration = int(state.get("iterations", 0))
        self._transition(state, CircuitState.OPEN)

    def _transition(self, state: AgentState, new_state: CircuitState) -> None:
        old_state = self.state
        if old_state == new_state:
            return
        self.state = new_state
        events = state.setdefault("harness_events", [])
        events.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "type": "circuit_breaker_state_change",
                "old_state": old_state.value,
                "new_state": new_state.value,
                "iteration": int(state.get("iterations", 0)),
            }
        )
        structlog.get_logger("harness").info(
            "circuit_breaker_state_change",
            old_state=old_state.value,
            new_state=new_state.value,
            iteration=int(state.get("iterations", 0)),
        )
