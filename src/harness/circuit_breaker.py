"""Circuit breaker for repeated harness failures."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Callable, TypeVar

from harness.state import AgentState


T = TypeVar("T")


class CircuitOpenError(RuntimeError):
    """Raised when the circuit is open and work should stop."""


class CircuitBreaker:
    """Stop an agent loop when bounded harness conditions are exceeded."""

    def __init__(
        self,
        max_iterations: int = 15,
        max_edits_per_file: int = 5,
        max_same_error: int = 3,
        max_failures: int | None = None,
    ) -> None:
        self.max_iterations = max_iterations
        self.max_edits_per_file = max_edits_per_file
        self.max_same_error = max_same_error
        self.max_failures = max_failures if max_failures is not None else max_same_error
        self.failure_count = 0
        self.opened = False

    def check(self, state: AgentState) -> tuple[bool, str]:
        iterations = int(state.get("iterations", 0))
        if iterations > self.max_iterations:
            self._log_event(
                state=state,
                condition="max_iterations",
                value=iterations,
                threshold=self.max_iterations,
            )
            return True, "iterations exceeded maximum"

        for filename, edit_count in state.get("file_edits", {}).items():
            if int(edit_count) > self.max_edits_per_file:
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

    def record_failure(self) -> None:
        self.failure_count += 1
        if self.failure_count >= self.max_failures:
            self.opened = True

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
