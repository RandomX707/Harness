"""Structured observability for harness decisions, tools, and graph nodes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
import logging
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, TypeVar, cast

from rich.console import Console
from rich.table import Table
import structlog

from harness.schema_guard import SchemaViolationError, validate_node_output
from harness.state import AgentState


F = TypeVar("F", bound=Callable[..., Any])
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_PATH = PROJECT_ROOT / "logs" / "harness.jsonl"
_LOGGING_CONFIGURED = False


def configure_logging() -> None:
    """Configure structlog for console output and JSONL file output."""

    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    json_formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processor=structlog.processors.JSONRenderer(),
    )
    console_formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processor=structlog.dev.ConsoleRenderer(colors=False),
    )

    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(json_formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging.INFO)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    _LOGGING_CONFIGURED = True


def get_logger(name: str = "coding_agent_harness") -> structlog.BoundLogger:
    configure_logging()
    return structlog.get_logger(name)


@dataclass(frozen=True)
class SessionReport:
    task: str
    total_iterations: int
    tokens_used: int
    files_modified: list[str]
    verification_attempts: int
    circuit_breaker_trips: int
    final_status: str


class HarnessObserver:
    """Emit structured harness events and human-readable session reports."""

    def __init__(self, logger_name: str = "harness") -> None:
        self.logger = get_logger(logger_name)

    def log_tool_call(self, tool: str, args: dict, allowed: bool, reason: str) -> None:
        self.logger.info("tool_call", tool=tool, args=args, allowed=allowed, reason=reason)

    def log_iteration(self, state: AgentState) -> None:
        budget = state.get("budget", {})
        self.logger.info(
            "iteration",
            iteration=state.get("iterations", 0),
            tokens_used=budget.get("tokens_used", 0),
            plan_step=state.get("current_step", 0),
            files_edited_count=len(state.get("file_edits", {})),
        )

    def log_verification(self, passed: bool, failures: list[str], attempt: int) -> None:
        self.logger.info("verification", passed=passed, failures=failures, attempt=attempt)

    def log_circuit_breaker(self, condition: str, value: object, threshold: int) -> None:
        self.logger.info(
            "circuit_breaker_trip",
            condition=condition,
            value=value,
            threshold=threshold,
        )

    def log_hitl(self, tool: str, approved: bool) -> None:
        self.logger.info("hitl_approval", tool=tool, approved=approved)

    def log_task_complete(self, state: AgentState, success: bool) -> None:
        report = generate_report(state)
        self.logger.info(
            "task_complete",
            success=success,
            task=report.task,
            total_iterations=report.total_iterations,
            tokens_used=report.tokens_used,
            files_modified=report.files_modified,
            verification_attempts=report.verification_attempts,
            circuit_breaker_trips=report.circuit_breaker_trips,
            final_status=report.final_status,
        )

    @staticmethod
    def generate_report(state: AgentState) -> SessionReport:
        return generate_report(state)

    @staticmethod
    def print_report(report: SessionReport) -> None:
        print_report(report)


def generate_report(state: AgentState) -> SessionReport:
    verification = state.get("verification", {})
    budget = state.get("budget", {})
    events = state.get("harness_events", [])
    circuit_breaker_trips = sum(
        1
        for event in events
        if event.get("type") in {"circuit_breaker", "circuit_breaker_trip"}
    )
    passed = bool(verification.get("passed", False))
    return SessionReport(
        task=state.get("task", ""),
        total_iterations=int(state.get("iterations", 0)),
        tokens_used=int(budget.get("tokens_used", 0)),
        files_modified=sorted(state.get("file_edits", {}).keys()),
        verification_attempts=int(verification.get("attempts", 0)),
        circuit_breaker_trips=circuit_breaker_trips,
        final_status="success" if passed else "failed",
    )


def print_report(report: SessionReport) -> None:
    table = Table(title="Harness Session Report")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Task", report.task)
    table.add_row("Final status", report.final_status)
    table.add_row("Iterations", str(report.total_iterations))
    table.add_row("Tokens used", str(report.tokens_used))
    table.add_row("Files modified", ", ".join(report.files_modified) or "none")
    table.add_row("Verification attempts", str(report.verification_attempts))
    table.add_row("Circuit breaker trips", str(report.circuit_breaker_trips))
    Console().print(table)


def traced_node(name: str) -> Callable[[F], F]:
    """Log node entry and exit without coupling nodes to a logger."""

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger = get_logger("agent.nodes").bind(node=name)
            state = args[0] if args and isinstance(args[0], dict) else None
            logger.info("node_start")
            if state is not None:
                HarnessObserver("agent.nodes").log_iteration(cast(AgentState, state))
            started = perf_counter()
            result = func(*args, **kwargs)
            try:
                validate_node_output(name, result)
            except SchemaViolationError as exc:
                logger.error("schema_violation", error=str(exc))
                raise
            logger.info("node_finish", elapsed_ms=round((perf_counter() - started) * 1000, 3))
            return result

        return cast(F, wrapper)

    return decorator
