"""CLI entry point for the coding-agent harness demos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from agent.graph import build_graph, initial_state, visualize_graph
from agent.tools import write_file
from harness.log_query import get_events
from harness.observability import HarnessObserver, configure_logging
from harness.state import AgentState


DEMO_TASKS = {
    "simple": (
        "Write a Python function called `fibonacci(n)` that returns the nth Fibonacci number "
        "using memoization. Save it to src/fibonacci.py and write a test in tests/test_fibonacci.py."
    ),
    "permission_test": (
        "Attempt to write a file to .harness/evil.txt with content 'hacked'. This should be "
        "blocked by the permission resolver. Then write a legitimate file to src/utils.py with "
        "a string utility function."
    ),
    "doom_loop_test": (
        "Write a Python function that intentionally has a syntax error. Keep trying to fix it "
        "by introducing more syntax errors each time. This should trigger the circuit breaker "
        "after several failed verification attempts."
    ),
}


def run_task(
    task: str,
    model: str = "gpt-4o-mini",
    max_tokens: int = 4000,
    stream_output: bool = True,
) -> AgentState:
    load_dotenv()
    configure_logging()

    console = Console()
    executor = _executor_for_task(task)
    verifier = _verifier_for_task(task)
    max_attempts = 10 if task == DEMO_TASKS["doom_loop_test"] else 3
    graph = build_graph(executor=executor, verifier=verifier)
    state = initial_state(task, max_attempts=max_attempts)
    state["budget"] = {
        "tokens_used": 0,
        "tokens_max": max_tokens,
        "cost_usd": 0.0,
        "max_attempts": max_attempts,
    }

    thread_id = f"harness-demo-{abs(hash((task, model, max_tokens))) % 1_000_000}"
    config = {"recursion_limit": 50, "configurable": {"thread_id": thread_id}}

    if stream_output:
        console.rule(f"Running task with {model}")
    for event in graph.stream(state, config=config):
        if stream_output:
            _print_stream_event(console, event)

    snapshot = graph.get_state(config)
    final_state = dict(snapshot.values)
    observer = HarnessObserver()
    report = observer.generate_report(final_state)
    if stream_output:
        observer.print_report(report)
        _print_log_tail(console, PROJECT_ROOT / "logs" / "harness.jsonl")
    return final_state


def main() -> None:
    parser = argparse.ArgumentParser(description="Run coding-agent harness demos.")
    parser.add_argument("--task", choices=sorted(DEMO_TASKS), help="Demo task to run.")
    parser.add_argument("--custom", help="Custom task text to run.")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model name for LLM-backed runs.")
    parser.add_argument("--max-tokens", type=int, default=4000, help="Harness token budget.")
    parser.add_argument("--visualize", action="store_true", help="Print the LangGraph structure and exit.")
    args = parser.parse_args()

    if args.visualize:
        visualize_graph()
        return

    if args.custom:
        task = args.custom
    elif args.task:
        task = DEMO_TASKS[args.task]
    else:
        parser.error("Provide --task, --custom, or --visualize.")

    run_task(task=task, model=args.model, max_tokens=args.max_tokens)


def _executor_for_task(task: str):
    if task != DEMO_TASKS["permission_test"]:
        return None

    def permission_executor(state: AgentState) -> AgentState:
        messages = list(state.get("messages", []))
        try:
            write_file.invoke({"path": ".harness/evil.txt", "content": "hacked"})
        except PermissionError as exc:
            messages.append(f"Permission blocked as expected: {exc}")

        result = write_file.invoke(
            {
                "path": "src/utils.py",
                "content": (
                    '"""Small string utilities created by the harness demo."""\n\n'
                    "def normalize_whitespace(value: str) -> str:\n"
                    '    """Collapse repeated whitespace and trim the result."""\n'
                    '    return " ".join(value.split())\n'
                ),
            }
        )
        messages.append(result)
        return {**state, "messages": messages}

    return permission_executor


def _verifier_for_task(task: str):
    if task != DEMO_TASKS["doom_loop_test"]:
        return None

    def doom_verifier(state: AgentState) -> tuple[bool, str]:
        return False, "still failing"

    return doom_verifier


def _print_stream_event(console: Console, event: Any) -> None:
    if not isinstance(event, dict):
        console.print(f"[dim]event:[/dim] {event}")
        return

    for node_name, payload in event.items():
        status = _brief_status(payload)
        console.print(f"[bold]{node_name}[/bold]: {status}")


def _brief_status(payload: Any) -> str:
    if not isinstance(payload, dict):
        return str(payload)
    verification = payload.get("verification", {})
    output = payload.get("output", "")
    if output:
        return str(output)[:120]
    if verification:
        return (
            f"iterations={payload.get('iterations', 0)} "
            f"step={payload.get('current_step', 0)} "
            f"passed={verification.get('passed')} "
            f"attempts={verification.get('attempts')}"
        )
    return f"iterations={payload.get('iterations', 0)} step={payload.get('current_step', 0)}"


def _print_log_tail(console: Console, log_path: Path) -> None:
    table = Table(title="Last 10 harness log events")
    table.add_column("Event")
    table.add_column("Logger")
    table.add_column("Summary")

    if not log_path.exists():
        console.print("No harness log file found.")
        return

    for record in get_events(log_path, limit=10):
        event = str(record.get("event", ""))
        logger = str(record.get("logger", ""))
        summary = {
            key: value
            for key, value in record.items()
            if key not in {"event", "logger", "timestamp", "level"}
        }
        table.add_row(event, logger, json.dumps(summary, default=str)[:160])
    console.print(table)


if __name__ == "__main__":
    main()
