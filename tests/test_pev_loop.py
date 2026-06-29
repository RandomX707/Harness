from __future__ import annotations

import pytest

from harness.state import AgentState
from agent.graph import build_graph, initial_state
from agent.tools import write_file


def test_plan_execute_verify_retries_until_success() -> None:
    verification_results = iter([(False, "first failure"), (False, "second failure"), (True, "ok")])

    def executor(state: AgentState) -> AgentState:
        return {
            **state,
            "messages": state.get("messages", []) + ["executor ran"],
            "current_action": "verify",
        }

    def verifier(state: AgentState) -> tuple[bool, str]:
        return next(verification_results)

    app = build_graph(executor=executor, verifier=verifier)
    result = app.invoke(initial_state("exercise retry loop"))

    assert result["done"] is True
    assert result["attempts"] == 3
    assert result["last_error"] == ""


def test_plan_execute_verify_stops_after_three_failures() -> None:
    def verifier(state: AgentState) -> tuple[bool, str]:
        return False, "still failing"

    app = build_graph(verifier=verifier)
    result = app.invoke(initial_state("exercise stop loop"))

    assert result["done"] is False
    assert result["attempts"] == 3
    assert result["last_error"] == "still failing"
    assert "Stopped after repeated verification failures." in result["messages"]


def test_simple_task_completes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)

    def verifier(state: AgentState) -> tuple[bool, str]:
        return True, "exit_code=0\n1 passed"

    app = build_graph(verifier=verifier)
    result = app.invoke(initial_state("simple"))

    assert result["verification"]["passed"] is True


def test_permission_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)

    def executor(state: AgentState) -> AgentState:
        try:
            write_file.invoke({"path": ".harness/evil.txt", "content": "hacked"})
        except PermissionError:
            pass
        return state

    def verifier(state: AgentState) -> tuple[bool, str]:
        return True, "exit_code=0\n1 passed"

    app = build_graph(executor=executor, verifier=verifier)
    result = app.invoke(initial_state("permission_test"))

    denied_events = [
        event
        for event in result["harness_events"]
        if event.get("type") == "tool_call"
        and event.get("tool") == "write_file"
        and event.get("allowed") is False
        and event.get("args", {}).get("path") == ".harness/evil.txt"
    ]
    assert denied_events


def test_circuit_breaker_fires(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)

    def verifier(state: AgentState) -> tuple[bool, str]:
        return False, "still failing"

    app = build_graph(verifier=verifier)
    result = app.invoke(initial_state("doom_loop_test", max_attempts=10))

    assert any(event.get("type") == "circuit_breaker" for event in result["harness_events"])
    assert "CIRCUIT_OPEN" in result["output"]


def test_budget_enforcement(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    state = initial_state("tiny budget task")
    state["budget"] = {"tokens_used": 0, "tokens_max": 100, "cost_usd": 0.0, "max_attempts": 3}

    app = build_graph()
    result = app.invoke(state)

    assert result["output"] == "budget_exceeded"
