from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from harness.state import AgentState
from harness.pricing import calculate_cost
from harness.inferential_verifier import InferentialVerificationResult
from agent.graph import build_graph, initial_state
from agent.rag_executor import rag_executor, rag_verifier
from agent.tools import write_file


def test_plan_execute_verify_retries_until_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    verification_results = iter([(False, "first failure"), (False, "second failure"), (True, "ok")])

    def executor(state: AgentState) -> AgentState:
        return {
            **state,
            "messages": state.get("messages", []) + ["executor ran"],
        }

    def verifier(state: AgentState) -> tuple[bool, str]:
        return next(verification_results)

    app = build_graph(executor=executor, verifier=verifier)
    result = app.invoke(initial_state("exercise retry loop"))

    assert result["done"] is True
    assert result["attempts"] == 3
    assert result["last_error"] == ""


def test_plan_execute_verify_stops_after_three_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)

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
    state["budget"] = {"tokens_used": 100, "tokens_max": 100, "cost_usd": 0.0, "max_attempts": 3}

    app = build_graph()
    result = app.invoke(state)

    assert result["output"] == "budget_exceeded"


def test_budget_uses_real_usage_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_API_KEY", "test-key")
    monkeypatch.setenv("LITELLM_BASE_URL", "https://example.com/v1")

    class FakeReactAgent:
        def invoke(self, payload):
            return {
                "messages": [
                    AIMessage(
                        content="done",
                        usage_metadata={
                            "input_tokens": 1000,
                            "output_tokens": 500,
                            "total_tokens": 1500,
                        },
                    )
                ]
            }

    monkeypatch.setattr("agent.nodes.create_react_agent", lambda *args, **kwargs: FakeReactAgent())

    def verifier(state: AgentState) -> tuple[bool, str]:
        return True, "exit_code=0\n1 passed"

    state = initial_state("usage tracking")
    state["plan"] = ["single step"]
    app = build_graph(verifier=verifier)
    result = app.invoke(state)

    assert result["budget"]["tokens_used"] == 1500
    assert result["budget"]["cost_usd"] == pytest.approx(calculate_cost("gpt-4o-mini", 1000, 500))


def test_inferential_verification_does_not_block_passing_task(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)

    class FakeRunTests:
        def invoke(self, args):
            return "exit_code=0\n1 passed"

    monkeypatch.setattr("agent.nodes.run_tests", FakeRunTests())
    monkeypatch.setattr(
        "agent.nodes.judge_code_quality",
        lambda plan_step, diff_summary: InferentialVerificationResult(
            score=0.2,
            reasoning="quality concern",
            flagged_issues=["soft warning"],
        ),
    )

    state = initial_state("soft inferential warning")
    state["plan"] = ["single step"]
    app = build_graph()
    result = app.invoke(state)

    assert result["verification"]["passed"] is True
    assert result["done"] is True
    assert result["verification"]["inferential"]["passed"] is False


def test_rag_executor_completes_via_pev_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    state = initial_state("What is harness engineering?")
    state["plan"] = [
        "Retrieve candidate context documents.",
        "Grade relevance and correct with stub search when needed.",
        "Generate and verify a grounded answer.",
    ]

    app = build_graph(executor=rag_executor, verifier=rag_verifier)
    result = app.invoke(state)

    assert result["done"] is True
    assert result["verification"]["passed"] is True
    assert "Stub RAG answer" in result["output"]
    assert any(event.get("type") == "rag_execution" for event in result["harness_events"])
    assert result["file_edits"] == {}
