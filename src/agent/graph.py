"""LangGraph assembly for the coding-agent harness demonstration."""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    Executor,
    Verifier,
    agent_node,
    feedback_injector_node,
    fail_node,
    harness_guard_node,
    make_execute_node,
    make_verify_node,
    plan_node,
    planner_node,
    reset_circuit_breaker,
    verify_node,
)
from harness.state import AgentState


def initial_state(task: str, max_attempts: int = 3) -> AgentState:
    state = {
        "messages": [],
        "plan": [],
        "current_step": 0,
        "iterations": 0,
        "file_edits": {},
        "verification": {"passed": False, "failures": [], "attempts": 0},
        "budget": {"tokens_used": 0, "tokens_max": 10000, "cost_usd": 0.0, "max_attempts": max_attempts},
        "harness_events": [],
        "task": task,
        "output": "",
    }
    return state


def route_after_verify(state: AgentState) -> str:
    verification = state.get("verification", {})
    if verification.get("passed") and verification.get("task_complete", True):
        return END
    if state.get("output") in {"budget_exceeded", "iterations exceeded maximum"}:
        return END
    max_attempts = int(state.get("budget", {}).get("max_attempts", 3))
    if verification.get("attempts", 0) >= max_attempts:
        return END
    if verification.get("passed"):
        return "harness_guard"
    return "feedback_injector"


def route_from_start(state: AgentState) -> str:
    if state.get("plan"):
        return "harness_guard"
    return "planner"


def route_after_guard(state: AgentState) -> str:
    if state.get("output") == "budget_exceeded":
        return END
    if str(state.get("output", "")).startswith("CIRCUIT_OPEN"):
        return END
    return "agent"


class CompiledCodingGraph:
    def __init__(self, graph: Any) -> None:
        self._graph = graph

    def invoke(self, state: AgentState, config: dict | None = None) -> dict:
        runtime_config = self._runtime_config(config)
        result = dict(self._graph.invoke(state, config=runtime_config))
        verification = result.get("verification", {})
        failures = verification.get("failures", [])
        result["done"] = bool(verification.get("passed", False))
        result["attempts"] = int(verification.get("attempts", 0))
        result["last_error"] = "" if result["done"] else (failures[-1] if failures else "")
        result["messages"] = [self._message_content(message) for message in result.get("messages", [])]
        return result

    def get_graph(self) -> Any:
        return self._graph.get_graph()

    def stream(self, state: AgentState, config: dict | None = None) -> Any:
        runtime_config = self._runtime_config(config)
        return self._graph.stream(state, config=runtime_config)

    def get_state(self, config: dict | None = None) -> Any:
        runtime_config = self._runtime_config(config)
        return self._graph.get_state(runtime_config)

    def _message_content(self, message: Any) -> Any:
        return getattr(message, "content", message)

    def _runtime_config(self, config: dict | None = None) -> dict:
        runtime_config = dict(config or {})
        configurable = dict(runtime_config.get("configurable", {}))
        configurable.setdefault("thread_id", "coding-agent-harness-default")
        runtime_config["configurable"] = configurable
        return runtime_config


def build_graph(
    executor: Executor | None = None,
    verifier: Verifier | None = None,
) -> CompiledCodingGraph:
    reset_circuit_breaker()
    graph = StateGraph(AgentState)
    agent_runnable = make_execute_node(executor) if executor is not None else agent_node
    verify_runnable = make_verify_node(verifier) if verifier is not None else verify_node

    graph.add_node("plan", plan_node)
    graph.add_node("planner", planner_node)
    graph.add_node("harness_guard", harness_guard_node)
    graph.add_node("agent", agent_runnable)
    graph.add_node("verify", verify_runnable)
    graph.add_node("feedback_injector", feedback_injector_node)
    graph.add_node("fail", fail_node)

    graph.add_conditional_edges(START, route_from_start)
    graph.add_edge("planner", "harness_guard")
    graph.add_conditional_edges("harness_guard", route_after_guard)
    graph.add_edge("agent", "verify")
    graph.add_conditional_edges("verify", route_after_verify)
    graph.add_edge("feedback_injector", "harness_guard")
    return CompiledCodingGraph(graph.compile(checkpointer=MemorySaver()))


def visualize_graph() -> None:
    graph = build_graph()
    try:
        graph.get_graph().print_ascii()
    except ImportError:
        print(
            "START\n"
            "  -> planner_node | harness_guard_node\n"
            "  -> harness_guard_node\n"
            "  -> agent_node\n"
            "  -> verify_node\n"
            "  -> feedback_injector_node -> harness_guard_node\n"
            "  -> fail_node\n"
            "  -> END"
        )
