from __future__ import annotations

from agent.graph import build_graph


def test_to_mermaid_contains_graph_td() -> None:
    graph = build_graph()

    assert "graph TD" in graph.to_mermaid()


def test_to_mermaid_contains_known_nodes() -> None:
    graph = build_graph()
    mermaid = graph.to_mermaid()

    for node_name in ["planner", "harness_guard", "agent", "verify", "feedback_injector", "fail"]:
        assert node_name in mermaid


def test_to_mermaid_contains_transition_arrows() -> None:
    graph = build_graph()
    mermaid = graph.to_mermaid()

    assert "START -->" in mermaid
    assert "planner --> harness_guard" in mermaid
    assert "harness_guard -->" in mermaid
    assert "agent --> verify" in mermaid
    assert "verify -->" in mermaid
    assert "feedback_injector --> harness_guard" in mermaid
