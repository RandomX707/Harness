"""Runtime validation for LangGraph node outputs."""

from __future__ import annotations

from harness.state import AgentState


class SchemaViolationError(RuntimeError):
    """Raised when a node emits keys outside the declared AgentState schema."""


def validate_node_output(node_name: str, output: dict) -> None:
    declared_keys = set(AgentState.__annotations__)
    undeclared_keys = sorted(key for key in output if key not in declared_keys)
    if undeclared_keys:
        raise SchemaViolationError(
            f"Node {node_name} returned undeclared AgentState keys: {undeclared_keys}. "
            f"Declared keys: {sorted(declared_keys)}"
        )
