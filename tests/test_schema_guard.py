from __future__ import annotations

import pytest

from harness.schema_guard import SchemaViolationError, validate_node_output
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
        "task": "test",
        "output": "",
    }


def test_declared_agent_state_keys_pass() -> None:
    validate_node_output("good_node", make_state())


def test_undeclared_agent_state_key_raises() -> None:
    output = {**make_state(), "task_complete": True}

    with pytest.raises(SchemaViolationError):
        validate_node_output("bad_node", output)


def test_exception_message_names_offending_key_and_node() -> None:
    output = {**make_state(), "task_complete": True}

    with pytest.raises(SchemaViolationError) as exc_info:
        validate_node_output("bad_node", output)

    message = str(exc_info.value)
    assert "bad_node" in message
    assert "task_complete" in message
