"""Shared LangGraph state for the demonstration coding agent."""

from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """State carried through plan, execute, and verify nodes."""

    messages: Annotated[list, add_messages]
    plan: list[str]
    current_step: int
    iterations: int
    file_edits: dict[str, int]
    verification: dict
    budget: dict
    harness_events: list[dict]
    task: str
    output: str
