from __future__ import annotations

import pytest

from agent.reviewer import REVIEWER_CIRCUIT_BREAKER, REVIEWER_PERMISSIONS, review_changes
from agent.tools import TOOLS, request_review
from harness.state import AgentState


def make_state() -> AgentState:
    return {
        "messages": [],
        "plan": [],
        "current_step": 0,
        "iterations": 0,
        "file_edits": {"src/example.py": 1},
        "verification": {"passed": False, "failures": [], "attempts": 0},
        "budget": {"tokens_used": 0, "tokens_max": 1000, "cost_usd": 0.0},
        "harness_events": [],
        "task": "review this change",
        "output": "",
    }


def test_review_changes_returns_stub_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)

    result = review_changes(make_state())

    assert result == {"approved": True, "summary": "reviewer skipped: no API key", "concerns": []}


def test_reviewer_permission_resolver_rejects_write_attempt() -> None:
    allowed, reason = REVIEWER_PERMISSIONS.check("write_file", {"path": "src/example.py"})

    assert allowed is False
    assert reason == "unknown tool: write_file"


def test_reviewer_circuit_breaker_has_tight_limit() -> None:
    assert REVIEWER_CIRCUIT_BREAKER.max_iterations == 5


def test_request_review_tool_is_present_and_permission_gated(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict]] = []

    class FakePermissions:
        blocked_patterns: list[str] = []

        def check(self, tool_name: str, args: dict) -> tuple[bool, str]:
            calls.append((tool_name, args))
            return True, "allowed"

    monkeypatch.setattr("agent.tools.PERMISSIONS", FakePermissions())
    monkeypatch.setattr(
        "agent.tools.review_changes",
        lambda state: {"approved": True, "summary": "ok", "concerns": []},
    )

    assert any(tool.name == "request_review" for tool in TOOLS)
    output = request_review.invoke({"reason": "before finishing"})

    assert "review approved=True" in output
    assert calls == [("request_review", {"reason": "before finishing"})]
