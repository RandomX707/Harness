"""Read-only reviewer subagent with its own scoped harness."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from harness.circuit_breaker import CircuitBreaker
from harness.permission_resolver import PermissionResolver, RiskLevel, ToolPolicy
from harness.state import AgentState


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REVIEWER_PERMISSIONS = PermissionResolver(
    [
        ToolPolicy("read_file", RiskLevel.READ_ONLY, None, False),
        ToolPolicy("list_files", RiskLevel.READ_ONLY, None, False),
        ToolPolicy("request_review", RiskLevel.READ_ONLY, None, False),
    ],
    project_root=PROJECT_ROOT,
)
REVIEWER_CIRCUIT_BREAKER = CircuitBreaker(max_iterations=5)


def review_changes(state: AgentState) -> dict:
    """Inspect edited files and return a structured read-only review."""

    if not os.getenv("LITELLM_API_KEY"):
        return {"approved": True, "summary": "reviewer skipped: no API key", "concerns": []}

    is_open, reason = REVIEWER_CIRCUIT_BREAKER.check({**state, "iterations": 1})
    if is_open:
        return {"approved": False, "summary": f"reviewer circuit open: {reason}", "concerns": [reason]}

    file_context = _file_context(state.get("file_edits", {}))
    system_prompt = (
        "Review the following file changes for correctness, style, and whether they "
        "match the stated task. Be specific about concerns. Return ONLY JSON with "
        "approved, summary, and concerns keys."
    )
    human_prompt = f"Task:\n{state.get('task', '')}\n\nFiles:\n{file_context}"

    try:
        model = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=os.getenv("LITELLM_API_KEY"),
            base_url=os.getenv("LITELLM_BASE_URL"),
        )
        response = model.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)])
        return _parse_review(_message_content(response))
    except Exception as exc:
        return {"approved": False, "summary": f"reviewer failed: {exc}", "concerns": [str(exc)]}


def _file_context(file_edits: dict[str, int]) -> str:
    if not file_edits:
        return "No file edits recorded."

    chunks: list[str] = []
    for path in sorted(file_edits):
        content = _reviewer_read_file(path)
        chunks.append(f"## {path}\n{content[:4000]}")
    return "\n\n".join(chunks)


def _reviewer_read_file(path: str) -> str:
    allowed, reason = REVIEWER_PERMISSIONS.check("read_file", {"path": path})
    if not allowed:
        raise PermissionError(f"reviewer read_file denied: {reason}")
    target = (PROJECT_ROOT / path).resolve()
    return target.read_text(encoding="utf-8")


def _parse_review(content: str) -> dict:
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise ValueError("review response was not a JSON object")
    concerns = parsed.get("concerns", [])
    if not isinstance(concerns, list):
        concerns = [str(concerns)]
    return {
        "approved": bool(parsed.get("approved", False)),
        "summary": str(parsed.get("summary", "")),
        "concerns": [str(concern) for concern in concerns],
    }


def _message_content(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    return str(content)
