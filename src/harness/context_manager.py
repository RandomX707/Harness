"""Context assembly for a constrained coding-agent prompt."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harness.state import AgentState


@dataclass
class HarnessContextManager:
    project_root: Path | None = None
    max_messages: int = 8

    def load_agent_instructions(self) -> str:
        root = self.project_root or Path.cwd()
        agents_file = root / "AGENTS.md"
        return agents_file.read_text(encoding="utf-8")

    def build_context(self, state: AgentState) -> str:
        recent_messages = state.get("messages", [])[-self.max_messages :]
        touched = ", ".join(state.get("file_edits", {}).keys()) or "none"
        return "\n".join(
            [
                "# Task",
                state["task"],
                "# Recent messages",
                *recent_messages,
                "# Files touched",
                touched,
                "# Harness instructions",
                self.load_agent_instructions(),
            ]
        )

    def inject_harness_context(self, state: AgentState) -> str:
        budget = state.get("budget", {})
        tokens_used = int(budget.get("tokens_used", 0))
        tokens_max = int(budget.get("tokens_max", 0))
        tokens_remaining = max(tokens_max - tokens_used, 0)
        edited_files = sorted(state.get("file_edits", {}).keys())
        warning = self._latest_circuit_breaker_warning(state)

        return "\n".join(
            [
                "# Harness Context",
                f"Current iterations: {state.get('iterations', 0)}",
                f"Budget remaining: {tokens_remaining} tokens",
                f"Circuit breaker warnings: {warning}",
                "Files edited this session:",
                *(edited_files or ["none"]),
            ]
        )

    def should_compact(self, state: AgentState) -> bool:
        budget = state.get("budget", {})
        tokens_used = float(budget.get("tokens_used", 0))
        tokens_max = float(budget.get("tokens_max", 0))
        return tokens_max > 0 and tokens_used > 0.7 * tokens_max

    def compact_messages(self, state: AgentState) -> list:
        messages = state.get("messages", [])
        system_message = self._first_message_with_role(messages, "system")
        tool_errors = [
            message
            for message in messages
            if self._message_role(message) == "tool" and self._contains_error(message)
        ]
        selected = []
        for message in [system_message, *tool_errors, *messages[-10:]]:
            if message is not None and not any(message is existing for existing in selected):
                selected.append(message)
        return selected

    def _first_message_with_role(self, messages: list, role: str) -> Any | None:
        for message in messages:
            if self._message_role(message) == role:
                return message
        return None

    def _message_role(self, message: Any) -> str:
        if isinstance(message, dict):
            return str(message.get("role", ""))
        return str(getattr(message, "role", getattr(message, "type", "")))

    def _contains_error(self, message: Any) -> bool:
        if isinstance(message, dict):
            content = str(message.get("content", ""))
        else:
            content = str(getattr(message, "content", message))
        lowered = content.lower()
        return any(marker in lowered for marker in ["error", "exception", "traceback", "failed"])

    def _latest_circuit_breaker_warning(self, state: AgentState) -> str:
        for event in reversed(state.get("harness_events", [])):
            if event.get("type") == "circuit_breaker":
                return str(event.get("condition", "open"))
        return "none"


ContextManager = HarnessContextManager
