"""LangChain tools exposed to the coding agent."""

from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path
import subprocess
import sys

from langchain_core.tools import tool

from agent.reviewer import review_changes
from harness.observability import HarnessObserver
from harness.permission_resolver import PermissionResolver
from harness.state import AgentState


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PERMISSIONS = PermissionResolver(PermissionResolver.default_policy(), project_root=PROJECT_ROOT)
OBSERVER = HarnessObserver("agent.tools")
_ACTIVE_STATE: AgentState | None = None


def bind_tool_state(state: AgentState | None) -> None:
    """Attach graph state for tool-side edit accounting without changing tool signatures."""

    global _ACTIVE_STATE
    _ACTIVE_STATE = state


def _check_permission(tool_name: str, args: dict) -> None:
    allowed, reason = PERMISSIONS.check(tool_name, args)
    OBSERVER.log_tool_call(tool_name, args, allowed, reason)
    if _ACTIVE_STATE is not None:
        events = _ACTIVE_STATE.setdefault("harness_events", [])
        events.append(
            {
                "type": "tool_call",
                "tool": tool_name,
                "args": args,
                "allowed": allowed,
                "reason": reason,
            }
        )
    if not allowed:
        raise PermissionError(f"{tool_name} denied: {reason}")


def _resolve_project_path(path: str) -> Path:
    target = Path(path)
    if not target.is_absolute():
        target = PROJECT_ROOT / target
    return target.resolve()


def _relative_path(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n... truncated to {limit} chars ..."


def _is_blocked(relative: str) -> bool:
    for pattern in PERMISSIONS.blocked_patterns:
        if fnmatch(relative, pattern):
            return True
        if pattern.endswith("/*"):
            blocked_root = pattern[:-2]
            if relative == blocked_root or relative.startswith(f"{blocked_root}/"):
                return True
    return False


def _record_file_edit(relative: str) -> None:
    if _ACTIVE_STATE is None:
        return
    edits = _ACTIVE_STATE.setdefault("file_edits", {})
    edits[relative] = int(edits.get(relative, 0)) + 1


@tool
def read_file(path: str) -> str:
    """Read a UTF-8 text file relative to the project root."""

    args = {"path": path}
    _check_permission("read_file", args)
    try:
        target = _resolve_project_path(path)
        return target.read_text(encoding="utf-8")
    except Exception as exc:
        return f"error reading {path}: {exc}"


@tool
def write_file(path: str, content: str) -> str:
    """Write UTF-8 text under allowed project paths."""

    args = {"path": path, "content_length": len(content)}
    _check_permission("write_file", args)
    try:
        target = _resolve_project_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        relative = _relative_path(target)
        _record_file_edit(relative)
        return f"written: {relative}"
    except Exception as exc:
        return f"error writing {path}: {exc}"


@tool
def run_code(code: str, timeout: int = 10) -> str:
    """Execute Python code in a subprocess and return truncated output."""

    args = {"code": code, "timeout": timeout}
    _check_permission("run_code", args)
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        output = completed.stdout + completed.stderr
        return _truncate(output, 2000)
    except Exception as exc:
        return _truncate(f"error running code: {exc}", 2000)


@tool
def list_files(directory: str = "src/") -> str:
    """List project files under a directory while respecting blocked patterns."""

    args = {"path": directory}
    _check_permission("list_files", args)
    try:
        root = _resolve_project_path(directory)
        if not root.exists():
            return f"error listing {directory}: directory does not exist"
        if not root.is_dir():
            return f"error listing {directory}: not a directory"

        files: list[str] = []
        for item in sorted(root.rglob("*")):
            if not item.is_file():
                continue
            relative = _relative_path(item)
            if not _is_blocked(relative):
                files.append(relative)
        return "\n".join(files)
    except Exception as exc:
        return f"error listing {directory}: {exc}"


@tool
def run_tests(test_path: str = "tests/") -> str:
    """Run pytest on a project test path and return exit code plus truncated output."""

    args = {"path": test_path}
    _check_permission("run_tests", args)
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", test_path],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        output = completed.stdout + completed.stderr
        return _truncate(f"exit_code={completed.returncode}\n{output}", 3000)
    except Exception as exc:
        return _truncate(f"error running tests: {exc}", 3000)


@tool
def request_review(reason: str) -> str:
    """Ask the read-only reviewer subagent to inspect current file edits."""

    args = {"reason": reason}
    _check_permission("request_review", args)
    state = _ACTIVE_STATE or {
        "messages": [],
        "plan": [],
        "current_step": 0,
        "iterations": 0,
        "file_edits": {},
        "verification": {"passed": False, "failures": [], "attempts": 0},
        "budget": {"tokens_used": 0, "tokens_max": 0, "cost_usd": 0.0},
        "harness_events": [],
        "task": reason,
        "output": "",
    }
    result = review_changes(state)
    concerns = result.get("concerns", [])
    concern_text = "; ".join(str(concern) for concern in concerns) or "none"
    return (
        f"review approved={result.get('approved')}\n"
        f"summary={result.get('summary')}\n"
        f"concerns={concern_text}"
    )


read_file_tool = read_file
write_file_tool = write_file
run_code_tool = run_code
list_files_tool = list_files
run_tests_tool = run_tests
request_review_tool = request_review

TOOLS = [
    read_file_tool,
    write_file_tool,
    run_code_tool,
    list_files_tool,
    run_tests_tool,
    request_review_tool,
]
