"""Path permission checks for agent tools."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fnmatch import fnmatch
from pathlib import Path
import time

from harness.approval_queue import ApprovalQueue


class RiskLevel(Enum):
    READ_ONLY = 1
    WRITE = 2
    NETWORK = 3
    EXECUTE = 4
    DESTRUCTIVE = 5


@dataclass(frozen=True)
class ToolPolicy:
    tool_name: str
    risk_level: RiskLevel
    allowed_paths: list[str] | None
    requires_hitl: bool


@dataclass(frozen=True)
class PermissionDecision:
    allowed: bool
    reason: str


class PermissionResolver:
    """Resolve tool permissions against path policy and HITL requirements."""

    def __init__(
        self,
        policy: list[ToolPolicy] | Path | None = None,
        hitl_threshold: RiskLevel = RiskLevel.EXECUTE,
        project_root: Path | None = None,
        blocked_patterns: list[str] | None = None,
        use_async_approval: bool = False,
        approval_timeout: float = 10.0,
        approval_queue: ApprovalQueue | None = None,
    ) -> None:
        if isinstance(policy, Path):
            self.project_root = policy.resolve()
            self.policy = self.default_policy()
        else:
            self.project_root = (project_root or Path.cwd()).resolve()
            self.policy = policy if policy is not None else self.default_policy()
        self.hitl_threshold = hitl_threshold
        self.blocked_patterns = blocked_patterns if blocked_patterns is not None else self._load_blocked_patterns()
        self.use_async_approval = use_async_approval
        self.approval_timeout = approval_timeout
        self.approval_queue = approval_queue
        self._policy_by_tool = {item.tool_name: item for item in self.policy}

    @classmethod
    def default_policy(cls) -> list[ToolPolicy]:
        return [
            ToolPolicy("read_file", RiskLevel.READ_ONLY, None, False),
            ToolPolicy("write_file", RiskLevel.WRITE, ["src/", "tests/"], False),
            ToolPolicy("list_files", RiskLevel.READ_ONLY, None, False),
            ToolPolicy("request_review", RiskLevel.READ_ONLY, None, False),
            ToolPolicy("run_code", RiskLevel.EXECUTE, None, False),
            ToolPolicy("run_tests", RiskLevel.EXECUTE, ["tests/"], False),
            ToolPolicy("run_terminal", RiskLevel.DESTRUCTIVE, None, True),
        ]

    def check(self, tool_name: str, args: dict) -> tuple[bool, str]:
        tool_policy = self._policy_by_tool.get(tool_name)
        if tool_policy is None:
            return False, f"unknown tool: {tool_name}"

        path_arg = args.get("path")
        if path_arg is not None:
            target = self._resolve(Path(str(path_arg)))
            if not self._is_inside_project(target):
                return False, "path is outside the project"

            relative = self._relative_posix(target)
            if self._is_blocked(relative):
                return False, f"path matches blocked pattern: {relative}"

            if tool_policy.allowed_paths is not None:
                allowed = [self._resolve(Path(path)) for path in tool_policy.allowed_paths]
                if not any(target.is_relative_to(root) or target == root for root in allowed):
                    return False, "path is outside allowed paths"

        if tool_policy.requires_hitl:
            approved, approval_reason = self._request_approval(tool_policy, args)
            if not approved:
                return False, approval_reason

        return True, "allowed"

    def can_read(self, target: str | Path) -> PermissionDecision:
        path = self._resolve(target)
        if not self._is_inside_project(path):
            return PermissionDecision(False, "path is outside the project")
        return PermissionDecision(True, "read allowed within project")

    def can_write(self, target: str | Path) -> PermissionDecision:
        allowed, reason = self.check("write_file", {"path": str(target)})
        return PermissionDecision(allowed, reason)

    def ensure_read(self, target: str | Path) -> Path:
        decision = self.can_read(target)
        if not decision.allowed:
            raise PermissionError(decision.reason)
        return self._resolve(target)

    def ensure_write(self, target: str | Path) -> Path:
        decision = self.can_write(target)
        if not decision.allowed:
            raise PermissionError(decision.reason)
        return self._resolve(target)

    def _load_blocked_patterns(self) -> list[str]:
        blocked_file = self.project_root / ".harness" / "blocked_patterns.txt"
        if not blocked_file.exists():
            return []
        return [
            line.strip()
            for line in blocked_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]

    def _resolve(self, target: str | Path) -> Path:
        path = Path(target)
        if not path.is_absolute():
            path = self.project_root / path
        return path.resolve()

    def _is_inside_project(self, path: Path) -> bool:
        return path == self.project_root or self.project_root in path.parents

    def _relative_posix(self, path: Path) -> str:
        return path.relative_to(self.project_root).as_posix()

    def _is_blocked(self, relative: str) -> bool:
        for pattern in self.blocked_patterns:
            if fnmatch(relative, pattern):
                return True
            if pattern.endswith("/*"):
                blocked_root = pattern[:-2]
                if relative == blocked_root or relative.startswith(f"{blocked_root}/"):
                    return True
        return False

    def _is_under_allowed_write_root(self, path: Path) -> bool:
        allowed_roots = [self.project_root / "src", self.project_root / "tests"]
        return any(path == root or root in path.parents for root in allowed_roots)

    def _request_approval(self, tool_policy: ToolPolicy, args: dict) -> tuple[bool, str]:
        if self.use_async_approval:
            started = time.monotonic()
            approval_queue = self.approval_queue or ApprovalQueue(self.project_root / ".harness" / "approvals.json")
            request_id = approval_queue.submit(
                tool_policy.tool_name,
                args,
                tool_policy.risk_level.name,
            )
            resolved = approval_queue.wait_for_resolution(
                request_id,
                timeout=self.approval_timeout,
                poll_interval=min(0.05, max(self.approval_timeout / 4, 0.01)),
            )
            if resolved.status == "approved":
                return True, "approved"
            if time.monotonic() - started >= self.approval_timeout:
                return False, "approval timed out"
            return False, "denied"

        print(
            "[HITL approval]\n"
            f"Tool: {tool_policy.tool_name}\n"
            f"Risk: {tool_policy.risk_level.name}\n"
            f"Args: {args}\n"
            "Approve? Type y to continue: ",
            end="",
        )
        if input().strip() == "y":
            return True, "approved"
        return False, "denied"
