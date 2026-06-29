from __future__ import annotations

from pathlib import Path

from harness.permission_resolver import PermissionResolver, RiskLevel, ToolPolicy


def make_resolver(tmp_path: Path) -> PermissionResolver:
    (tmp_path / ".harness").mkdir()
    (tmp_path / ".harness" / "blocked_patterns.txt").write_text(
        ".harness/*\npyproject.toml\n*.env\n.git/*\n",
        encoding="utf-8",
    )
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    return PermissionResolver(tmp_path)


def test_read_is_allowed_inside_project(tmp_path: Path) -> None:
    resolver = make_resolver(tmp_path)
    assert resolver.can_read("pyproject.toml").allowed


def test_write_is_limited_to_src_and_tests(tmp_path: Path) -> None:
    resolver = make_resolver(tmp_path)

    assert resolver.can_write("src/app.py").allowed
    assert resolver.can_write("tests/test_app.py").allowed
    assert not resolver.can_write("README.md").allowed


def test_blocked_patterns_override_write_roots(tmp_path: Path) -> None:
    resolver = make_resolver(tmp_path)

    assert not resolver.can_write("pyproject.toml").allowed
    assert not resolver.can_write(".env").allowed
    assert not resolver.can_write(".harness/allowlist.txt").allowed


def test_path_traversal_is_rejected(tmp_path: Path) -> None:
    resolver = make_resolver(tmp_path)

    assert not resolver.can_write("../outside.py").allowed


def test_check_allows_default_read_policy_inside_project(tmp_path: Path) -> None:
    resolver = make_resolver(tmp_path)

    allowed, reason = resolver.check("read_file", {"path": "src/app.py"})

    assert allowed is True
    assert reason == "allowed"


def test_check_rejects_blocked_patterns_for_any_path_tool(tmp_path: Path) -> None:
    resolver = make_resolver(tmp_path)

    allowed, reason = resolver.check("read_file", {"path": "pyproject.toml"})

    assert allowed is False
    assert "blocked pattern" in reason


def test_check_enforces_allowed_paths_for_write_policy(tmp_path: Path) -> None:
    resolver = make_resolver(tmp_path)

    allowed, reason = resolver.check("write_file", {"path": "README.md"})

    assert allowed is False
    assert reason == "path is outside allowed paths"


def test_check_allows_sandboxed_run_code_without_hitl(tmp_path: Path) -> None:
    resolver = make_resolver(tmp_path)

    allowed, reason = resolver.check("run_code", {"command": "python -m pytest tests/"})

    assert allowed is True
    assert reason == "allowed"


def test_check_rejects_unknown_tools(tmp_path: Path) -> None:
    resolver = make_resolver(tmp_path)

    allowed, reason = resolver.check("delete_everything", {})

    assert allowed is False
    assert reason == "unknown tool: delete_everything"


def test_hitl_denies_when_user_does_not_approve(tmp_path: Path, monkeypatch) -> None:
    resolver = make_resolver(tmp_path)
    monkeypatch.setattr("builtins.input", lambda: "n")

    allowed, reason = resolver.check("run_terminal", {"command": "rm -rf build"})

    assert allowed is False
    assert reason == "denied"


def test_hitl_allows_when_user_approves(tmp_path: Path, monkeypatch) -> None:
    resolver = make_resolver(tmp_path)
    monkeypatch.setattr("builtins.input", lambda: "y")

    allowed, reason = resolver.check("run_terminal", {"command": "ls"})

    assert allowed is True
    assert reason == "allowed"


def test_custom_network_policy_can_be_registered(tmp_path: Path) -> None:
    resolver = PermissionResolver(
        [
            ToolPolicy("fetch_docs", RiskLevel.NETWORK, None, False),
        ],
        project_root=tmp_path,
    )

    allowed, reason = resolver.check("fetch_docs", {"url": "https://docs.python.org"})

    assert allowed is True
    assert reason == "allowed"
