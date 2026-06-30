from __future__ import annotations

import json
from pathlib import Path

from harness.log_query import count_events, get_events


def write_log(path: Path) -> None:
    events = [
        {"timestamp": "2026-01-01T00:00:00Z", "event": "node_start"},
        {"timestamp": "2026-01-01T00:00:01Z", "event": "tool_call", "tool": "read_file"},
        {"timestamp": "2026-01-01T00:00:02Z", "event": "node_finish"},
        {"timestamp": "2026-01-01T00:00:03Z", "event": "tool_call", "tool": "write_file"},
        {"timestamp": "2026-01-01T00:00:04Z", "event": "circuit_breaker_trip"},
        {"timestamp": "2026-01-01T00:00:05Z", "event": "node_start"},
        {"timestamp": "2026-01-01T00:00:06Z", "event": "node_finish"},
        {"timestamp": "2026-01-01T00:00:07Z", "event": "tool_call", "tool": "run_tests"},
    ]
    path.write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")


def test_get_events_filters_tool_calls_before_limiting(tmp_path: Path) -> None:
    log_path = tmp_path / "harness.jsonl"
    write_log(log_path)

    events = get_events(log_path, event_type="tool_call")

    assert [event["event"] for event in events] == ["tool_call", "tool_call", "tool_call"]
    assert [event["tool"] for event in events] == ["read_file", "write_file", "run_tests"]


def test_get_events_limit_returns_recent_matching_events(tmp_path: Path) -> None:
    log_path = tmp_path / "harness.jsonl"
    write_log(log_path)

    events = get_events(log_path, event_type="tool_call", limit=2)

    assert [event["tool"] for event in events] == ["write_file", "run_tests"]


def test_count_events(tmp_path: Path) -> None:
    log_path = tmp_path / "harness.jsonl"
    write_log(log_path)

    assert count_events(log_path, "tool_call") == 3
