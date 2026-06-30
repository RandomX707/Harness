"""Queryable helpers for harness JSONL logs."""

from __future__ import annotations

import json
from pathlib import Path


def get_events(
    log_path: Path,
    event_type: str | None = None,
    since: str | None = None,
    limit: int | None = None,
) -> list[dict]:
    if not log_path.exists():
        return []

    matches: list[dict] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event_type is not None and event.get("event") != event_type:
            continue
        if since is not None and str(event.get("timestamp", "")) < since:
            continue
        matches.append(event)

    if limit is None:
        return matches
    return matches[-limit:]


def count_events(log_path: Path, event_type: str) -> int:
    return len(get_events(log_path, event_type=event_type))
