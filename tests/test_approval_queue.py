from __future__ import annotations

import json
from pathlib import Path
import threading
import time

from harness.approval_queue import ApprovalQueue


def test_submit_poll_resolve_round_trip(tmp_path: Path) -> None:
    queue = ApprovalQueue(tmp_path / "approvals.json")

    request_id = queue.submit("run_terminal", {"command": "ls"}, "DESTRUCTIVE")
    pending = queue.poll(request_id)

    assert pending is not None
    assert pending.status == "pending"
    assert queue.resolve(request_id, approved=True) is True
    resolved = queue.poll(request_id)
    assert resolved is not None
    assert resolved.status == "approved"
    assert resolved.resolved_at is not None


def test_wait_for_resolution_observes_separate_queue_instance(tmp_path: Path) -> None:
    path = tmp_path / "approvals.json"
    waiting_queue = ApprovalQueue(path)
    resolving_queue = ApprovalQueue(path)
    request_id = waiting_queue.submit("run_terminal", {"command": "ls"}, "DESTRUCTIVE")

    def resolve_later() -> None:
        time.sleep(0.05)
        resolving_queue.resolve(request_id, approved=True)

    thread = threading.Thread(target=resolve_later)
    thread.start()
    resolved = waiting_queue.wait_for_resolution(request_id, timeout=1.0, poll_interval=0.01)
    thread.join()

    assert resolved.status == "approved"


def test_wait_for_resolution_times_out_and_auto_denies(tmp_path: Path) -> None:
    queue = ApprovalQueue(tmp_path / "approvals.json")
    request_id = queue.submit("run_terminal", {"command": "rm -rf build"}, "DESTRUCTIVE")

    resolved = queue.wait_for_resolution(request_id, timeout=0.01, poll_interval=0.005)

    assert resolved.status == "denied"
    assert resolved.resolved_at is not None


def test_atomic_writes_keep_valid_json_and_pending_requests(tmp_path: Path) -> None:
    path = tmp_path / "approvals.json"
    queue = ApprovalQueue(path)

    first_id = queue.submit("run_terminal", {"command": "one"}, "DESTRUCTIVE")
    corrupt_temp = path.with_name(".approvals.json.crash.tmp")
    corrupt_temp.write_text("{not-json", encoding="utf-8")
    second_id = queue.submit("run_terminal", {"command": "two"}, "DESTRUCTIVE")

    data = json.loads(path.read_text(encoding="utf-8"))
    ids = {item["id"] for item in data}

    assert first_id in ids
    assert second_id in ids
    assert queue.poll(first_id) is not None
    assert queue.poll(second_id) is not None
