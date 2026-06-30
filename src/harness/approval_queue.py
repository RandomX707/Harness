"""File-backed approval queue for asynchronous HITL decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from uuid import uuid4


@dataclass(frozen=True)
class ApprovalRequest:
    id: str
    tool_name: str
    args: dict
    risk_level: str
    created_at: str
    status: str
    resolved_at: str | None


class ApprovalQueue:
    """Persist approval requests to a JSON file using atomic writes."""

    def __init__(self, path: Path | None = None) -> None:
        project_root = Path(__file__).resolve().parents[2]
        self.path = path or project_root / ".harness" / "approvals.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write_requests([])

    def submit(self, tool_name: str, args: dict, risk_level: str) -> str:
        requests = self._read_requests()
        request = ApprovalRequest(
            id=str(uuid4()),
            tool_name=tool_name,
            args=args,
            risk_level=risk_level,
            created_at=_now_iso(),
            status="pending",
            resolved_at=None,
        )
        requests.append(request)
        self._write_requests(requests)
        return request.id

    def poll(self, request_id: str) -> ApprovalRequest | None:
        for request in self._read_requests():
            if request.id == request_id:
                return request
        return None

    def resolve(self, request_id: str, approved: bool) -> bool:
        requests = self._read_requests()
        updated: list[ApprovalRequest] = []
        found = False
        for request in requests:
            if request.id == request_id:
                found = True
                updated.append(
                    ApprovalRequest(
                        id=request.id,
                        tool_name=request.tool_name,
                        args=request.args,
                        risk_level=request.risk_level,
                        created_at=request.created_at,
                        status="approved" if approved else "denied",
                        resolved_at=_now_iso(),
                    )
                )
            else:
                updated.append(request)
        if not found:
            return False
        self._write_requests(updated)
        return True

    def wait_for_resolution(
        self,
        request_id: str,
        timeout: float = 30.0,
        poll_interval: float = 0.5,
    ) -> ApprovalRequest:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            request = self.poll(request_id)
            if request is None:
                raise ValueError(f"approval request not found: {request_id}")
            if request.status != "pending":
                return request
            time.sleep(poll_interval)

        self.resolve(request_id, approved=False)
        request = self.poll(request_id)
        if request is None:
            raise ValueError(f"approval request not found: {request_id}")
        return request

    def _read_requests(self) -> list[ApprovalRequest]:
        if not self.path.exists():
            return []
        data = json.loads(self.path.read_text(encoding="utf-8") or "[]")
        return [ApprovalRequest(**item) for item in data]

    def _write_requests(self, requests: list[ApprovalRequest]) -> None:
        payload = [asdict(request) for request in requests]
        temp_path = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
        temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temp_path, self.path)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
