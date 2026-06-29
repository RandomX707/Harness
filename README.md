# coding-agent-harness

This project demonstrates harness engineering for a coding agent built with Python, LangGraph, and LangChain. It shows how to place safety and reliability layers around an agent loop: permission checks, circuit breakers, budget guards, context compaction, structured observability, verification, and feedback injection.

## Setup

```bash
pip install -e .
export LITELLM_API_KEY=your_key_here
export LITELLM_BASE_URL=https://your-proxy.litellm.ai
```

The demos also run without `LITELLM_API_KEY`; in that mode the graph falls back to deterministic executor/verifier hooks so the harness remains testable without network access.

## Run Demos

```bash
PYTHONPATH=src python3 src/main.py --task simple
PYTHONPATH=src python3 src/main.py --task permission_test
PYTHONPATH=src python3 src/main.py --task doom_loop_test
PYTHONPATH=src python3 src/main.py --custom "your task here"
PYTHONPATH=src python3 src/main.py --visualize
```

You can also run the module form from the project root:

```bash
python3 -m src.main --task simple
```

## What To Observe

- `simple`: clean plan-execute-verify flow with no harness intervention.
- `permission_test`: attempts to write `.harness/evil.txt`; the permission resolver blocks the write and records a denied `tool_call`.
- `doom_loop_test`: injects a verifier that always returns `still failing`; repeated verification failures trip the circuit breaker.

## Logs

Structured harness logs are written to `logs/harness.jsonl`. Each line is a JSON event from the observer layer. Useful events include `tool_call`, `iteration`, `verification`, `circuit_breaker_trip`, and `task_complete`.

The CLI prints the last 10 log events after each run.
