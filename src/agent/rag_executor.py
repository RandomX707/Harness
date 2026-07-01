"""RAG executor adapter for running CRAG through the existing harness loop."""

from __future__ import annotations

from collections import Counter
from typing import Any

from harness.inferential_verifier import judge_rag_answer
from harness.state import AgentState
from rag.graph import run_crag


def rag_executor(state: AgentState) -> AgentState:
    question = state["task"]
    rag_state = run_crag(question)
    documents = rag_state.get("documents", [])
    relevance_counts = dict(Counter(str(document.get("relevance", "unknown")) for document in documents))
    verification = dict(state.get("verification", {}))
    verification["rag"] = {
        "documents": documents,
        "web_search_needed": bool(rag_state.get("web_search_needed", False)),
        "iterations": int(rag_state.get("iterations", 0)),
        "relevance_counts": relevance_counts,
    }

    events = list(state.get("harness_events", []))
    events.append(
        {
            "type": "rag_execution",
            "documents_retrieved": len(documents),
            "web_search_needed": bool(rag_state.get("web_search_needed", False)),
            "relevance_counts": relevance_counts,
        }
    )
    if rag_state.get("web_search_needed", False):
        events.append({"type": "stub_web_search", "question": question})

    # Friction point: AgentState.file_edits is intentionally left empty for RAG.
    # Treating retrieved documents as file edits would make the circuit breaker
    # metrics misleading rather than more general.
    return {
        **state,
        "output": rag_state.get("answer", ""),
        "verification": verification,
        "harness_events": events,
        "messages": state.get("messages", []) + ["CRAG executor completed."],
    }


def rag_verifier(state: AgentState) -> tuple[bool, str]:
    verification = dict(state.get("verification", {}))
    rag_details = dict(verification.get("rag", {}))
    documents = _documents(rag_details.get("documents", []))
    result = judge_rag_answer(state["task"], state.get("output", ""), documents)
    verification["inferential"] = result.as_dict()
    state["verification"] = verification

    # Architectural seam: in the coding path pytest is the hard gate and the LLM
    # judge is soft. For RAG there is no computational test suite equivalent, so
    # the inferential faithfulness judge necessarily becomes the hard gate.
    if result.passed:
        return True, state.get("output", "")
    return False, result.reasoning


def _documents(value: Any) -> list[dict]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]
