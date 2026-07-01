"""Clearly labeled corrective web-search stub for CRAG."""

from __future__ import annotations

import structlog


def web_search(question: str) -> list[dict]:
    structlog.get_logger("rag.web_search_stub").info("stub_web_search", question=question)
    return [
        {
            "content": (
                "Synthetic web-search result: no real network search was performed. "
                f"The question was: {question}"
            ),
            "source": "stub:web_search",
            "relevance": "relevant",
        }
    ]

