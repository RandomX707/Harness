"""State for the standalone CRAG subgraph."""

from __future__ import annotations

from typing import TypedDict


class RAGState(TypedDict):
    question: str
    documents: list[dict]
    web_search_needed: bool
    answer: str
    iterations: int

