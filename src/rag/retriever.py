"""Deliberately small in-memory retriever for CRAG experiments."""

from __future__ import annotations

import re


KNOWLEDGE_BASE: list[dict] = [
    {
        "content": "Harness engineering wraps an agent with safety, permissions, verification, and observability.",
        "source": "kb:harness-engineering",
    },
    {
        "content": "A plan-execute-verify loop decomposes work, performs an action, and checks the result before continuing.",
        "source": "kb:pev-loop",
    },
    {
        "content": "LangGraph models agent workflows as stateful graphs with nodes, edges, and conditional routing.",
        "source": "kb:langgraph",
    },
    {
        "content": "A circuit breaker stops repeated failures such as too many iterations or repeated verification errors.",
        "source": "kb:circuit-breaker",
    },
    {
        "content": "A permission resolver gates tool calls by risk, path policy, and human approval requirements.",
        "source": "kb:permissions",
    },
    {
        "content": "Observability records structured events for tool calls, node execution, verification, and task completion.",
        "source": "kb:observability",
    },
    {
        "content": "Corrective RAG retrieves documents, grades relevance, performs a corrective search if needed, then generates.",
        "source": "kb:crag",
    },
    {
        "content": "Python pathlib.Path provides object-oriented filesystem paths and is preferred over ad hoc os.path handling.",
        "source": "kb:pathlib",
    },
    {
        "content": "Token and cost accounting tracks model usage so an agent can stop before exceeding a configured budget.",
        "source": "kb:budget",
    },
]


def retrieve(question: str, k: int = 3) -> list[dict]:
    query_terms = _terms(question)
    scored: list[tuple[int, dict]] = []
    for document in KNOWLEDGE_BASE:
        score = len(query_terms.intersection(_terms(str(document["content"]))))
        scored.append((score, document))

    ranked = sorted(scored, key=lambda item: (-item[0], str(item[1]["source"])))
    return [
        {
            "content": str(document["content"]),
            "source": str(document["source"]),
            "relevance": "ambiguous",
        }
        for score, document in ranked[:k]
        if score > 0
    ] or [
        {
            "content": str(document["content"]),
            "source": str(document["source"]),
            "relevance": "ambiguous",
        }
        for _, document in ranked[:k]
    ]


def _terms(text: str) -> set[str]:
    return {term for term in re.findall(r"[a-z0-9]+", text.lower()) if len(term) > 2}

