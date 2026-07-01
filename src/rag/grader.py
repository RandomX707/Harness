"""Document relevance grading for the CRAG experiment."""

from __future__ import annotations

import json
import os
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


def grade_documents(question: str, documents: list[dict]) -> list[dict]:
    if not os.getenv("LITELLM_API_KEY"):
        return [{**document, "relevance": "relevant"} for document in documents]

    graded: list[dict] = []
    for document in documents:
        graded.append(_grade_one(question, document))
    return graded


def _grade_one(question: str, document: dict) -> dict:
    prompt = (
        "Grade whether the document is relevant to the question. Return ONLY JSON "
        'with {"relevance": "relevant|irrelevant|ambiguous"}.'
    )
    body = f"Question:\n{question}\n\nDocument:\n{document.get('content', '')}"
    try:
        model = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=os.getenv("LITELLM_API_KEY"),
            base_url=os.getenv("LITELLM_BASE_URL"),
        )
        response = model.invoke([SystemMessage(content=prompt), HumanMessage(content=body)])
        parsed = json.loads(_message_content(response))
        relevance = str(parsed.get("relevance", "ambiguous"))
        if relevance not in {"relevant", "irrelevant", "ambiguous"}:
            relevance = "ambiguous"
        return {**document, "relevance": relevance}
    except Exception:
        return {**document, "relevance": "ambiguous"}


def _message_content(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    return str(content)

