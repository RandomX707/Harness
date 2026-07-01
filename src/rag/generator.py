"""Answer generation for the minimal CRAG pipeline."""

from __future__ import annotations

import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


def generate_answer(question: str, documents: list[dict]) -> str:
    relevant_documents = [doc for doc in documents if doc.get("relevance") == "relevant"]
    context_documents = relevant_documents or documents
    if not os.getenv("LITELLM_API_KEY"):
        sources = ", ".join(str(document.get("source", "unknown")) for document in context_documents)
        return f"Stub RAG answer for '{question}' using documents: {sources}"

    context = "\n\n".join(
        f"Source: {document.get('source', 'unknown')}\n{document.get('content', '')}"
        for document in context_documents
    )
    prompt = "Answer the question using only the provided context. Be concise and cite sources by source name."
    model = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.getenv("LITELLM_API_KEY"),
        base_url=os.getenv("LITELLM_BASE_URL"),
    )
    response = model.invoke(
        [
            SystemMessage(content=prompt),
            HumanMessage(content=f"Question:\n{question}\n\nContext:\n{context}"),
        ]
    )
    content = getattr(response, "content", response)
    return content if isinstance(content, str) else str(content)

