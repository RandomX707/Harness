from __future__ import annotations

import pytest

from rag.generator import generate_answer
from rag.grader import grade_documents
from rag.graph import run_crag
from rag.retriever import retrieve


def test_retrieve_returns_documents_ranked_by_keyword_overlap() -> None:
    documents = retrieve("How does LangGraph model agent workflows?", k=2)

    assert documents
    assert documents[0]["source"] == "kb:langgraph"


def test_grade_documents_stub_marks_everything_relevant(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    documents = [{"content": "anything", "source": "test", "relevance": "ambiguous"}]

    graded = grade_documents("question", documents)

    assert graded == [{"content": "anything", "source": "test", "relevance": "relevant"}]


def test_full_graph_routes_to_web_search_when_document_irrelevant(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)

    def fake_grade_documents(question: str, documents: list[dict]) -> list[dict]:
        return [{**document, "relevance": "irrelevant"} for document in documents]

    monkeypatch.setattr("rag.graph.grade_documents", fake_grade_documents)

    final_state = run_crag("What is harness engineering?")

    assert final_state["web_search_needed"] is True
    assert any(document["source"] == "stub:web_search" for document in final_state["documents"])


def test_full_graph_skips_web_search_when_all_documents_relevant(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)

    def fake_grade_documents(question: str, documents: list[dict]) -> list[dict]:
        return [{**document, "relevance": "relevant"} for document in documents]

    monkeypatch.setattr("rag.graph.grade_documents", fake_grade_documents)

    final_state = run_crag("What is harness engineering?")

    assert final_state["web_search_needed"] is False
    assert not any(document["source"] == "stub:web_search" for document in final_state["documents"])


def test_generate_answer_stub_path_when_no_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)

    answer = generate_answer(
        "What is harness engineering?",
        [{"content": "Harnesses wrap agents.", "source": "kb:test", "relevance": "relevant"}],
    )

    assert "Stub RAG answer" in answer
    assert "kb:test" in answer
