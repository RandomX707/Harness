"""LangGraph assembly for a minimal Corrective RAG pipeline."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from rag.generator import generate_answer
from rag.grader import grade_documents
from rag.retriever import retrieve
from rag.state import RAGState
from rag.web_search_stub import web_search


def retrieve_node(state: RAGState) -> RAGState:
    return {
        **state,
        "documents": retrieve(state["question"]),
        "iterations": int(state.get("iterations", 0)) + 1,
    }


def grade_documents_node(state: RAGState) -> RAGState:
    graded = grade_documents(state["question"], state.get("documents", []))
    web_search_needed = any(
        document.get("relevance") in {"irrelevant", "ambiguous"}
        for document in graded
    )
    return {**state, "documents": graded, "web_search_needed": web_search_needed}


def web_search_node(state: RAGState) -> RAGState:
    return {
        **state,
        "documents": state.get("documents", []) + web_search(state["question"]),
        "web_search_needed": True,
    }


def generate_node(state: RAGState) -> RAGState:
    return {
        **state,
        "answer": generate_answer(state["question"], state.get("documents", [])),
    }


def route_after_grade(state: RAGState) -> str:
    if state.get("web_search_needed", False):
        return "web_search"
    return "generate"


class CompiledRAGGraph:
    def __init__(self, graph: Any) -> None:
        self._graph = graph

    def invoke(self, state: RAGState, config: dict | None = None) -> RAGState:
        return self._graph.invoke(state, config=config)


def build_rag_graph() -> CompiledRAGGraph:
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade_documents", grade_documents_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("generate", generate_node)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges("grade_documents", route_after_grade)
    graph.add_edge("web_search", "generate")
    graph.add_edge("generate", END)
    return CompiledRAGGraph(graph.compile())


def run_crag(question: str) -> RAGState:
    initial_state: RAGState = {
        "question": question,
        "documents": [],
        "web_search_needed": False,
        "answer": "",
        "iterations": 0,
    }
    return build_rag_graph().invoke(initial_state)
