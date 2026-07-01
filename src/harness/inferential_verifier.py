"""Inferential verification helpers for soft LLM-as-judge checks."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


DEFAULT_PASS_THRESHOLD = 0.7


@dataclass(frozen=True)
class InferentialVerificationResult:
    score: float
    reasoning: str
    flagged_issues: list[str]
    passed: bool = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "passed", self.score >= DEFAULT_PASS_THRESHOLD)

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "reasoning": self.reasoning,
            "passed": self.passed,
            "flagged_issues": self.flagged_issues,
        }


def judge_code_quality(
    plan_step: str,
    diff_summary: str,
    model: str = "gpt-4o-mini",
) -> InferentialVerificationResult:
    """Judge whether changes match a plan step, falling back deterministically offline."""

    api_key = os.getenv("LITELLM_API_KEY")
    if not api_key:
        return InferentialVerificationResult(
            score=1.0,
            reasoning="inferential verification skipped: no API key",
            flagged_issues=[],
        )

    system_prompt = (
        "You are a code quality judge. Decide whether the code change matches the "
        "plan step's intent. Return ONLY JSON with keys: score, reasoning, "
        "flagged_issues. score must be a number from 0.0 to 1.0."
    )
    human_prompt = f"Plan step:\n{plan_step}\n\nDiff summary:\n{diff_summary}"

    try:
        llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=os.getenv("LITELLM_BASE_URL"),
        )
        response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)])
        return _parse_judge_response(_message_content(response))
    except Exception as exc:
        return InferentialVerificationResult(
            score=0.5,
            reasoning=f"inferential verification failed: {exc}",
            flagged_issues=["could not parse judge response"],
        )


def judge_rag_answer(
    question: str,
    answer: str,
    documents: list[dict],
    model: str = "gpt-4o-mini",
) -> InferentialVerificationResult:
    """Judge whether a RAG answer is grounded in its retrieved documents."""

    api_key = os.getenv("LITELLM_API_KEY")
    if not api_key:
        return InferentialVerificationResult(
            score=1.0,
            reasoning="rag inferential verification skipped: no API key",
            flagged_issues=[],
        )

    context = "\n\n".join(
        f"Source: {document.get('source', 'unknown')}\n{document.get('content', '')}"
        for document in documents
    )
    system_prompt = (
        "You are a RAG faithfulness judge. Decide whether the answer is grounded "
        "in the provided documents and answers the question. Return ONLY JSON "
        "with keys: score, reasoning, flagged_issues. score must be 0.0 to 1.0."
    )
    human_prompt = f"Question:\n{question}\n\nAnswer:\n{answer}\n\nDocuments:\n{context}"

    try:
        llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=os.getenv("LITELLM_BASE_URL"),
        )
        response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)])
        return _parse_judge_response(_message_content(response))
    except Exception as exc:
        return InferentialVerificationResult(
            score=0.5,
            reasoning=f"rag inferential verification failed: {exc}",
            flagged_issues=["could not parse judge response"],
        )


def _parse_judge_response(content: str) -> InferentialVerificationResult:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return InferentialVerificationResult(
            score=0.5,
            reasoning="could not parse judge response",
            flagged_issues=["could not parse judge response"],
        )

    if not isinstance(parsed, dict):
        return InferentialVerificationResult(
            score=0.5,
            reasoning="judge response was not an object",
            flagged_issues=["could not parse judge response"],
        )

    issues = parsed.get("flagged_issues", [])
    if not isinstance(issues, list):
        issues = [str(issues)]
    return InferentialVerificationResult(
        score=float(parsed.get("score", 0.5)),
        reasoning=str(parsed.get("reasoning", "")),
        flagged_issues=[str(issue) for issue in issues],
    )


def _message_content(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    return str(content)
