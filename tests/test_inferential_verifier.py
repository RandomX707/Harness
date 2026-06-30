from __future__ import annotations

import json

from langchain_core.messages import AIMessage
import pytest

from harness.inferential_verifier import (
    DEFAULT_PASS_THRESHOLD,
    InferentialVerificationResult,
    judge_code_quality,
)


def test_judge_code_quality_returns_stub_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)

    result = judge_code_quality("write code", "src/example.py: 1 edit")

    assert result.score == 1.0
    assert result.passed is True
    assert result.reasoning == "inferential verification skipped: no API key"


def test_result_fails_below_threshold() -> None:
    result = InferentialVerificationResult(
        score=DEFAULT_PASS_THRESHOLD - 0.1,
        reasoning="weak match",
        flagged_issues=["missing test"],
    )

    assert result.passed is False


def test_judge_code_quality_parses_mocked_json_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_API_KEY", "test-key")
    monkeypatch.setenv("LITELLM_BASE_URL", "https://example.com/v1")

    class FakeChatOpenAI:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def invoke(self, messages):
            return AIMessage(
                content=json.dumps(
                    {
                        "score": 0.82,
                        "reasoning": "matches the requested change",
                        "flagged_issues": ["consider edge cases"],
                    }
                )
            )

    monkeypatch.setattr("harness.inferential_verifier.ChatOpenAI", FakeChatOpenAI)

    result = judge_code_quality("add function", "src/example.py: 1 edit")

    assert result.score == 0.82
    assert result.passed is True
    assert result.reasoning == "matches the requested change"
    assert result.flagged_issues == ["consider edge cases"]


def test_malformed_judge_response_falls_back_safely(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_API_KEY", "test-key")
    monkeypatch.setenv("LITELLM_BASE_URL", "https://example.com/v1")

    class FakeChatOpenAI:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def invoke(self, messages):
            return AIMessage(content="not json")

    monkeypatch.setattr("harness.inferential_verifier.ChatOpenAI", FakeChatOpenAI)

    result = judge_code_quality("add function", "src/example.py: 1 edit")

    assert result.score == 0.5
    assert result.flagged_issues == ["could not parse judge response"]
