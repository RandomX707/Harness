"""LangGraph node functions for a plan-execute-verify coding loop."""

from __future__ import annotations

from collections.abc import Callable
import json
import os
from pathlib import Path
from typing import Any, cast

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from harness.circuit_breaker import CircuitBreaker, CircuitState
from harness.context_manager import HarnessContextManager
from harness.inferential_verifier import judge_code_quality
from harness.observability import HarnessObserver, traced_node
from harness.pricing import calculate_cost
from harness.state import AgentState

from agent.tools import TOOLS, bind_tool_state, run_tests


Executor = Callable[[AgentState], AgentState]
Verifier = Callable[[AgentState], tuple[bool, str]]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "gpt-4o-mini"
DEFAULT_CIRCUIT_BREAKER = CircuitBreaker()


@traced_node("plan")
def plan_node(state: AgentState) -> AgentState:
    if state.get("plan"):
        return state

    return {
        **state,
        "plan": [
            "Understand the requested change.",
            "Modify only allowed source or test files.",
            "Run verification before finishing.",
        ],
        "messages": state.get("messages", []) + ["Plan created."],
    }


def default_executor(state: AgentState) -> AgentState:
    return {
        **state,
        "messages": state.get("messages", []) + ["No-op demo executor completed."],
    }


def default_verifier(state: AgentState) -> tuple[bool, str]:
    try:
        output = run_tests.invoke({"test_path": "tests/"})
    except Exception as exc:
        return False, str(exc)
    return output.startswith("exit_code=0"), output


def make_execute_node(executor: Executor | None = None) -> Executor:
    selected = executor or default_executor

    @traced_node("execute")
    def execute_node(state: AgentState) -> AgentState:
        bind_tool_state(state)
        try:
            return selected(state)
        finally:
            bind_tool_state(None)

    return execute_node


def make_verify_node(verifier: Verifier | None = None) -> Executor:
    selected = verifier or default_verifier

    @traced_node("verify")
    def verify_node(state: AgentState) -> AgentState:
        passed, output = selected(state)
        verification = dict(state.get("verification", {}))
        attempts = int(verification.get("attempts", 0)) + 1
        failures = list(verification.get("failures", []))
        if not passed:
            failures.append(output)
        _record_trial_result(state, passed)
        messages = state.get("messages", []) + [f"Verification attempt {attempts}: {passed}"]
        if not passed and attempts >= 3:
            messages.append("Stopped after repeated verification failures.")
        next_verification = {
            **verification,
            "passed": passed,
            "failures": failures,
            "attempts": attempts,
            "task_complete": True,
        }
        return {
            **state,
            "verification": next_verification,
            "output": output if passed else "",
            "messages": messages,
        }

    return verify_node


@traced_node("fail")
def fail_node(state: AgentState) -> AgentState:
    return {
        **state,
        "output": state.get("output", ""),
        "messages": state.get("messages", []) + ["Stopped after repeated verification failures."],
    }


@traced_node("harness_guard")
def harness_guard_node(state: AgentState) -> AgentState:
    updated: dict[str, Any] = {
        **state,
        "iterations": int(state.get("iterations", 0)) + 1,
    }
    guarded_state = cast(AgentState, updated)

    is_open, reason = DEFAULT_CIRCUIT_BREAKER.check(guarded_state)
    HarnessContextManager().inject_harness_context(guarded_state)
    if is_open:
        updated["output"] = f"CIRCUIT_OPEN: {reason}"
        HarnessObserver().log_circuit_breaker("open", reason, 0)
        return cast(AgentState, updated)

    budget = guarded_state.get("budget", {})
    tokens_max = int(budget.get("tokens_max", 0))
    tokens_used = int(budget.get("tokens_used", 0))
    if tokens_max > 0 and tokens_used >= tokens_max:
        updated["output"] = "budget_exceeded"
        return cast(AgentState, updated)

    context_manager = HarnessContextManager()
    if context_manager.should_compact(guarded_state):
        updated["messages"] = context_manager.compact_messages(guarded_state)

    HarnessObserver().log_iteration(cast(AgentState, updated))
    return cast(AgentState, updated)


@traced_node("planner")
def planner_node(state: AgentState) -> AgentState:
    if state.get("plan"):
        return state

    prompt = (
        "You are a planning agent. Break the task into 3-7 concrete, verifiable steps. "
        "Return ONLY a JSON array of step strings, no other text."
    )
    plan = _fallback_plan(state["task"])
    api_key = os.getenv("LITELLM_API_KEY")
    if api_key:
        try:
            from langchain_openai import ChatOpenAI

            model = ChatOpenAI(
                model=MODEL_NAME,
                api_key=api_key,
                base_url=os.getenv("LITELLM_BASE_URL"),
            )
            response = model.invoke([SystemMessage(content=prompt), HumanMessage(content=state["task"])])
            parsed = json.loads(_message_content(response))
            if isinstance(parsed, list) and all(isinstance(step, str) for step in parsed):
                plan = parsed[:7]
        except Exception as exc:
            state = {
                **state,
                "harness_events": state.get("harness_events", [])
                + [{"type": "planner_fallback", "reason": str(exc)}],
            }

    return {
        **state,
        "plan": plan,
        "current_step": 0,
    }


@traced_node("agent")
def agent_node(state: AgentState) -> AgentState:
    if not os.getenv("LITELLM_API_KEY"):
        next_state = default_executor(state)
        return _record_estimated_usage(next_state)

    context_manager = HarnessContextManager(PROJECT_ROOT)
    instructions = context_manager.load_agent_instructions()
    harness_context = context_manager.inject_harness_context(state)
    current_step = _current_step_text(state)
    system_prompt = "\n\n".join([instructions, harness_context, current_step])

    try:
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(
            model=MODEL_NAME,
            api_key=os.getenv("LITELLM_API_KEY"),
            base_url=os.getenv("LITELLM_BASE_URL"),
        )
        react_agent = create_react_agent(model, TOOLS, prompt=system_prompt)
        messages = list(state.get("messages", []))
        if not messages:
            messages.append(HumanMessage(content=state["task"]))
        bind_tool_state(state)
        try:
            result = react_agent.invoke({"messages": messages})
        finally:
            bind_tool_state(None)
        response_messages = result.get("messages", []) if isinstance(result, dict) else []
        last_message = response_messages[-1] if response_messages else AIMessage(content="")
        next_state = _record_llm_usage(
            state,
            getattr(last_message, "usage_metadata", None),
        )
        return {
            **next_state,
            "messages": next_state.get("messages", []) + [last_message],
        }
    except Exception as exc:
        next_state = _record_estimated_usage(state)
        return {
            **next_state,
            "messages": next_state.get("messages", []) + [AIMessage(content=f"agent_error: {exc}")],
        }


@traced_node("verify")
def verify_node(state: AgentState) -> AgentState:
    output = run_tests.invoke({"test_path": "tests/"})
    passed = _pytest_passed(output)
    verification = dict(state.get("verification", {}))
    failures = list(verification.get("failures", []))
    attempts = int(verification.get("attempts", 0))
    current_step = int(state.get("current_step", 0))
    plan = state.get("plan", [])
    messages = list(state.get("messages", []))

    if passed:
        inferential = judge_code_quality(
            _current_step_text(state),
            _diff_summary(state),
        ).as_dict()
        HarnessObserver().log_inferential_verification(inferential)
        _record_trial_result(state, True)
        task_complete = current_step >= max(len(plan) - 1, 0)
        if not task_complete:
            current_step += 1
        verification = {
            "passed": True,
            "failures": failures,
            "attempts": attempts,
            "task_complete": task_complete,
            "inferential": inferential,
        }
        next_state: dict[str, Any] = {
            **state,
            "current_step": current_step,
            "verification": verification,
            "output": output if task_complete else state.get("output", ""),
            "messages": messages + [f"Verification passed: step {current_step + 1}"],
        }
        HarnessObserver().log_verification(True, failures, attempts)
        if task_complete:
            HarnessObserver().log_task_complete(cast(AgentState, next_state), True)
        return cast(AgentState, next_state)

    attempts += 1
    _record_trial_result(state, False)
    failure_summary = _summarize_failure(output)
    failures.append(failure_summary)
    verification = {
        "passed": False,
        "failures": failures,
        "attempts": attempts,
        "task_complete": False,
        "inferential": verification.get("inferential", {}),
    }
    next_state = {
        **state,
        "verification": verification,
        "output": failure_summary if attempts >= 3 else state.get("output", ""),
        "messages": messages,
    }
    HarnessObserver().log_verification(False, failures, attempts)
    if attempts >= 3:
        next_state["messages"] = messages + ["Stopped after repeated verification failures."]
        HarnessObserver().log_task_complete(cast(AgentState, next_state), False)
    return cast(AgentState, next_state)


@traced_node("feedback_injector")
def feedback_injector_node(state: AgentState) -> AgentState:
    verification = state.get("verification", {})
    attempts = int(verification.get("attempts", 0))
    failures = list(verification.get("failures", []))
    failure_summary = failures[-1] if failures else "unknown failure"
    message = HumanMessage(
        content=(
            f"Verification failed (attempt {attempts}): {failure_summary}. "
            "Re-read AGENTS.md constraints and fix the issue."
        )
    )
    return {
        **state,
        "messages": state.get("messages", []) + [message],
    }


def _fallback_plan(task: str) -> list[str]:
    return [
        f"Inspect the task requirements: {task}",
        "Make the smallest allowed source or test change.",
        "Run the verification suite and address failures.",
    ]


def _current_step_text(state: AgentState) -> str:
    plan = state.get("plan", [])
    current_step = int(state.get("current_step", 0))
    if not plan:
        return "Current step: no plan available"
    bounded_index = min(current_step, len(plan) - 1)
    return f"Current step: {plan[bounded_index]}"


def _message_content(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    return str(content)


def _pytest_passed(output: str) -> bool:
    lowered = output.lower()
    return "exit_code=0" in lowered and " passed" in lowered and " failed" not in lowered


def _summarize_failure(output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    for line in reversed(lines):
        lowered = line.lower()
        if "failed" in lowered or "error" in lowered or "exit_code=" in lowered:
            return line[:500]
    return output[:500]


def _diff_summary(state: AgentState) -> str:
    edits = state.get("file_edits", {})
    if not edits:
        return "No file edits recorded."
    return "\n".join(f"{path}: {count} edit(s)" for path, count in sorted(edits.items()))


def _estimate_iteration_tokens(state: AgentState) -> int:
    text = " ".join(
        [
            state.get("task", ""),
            " ".join(state.get("plan", [])),
            " ".join(str(getattr(message, "content", message)) for message in state.get("messages", [])[-5:]),
        ]
    )
    return max(128, len(text) // 4)


def reset_circuit_breaker() -> None:
    global DEFAULT_CIRCUIT_BREAKER
    DEFAULT_CIRCUIT_BREAKER = CircuitBreaker()


def _record_trial_result(state: AgentState, passed: bool) -> None:
    if DEFAULT_CIRCUIT_BREAKER.state != CircuitState.HALF_OPEN:
        return
    if passed:
        DEFAULT_CIRCUIT_BREAKER.record_trial_success(state)
    else:
        DEFAULT_CIRCUIT_BREAKER.record_trial_failure(state)


def _record_llm_usage(state: AgentState, usage_metadata: Any) -> AgentState:
    if not usage_metadata:
        return _record_estimated_usage(state)

    input_tokens = int(usage_metadata.get("input_tokens", 0))
    output_tokens = int(usage_metadata.get("output_tokens", 0))
    total_tokens = int(usage_metadata.get("total_tokens", input_tokens + output_tokens))
    budget = dict(state.get("budget", {}))
    budget["tokens_used"] = int(budget.get("tokens_used", 0)) + total_tokens
    budget["cost_usd"] = float(budget.get("cost_usd", 0.0)) + calculate_cost(
        MODEL_NAME,
        input_tokens,
        output_tokens,
    )
    return {**state, "budget": budget}


def _record_estimated_usage(state: AgentState) -> AgentState:
    budget = dict(state.get("budget", {}))
    estimated_tokens = _estimate_iteration_tokens(state)
    budget["tokens_used"] = int(budget.get("tokens_used", 0)) + estimated_tokens
    events = state.get("harness_events", []) + [
        {
            "type": "token_usage",
            "source": "estimated, not measured",
            "tokens": estimated_tokens,
        }
    ]
    return {**state, "budget": budget, "harness_events": events}
