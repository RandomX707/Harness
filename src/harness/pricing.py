"""Model pricing helpers for budget accounting."""

from __future__ import annotations


MODEL_PRICING = {
    "gpt-4o-mini": {
        "input_per_1k": 0.000150,
        "output_per_1k": 0.000600,
    }
}


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    if model not in MODEL_PRICING:
        raise ValueError(f"Unknown model pricing for {model}")
    pricing = MODEL_PRICING[model]
    input_cost = (input_tokens / 1000) * pricing["input_per_1k"]
    output_cost = (output_tokens / 1000) * pricing["output_per_1k"]
    return input_cost + output_cost
