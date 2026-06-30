from __future__ import annotations

import pytest

from harness.pricing import calculate_cost


def test_calculate_cost_uses_input_and_output_rates() -> None:
    assert calculate_cost("gpt-4o-mini", input_tokens=1000, output_tokens=500) == pytest.approx(0.00045)


def test_calculate_cost_unknown_model_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="Unknown model pricing"):
        calculate_cost("unknown-model", input_tokens=1000, output_tokens=1000)
