from __future__ import annotations

import pytest

from chat_alpaca.forecasting import simulate_portfolio_projection


def test_projection_without_volatility_has_known_month_end_values() -> None:
    result = simulate_portfolio_projection(
        current_value=100.0,
        annual_return=0.12,
        annual_volatility=0.0,
        monthly_contribution=10.0,
        horizon_years=1,
        simulations=3,
    )

    expected = 100.0
    monthly_gross_return = 1.12 ** (1 / 12)
    for _ in range(12):
        expected = expected * monthly_gross_return + 10.0
    assert result.annual_percentiles.loc[1].tolist() == pytest.approx([expected] * 5)


def test_projection_is_seeded_and_reports_target_probability() -> None:
    arguments = dict(
        current_value=100_000.0,
        annual_return=0.07,
        annual_volatility=0.15,
        monthly_contribution=500.0,
        horizon_years=2,
        target_value=115_000.0,
        simulations=1_000,
        seed=7,
    )
    first = simulate_portfolio_projection(**arguments)
    second = simulate_portfolio_projection(**arguments)

    assert first.monthly_percentiles.equals(second.monthly_percentiles)
    assert 0 <= first.target_probability <= 1
    assert list(first.annual_percentiles.index) == [0, 1, 2]


@pytest.mark.parametrize(
    "arguments",
    [
        {"current_value": 0},
        {"annual_return": -1},
        {"annual_volatility": -0.1},
        {"monthly_contribution": -1},
        {"horizon_years": 11},
    ],
)
def test_projection_rejects_invalid_assumptions(arguments: dict[str, float]) -> None:
    values = {
        "current_value": 100.0,
        "annual_return": 0.07,
        "annual_volatility": 0.15,
        "monthly_contribution": 0.0,
        "horizon_years": 1,
    }
    values.update(arguments)

    with pytest.raises(ValueError):
        simulate_portfolio_projection(**values)
