from __future__ import annotations

from datetime import date
from decimal import Decimal

import pandas as pd
import pytest

from chat_alpaca.forecasting import (
    ForecastAssumptions,
    build_forecast_request,
    simulate_portfolio_projection,
)
from chat_alpaca.models import HoldingLot, Portfolio


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


def test_forecast_request_discloses_cost_basis_fallback() -> None:
    portfolio = Portfolio(id=1, name="Example", cash=Decimal("5"))
    portfolio.holdings = [
        HoldingLot(
            symbol="ABC",
            shares=Decimal("2"),
            acquired_on=date(2026, 1, 1),
            cost_basis=Decimal("10"),
        )
    ]
    assumptions = ForecastAssumptions(0.07, 0.12, 0, 10)

    request = build_forecast_request([portfolio], pd.DataFrame(), assumptions)

    assert request.current_value == 25
    assert request.valuation_basis == "cost_basis_plus_cash"
    assert any("explicitly uses cost basis" in warning for warning in request.warnings)


def test_forecast_request_does_not_silently_fill_incomplete_prices() -> None:
    portfolio = Portfolio(id=1, name="Example", cash=Decimal("5"))
    portfolio.holdings = [
        HoldingLot(
            symbol="ABC",
            shares=Decimal("2"),
            acquired_on=date(2026, 1, 1),
            cost_basis=Decimal("10"),
        )
    ]
    closes = pd.DataFrame({"OTHER": [10.0]}, index=pd.to_datetime(["2026-01-02"]))

    with pytest.raises(ValueError, match="Incomplete portfolios: Example"):
        build_forecast_request([portfolio], closes, ForecastAssumptions(0.07, 0.12, 0, 10))


def test_forecast_request_uses_common_confirmed_household_value() -> None:
    first = Portfolio(id=1, name="First", cash=Decimal("0"))
    first.holdings = [
        HoldingLot(
            symbol="AAA",
            shares=Decimal("2"),
            acquired_on=date(2026, 1, 1),
            cost_basis=Decimal("80"),
        )
    ]
    second = Portfolio(id=2, name="Second", cash=Decimal("0"))
    second.holdings = [
        HoldingLot(
            symbol="BBB",
            shares=Decimal("3"),
            acquired_on=date(2026, 1, 1),
            cost_basis=Decimal("40"),
        )
    ]
    closes = pd.DataFrame(
        {"AAA": [100.0, 110.0], "BBB": [50.0, float("nan")]},
        index=pd.to_datetime(["2026-01-02", "2026-01-03"]),
    )

    request = build_forecast_request(
        [first, second], closes, ForecastAssumptions(0.07, 0.12, 0, 10)
    )

    assert request.current_value == 350.0
    assert "2026-01-02" in request.coverage
